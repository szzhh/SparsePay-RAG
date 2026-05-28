"""
Core SparsePay-RAG generation ensemble.

Contains:
  - PAV isotonic regression (Sec. 3.3)
  - ITR rejection scoring (Sec. 3.3)
  - DP Contrastive Decoding (Sec. 3.4)
  - Main generation loop ``sparsepay_generate``
"""

import numpy as np
import torch
import torch.nn.functional as F

from .dp import DPExpenseOverflow
from .model_hook_utils import (
    get_model_info, register_hooks, process_all_layers_for_itr, cleanup_hooks,
)


# ===========================================================================
#  PAV (Pool Adjacent Violators) algorithm — isotonic regression
# ===========================================================================

def _pav_isotonic_nondecreasing(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators for monotone non-decreasing isotonic regression.

    Given observations y[0..n-1], returns mu that minimizes
        sum_i (y_i - mu_i)^2
    subject to mu_0 <= mu_1 <= ... <= mu_{n-1}.

    Args:
        y: 1D numpy array of observations.

    Returns:
        1D array of fitted values, same length as y.
    """
    n = len(y)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    sums: list[float] = []
    counts: list[int] = []

    for v in y:
        sums.append(float(v))
        counts.append(1)
        # Merge backward while monotonicity is violated.
        while (len(sums) >= 2
               and (sums[-2] / counts[-2]) > (sums[-1] / counts[-1])):
            s2 = sums.pop(); c2 = counts.pop()
            s1 = sums.pop(); c1 = counts.pop()
            sums.append(s1 + s2)
            counts.append(c1 + c2)

    mu = np.empty(n, dtype=np.float64)
    i = 0
    for s, c in zip(sums, counts):
        avg = s / c
        mu[i:i + c] = avg
        i += c
    return mu


# ===========================================================================
#  ITR: Isotonic Trajectory Reliability (Sec. 3.3)
# ===========================================================================

def compute_itr_rejection(
    layers_list: list[dict],
    alpha: float = 0.2,
    theta: float = 0.5,
) -> tuple[bool, int | None]:
    """Isotonic Trajectory Reliability (ITR) for adaptive private access triggering.

    For each candidate x in the dynamic head vocabulary V_head, we:
      1. Extract its cross-layer probability trajectory p^(l)(x) over the
         trailing window [l_start, L].
      2. Fit isotonic regression mu(x) via PAV.
      3. Score: S(x) = mu_L(x) - RMSE(p(x), mu(x)).
      4. If max S(x) < theta_reject -> trigger private access.

    Args:
        layers_list: Output of `process_all_layers_for_itr` (list of per-layer
            dicts with 'top1000_ids' and 'top1000_probs'). Window length L_win.
        alpha: Head vocabulary truncation ratio (Eq. 3).
        theta: Rejection threshold theta_reject.

    Returns:
        (use_dprag, best_token):
            use_dprag: True = trigger private access, False = emit best_token.
            best_token: argmax-score candidate x* (or None).
    """
    L_win = len(layers_list)
    if L_win == 0:
        return True, None

    # Final (mature) layer probabilities.
    final_layer = layers_list[-1]
    final_probs = torch.tensor(final_layer['top1000_probs'], dtype=torch.float32)
    final_tokens = final_layer['top1000_ids']

    if len(final_probs) == 0:
        return True, None

    # Step 1: Build V_head (Eq. 3)
    max_prob = final_probs.max().item()
    head_threshold = alpha * max_prob
    head_mask = final_probs >= head_threshold
    head_indices = torch.where(head_mask)[0]
    if len(head_indices) == 0:
        return True, None

    # Step 2: Per-layer lookup tables (token_id -> prob).
    layer_lookups = [
        dict(zip(layer['top1000_ids'], layer['top1000_probs']))
        for layer in layers_list
    ]

    # Step 3: Score each candidate x in V_head.
    best_token = None
    best_score = -float('inf')

    for idx in head_indices:
        token_id = int(final_tokens[idx])
        traj = np.fromiter(
            (layer_lookups[l].get(token_id, 0.0) for l in range(L_win)),
            dtype=np.float64, count=L_win,
        )
        mu = _pav_isotonic_nondecreasing(traj)
        residual = float(np.sqrt(np.mean((traj - mu) ** 2)))
        score = float(mu[-1] - residual)          # S(x) = mu_L - RMSE
        if score > best_score:
            best_score = score
            best_token = token_id

    if best_token is None:
        return True, None

    use_dprag = bool(best_score < theta)
    return use_dprag, best_token


# ===========================================================================
#  DP Contrastive Decoding (Sec. 3.4)
# ===========================================================================

def dp_contrastive_decoding(
    args,
    next_token_logits: torch.Tensor,
    dp_engine,
    bt_dict: dict | None = None,
) -> int | None:
    """DP Contrastive Decoding for a single token generation step.

    All branches restrict token selection to V_head for consistency with ITR.

    Strategy (Sec. 3.4):
      1. Compute private delta:  log p_priv^(i) - log p_pub
      2. Clip delta to [-C, C]
      3. Aggregate:  u = log p_pub + mean(clipped delta)
      4. Sample via exponential mechanism (softmax with temperature)

    Args:
        args: Arguments with itr_alpha, etc.
        next_token_logits: [num_models, vocab_size]; last row is public model.
        dp_engine: ClippedLogitsDP instance, or None (no-DP mode).
        bt_dict: Dict with 'best_token' from ITR for ada_rag fallback.

    Returns:
        Sampled token id, or None if budget exhausted (non-ada mode).
    """
    pri_logits = next_token_logits[:-1]   # [N_priv, vocab]
    pub_logits = next_token_logits[-1]    # [vocab]

    # --- Build V_head ---
    alpha = getattr(args, 'itr_alpha', 0.2)
    pub_probs = F.softmax(pub_logits, dim=-1)
    p_top1 = pub_probs.max().item()
    cand_mask = pub_probs >= alpha * p_top1
    cand_indices = torch.where(cand_mask)[0]
    if len(cand_indices) == 0:
        cand_indices = torch.argmax(pub_probs).unsqueeze(0)

    # --- No-DP mode (for debugging / non-private RAG) ---
    if dp_engine is None:
        pri_probs = F.softmax(pri_logits, dim=-1)
        prob_avg = pri_probs.mean(dim=0)
        return int(torch.argmax(prob_avg).item())

    # --- Budget exhausted: fallback to public model ---
    if dp_engine.budget_exhausted:
        if getattr(args, 'ada_rag', False):
            if bt_dict and bt_dict.get('best_token') is not None:
                return int(bt_dict['best_token'])
            return int(torch.argmax(pub_probs).item())
        return None

    # --- DP mode: clip delta, aggregate, sample ---
    try:
        pri_logprobs = F.log_softmax(pri_logits, dim=-1)[:, cand_indices]
        pub_logprobs = F.log_softmax(pub_logits, dim=-1)[cand_indices]
        diff = pri_logprobs - pub_logprobs.unsqueeze(0)          # private delta
        clipped_diff = torch.clamp(diff, -dp_engine.clip_norm, dp_engine.clip_norm)
        avg_clipped_diff = clipped_diff.mean(dim=0)
        adjusted = pub_logprobs + avg_clipped_diff               # u = pub + mean(delta)
        scaled = adjusted / dp_engine.temperature
        probs = F.softmax(scaled, dim=-1)
        idx_in_cand = torch.multinomial(probs, num_samples=1).squeeze(-1)
        token = int(cand_indices[idx_in_cand].item())
        dp_engine.tokens_generated += 1
        dp_engine.check_budget()
        return token

    except DPExpenseOverflow:
        eps, delta = dp_engine.get_dp_expense()
        print(f"[DP] Budget exhausted: eps={eps:.4f}, delta={delta:.2e}")
        if getattr(args, 'ada_rag', False):
            print("[DP] Falling back to public model.")
            if bt_dict and bt_dict.get('best_token') is not None:
                return int(bt_dict['best_token'])
            return int(torch.argmax(pub_probs).item())
        print("[DP] Stopping generation.")
        return None


# ===========================================================================
#  Temperature sampling helper
# ===========================================================================

def _decode_temperature_sampling(logits: torch.Tensor, temperature: float) -> int:
    """Sample a token from logits with temperature scaling."""
    if temperature == 0:
        return int(torch.argmax(logits, dim=-1).item())
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).squeeze(-1).item())


# ===========================================================================
#  Main SparsePay-RAG generation loop
# ===========================================================================

def sparsepay_generate(
    args,
    hf_model,
    tokenizer,
    prompt_list: list[str],
    dp_engine=None,
    bt_dict: dict | None = None,
) -> tuple[list[int], int]:
    """Main SparsePay-RAG auto-regressive generation loop.

    At each decoding step:
      1. Forward pass with hooks captures cross-layer hidden states.
      2. ITR rejection (Sec. 3.3): compute confidence scores and decide
         whether to trigger private access.
      3. If not triggered: emit the ITR best token directly (no privacy cost).
      4. If triggered: DP contrastive decoding (Sec. 3.4).

    Args:
        args: Configuration with n_docs, n_split, decoding_method, itr_*,
              max_length, temperature, ada_rag, hook_flags, etc.
        hf_model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        prompt_list: [private_1, ..., private_N, public].
        dp_engine: ClippedLogitsDP instance (None = no DP).
        bt_dict: Dictionary for tracking DP flags and ITR best_token.

    Returns:
        (generated_token_ids, exit_status):
            exit_status: 0 = normal, 1 = budget exhausted / error.
    """
    device = hf_model.device
    batch_size = len(prompt_list)

    generated_tokens: list[int] = []
    exit_status = 0
    past_key_values = None

    # Reserve space to avoid exceeding position embeddings.
    model_max_len = getattr(hf_model.config, 'max_position_embeddings', 2048)
    max_input_len = model_max_len - args.max_length

    inputs = tokenizer(
        prompt_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len,
    ).to(device)

    current_input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Register hooks for ITR cross-layer trajectory extraction.
    use_ada = getattr(args, 'ada_rag', False)
    if use_ada:
        model_info = get_model_info(hf_model)
        hidden_states_cache, hooks = register_hooks(hf_model, model_info)

    # --- Auto-regressive loop ---
    for _step in range(args.max_length):
        with torch.no_grad():
            outputs = hf_model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]   # [batch, vocab]

        # ---- ITR: adaptive triggering (Sec. 3.3) ----
        use_dprag = True       # default: trigger private access
        best_token = None

        if use_ada:
            # Public prompt is the last batch item.
            pub_batch_item = batch_size - 1
            layers_list = process_all_layers_for_itr(
                model_info=model_info,
                hidden_states_cache=hidden_states_cache,
                batch_item=pub_batch_item,
                selected_L=getattr(args, 'itr_selected_L', None),
            )
            use_dprag, best_token = compute_itr_rejection(
                layers_list=layers_list,
                alpha=getattr(args, 'itr_alpha', 0.2),
                theta=getattr(args, 'itr_theta', 0.5),
            )
            if bt_dict is not None:
                bt_dict['best_token'] = best_token

        # ---- Token selection ----
        if args.n_docs == 0:
            # Non-RAG: just sample from the public logits.
            majority_token = _decode_temperature_sampling(
                next_token_logits.squeeze(0), args.temperature
            )
        elif args.decoding_method == "CD":
            if use_ada and not use_dprag:
                # ITR says public model is confident: emit best_token for free.
                if best_token is not None:
                    majority_token = best_token
                else:
                    majority_token = _decode_temperature_sampling(
                        next_token_logits[-1], args.temperature
                    )
            else:
                majority_token = dp_contrastive_decoding(
                    args, next_token_logits, dp_engine, bt_dict
                )
        else:
            raise NotImplementedError(
                f"Decoding method '{args.decoding_method}' not supported. "
                f"Use 'CD' for SparsePay-RAG."
            )

        if majority_token is None:
            exit_status = 1
            break

        generated_tokens.append(majority_token)

        if majority_token == tokenizer.eos_token_id:
            break

        # Prepare for next step.
        current_input_ids = torch.full(
            (batch_size, 1), majority_token,
            device=device, dtype=torch.long,
        )
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones(batch_size, 1, device=device, dtype=torch.long)],
            dim=1,
        )

    # Cleanup hooks.
    if use_ada:
        cleanup_hooks(hooks)

    return generated_tokens, exit_status