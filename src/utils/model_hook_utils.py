"""
Model hook utilities for the SparsePay-RAG adaptive triggering module.

Provides:
  - `get_model_info`: detect model architecture and return config.
  - `register_hooks` / `cleanup_hooks`: register/remove forward hooks on all layers.
  - `process_all_layers_for_itr`: fast batched extraction of cross-layer
    probability trajectories for isotonic regression fitting (Sec. 3.3).

Supports: OPT, GPT-NeoX (Pythia), GPT-2, Llama, Mistral, Qwen.
"""

import torch
import torch.nn.functional as F


# ===========================================================================
#  Model architecture detection
# ===========================================================================

def get_model_info(model) -> dict:
    """Detect model architecture and return configuration dictionary.

    Returns keys:
        model_type    : str ('opt', 'gpt_neox', 'gpt2', 'llama', 'mistral', 'qwen')
        num_layers    : int
        layer_module_path : list[str]  (navigational path to the layer list)
        decoder_name  : str
        norm_fn       : callable  (final layer norm)
        lm_head       : nn.Module
        hidden_size   : int
    """
    class_name = model.__class__.__name__

    if 'OPT' in class_name:
        return {
            'model_type': 'opt',
            'num_layers': len(model.model.decoder.layers),
            'layer_module_path': ['model', 'decoder', 'layers'],
            'decoder_name': 'decoder',
            'norm_fn': model.model.decoder.final_layer_norm,
            'lm_head': model.lm_head,
            'hidden_size': model.config.hidden_size,
        }

    if 'GPTNeoX' in class_name:                    # Pythia
        return {
            'model_type': 'gpt_neox',
            'num_layers': len(model.gpt_neox.layers),
            'layer_module_path': ['gpt_neox', 'layers'],
            'decoder_name': 'gpt_neox',
            'norm_fn': model.gpt_neox.final_layer_norm,
            'lm_head': model.embed_out,
            'hidden_size': model.config.hidden_size,
        }

    if 'GPT2' in class_name:
        return {
            'model_type': 'gpt2',
            'num_layers': len(model.transformer.h),
            'layer_module_path': ['transformer', 'h'],
            'decoder_name': 'transformer',
            'norm_fn': model.transformer.ln_f,
            'lm_head': model.lm_head,
            'hidden_size': model.config.hidden_size,
        }

    if 'Llama' in class_name:
        return {
            'model_type': 'llama',
            'num_layers': len(model.model.layers) if hasattr(model, 'model') else 0,
            'layer_module_path': ['model', 'layers'] if hasattr(model, 'model')
                                  else ['transformer', 'h'],
            'decoder_name': 'model' if hasattr(model, 'model') else 'transformer',
            'norm_fn': (model.model.norm if hasattr(model, 'model')
                        else model.transformer.ln_f),
            'lm_head': model.lm_head,
            'hidden_size': model.config.hidden_size,
        }

    if 'Mistral' in class_name:
        return {
            'model_type': 'mistral',
            'num_layers': len(model.model.layers),
            'layer_module_path': ['model', 'layers'],
            'decoder_name': 'model',
            'norm_fn': model.model.norm,
            'lm_head': model.lm_head,
            'hidden_size': model.config.hidden_size,
        }

    if 'Qwen' in class_name:
        return {
            'model_type': 'qwen',
            'num_layers': len(model.model.layers),
            'layer_module_path': ['model', 'layers'],
            'decoder_name': 'model',
            'norm_fn': model.model.norm,
            'lm_head': model.lm_head,
            'hidden_size': model.config.hidden_size,
        }

    raise NotImplementedError(f"Model architecture '{class_name}' not supported.")


def _get_layer_module(model, model_info: dict):
    """Navigate to the list of transformer layers."""
    container = model
    for p in model_info['layer_module_path']:
        container = getattr(container, p)
    return container


# ===========================================================================
#  Forward hooks for hidden state collection
# ===========================================================================

def register_hooks(model, model_info: dict, collect_all_batch_items: bool = False):
    """Register forward hooks on all transformer layers to capture hidden states.

    The hooks capture the last-token hidden state from each layer after every
    forward pass. These are consumed by `process_all_layers_for_itr` to build
    cross-layer probability trajectories.

    Args:
        model: HuggingFace model.
        model_info: Result of `get_model_info(model)`.
        collect_all_batch_items: If True, store states for ALL batch items
            keyed as (layer_idx, batch_idx). If False, only store the last
            batch item keyed as layer_idx.

    Returns:
        (hidden_states_cache, hooks):
            hidden_states_cache: dict storing per-layer hidden states.
            hooks: list of hook handles for later cleanup.
    """
    hidden_states_cache = {}
    hooks = []

    def _make_hook(layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            if collect_all_batch_items and hidden.ndim == 3:
                bs = hidden.shape[0]
                for bi in range(bs):
                    hidden_states_cache[(layer_idx, bi)] = hidden[bi, -1, :].detach()
            else:
                if hidden.ndim == 3:
                    vec = hidden[-1, -1, :]
                elif hidden.ndim == 2:
                    vec = hidden[-1, :]
                else:
                    vec = hidden[-1]
                hidden_states_cache[layer_idx] = vec.detach()
        return hook

    layers_module = _get_layer_module(model, model_info)
    for i in range(model_info['num_layers']):
        h = layers_module[i].register_forward_hook(_make_hook(i))
        hooks.append(h)

    return hidden_states_cache, hooks


def cleanup_hooks(hooks: list) -> None:
    """Remove all registered forward hooks."""
    for h in hooks:
        h.remove()


# ===========================================================================
#  Fast batched cross-layer trajectory extraction (for ITR)
# ===========================================================================

def process_all_layers_for_itr(
    model_info: dict,
    hidden_states_cache: dict,
    batch_item: int | None = None,
    selected_L: int | None = None,
    topk: int = 1000,
) -> list[dict]:
    """Fast minimal processor for ITR rejection (Sec. 3.3).

    Stacks trailing ``selected_L`` hidden states and runs ``norm_fn`` +
    ``lm_head`` **once** on the whole stack, then extracts top-k probs and ids
    per layer. This eliminates per-layer Python overhead.

    Args:
        model_info: Architecture info from `get_model_info()`.
        hidden_states_cache: Populated by `register_hooks()`.
        batch_item: Batch index when cache is keyed by (layer_idx, batch_idx).
            ``None`` means use plain ``layer_idx`` keys.
        selected_L: Number of trailing layers to process (window length L_win).
            Defaults to ``num_layers // 2``.
        topk: Number of top-k entries to keep per layer (default 1000).

    Returns:
        List of dicts, each with ``top1000_ids`` (list[int]) and
        ``top1000_probs`` (list[float]). Consumed by `compute_itr_rejection`.
    """
    num_layers = model_info['num_layers']
    if selected_L is None:
        selected_L = num_layers // 2
    selected_L = min(selected_L, num_layers)
    start_layer = num_layers - selected_L

    # Gather hidden states for the trailing window.
    hiddens = []
    for i in range(start_layer, num_layers):
        cache_key = (i, batch_item) if batch_item is not None else i
        if cache_key not in hidden_states_cache:
            continue
        hiddens.append(hidden_states_cache[cache_key])

    if not hiddens:
        return []

    # Stack -> [L_win, hidden_dim]; single norm + lm_head call.
    stacked = torch.stack(hiddens, dim=0)
    norm_fn = model_info['norm_fn']
    lm_head = model_info['lm_head']

    with torch.no_grad():
        stacked_logits = lm_head(norm_fn(stacked))            # [L_win, vocab]
        stacked_probs = F.softmax(stacked_logits, dim=-1)     # [L_win, vocab]
        k = min(topk, stacked_probs.shape[-1])
        top_probs, top_ids = torch.topk(stacked_probs, k, dim=-1)

    # Single batched CPU transfer.
    top_ids_np = top_ids.detach().cpu().numpy()
    top_probs_np = top_probs.detach().cpu().numpy()

    layers_list = [
        {
            "top1000_ids": top_ids_np[l].tolist(),
            "top1000_probs": top_probs_np[l].tolist(),
        }
        for l in range(top_ids_np.shape[0])
    ]
    return layers_list