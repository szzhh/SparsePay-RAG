#!/usr/bin/env python3
"""
SparsePay-RAG generation script for Natural Questions (NQ).

Usage:
    python scripts/generate_nq.py \
        --model_name_or_path models/opt-1.3b \
        --evaluation_set data/nq/nq-dev-small.questions \
        --gold_path data/nq/nq-dev-small.qa.csv \
        --retrieved_docs_dir data/nq/retrieval_dpr \
        --prediction_dir output/nq \
        --n_docs 1 --n_split 50 \
        --total_eps 1 --dp_eps 0.1 --dp_delta 1e-5 \
        --itr_alpha 0.2 --itr_theta 0.5 \
        --em_temperature 0.4 --device cuda
"""

import argparse
import logging
import os
import sys

import torch
from tqdm import tqdm

# Add project root to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.random_seed import set_seed
from src.utils.model import load_model_hf
from src.utils.prompt_builder import build_prompt_list
from src.utils.ensemble import sparsepay_generate
from src.utils.dp import ClippedLogitsDP
from src.utils.eval import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="SparsePay-RAG generation for Natural Questions"
    )

    # --- Paths ---
    parser.add_argument("--model_name_or_path", required=True,
                        help="HuggingFace model name or local path.")
    parser.add_argument("--evaluation_set", required=True,
                        help="File with one question per line.")
    parser.add_argument("--gold_path", required=True,
                        help="Path to gold answers (tab-separated CSV).")
    parser.add_argument("--retrieved_docs_dir", required=True,
                        help="Directory with retrieved doc text files.")
    parser.add_argument("--prediction_dir", required=True,
                        help="Directory to save predictions.")

    # --- Retrieval config ---
    parser.add_argument("--n_docs", type=int, default=1,
                        help="Documents per private prompt split.")
    parser.add_argument("--n_split", type=int, default=50,
                        help="Number of prompt splits (K in the paper).")

    # --- Generation config ---
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for non-DP sampling (0 = greedy).")

    # --- SparsePay-RAG flags ---
    parser.add_argument("--ada_rag", action="store_true", default=True,
                        help="Enable adaptive RAG triggering (ITR). Default: True.")
    parser.add_argument("--itr_alpha", type=float, default=0.2,
                        help="Head vocab truncation ratio alpha.")
    parser.add_argument("--itr_theta", type=float, default=0.5,
                        help="ITR rejection threshold theta_reject.")
    parser.add_argument("--itr_selected_L", type=int, default=None,
                        help="Number of trailing layers for ITR (None = L/2).")

    # --- DP config ---
    parser.add_argument("--total_eps", type=float, default=1.0,
                        help="Total epsilon budget per query.")
    parser.add_argument("--total_delta", type=float, default=1e-5,
                        help="Total delta budget per query.")
    parser.add_argument("--dp_eps", type=float, default=0.1,
                        help="Epsilon per DP token generation step.")
    parser.add_argument("--dp_delta", type=float, default=1e-5,
                        help="Delta per DP token generation step.")
    parser.add_argument("--em_temperature", type=float, default=0.4,
                        help="Temperature tau for exponential mechanism sampling.")

    # --- Misc ---
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_predictions", action="store_true", default=True)

    args = parser.parse_args()

    # Fixed config for SparsePay-RAG (paper defaults).
    args.decoding_method = "CD"         # Contrastive Decoding
    args.decoding_framework = "hf"      # HuggingFace (full logit access)
    args.new_instruction = False        # Use paper prompt template

    set_seed(args.seed)
    os.makedirs(args.prediction_dir, exist_ok=True)

    # --- Load model ---
    logger.info(f"Loading model: {args.model_name_or_path}")
    hf_model, tokenizer = load_model_hf(args.model_name_or_path, device=args.device)

    # --- Initialize DP engine ---
    # Budget split in rho-space (paper Sec. 4, App B.5):
    #   rho_total = rho_cls + rho_gen, where rho_cls = 0.2 * rho_total (1:4 ratio).
    #   The generation script receives 4/5 of rho_total as its privacy budget.
    # All budget tracking is done in rho-space; no eps->rho->eps conversion.
    helper = ClippedLogitsDP(0, 0, 0, 0, 1, 1)
    rho_total = helper._cdp_rho(args.total_eps, args.total_delta)
    rho_cls = 0.2 * rho_total
    generation_rho = 0.8 * rho_total
    # Per-token rho from dp_eps (still passed as CLI arg for compatibility):
    rho_per_token = helper._cdp_rho(args.dp_eps, args.dp_delta)
    logger.info(
        f"Budget split (1:4 rho-space): "
        f"rho_total={rho_total:.6f}, rho_cls={rho_cls:.6f}, "
        f"rho_gen={generation_rho:.6f}, rho_token={rho_per_token:.6f}"
    )
    dp_engine = ClippedLogitsDP(
        rho_per_token=rho_per_token,
        target_rho=generation_rho,
        target_delta=args.total_delta,
        num_private_models=args.n_split,    # K in the paper
        temperature=args.em_temperature,
        fail_mode='stop',
    )
    logger.info(f"DP engine: C={dp_engine.clip_norm:.4f}, "
                f"rho_token={rho_per_token:.6f}, K={args.n_split}, tau={args.em_temperature}")

    # --- Read questions ---
    with open(args.evaluation_set, "r", encoding="utf-8") as f:
        questions = [l.strip() for l in f if l.strip()]

    predictions = []
    exit_stats = {"normal": 0, "budget_exhausted": 0}

    for i, question in enumerate(tqdm(questions, desc="Generating")):
        # Build prompts: private prompts + public prompt.
        retrieved_docs_path = os.path.join(
            args.retrieved_docs_dir, f'{i}_doc_texts.txt'
        )
        prompt_list = build_prompt_list(args, tokenizer, question, retrieved_docs_path)

        # SparsePay-RAG token-level dictionary.
        bt_dict: dict = {'best_token': None}

        # Generate.
        token_ids, exit_status = sparsepay_generate(
            args, hf_model, tokenizer, prompt_list,
            dp_engine=dp_engine, bt_dict=bt_dict,
        )

        answer = tokenizer.decode(token_ids, skip_special_tokens=True).replace("\n", "\t")
        predictions.append(answer)

        if exit_status == 0:
            exit_stats["normal"] += 1
        else:
            exit_stats["budget_exhausted"] += 1

    # --- Save predictions ---
    pred_path = os.path.join(args.prediction_dir, "predictions.txt")
    with open(pred_path, 'w', encoding='utf-8') as f:
        for p in predictions:
            f.write(p + '\n')

    # --- Evaluate ---
    logger.info("Evaluating Match Accuracy...")
    result = evaluate(args.gold_path, pred_path, metric="match_accuracy")

    # --- Summary ---
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Budget (1:4 split): total_eps={args.total_eps}, "
                f"rho_cls={rho_cls:.6f}, rho_gen={generation_rho:.6f}")
    logger.info(f"Delta={args.total_delta}, K={args.n_split}")
    logger.info(f"Per-token rho={rho_per_token:.6f}, tau={args.em_temperature}")
    logger.info(f"ITR alpha={args.itr_alpha}, theta={args.itr_theta}")
    logger.info(f"Normal exits: {exit_stats['normal']}")
    logger.info(f"Budget-exhausted: {exit_stats['budget_exhausted']}")
    if result:
        logger.info(f"{result['metric']}: {result['value']:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()