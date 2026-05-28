#!/usr/bin/env python3
"""
Retrieve documents for TriviaQA using DPR with topic-guided clustering.

This script performs the two-stage retrieval described in Sec. 3.2:
  Stage 1: Route the query to relevant topic clusters (cluster retrieval).
  Stage 2: Retrieve top-K documents from matched clusters (fine-grained retrieval).

Output: One file per query under `<retrieved_docs_dir>/{i}_doc_texts.txt`
"""

import argparse
import logging
import os
import sys

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk

from transformers import RagRetriever, RagSequenceForGeneration
import transformers.utils.import_utils

transformers.utils.import_utils._faiss_available = True

# Add project root to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.dp import ClippedLogitsDP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def read_cluster_mapping(path: str) -> list[list[int]]:
    """Read cluster-to-doc-ids mapping.

    Each line: comma-separated doc indices belonging to that cluster.
    """
    mapping = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                mapping.append([])
            else:
                mapping.append([int(x.strip()) for x in line.split(',')])
    return mapping


# ---------------------------------------------------------------------------
#  Two-stage retrieval
# ---------------------------------------------------------------------------

def retrieve_clusters(args, cluster_retriever, question_encoder, question: str):
    """Stage 1: Retrieve top-matching topic clusters for a query."""
    inputs_dict = cluster_retriever.question_encoder_tokenizer(
        question, return_tensors="pt"
    )
    input_ids = inputs_dict.input_ids.to(args.device)
    q_hidden = question_encoder(input_ids)[0]

    clusters_dict = cluster_retriever(
        input_ids.cpu().numpy(),
        q_hidden.detach().cpu().numpy(),
        return_tensors="pt",
    )
    scores = torch.bmm(
        q_hidden.unsqueeze(1).detach().cpu(),
        clusters_dict["retrieved_doc_embeds"].float().transpose(1, 2),
    ).squeeze(1).squeeze(0)
    _, indices = torch.sort(scores, descending=True)
    cluster_ids = clusters_dict['doc_ids'].squeeze(0)[indices]
    return cluster_ids


def retrieve_docs(args, text_dataset, question_encoder, question: str,
                  raw_doc_ids: list[int]) -> tuple[list[int], list[str]]:
    """Stage 2: Retrieve top-K documents from candidate set by embedding similarity."""
    inputs = question_encoder.tokenizer(question, return_tensors="pt")
    input_ids = inputs.input_ids.to(args.device)
    with torch.no_grad():
        q_embed = question_encoder(input_ids)[0]

    candidate_subset = text_dataset.select(raw_doc_ids)
    doc_embeds = torch.tensor(
        np.array(candidate_subset["embeddings"]), dtype=torch.float32
    ).to(args.device)

    scores = torch.matmul(q_embed, doc_embeds.T).squeeze(0)
    _, sorted_indices = torch.sort(scores, descending=True)
    top_k = min(args.n_retrieval, len(raw_doc_ids))
    sorted_indices = sorted_indices[:top_k].cpu().numpy()

    final_doc_ids = [raw_doc_ids[int(i)] for i in sorted_indices]
    final_doc_texts = [candidate_subset[int(i)]["text"] for i in sorted_indices]
    return final_doc_ids, final_doc_texts


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage retrieval for TriviaQA with topic-guided clustering."
    )
    parser.add_argument("--retriever_checkpoint", default="models/rag-sequence-nq",
                        help="Path to DPR retriever checkpoint.")
    parser.add_argument("--index_name", default="custom",
                        choices=["custom", "exact", "compressed", "legacy"])
    parser.add_argument("--text_index_path", default="./rag_database/wiki_dpr/text/embeddings.faiss")
    parser.add_argument("--text_passages_path", default="./rag_database/wiki_dpr/text")
    parser.add_argument("--cluster_index_path", default="./rag_database/wiki_dpr/cluster/embeddings.faiss")
    parser.add_argument("--cluster_passages_path", default="./rag_database/wiki_dpr/cluster")
    parser.add_argument("--cluster_mapping_path", default="data/wiki/clustering_result.txt",
                        help="Path to cluster-to-doc mapping.")
    parser.add_argument("--n_retrieval", type=int, default=50,
                        help="Number of documents to retrieve (final K).")
    parser.add_argument("--cluster_n_retrieval", type=int, default=100,
                        help="Number of clusters to retrieve in stage 1.")
    parser.add_argument("--n_coarse", type=int, default=10000,
                        help="Coarse candidate pool size (N_coarse).")
    parser.add_argument("--evaluation_set", required=True,
                        help="File with one question per line.")
    parser.add_argument("--retrieved_docs_dir", required=True,
                        help="Directory to save retrieved doc text files.")
    parser.add_argument("--total_eps", type=float, default=None,
                        help="Total epsilon budget (for cluster-size DP noise; "
                             "1/5 of this is spent on retrieval, 4/5 on generation).")
    parser.add_argument("--total_delta", type=float, default=1e-5,
                        help="Total delta budget for cluster-size DP noise.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for cluster-size noise.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data.
    text_dataset = load_from_disk(args.text_passages_path)
    cluster_mapping = read_cluster_mapping(args.cluster_mapping_path)
    cluster_sizes = [len(c) for c in cluster_mapping]

    # --- DP cluster-size noise (Sec. 3.2: 1/5 of total budget) ---
    if args.total_eps is not None and args.total_eps > 0:
        sigma, rho_cls = ClippedLogitsDP.compute_cluster_noise_sigma(
            args.total_eps, args.total_delta, budget_ratio=0.2
        )
        rng = np.random.default_rng(args.seed if hasattr(args, 'seed') else 42)
        noisy_cluster_sizes = np.maximum(
            np.array(cluster_sizes, dtype=np.float64) + rng.normal(0, sigma, len(cluster_sizes)),
            0.0,
        ).astype(np.int64)
        logger.info(
            f"Cluster-size DP: rho_cls={rho_cls:.6f}, sigma={sigma:.2f}, "
            f"raw max={max(cluster_sizes)}, noisy max={max(noisy_cluster_sizes)}"
        )
    else:
        noisy_cluster_sizes = np.array(cluster_sizes, dtype=np.int64)
        logger.warning("No total_eps provided; cluster sizes are released WITHOUT DP noise.")

    # Load cluster retriever (stage 1).
    cluster_kwargs = {
        'n_docs': args.cluster_n_retrieval,
        "index_name": args.index_name,
        "passages_path": args.cluster_passages_path,
        "index_path": args.cluster_index_path,
    }
    cluster_retriever = RagRetriever.from_pretrained(
        args.retriever_checkpoint, **cluster_kwargs
    )
    cluster_retriever.init_retrieval()
    q_encoder = RagSequenceForGeneration.from_pretrained(
        args.retriever_checkpoint, **cluster_kwargs
    ).question_encoder.to(args.device)
    q_encoder.tokenizer = cluster_retriever.question_encoder_tokenizer

    os.makedirs(args.retrieved_docs_dir, exist_ok=True)

    with open(args.evaluation_set, "r", encoding="utf-8") as eval_f:
        lines = [l.strip() for l in eval_f if l.strip()]

    for i, question in enumerate(tqdm(lines, desc="Retrieving")):
        # Stage 1: retrieve clusters.
        cluster_ids = retrieve_clusters(args, cluster_retriever, q_encoder, question)

        # Collect candidate doc IDs until N_coarse is reached.
        selected_cids = []
        accumulated = 0
        for cid_t in cluster_ids:
            cid = int(cid_t.item())
            selected_cids.append(cid)
            accumulated += int(noisy_cluster_sizes[cid])
            if accumulated >= args.n_coarse:
                break

        raw_doc_ids = []
        for cid in selected_cids:
            raw_doc_ids.extend(cluster_mapping[cid])

        # Stage 2: fine-grained retrieval.
        doc_ids, doc_texts = retrieve_docs(
            args, text_dataset, q_encoder, question, raw_doc_ids
        )

        output_file = os.path.join(args.retrieved_docs_dir, f'{i}_doc_texts.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(doc_texts))

    logger.info(f"Retrieval complete. Results saved to {args.retrieved_docs_dir}")


if __name__ == "__main__":
    main()