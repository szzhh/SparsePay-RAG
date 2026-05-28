#!/usr/bin/env python3
"""
Retrieve documents for ChatDoctor using SBERT with topic-guided clustering.

This script performs the two-stage retrieval described in Sec. 3.2, adapted
for the ChatDoctor medical dialogue corpus (~115K dialogues).

Stage 1: Route the query to relevant topic clusters using SBERT embeddings.
Stage 2: Retrieve top-K documents via cosine similarity within matched clusters.

Output: One file per query under `<retrieved_docs_dir>/{i}_doc_texts.txt`
"""

import argparse
import logging
import os
import sys
import json

import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Add project root to path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.dp import ClippedLogitsDP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def load_dialogue_corpus(corpus_path: str) -> list[dict]:
    """Load the ChatDoctor dialogue corpus from a JSON file."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_topic_embeddings(emb_path: str) -> np.ndarray:
    """Load pre-computed topic embeddings."""
    return np.load(emb_path)


def load_doc_embeddings(emb_path: str) -> np.ndarray:
    """Load pre-computed document embeddings."""
    return np.load(emb_path)


def read_cluster_mapping(path: str) -> list[list[int]]:
    """Read cluster-to-doc-ids mapping."""
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
#  Two-stage retrieval (SBERT-based)
# ---------------------------------------------------------------------------

def retrieve_clusters_sbert(
    q_embed: np.ndarray,
    topic_embeddings: np.ndarray,
    n_clusters: int,
) -> list[int]:
    """Stage 1: Find top-matching topic clusters by cosine similarity."""
    # Normalize for cosine similarity.
    q_norm = q_embed / (np.linalg.norm(q_embed) + 1e-12)
    t_norm = topic_embeddings / (np.linalg.norm(topic_embeddings, axis=1, keepdims=True) + 1e-12)
    scores = np.dot(t_norm, q_norm)
    top_indices = np.argsort(-scores)[:n_clusters]
    return top_indices.tolist()


def retrieve_docs_sbert(
    q_embed: np.ndarray,
    doc_embeddings: np.ndarray,
    raw_doc_ids: list[int],
    n_retrieval: int,
) -> tuple[list[int], list[int]]:
    """Stage 2: Retrieve top-K docs from candidate set by cosine similarity."""
    candidate_embeds = doc_embeddings[raw_doc_ids]
    q_norm = q_embed / (np.linalg.norm(q_embed) + 1e-12)
    c_norm = candidate_embeds / (np.linalg.norm(candidate_embeds, axis=1, keepdims=True) + 1e-12)
    scores = np.dot(c_norm, q_norm)
    top_k = min(n_retrieval, len(raw_doc_ids))
    top_indices = np.argsort(-scores)[:top_k]
    sorted_doc_ids = [raw_doc_ids[int(i)] for i in top_indices]
    return sorted_doc_ids, top_indices


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage retrieval for ChatDoctor with topic-guided clustering."
    )
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2",
                        help="SBERT model for embeddings.")
    parser.add_argument("--corpus_path", required=True,
                        help="Path to ChatDoctor dialogue corpus JSON.")
    parser.add_argument("--topic_embeddings_path", required=True,
                        help="Path to pre-computed topic embeddings (.npy).")
    parser.add_argument("--doc_embeddings_path", required=True,
                        help="Path to pre-computed document embeddings (.npy).")
    parser.add_argument("--cluster_mapping_path", required=True,
                        help="Path to cluster-to-doc mapping.")
    parser.add_argument("--n_retrieval", type=int, default=50,
                        help="Number of documents to retrieve (final K).")
    parser.add_argument("--cluster_n_retrieval", type=int, default=50,
                        help="Number of clusters to retrieve in stage 1.")
    parser.add_argument("--n_coarse", type=int, default=2000,
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

    # Load embedding model.
    embedder = SentenceTransformer(args.embedding_model, device=args.device)

    # Load corpus and pre-computed embeddings.
    corpus = load_dialogue_corpus(args.corpus_path)
    topic_embeddings = load_topic_embeddings(args.topic_embeddings_path)
    doc_embeddings = load_doc_embeddings(args.doc_embeddings_path)
    cluster_mapping = read_cluster_mapping(args.cluster_mapping_path)
    cluster_sizes = [len(c) for c in cluster_mapping]

    # --- DP cluster-size noise (Sec. 3.2: 1/5 of total budget) ---
    # Convert eps→rho once at script level; all downstream budget ops use rho directly.
    if args.total_eps is not None and args.total_eps > 0:
        helper = ClippedLogitsDP(0, 0, 0, 0, 1, 1)
        rho_total = helper._cdp_rho(args.total_eps, args.total_delta)
        sigma, rho_cls = ClippedLogitsDP.compute_cluster_noise_sigma(
            rho_total, budget_ratio=0.2
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

    os.makedirs(args.retrieved_docs_dir, exist_ok=True)

    with open(args.evaluation_set, "r", encoding="utf-8") as eval_f:
        lines = [l.strip() for l in eval_f if l.strip()]

    for i, question in enumerate(tqdm(lines, desc="Retrieving")):
        q_embed = embedder.encode(question, normalize_embeddings=True)

        # Stage 1: retrieve clusters.
        selected_cids = retrieve_clusters_sbert(
            q_embed, topic_embeddings, args.cluster_n_retrieval
        )

        # Collect candidate doc IDs until N_coarse is reached.
        raw_doc_ids = []
        accumulated = 0
        for cid in selected_cids:
            raw_doc_ids.extend(cluster_mapping[cid])
            accumulated += int(noisy_cluster_sizes[cid])
            if accumulated >= args.n_coarse:
                break

        # Stage 2: fine-grained retrieval.
        sorted_doc_ids, _ = retrieve_docs_sbert(
            q_embed, doc_embeddings, raw_doc_ids, args.n_retrieval
        )

        # Format each dialogue as text.
        doc_texts = []
        for did in sorted_doc_ids:
            entry = corpus[did]
            if isinstance(entry, dict):
                text = f"Patient: {entry.get('input', '')}\nDoctor: {entry.get('output', '')}"
            else:
                text = str(entry)
            doc_texts.append(text)

        output_file = os.path.join(args.retrieved_docs_dir, f'{i}_doc_texts.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(doc_texts))

    logger.info(f"Retrieval complete. Results saved to {args.retrieved_docs_dir}")


if __name__ == "__main__":
    main()