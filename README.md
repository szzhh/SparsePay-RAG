# SparsePay-RAG: On-Demand Privacy Budget Payment for Differentially Private RAG

Official implementation of the paper
**"Only Pay What You Must Spend: On-Demand Privacy Budget Payment for Differentially Private RAG"**.

> SparsePay-RAG sparsifies privacy budget consumption across three orthogonal dimensions:
> **(1)** localized two-stage retrieval via public topic-guided clustering (Sec 3.2),
> **(2)** adaptive private-access triggering via isotonic cross-layer trajectory fitting (Sec 3.3),
> **(3)** per-token budget compression via DP contrastive decoding (Sec 3.4).

---

## Project Structure

```
SparsePay-RAG/
├── src/                            # Core library (SparsePay-RAG engine)
│   └── utils/
│       ├── dp.py                   # ClippedLogitsDP: zCDP accounting & exponential mechanism
│       ├── ensemble.py             # PAV isotonic regression, ITR rejection, DP contrastive
│       │                           #   decoding, main autoregressive generation loop
│       ├── model_hook_utils.py     # Model architecture detection, forward hooks,
│       │                           #   fast batched ITR trajectory extraction
│       ├── model.py                # HuggingFace model loading (OPT/Pythia/Llama/Mistral/Qwen)
│       ├── prompt_builder.py       # Prompt templates for NQ, TriviaQA, ChatDoctor
│       ├── eval.py                 # Match Accuracy (NQ/TriviaQA) & BERTScore F1 (ChatDoctor)
│       └── random_seed.py          # Reproducibility (Python + NumPy + PyTorch)
│
├── scripts/                        # Executable scripts (one per dataset)
│   ├── retrieve_nq.py              # Two-stage DPR retrieval for NQ
│   ├── retrieve_triviaqa.py        # Two-stage DPR retrieval for TriviaQA
│   ├── retrieve_chatdoctor.py      # Two-stage SBERT retrieval for ChatDoctor
│   ├── generate_nq.py              # SparsePay-RAG generation for NQ
│   ├── generate_triviaqa.py        # SparsePay-RAG generation for TriviaQA
│   └── generate_chatdoctor.py      # SparsePay-RAG generation for ChatDoctor
│
├── run/                            # Shell launch scripts
│   ├── run_nq.sh                   # NQ
│   ├── run_triviaqa.sh             # TriviaQA
│   └── run_chatdoctor.sh           # ChatDoctor
│
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Expected data layout:

```
data/
├── nq/
│   ├── nq-dev-small.questions          # One question per line
│   ├── nq-dev-small.qa.csv             # TSV: question\t[answer_list]
│   └── retrieval_dpr/                  # Output of retrieve_nq.py
├── triviaqa/
│   ├── triviaqa-dev-small.questions
│   ├── triviaqa-dev-small.qa.csv
│   └── retrieval_dpr/                  # Output of retrieve_triviaqa.py
└── chatdoctor/
    ├── eval.questions                   # One question per line
    ├── eval.json                        # [{"input":..., "output":...}, ...]
    └── retrieval_sbert/                # Output of retrieve_chatdoctor.py
```

### 3. Run Retrieval (Stage 1: cluster routing; Stage 2: fine-grained search)

```bash
# NQ (with DP cluster-size noise; omit --total_eps for non-DP retrieval)
python scripts/retrieve_nq.py \
    --evaluation_set data/nq/nq-dev-small.questions \
    --retrieved_docs_dir data/nq/retrieval_dpr \
    --n_retrieval 50 --cluster_n_retrieval 100 --n_coarse 10000 \
    --total_eps 1 --total_delta 1e-5

# TriviaQA
python scripts/retrieve_triviaqa.py \
    --evaluation_set data/triviaqa/triviaqa-dev-small.questions \
    --retrieved_docs_dir data/triviaqa/retrieval_dpr \
    --n_retrieval 50 --cluster_n_retrieval 100 --n_coarse 10000 \
    --total_eps 1 --total_delta 1e-5

# ChatDoctor
python scripts/retrieve_chatdoctor.py \
    --corpus_path data/chatdoctor/corpus.json \
    --topic_embeddings_path data/chatdoctor/topic_embeddings.npy \
    --doc_embeddings_path data/chatdoctor/doc_embeddings.npy \
    --cluster_mapping_path data/chatdoctor/cluster_mapping.txt \
    --evaluation_set data/chatdoctor/eval.questions \
    --retrieved_docs_dir data/chatdoctor/retrieval_sbert \
    --n_retrieval 50 --cluster_n_retrieval 50 --n_coarse 2000 \
    --total_eps 1 --total_delta 1e-5
```

Note: The retrieval scripts convert the supplied `--total_eps`/`--total_delta` (ε,δ) into zCDP (`rho_total`), and allocate the budget in rho-space by default at a 1:4 ratio (rho_cls:rho_gen) between retrieval and generation. The cluster-noise standard deviation for retrieval is determined by rho_cls as σ = sqrt(1/(2 * rho_cls)).

### 4. Run Generation (SparsePay-RAG)

**Single model, single budget** (e.g., OPT-1.3B with ε=1 on NQ):

```bash
python scripts/generate_nq.py \
    --model_name_or_path models/opt-1.3b \
    --evaluation_set data/nq/nq-dev-small.questions \
    --gold_path data/nq/nq-dev-small.qa.csv \
    --retrieved_docs_dir data/nq/retrieval_dpr \
    --prediction_dir output/nq/opt-1.3b/eps1 \
    --n_docs 1 --n_split 50 \
    --total_eps 1 --dp_eps 0.1 --dp_delta 1e-5 \
    --em_temperature 0.4 --itr_alpha 0.2 --itr_theta 0.50 \
    --device cuda
```

Note: `--dp_eps`/`--dp_delta` are the external (ε,δ) parameters for per-token privatization; the scripts convert them to `rho_per_token`. The scripts also convert `--total_eps`/`--total_delta` to `rho_total` and allocate `target_rho` in rho-space (default 20% retrieval, 80% generation). The internal class `ClippedLogitsDP` performs accounting using `rho_per_token` and `target_rho`; logs primarily report rho values (which can be converted back to (ε,δ) for reporting).

### 5. Run Full Experiment Suite

```bash
# All models & budgets for each dataset
bash run/run_nq.sh
bash run/run_triviaqa.sh
bash run/run_chatdoctor.sh

# Or single model / single budget:
bash run/run_nq.sh opt 1          # OPT-1.3B, ε=1 only
bash run/run_nq.sh llama 10       # Llama-3.1-8B, ε=10 only
bash run/run_chatdoctor.sh mistral 5
```
