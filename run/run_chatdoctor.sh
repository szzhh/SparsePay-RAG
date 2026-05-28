#!/usr/bin/env bash
# =============================================================================
#  SparsePay-RAG: ChatDoctor experiment suite
#
#  Runs SparsePay-RAG across 4 backbone models × 4 privacy budgets (eps).
#  Supports: OPT-1.3B, Pythia-1.4B, Mistral-7B, Llama-3.1-8B
#
#  Usage:
#    bash run/run_chatdoctor.sh              # run all
#    bash run/run_chatdoctor.sh opt          # only OPT-1.3B
#    bash run/run_chatdoctor.sh opt 1        # OPT-1.3B with eps=1
# =============================================================================

set -euo pipefail

# --- Configuration ---
EVAL_SET="data/chatdoctor/eval.questions"
GOLD_PATH="data/chatdoctor/eval.json"
RETRIEVED_DIR="data/chatdoctor/retrieval_sbert"
PRED_DIR_BASE="output/chatdoctor"
DEVICE="cuda"
SEED=42
MAX_TOKENS=50

# --- Model list ---
MODELS=(
    "models/opt-1.3b"
    "models/pythia-1.4b"
    "models/Mistral-7B"
    "models/Llama-3.1-8B"
)

# --- (eps, n_split, dp_eps, tau) configurations ---
# Note: ChatDoctor uses higher temperature (tau=1.2) for medical dialogue diversity.
CONFIGS=(
    "1      50         0.1        1.2"
    "5      30         0.2        1.2"
    "10     15         0.5        1.2"
    "100    10         1.0        1.2"
)

# --- ITR thresholds (per model; calibrated via EER) ---
declare -A ITR_THETA
ITR_THETA["models/opt-1.3b"]="0.50"
ITR_THETA["models/pythia-1.4b"]="0.52"
ITR_THETA["models/Mistral-7B"]="0.47"
ITR_THETA["models/Llama-3.1-8B"]="0.56"

# --- Filter by model argument (optional) ---
MODEL_FILTER="${1:-}"
EPS_FILTER="${2:-}"

# --- Retrieval config (SBERT-based) ---
N_RETRIEVAL=50
CLUSTER_N_RETRIEVAL=50
N_COARSE=2000
CORPUS_PATH="data/chatdoctor/corpus.json"
TOPIC_EMB_PATH="data/chatdoctor/topic_embeddings.npy"
DOC_EMB_PATH="data/chatdoctor/doc_embeddings.npy"
CLUSTER_MAP_PATH="data/chatdoctor/cluster_mapping.txt"

run_retrieval() {
    local eps="$1"
    echo "----------------------------------------------------------------------------"
    echo "  Retrieval (DP cluster-size noise) | ChatDoctor | eps=${eps}"
    echo "----------------------------------------------------------------------------"
    python scripts/retrieve_chatdoctor.py \
        --corpus_path "${CORPUS_PATH}" \
        --topic_embeddings_path "${TOPIC_EMB_PATH}" \
        --doc_embeddings_path "${DOC_EMB_PATH}" \
        --cluster_mapping_path "${CLUSTER_MAP_PATH}" \
        --evaluation_set "${EVAL_SET}" \
        --retrieved_docs_dir "${RETRIEVED_DIR}" \
        --n_retrieval "${N_RETRIEVAL}" \
        --cluster_n_retrieval "${CLUSTER_N_RETRIEVAL}" \
        --n_coarse "${N_COARSE}" \
        --total_eps "${eps}" \
        --total_delta 1e-5 \
        --device "${DEVICE}"
    echo ""
}

run_one() {
    local model="$1"
    local eps="$2"
    local n_split="$3"
    local dp_eps="$4"
    local tau="$5"

    local model_name
    model_name=$(basename "$model")
    local theta="${ITR_THETA[$model]:-0.50}"

    local pred_dir="${PRED_DIR_BASE}/${model_name}/eps${eps}"

    echo "----------------------------------------------------------------------------"
    echo "  SparsePay-RAG | ChatDoctor | ${model_name} | eps=${eps} | K=${n_split}"
    echo "----------------------------------------------------------------------------"

    python scripts/generate_chatdoctor.py \
        --model_name_or_path "${model}" \
        --evaluation_set "${EVAL_SET}" \
        --gold_path "${GOLD_PATH}" \
        --retrieved_docs_dir "${RETRIEVED_DIR}" \
        --prediction_dir "${pred_dir}" \
        --n_docs 1 \
        --n_split "${n_split}" \
        --max_length "${MAX_TOKENS}" \
        --total_eps "${eps}" \
        --total_delta 1e-5 \
        --dp_eps "${dp_eps}" \
        --dp_delta 1e-5 \
        --em_temperature "${tau}" \
        --itr_alpha 0.2 \
        --itr_theta "${theta}" \
        --seed "${SEED}" \
        --device "${DEVICE}"

    echo ""
}

# --- Main loop (retrieval per eps, then all models) ---
for cfg in "${CONFIGS[@]}"; do
    read -r eps n_split dp_eps tau <<< "${cfg}"
    if [[ -n "${EPS_FILTER}" && "${eps}" != "${EPS_FILTER}" ]]; then
        continue
    fi

    # Step 1: retrieval with DP cluster-size noise (1/5 of budget)
    run_retrieval "${eps}"

    # Step 2: generation for each model (using remaining 4/5 of budget)
    for model in "${MODELS[@]}"; do
        if [[ -n "${MODEL_FILTER}" && "${model}" != *"${MODEL_FILTER}"* ]]; then
            continue
        fi
        run_one "${model}" "${eps}" "${n_split}" "${dp_eps}" "${tau}"
    done
done

echo "All ChatDoctor experiments complete."