"""Model loading utilities for SparsePay-RAG.

Supports OPT, Pythia (GPT-NeoX), Llama, Mistral, and Qwen architectures.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_hf(checkpoint: str, device: str = "cuda"):
    """Load a HuggingFace causal LM and its tokenizer.

    Args:
        checkpoint: Model name or path (e.g. 'facebook/opt-1.3b').
        device: Device to load the model on.

    Returns:
        (model, tokenizer) tuple.
    """
    model_kwargs = {}
    pad_with_eos = False

    checkpoint_lower = checkpoint.lower()
    if any(k in checkpoint_lower for k in ['llama', 'pythia', 'gpt', 'qwen', 'mistral']):
        pad_with_eos = True
    if 'llama-2' in checkpoint_lower:
        model_kwargs = {'low_cpu_mem_usage': True, 'torch_dtype': torch.float16}

    model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if pad_with_eos:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return model, tokenizer


def get_model_max_length(model) -> int:
    """Get the maximum position embeddings length of a model."""
    return getattr(model.config, 'max_position_embeddings', 2048)