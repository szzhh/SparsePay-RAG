"""
Prompt builder for SparsePay-RAG.
Supports NQ, TriviaQA, and ChatDoctor datasets.
"""

import random


def build_prompt_list(args, tokenizer, question: str, retrieved_docs_path: str | None = None) -> list[str]:
    """Build prompt list for SparsePay-RAG generation.

    Constructs:
      - One public prompt (no context) for the public model's forward pass
      - Multiple private prompts (with context) for contrastive decoding

    Args:
        args: Configuration with n_docs, n_split, retrieved_docs_dir, etc.
        tokenizer: HuggingFace tokenizer (unused; kept for API compatibility).
        question: The input question string.
        retrieved_docs_path: Path to retrieved documents file (one doc per line).

    Returns:
        List of prompt strings: [private_1, ..., private_N, public].
    """
    is_nq_triviaqa = 'nq' in args.retrieved_docs_dir or 'triviaqa' in args.retrieved_docs_dir
    is_chatdoctor = 'chatdoctor' in args.retrieved_docs_dir

    prompt_list = []

    if args.n_docs > 0:
        # Load and shuffle context documents
        context_text = []
        if retrieved_docs_path:
            with open(retrieved_docs_path, 'r', encoding='utf-8') as f:
                for _ in range(args.n_split * args.n_docs):
                    line = f.readline()
                    if not line:
                        break
                    context_text.append(line.strip())
            random.shuffle(context_text)

        # Build private prompts with context (one per split)
        for split_id in range(args.n_split):
            start = split_id * args.n_docs
            end = (split_id + 1) * args.n_docs
            context_text_per_split = context_text[start:end]

            context_items = []
            for i, ctx in enumerate(context_text_per_split):
                context_items.append(f"{i+1}. " + ctx.split("//")[0])

            if is_nq_triviaqa:
                prompt = _build_nq_triviaqa_prompt_with_context(question, context_items, args)
            elif is_chatdoctor:
                prompt = _build_chatdoctor_prompt_with_context(question, context_items, args)
            else:
                raise ValueError(
                    f"Unknown dataset type. retrieved_docs_dir must contain "
                    f"'nq', 'triviaqa', or 'chatdoctor'. Got: {args.retrieved_docs_dir}"
                )
            prompt_list.append(prompt)

        # Add public prompt (no context) for contrastive decoding baseline
        if args.decoding_method in ["CD", "LA"]:
            if is_nq_triviaqa:
                prompt = _build_nq_triviaqa_prompt_without_context(question, args)
            elif is_chatdoctor:
                prompt = _build_chatdoctor_prompt_without_context(question, args)
            else:
                raise ValueError(
                    f"Unknown dataset type. retrieved_docs_dir must contain "
                    f"'nq', 'triviaqa', or 'chatdoctor'. Got: {args.retrieved_docs_dir}"
                )
            prompt_list.append(prompt)
    else:
        # Non-RAG mode: single public prompt
        if is_nq_triviaqa:
            prompt = _build_nq_triviaqa_prompt_without_context(question, args)
        elif is_chatdoctor:
            prompt = _build_chatdoctor_prompt_without_context(question, args)
        else:
            raise ValueError(
                f"Unknown dataset type. retrieved_docs_dir must contain "
                f"'nq', 'triviaqa', or 'chatdoctor'. Got: {args.retrieved_docs_dir}"
            )
        prompt_list.append(prompt)

    return prompt_list


# ---------------------------------------------------------------------------
#  NQ / TriviaQA prompt templates
# ---------------------------------------------------------------------------

def _build_nq_triviaqa_prompt_with_context(question: str, context_items: list[str], args) -> str:
    context_prompt = "Context: " + "; ".join(context_items) + ".\n"
    if getattr(args, 'new_instruction', False):
        instruction = (
            "Instruction: Give a simple short answer for the question based on the context. "
            "Provide ONLY the answer. Do not repeat the Question and Context. Do not explain.\n"
        )
        qa = f"{question}?\nThe answer is:"
    else:
        instruction = "Instruction: Give a simple short answer for the question based on the context\n"
        qa = f"Question: {question}\nAnswer:"
    return instruction + context_prompt + qa


def _build_nq_triviaqa_prompt_without_context(question: str, args) -> str:
    if getattr(args, 'new_instruction', False):
        instruction = (
            "Instruction: Give a simple short answer for the question. "
            "Provide ONLY the answer. Do not repeat the Question. Do not explain.\n"
        )
        qa = f"{question}?\nThe answer is:"
    else:
        instruction = "Instruction: Give a simple short answer for the question\n"
        qa = f"Question: {question}\nAnswer:"
    return instruction + qa


# ---------------------------------------------------------------------------
#  ChatDoctor prompt templates
# ---------------------------------------------------------------------------

def _build_chatdoctor_prompt_with_context(question: str, context_items: list[str], args) -> str:
    context_prompt = "\n".join(context_items)
    instruction = (
        "Instruction: If you are a doctor, please answer the medical questions "
        "based on the patient's description and the provided Patient-Doctor chat example(s).\n"
    )
    prompt = (
        f"{instruction}"
        f"Patient-Doctor chat example(s): {context_prompt}\n"
        f"Patient: {question}\n"
        f"Doctor:"
    )
    return prompt


def _build_chatdoctor_prompt_without_context(question: str, args) -> str:
    instruction = (
        "Instruction: If you are a doctor, please answer the medical questions "
        "based on the patient's description.\n"
    )
    prompt = f"{instruction}Patient: {question}\nDoctor:"
    return prompt