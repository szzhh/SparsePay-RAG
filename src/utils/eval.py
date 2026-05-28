"""Evaluation metrics: Match Accuracy (NQ/TriviaQA) and BERTScore (ChatDoctor)."""

import ast
import json
from bert_score import score


def normalize_text(s: str) -> str:
    """Lowercase and strip whitespace."""
    return s.lower().strip()


def load_gold_answers(csv_path: str) -> list[list[str]]:
    """Load gold answers from a tab-separated CSV file.

    Each line: <question>\t<list-of-acceptable-answers>
    """
    gold_answers = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"Invalid line: {line}")
            answers = ast.literal_eval(parts[1])
            gold_answers.append([normalize_text(a) for a in answers])
    return gold_answers


def load_predictions(pred_path: str) -> list[str]:
    """Load model predictions, one per line."""
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            preds.append(normalize_text(line))
    return preds


def compute_match_accuracy(gold_path: str, pred_path: str) -> dict:
    """Compute Match Accuracy for NQ and TriviaQA.

    A prediction is correct if it contains any acceptable gold answer.
    """
    gold_answers = load_gold_answers(gold_path)
    predictions = load_predictions(pred_path)

    assert len(gold_answers) == len(predictions), \
        f"Length mismatch: {len(gold_answers)} vs {len(predictions)}"

    match_count = 0
    for golds, pred in zip(gold_answers, predictions):
        if not pred:
            continue
        for ans in golds:
            if ans in pred:
                match_count += 1
                break

    acc = match_count / len(gold_answers)
    print("-" * 30)
    print(f"Match Accuracy: {acc:.4f}  ({match_count}/{len(gold_answers)})")
    print("-" * 30)
    return {
        "metric": "match_accuracy",
        "value": acc,
        "num_samples": len(gold_answers),
        "num_matches": match_count,
    }


def compute_bertscore(gold_path: str, pred_path: str, lang: str = 'en') -> dict | None:
    """Compute BERTScore F1 for ChatDoctor-style evaluation.

    The gold file is a JSON list of dicts with an 'output' field.
    """
    refs = []
    try:
        with open(gold_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if 'output' in item:
                    refs.append(item['output'])
                else:
                    print("Warning: 'output' field not found in a data entry, skipped.")
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

    cands = []
    try:
        with open(pred_path, 'r', encoding='utf-8') as f:
            cands = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error reading TXT file: {e}")
        return None

    if len(refs) != len(cands):
        print(f"⚠️  Warning: Mismatch in number of entries! refs={len(refs)}, preds={len(cands)}")
        min_len = min(len(refs), len(cands))
        refs = refs[:min_len]
        cands = cands[:min_len]
        print(f"Truncated to first {min_len} entries.")

    print(f"Calculating BERTScore for {len(cands)} entries...")

    P, R, F1 = score(cands, refs, lang=lang, model_type='models/roberta-large',
                     num_layers=17, verbose=True)
    mean_p = P.mean().item()
    mean_r = R.mean().item()
    mean_f1 = F1.mean().item()

    print("-" * 30)
    print(f"BERTScore F1: {mean_f1:.4f}  (P={mean_p:.4f}, R={mean_r:.4f})")
    print("-" * 30)
    return {
        "metric": "bertscore_f1",
        "value": mean_f1,
        "precision": mean_p,
        "recall": mean_r,
    }


def evaluate(gold_path: str, pred_path: str, metric: str = "match_accuracy",
             lang: str = 'en') -> dict | None:
    """Unified evaluation entry point.

    Args:
        gold_path: Path to gold labels.
        pred_path: Path to predictions.
        metric: "match_accuracy" or "bertscore".
        lang: Language for BERTScore.

    Returns:
        Dict with metric results, or None on failure.
    """
    if metric == "match_accuracy":
        return compute_match_accuracy(gold_path, pred_path)
    elif metric == "bertscore":
        return compute_bertscore(gold_path, pred_path, lang=lang)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'match_accuracy' or 'bertscore'.")