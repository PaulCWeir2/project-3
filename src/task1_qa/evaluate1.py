"""
Evaluation script for Task 1: Question Answering
"""
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from peft import PeftModel
from datasets import Dataset
from tqdm import tqdm
from typing import Dict, List, Tuple
import string
import re
from src.task1_qa.preprocess1 import prepare_qa_dataset
from src.task1_qa.config1 import QATrainingConfig


def normalize_answer(s: str) -> str:
    """
    Normalize answer text for comparison.
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """
    Calculate exact match score (1 if match, 0 otherwise).

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1 if exact match (after normalization), 0 otherwise
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def get_answer_from_logits(
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer: AutoTokenizer,
        max_answer_length: int = 30
) -> str:
    """
    Extract answer text from start and end logits.

    Args:
        start_logits: Start position logits
        end_logits: End position logits
        input_ids: Input token IDs
        tokenizer: Tokenizer
        max_answer_length: Maximum answer length in tokens

    Returns:
        Predicted answer string
    """
    # Get the most likely start and end positions
    start_idx = torch.argmax(start_logits).item()
    end_idx = torch.argmax(end_logits).item()

    # Ensure valid span
    if end_idx < start_idx:
        end_idx = start_idx

    if end_idx - start_idx + 1 > max_answer_length:
        end_idx = start_idx + max_answer_length - 1

    # Extract answer tokens
    answer_tokens = input_ids[start_idx:end_idx + 1]

    # Decode to text
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer.strip()


def evaluate_qa_model(
        model_path: str,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer = None,
        device: str = None,
        is_peft: bool = True
) -> Dict[str, float]:
    """
    Evaluate the QA model on test dataset.

    Args:
        model_path: Path to saved model
        test_dataset: Test dataset
        tokenizer: Tokenizer (will be loaded if not provided)
        device: Device to use (cuda/cpu)
        is_peft: Whether model is a PEFT model

    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    print(f"Loading model from {model_path}")
    if is_peft:
        base_model = AutoModelForQuestionAnswering.from_pretrained(
            "roberta-base"  # Load base model first
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    model.to(device)
    model.eval()

    # Evaluation
    predictions = []
    ground_truths = []
    exact_matches = []

    print("Evaluating model...")
    with torch.no_grad():
        for example in tqdm(test_dataset, desc="Evaluating"):
            # Prepare inputs
            input_ids = torch.tensor([example["input_ids"]]).to(device)
            attention_mask = torch.tensor([example["attention_mask"]]).to(device)

            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Extract answer
            pred_answer = get_answer_from_logits(
                outputs.start_logits[0],
                outputs.end_logits[0],
                input_ids[0],
                tokenizer
            )

            # Get ground truth (reconstruct from positions)
            start_pos = example["start_positions"]
            end_pos = example["end_positions"]

            if start_pos == 0 and end_pos == 0:  # CLS index means no answer
                true_answer = ""
            else:
                true_tokens = example["input_ids"][start_pos:end_pos + 1]
                true_answer = tokenizer.decode(true_tokens, skip_special_tokens=True)

            predictions.append(pred_answer)
            ground_truths.append(true_answer)

            # Calculate exact match
            em_score = exact_match_score(pred_answer, true_answer)
            exact_matches.append(em_score)

    # Calculate overall metrics
    avg_exact_match = np.mean(exact_matches) * 100

    results = {
        "exact_match": avg_exact_match,
        "num_examples": len(test_dataset),
        "predictions_sample": predictions[:10],
        "ground_truths_sample": ground_truths[:10],
    }

    print(f"\nEvaluation Results:")
    print(f"Exact Match: {avg_exact_match:.2f}%")
    print(f"Number of examples: {len(test_dataset)}")

    return results, predictions, ground_truths


def save_evaluation_results(
        results: Dict,
        predictions: List[str],
        ground_truths: List[str],
        output_path: str
):
    """
    Save evaluation results to JSON file.
    """
    output_data = {
        "metrics": results,
        "predictions": predictions,
        "ground_truths": ground_truths,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    from preprocess1 import prepare_qa_dataset
    from config1 import QATrainingConfig

    config = QATrainingConfig()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Prepare dataset
    datasets = prepare_qa_dataset(config, tokenizer)

    # Evaluate
    results, preds, truths = evaluate_qa_model(
        model_path=config.output_dir,
        test_dataset=datasets["test"],
        tokenizer=tokenizer,
    )

    # Save results
    save_evaluation_results(
        results,
        preds,
        truths,
        os.path.join(config.results_dir, "evaluation_results.json")
    )