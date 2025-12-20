"""
Task 2: Evaluation Script
Generate predictions and evaluate with BLEU score and code execution tests
"""

import os
import json
import re
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2: Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="JetBrains/Mellum-4b-base")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Path to test dataset JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate (None for all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--is_pretrained", action="store_true", help="Evaluate pretrained model without LoRA")

    return parser.parse_args()


def extract_python_code(text: str) -> str:
    """
    Extract Python code from the generated text.
    Looks for code between ```python and ``` or between ``` and ```
    """
    # Remove the opening ```python if present at the start
    text = text.strip()

    # Pattern 1: Code blocks with ```python
    pattern1 = r'```python\s*(.*?)```'
    matches = re.findall(pattern1, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Pattern 2: Code blocks with just ```
    pattern2 = r'```\s*(.*?)```'
    matches = re.findall(pattern2, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Pattern 3: Text up to closing ``` (for cases where opening ``` was in prompt)
    if '```' in text:
        code = text.split('```')[0].strip()
        return code

    # Pattern 4: No code blocks, return cleaned text
    # Remove common non-code prefixes
    text = re.sub(r'^(Here\'s|Here is|Sure|Certainly).*?:\s*', '', text, flags=re.IGNORECASE)

    return text.strip()


def test_code_execution(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Test if the generated code can be executed without errors
    Returns (success, error_message)
    """
    try:
        # Create a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            temp_file = f.name
            f.write(code)

        # Try to execute the code
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Clean up
        os.unlink(temp_file)

        # Check if execution was successful
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr

    except subprocess.TimeoutExpired:
        os.unlink(temp_file)
        return False, "Execution timeout"
    except Exception as e:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return False, str(e)


def calculate_bleu(reference: str, hypothesis: str) -> float:
    """
    Calculate BLEU score between reference and hypothesis
    """
    # Tokenize
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()

    # Use smoothing function to avoid zero scores
    smoothie = SmoothingFunction().method4

    # Calculate BLEU score
    score = sentence_bleu(
        [reference_tokens],
        hypothesis_tokens,
        smoothing_function=smoothie
    )

    return score


def generate_predictions(model, tokenizer, test_dataset, args):
    """
    Generate predictions for the test dataset
    """
    predictions = []
    references = []
    instructions = []

    # Limit samples if specified
    test_data = test_dataset
    if args.num_samples:
        test_data = test_dataset.select(range(min(args.num_samples, len(test_dataset))))

    print(f"Generating predictions for {len(test_data)} samples...")

    model.eval()
    with torch.no_grad():
        for example in tqdm(test_data):
            instruction = example['instruction']
            input_text = example.get('input', '')
            reference = example['output']

            # Format prompt to match training format
            prompt = instruction
            if input_text and input_text.strip():
                prompt = f"{instruction} {input_text}"
            prompt = f"{prompt}\n```python\n"

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part (remove prompt)
            generated_text = generated_text[len(prompt):]

            predictions.append(generated_text)
            references.append(reference)
            instructions.append(instruction)

    return predictions, references, instructions


def evaluate_predictions(predictions, references, instructions, output_dir):
    """
    Evaluate predictions using BLEU and code execution tests
    """
    results = []
    bleu_scores = []
    execution_success = []

    print("\nEvaluating predictions...")

    for i, (pred, ref, inst) in enumerate(tqdm(zip(predictions, references, instructions))):
        # Extract Python code from prediction
        extracted_code = extract_python_code(pred)

        # Calculate BLEU score (compare extracted code with reference)
        bleu_score = calculate_bleu(ref, extracted_code)
        bleu_scores.append(bleu_score)

        # Test code execution
        can_execute, error_msg = test_code_execution(extracted_code)
        execution_success.append(can_execute)

        # Store result
        result = {
            "index": i,
            "instruction": inst,
            "reference": ref,
            "prediction_raw": pred,
            "prediction_extracted": extracted_code,
            "bleu_score": bleu_score,
            "executable": can_execute,
            "execution_error": error_msg if not can_execute else ""
        }
        results.append(result)

    # Calculate overall metrics
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    execution_rate = sum(execution_success) / len(execution_success)

    metrics = {
        "num_samples": len(predictions),
        "average_bleu": avg_bleu,
        "execution_success_rate": execution_rate,
        "num_executable": sum(execution_success),
        "num_failed": len(execution_success) - sum(execution_success)
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    with open(output_path / "predictions_detailed.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save metrics
    with open(output_path / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Average BLEU score: {metrics['average_bleu']:.4f}")
    print(f"Execution success rate: {metrics['execution_success_rate']:.2%}")
    print(f"Executable codes: {metrics['num_executable']}")
    print(f"Failed executions: {metrics['num_failed']}")
    print("=" * 50)

    return metrics, results


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    print(f"Loading test dataset from {args.test_dataset_path}")
    test_dataset = Dataset.from_json(args.test_dataset_path)
    print(f"Test dataset size: {len(test_dataset)}")

    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path if not args.is_pretrained else args.base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {args.model_path}...")
    if args.is_pretrained:
        # Load base model without fine-tuning
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # Load fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    # Generate predictions
    predictions, references, instructions = generate_predictions(
        model, tokenizer, test_dataset, args
    )

    # Evaluate predictions
    metrics, results = evaluate_predictions(
        predictions, references, instructions, output_dir
    )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()