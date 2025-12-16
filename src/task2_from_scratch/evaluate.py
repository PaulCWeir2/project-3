"""
Evaluation script for Task 2
"""
import os
import json
import re
import subprocess
import tempfile
from typing import List, Tuple, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from config import Task2Config


def extract_python_code(text: str) -> str:
    """
    Extract Python code from generated text.
    Handles markdown code blocks and plain code.
    """
    # Try markdown with python specifier
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try markdown without specifier
    match = re.search(r'```\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return as-is if no code blocks found
    return text.strip()


def calculate_bleu(prediction: str, reference: str) -> float:
    """Calculate BLEU score between prediction and reference."""
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    smoothing = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)

    return score


def test_code_execution(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Test if code can execute without errors.

    Returns:
        (success, error_message)
    """
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        success = (result.returncode == 0)
        error = "" if success else result.stderr

        return success, error

    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def generate_code(model, tokenizer, instruction: str, config: Task2Config, device: str) -> str:
    """Generate code for given instruction."""
    # Format prompt
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode and remove prompt
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in generated:
        generated = generated.replace(prompt, "").strip()

    return generated


def evaluate_model(
        model_path: str,
        test_dataset,
        config: Task2Config,
        is_pretrained: bool = False
) -> Dict:
    """
    Evaluate a model on the test set.

    Args:
        model_path: Path to model directory
        test_dataset: Test dataset
        config: Config object
        is_pretrained: If True, evaluate base model without LoRA

    Returns:
        Dictionary with metrics and results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {os.path.basename(model_path)}")
    print(f"Device: {device}")
    print(f"{'=' * 60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load model
    if is_pretrained:
        print("Loading pretrained base model...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=config.cache_dir
        )
    else:
        print("Loading fine-tuned model with LoRA...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=config.cache_dir
        )
        model = PeftModel.from_pretrained(base_model, model_path)

    model.eval()

    # Storage for results
    results = []
    bleu_scores = []
    exec_successes = []

    print(f"\nGenerating predictions on {len(test_dataset)} examples...")

    for i, example in enumerate(tqdm(test_dataset, desc="Evaluating")):
        instruction = example['instruction']
        reference_output = example['output']

        # Generate
        generated_text = generate_code(model, tokenizer, instruction, config, device)

        # Extract code
        pred_code = extract_python_code(generated_text)
        ref_code = extract_python_code(reference_output)

        # Calculate BLEU
        bleu = calculate_bleu(pred_code, ref_code)

        # Test execution
        exec_success, exec_error = test_code_execution(pred_code)

        # Store
        results.append({
            'instruction': instruction,
            'reference': reference_output,
            'generated': generated_text,
            'pred_code': pred_code,
            'ref_code': ref_code,
            'bleu': bleu,
            'exec_success': exec_success,
            'exec_error': exec_error
        })

        bleu_scores.append(bleu)
        exec_successes.append(exec_success)

    # Calculate metrics
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    exec_rate = sum(exec_successes) / len(exec_successes) * 100

    metrics = {
        'average_bleu': avg_bleu,
        'execution_success_rate': exec_rate,
        'num_examples': len(test_dataset)
    }

    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Execution Success Rate: {exec_rate:.2f}%")
    print(f"Number of examples: {len(test_dataset)}")
    print(f"{'=' * 60}\n")

    return {
        'metrics': metrics,
        'results': results
    }


def save_results(eval_output: Dict, output_path: str):
    """Save evaluation results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare output (limit full results to save space)
    output_data = {
        'metrics': eval_output['metrics'],
        'sample_results': eval_output['results'][:10],  # First 10 examples
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Results saved to: {output_path}")


if __name__ == "__main__":
    # Test evaluation
    from data_loader import prepare_dataset

    config = Task2Config()
    datasets = prepare_dataset(config)

    # This would evaluate a trained model
    # eval_output = evaluate_model(
    #     model_path="./outputs/lora_r8_full",
    #     test_dataset=datasets['test'],
    #     config=config
    # )

    print("✓ Evaluation script ready!")