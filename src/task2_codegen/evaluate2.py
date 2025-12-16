"""
Evaluation script for Task 2: Code Generation
"""
import os
import json
import re
import subprocess
import tempfile
from typing import Dict, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def extract_python_code(text: str) -> str:
    """
    Extract Python code from text.
    Handles markdown code blocks and tries to extract raw code.

    Args:
        text: Generated text that may contain code

    Returns:
        Extracted Python code
    """
    # Try to find code in markdown code blocks
    pattern = r'```python\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Try without language specifier
    pattern = r'```\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        return matches[0].strip()

    # If no code blocks, try to extract lines that look like Python code
    # This is a heuristic approach
    lines = text.split('\n')
    code_lines = []

    for line in lines:
        # Skip lines that look like natural language
        if any(starter in line.lower() for starter in ['here is', 'here\'s', 'this code', 'the code', 'this function']):
            continue
        # Include lines that look like code
        if line.strip() and (
                line.strip().startswith(('def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', '#')) or
                '=' in line or
                line.strip().startswith((' ', '\t'))
        ):
            code_lines.append(line)

    if code_lines:
        return '\n'.join(code_lines).strip()

    # Return the whole text as fallback
    return text.strip()


def calculate_bleu(prediction: str, reference: str) -> float:
    """
    Calculate BLEU score between prediction and reference.

    Args:
        prediction: Predicted code
        reference: Reference code

    Returns:
        BLEU score (0-1)
    """
    # Tokenize by splitting on whitespace and special characters
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    # Use smoothing function to handle zero counts
    smoothing = SmoothingFunction().method1

    # Calculate BLEU score
    score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)

    return score


def test_code_execution(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Test if Python code can be executed without errors.

    Args:
        code: Python code to test
        timeout: Maximum execution time in seconds

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Try to execute the code
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Check if execution was successful
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr

    except subprocess.TimeoutExpired:
        return False, "Execution timeout"
    except Exception as e:
        return False, str(e)
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def generate_code(
        model,
        tokenizer,
        instruction: str,
        config,
        device: str = "cuda"
) -> str:
    """
    Generate code for a given instruction.

    Args:
        model: The language model
        tokenizer: Tokenizer
        instruction: User instruction
        config: Configuration object
        device: Device to use

    Returns:
        Generated code
    """
    # Format as chat
    messages = [{"role": "user", "content": instruction}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
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

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the generated text
    if prompt in generated_text:
        generated_text = generated_text.replace(prompt, "").strip()

    return generated_text


def evaluate_code_generation(
        model_path: str,
        test_dataset: Dataset,
        config,
        tokenizer: AutoTokenizer = None,
        device: str = None,
        is_peft: bool = True
) -> Dict:
    """
    Evaluate code generation model.

    Args:
        model_path: Path to saved model
        test_dataset: Test dataset
        config: Configuration object
        tokenizer: Tokenizer
        device: Device to use
        is_peft: Whether model is a PEFT model

    Returns:
        Dictionary with evaluation results
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {model_path}")
    if is_peft:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    model.eval()

    # Results storage
    results = {
        "predictions": [],
        "references": [],
        "extracted_code_pred": [],
        "extracted_code_ref": [],
        "bleu_scores": [],
        "execution_success": [],
        "execution_errors": []
    }

    print("Generating and evaluating code...")
    for example in tqdm(test_dataset, desc="Evaluating"):
        instruction = example["instruction"]
        reference_output = example["output"]

        # Generate code
        generated_text = generate_code(model, tokenizer, instruction, config, device)

        # Extract code from both prediction and reference
        pred_code = extract_python_code(generated_text)
        ref_code = extract_python_code(reference_output)

        # Calculate BLEU score
        bleu = calculate_bleu(pred_code, ref_code)

        # Test execution
        exec_success, exec_error = test_code_execution(pred_code)

        # Store results
        results["predictions"].append(generated_text)
        results["references"].append(reference_output)
        results["extracted_code_pred"].append(pred_code)
        results["extracted_code_ref"].append(ref_code)
        results["bleu_scores"].append(bleu)
        results["execution_success"].append(exec_success)
        results["execution_errors"].append(exec_error)

    # Calculate aggregate metrics
    avg_bleu = sum(results["bleu_scores"]) / len(results["bleu_scores"])
    exec_rate = sum(results["execution_success"]) / len(results["execution_success"]) * 100

    metrics = {
        "average_bleu": avg_bleu,
        "execution_success_rate": exec_rate,
        "num_examples": len(test_dataset),
    }

    print(f"\nEvaluation Results:")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Execution Success Rate: {exec_rate:.2f}%")
    print(f"Number of examples: {len(test_dataset)}")

    return metrics, results


def save_evaluation_results(
        metrics: Dict,
        results: Dict,
        output_path: str
):
    """
    Save evaluation results to JSON file.
    """
    output_data = {
        "metrics": metrics,
        "detailed_results": {
            "bleu_scores": results["bleu_scores"],
            "execution_success": results["execution_success"],
            "sample_predictions": results["predictions"][:10],
            "sample_references": results["references"][:10],
        },
        "full_results": results  # Include everything
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    from preprocess2 import prepare_code_dataset
    from config2 import CodeGenTrainingConfig

    config = CodeGenTrainingConfig()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    datasets = prepare_code_dataset(config, tokenizer)

    # Evaluate
    metrics, results = evaluate_code_generation(
        model_path=config.output_dir,
        test_dataset=datasets["test"],
        config=config,
        tokenizer=tokenizer,
    )

    # Save results
    save_evaluation_results(
        metrics,
        results,
        os.path.join(config.results_dir, "evaluation_results.json")
    )