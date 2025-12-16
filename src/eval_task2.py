import sys, re, json, subprocess, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu


def extract_code(text):
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.split("### Response:")[-1].strip()


def run_eval(adapter_path):
    model_id = "JetBrains/Mellum-4b-base"
    dataset = load_dataset("flytech/python-codes-25k", split='train')
    test_ds = dataset.train_test_split(test_size=0.2, seed=42)['test'].train_test_split(test_size=0.5, seed=42)['test']

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    if adapter_path != "baseline":
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    results = []
    for entry in test_ds.select(range(100)):  # Evaluate subset for speed
        prompt = f"### Instruction: {entry['instruction']}\n### Response: "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=128)

        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        code = extract_code(gen_text)

        # Simple execution check
        try:
            subprocess.run(["python3", "-c", code], timeout=2, check=True, capture_output=True)
            exec_ok = True
        except:
            exec_ok = False

        results.append({"bleu": sentence_bleu([entry['output'].split()], code.split()), "exec": exec_ok})

    print(
        f"Results for {adapter_path}: BLEU: {sum(r['bleu'] for r in results) / 100:.4f}, Exec: {sum(r['exec'] for r in results)}%")


if __name__ == "__main__":
    run_eval(sys.argv[1])