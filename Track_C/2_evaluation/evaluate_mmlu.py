"""
Evaluate LoRA-adapted Llama 3.2-1B model on MMLU benchmark.

Usage:
    python evaluate_mmlu.py --lora_path <path_to_lora> [options]
    
    Or use default path:
    python evaluate_mmlu.py
"""

import argparse
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datetime import datetime
from tqdm import tqdm

try:
    # Try newer API first
    try:
        from lm_eval import tasks
        from lm_eval.evaluator import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        LM_EVAL_AVAILABLE = True
        LM_EVAL_VERSION = "new"
    except ImportError:
        # Try older API
        try:
            from lm_eval import simple_evaluate
            from lm_eval.models import HFLM
            LM_EVAL_AVAILABLE = True
            LM_EVAL_VERSION = "old"
        except ImportError:
            LM_EVAL_AVAILABLE = False
            LM_EVAL_VERSION = None
except Exception as e:
    LM_EVAL_AVAILABLE = False
    LM_EVAL_VERSION = None
    print(f"Warning: Could not import lm-eval: {e}")

if not LM_EVAL_AVAILABLE:
    print("Warning: lm-eval not installed. Install with: pip install lm-eval")


def load_lora_model(lora_path, base_model="meta-llama/Llama-3.2-1B-Instruct", 
                   use_quantization=True, device_map="auto"):
    """
    Load base model and LoRA adapter.
    
    Args:
        lora_path: Path to LoRA adapter directory
        base_model: Base model name or path
        use_quantization: Whether to use 4-bit quantization
        device_map: Device mapping strategy
        
    Returns:
        model, tokenizer
    """
    print(f"Loading base model: {base_model}")
    
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            dtype='auto',
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    print("Base model loaded successfully!")
    
    print(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    print("LoRA adapter loaded successfully!")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def evaluate_with_lm_eval(model, tokenizer, tasks_list=None, limit=None, batch_size=1):
    """
    Evaluate model using lm-evaluation-harness.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        tasks_list: List of MMLU tasks to evaluate (None = all MMLU tasks)
        limit: Limit number of examples per task (None = all)
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    if not LM_EVAL_AVAILABLE:
        raise ImportError("lm-eval is required. Install with: pip install lm-eval")
    
    # Default to all MMLU tasks if not specified
    if tasks_list is None:
        tasks_list = [
            "mmlu",
            # Or specify individual subjects:
            # "mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_astronomy", etc.
        ]
    
    print(f"Evaluating on tasks: {tasks_list}")
    
    try:
        # Create HFLM model wrapper
        # Note: API may vary by version - try different approaches
        try:
            # Newer API
            model_wrapper = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                device=next(model.parameters()).device,
            )
        except TypeError:
            # Older API or different signature
            model_wrapper = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
            )
        
        # Run evaluation
        results = simple_evaluate(
            model=model_wrapper,
            tasks=tasks_list,
            limit=limit,
            batch_size=batch_size,
        )
        
        return results
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("\nTip: If you encounter API errors, try:")
        print("  1. Update lm-eval: pip install --upgrade lm-eval")
        print("  2. Use --manual flag for simplified evaluation")
        print("  3. Use lm-eval CLI directly (see README)")
        raise


def evaluate_manual_mmlu(model, tokenizer, num_shots=5, limit=None):
    """
    Manual MMLU evaluation (alternative if lm-eval is not available).
    This is a simplified version - for full evaluation, use lm-eval.
    """
    from datasets import load_dataset
    
    print("Loading MMLU dataset...")
    # Load MMLU dataset
    # Note: This is a simplified example - full MMLU has many subjects
    try:
        dataset = load_dataset("cais/mmlu", "all", split="test")
    except:
        print("Could not load MMLU dataset. Please install: pip install datasets")
        return None
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} examples...")
    
    correct = 0
    total = 0
    
    model.eval()
    for example in tqdm(dataset, desc="Evaluating"):
        question = example["question"]
        choices = example["choices"]
        correct_answer = example["answer"]
        
        # Format as multiple choice question
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        
        # Format for chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and extract answer
        generated = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Check if answer matches (simple check)
        predicted = generated[0].upper() if generated else None
        if predicted == correct_answer:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def make_json_serializable(obj):
    """
    Recursively convert non-JSON-serializable objects to serializable ones.
    Handles numpy types, torch tensors, and other common types.
    """
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.dtype):
        return str(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Try to convert to string as fallback
        try:
            return str(obj)
        except:
            return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model on MMLU")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=r"C:\Projects\Great_Hack\Doctor\5_models\global_chain_patch\lora_20251116_064822",
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (default: True)"
    )
    parser.add_argument(
        "--no_quantization",
        action="store_false",
        dest="use_quantization",
        help="Disable quantization"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="mmlu",
        help="Comma-separated list of tasks (default: mmlu for all MMLU tasks)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for quick testing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: same as lora_path)"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use manual evaluation instead of lm-eval (simpler but less accurate)"
    )
    
    args = parser.parse_args()
    
    # Load model
    print("="*80)
    print("MMLU Evaluation for LoRA Model")
    print("="*80)
    model, tokenizer = load_lora_model(
        args.lora_path,
        args.base_model,
        use_quantization=args.use_quantization
    )
    
    # Evaluate
    if args.manual or not LM_EVAL_AVAILABLE:
        print("\nUsing manual evaluation...")
        results = evaluate_manual_mmlu(model, tokenizer, limit=args.limit)
        if results:
            print(f"\nResults: {results}")
    else:
        print("\nUsing lm-evaluation-harness...")
        tasks_list = [t.strip() for t in args.tasks.split(",")]
        results = evaluate_with_lm_eval(
            model, tokenizer,
            tasks_list=tasks_list,
            limit=args.limit,
            batch_size=args.batch_size
        )
        
        # Print results
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        if isinstance(results, dict) and "results" in results:
            for task, task_results in results["results"].items():
                print(f"\n{task}:")
                if isinstance(task_results, dict):
                    for key, value in task_results.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
        else:
            print(json.dumps(results, indent=2))
    
    # Save results
    output_dir = args.output_dir or os.path.dirname(args.lora_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"mmlu_results_{timestamp}.json")
    
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "lora_path": args.lora_path,
            "base_model": args.base_model,
            "tasks": args.tasks,
            "timestamp": timestamp,
            "results": make_json_serializable(results)
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()

