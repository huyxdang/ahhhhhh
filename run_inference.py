"""
Run inference on datasets (MATH, MedQA, BoolQ) using vLLM for fast batch processing.
No confidence prompting - just answer questions.
"""
import json
import argparse
import os
from vllm import LLM, SamplingParams
from datasets import load_dataset


# Dataset configurations
DATASET_CONFIGS = {
    'math': {
        'hf_path': 'huyxdang/math-split',
        'question_field': 'problem',
        'answer_field': 'solution',
        'max_tokens': 1024,
        'domain': 'MATH'
    },
    'medqa': {
        'hf_path': 'huyxdang/medqa-split',
        'question_field': 'question',
        'answer_field': 'answer_idx',
        'max_tokens': 64,
        'domain': 'MED',
        'has_options': True,
        'options_field': 'options'
    }
}


def create_prompt(example, dataset_name):
    """Create appropriate prompt based on dataset type."""
    config = DATASET_CONFIGS[dataset_name]
    
    if dataset_name == 'math':
        # Simple problem prompt
        prompt = f"""Solve the following math problem. At the end, on the last line, output ONLY the final answer in LaTeX in the form: 
        \\boxed{{<final_answer>}}

Problem: {example[config['question_field']]}

Solution:"""
        
    elif dataset_name == 'medqa':
        # Multiple choice with options
        question = example[config['question_field']]
        options = example[config['options_field']]
        
        # Format options
        options_text = '\n'.join([f"{key}: {value}" for key, value in options.items()])
        
        prompt = f"""Answer the following medical question by selecting the correct option.

Question: {question}

Options:
{options_text}

Answer:"""
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return prompt


def load_dataset_split(dataset_name, split='train'):
    """Load dataset from HuggingFace."""
    config = DATASET_CONFIGS[dataset_name]
    print(f"Loading {dataset_name} dataset from {config['hf_path']} (split: {split})")
    
    dataset = load_dataset(config['hf_path'], split=split)
    
    # Convert to list of dicts
    examples = []
    for i, example in enumerate(dataset):
        example_dict = dict(example)
        example_dict['index'] = i
        examples.append(example_dict)
    
    print(f"Loaded {len(examples)} examples from {dataset_name}")
    return examples


def run_inference(model_name, dataset_name, split='train', output_file=None, 
                  tensor_parallel_size=1, batch_size=50):
    """Run inference on dataset using vLLM with partial progress saving."""
    
    # Get dataset config
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be one of: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    # Set default output file if not provided
    if output_file is None:
        model_short = model_name.split('/')[-1].lower().replace('-', '').replace('.', '')
        output_file = f"predictions/{model_short}_{dataset_name}_{split}.json"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    examples = load_dataset_split(dataset_name, split)
    
    print(f"\nModel: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Max tokens: {config['max_tokens']}")
    print(f"Total examples: {len(examples)}")
    print(f"Output file: {output_file}")
    
    # Load existing predictions if resuming
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            predictions = json.load(f)
        print(f"\nResuming from existing file with {len(predictions)} predictions")
    else:
        predictions = {}
    
    # Initialize vLLM
    print(f"\nLoading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True
    )
    
    # Sampling params for deterministic generation
    params = SamplingParams(
        temperature=0.0,
        max_tokens=config['max_tokens'],
        skip_special_tokens=True
    )
    
    # Filter out already processed examples
    examples_to_process = [ex for ex in examples if str(ex['index']) not in predictions]
    
    if not examples_to_process:
        print("\nAll examples already processed!")
        return
    
    print(f"Processing {len(examples_to_process)} remaining examples...")
    
    # Process in batches and save progress
    total_to_process = len(examples_to_process)
    all_outputs = []
    
    for batch_start in range(0, total_to_process, batch_size):
        batch_end = min(batch_start + batch_size, total_to_process)
        batch_examples = examples_to_process[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start + 1}-{batch_end}/{total_to_process}...")
        
        # Create prompts for this batch
        batch_prompts = [create_prompt(ex, dataset_name) for ex in batch_examples]
        
        # Generate for this batch
        batch_outputs = llm.generate(batch_prompts, params)
        all_outputs.extend(batch_outputs)
        
        # Add to predictions
        for i, output in enumerate(batch_outputs):
            example = batch_examples[i]
            example_index = example['index']
            
            predictions[str(example_index)] = {
                "response": output.outputs[0].text.strip(),
                "question": example[config['question_field']],
                "correct_answer": example[config['answer_field']]
            }
        
        # Save partial progress
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=2)
        
        print(f"Saved {len(predictions)}/{len(examples)} predictions")
    
    print(f"\n{'='*60}")
    print(f"Completed! Saved all {len(predictions)} predictions to {output_file}")
    print(f"{'='*60}")
    
    # Calculate average tokens
    if all_outputs:
        avg_tokens = sum(len(o.outputs[0].token_ids) for o in all_outputs) / len(all_outputs)
        print(f"Average tokens per response: {avg_tokens:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on datasets without confidence prompting"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., meta-llama/Meta-Llama-3-8B-Instruct)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['math', 'medqa', 'boolq'],
        help="Dataset to run inference on"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for predictions (default: predictions/{model}_{dataset}_{split}.json)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of questions to process before saving progress (default: 50)"
    )
    
    args = parser.parse_args()
    
    run_inference(
        args.model_name,
        args.dataset,
        args.split,
        args.output,
        args.tensor_parallel_size,
        args.batch_size
    )