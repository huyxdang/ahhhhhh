"""
Create difficulty-labeled datasets from existing inference outputs.

Takes prediction files from run_inference.py and creates labeled datasets:
- Label 0 (easy): SLM gets it correct
- Label 1 (hard): SLM gets it incorrect, but LLM gets it correct  
- Label 2 (omit): Both SLM and LLM get it incorrect
"""
import json
import argparse
import os
import re
from datasets import Dataset
from typing import Dict, List


def extract_boxed_answer(text):
    """Extract answer from \\boxed{} in MATH dataset."""
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_medqa_answer(text):
    """Extract answer option (A, B, C, D, E) from MedQA response."""
    text = text.strip().upper()
    # Look for single letter answer at the start
    if len(text) > 0 and text[0] in ['A', 'B', 'C', 'D', 'E']:
        return text[0]
    # Look for pattern like "Answer: A" or "The answer is A"
    match = re.search(r'(?:answer|option)[\s:]+([A-E])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def check_correctness_math(response, ground_truth):
    """Check if MATH response is correct."""
    predicted = extract_boxed_answer(response)
    actual = extract_boxed_answer(ground_truth)
    
    if predicted is None or actual is None:
        return False
    
    # Normalize whitespace and compare
    predicted = predicted.strip().replace(' ', '')
    actual = actual.strip().replace(' ', '')
    return predicted == actual


def check_correctness_medqa(response, ground_truth):
    """Check if MedQA response is correct."""
    predicted = extract_medqa_answer(response)
    return predicted == ground_truth


def load_predictions(prediction_file):
    """Load predictions from JSON file."""
    if not os.path.exists(prediction_file):
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
    
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions from {prediction_file}")
    return predictions


def evaluate_predictions(predictions, dataset_type):
    """Evaluate predictions and add correctness flag."""
    check_fn = check_correctness_math if dataset_type == 'math' else check_correctness_medqa
    
    correct_count = 0
    for idx, pred in predictions.items():
        is_correct = check_fn(pred['response'], pred['correct_answer'])
        pred['correct'] = is_correct
        if is_correct:
            correct_count += 1
    
    accuracy = correct_count / len(predictions) * 100 if predictions else 0
    print(f"  Accuracy: {correct_count}/{len(predictions)} = {accuracy:.2f}%")
    
    return predictions


def create_difficulty_labels(slm1_preds, slm2_preds, llm_preds, dataset_type):
    """
    Create difficulty labels based on model correctness.
    
    Labels:
    - 0 (easy): At least one SLM gets it correct
    - 1 (hard): Both SLMs get it incorrect, but LLM gets it correct
    - 2 (omit): Both SLMs and LLM get it incorrect
    """
    # Ensure all prediction files have same examples
    indices = set(slm1_preds.keys()) & set(slm2_preds.keys()) & set(llm_preds.keys())
    
    if len(indices) != len(slm1_preds):
        print(f"Warning: Not all predictions match. Using intersection of {len(indices)} examples")
    
    labeled_data = []
    label_counts = {0: 0, 1: 0, 2: 0}
    
    for idx in sorted(indices, key=int):
        slm1_correct = slm1_preds[idx]['correct']
        slm2_correct = slm2_preds[idx]['correct']
        llm_correct = llm_preds[idx]['correct']
        
        # Determine if any SLM got it correct
        slm_correct = slm1_correct or slm2_correct
        
        # Assign label
        if slm_correct:
            label = 0  # easy
        elif llm_correct:
            label = 1  # hard
        else:
            label = 2  # omit
        
        label_counts[label] += 1
        
        # Create data point
        data_point = {
            'index': int(idx),
            'question': slm1_preds[idx]['question'],
            'correct_answer': slm1_preds[idx]['correct_answer'],
            'label': label,
            'slm1_correct': slm1_correct,
            'slm2_correct': slm2_correct,
            'llm_correct': llm_correct,
            'slm1_response': slm1_preds[idx]['response'],
            'slm2_response': slm2_preds[idx]['response'],
            'llm_response': llm_preds[idx]['response']
        }
        
        labeled_data.append(data_point)
    
    print(f"\nLabel distribution:")
    total = len(labeled_data)
    print(f"  Easy (0): {label_counts[0]} ({label_counts[0]/total*100:.1f}%)")
    print(f"  Hard (1): {label_counts[1]} ({label_counts[1]/total*100:.1f}%)")
    print(f"  Omit (2): {label_counts[2]} ({label_counts[2]/total*100:.1f}%)")
    
    return labeled_data, label_counts


def save_labeled_dataset(labeled_data, output_file, include_omit=False):
    """Save labeled data as JSON and parquet."""
    # Save full dataset as JSON
    with open(output_file, 'w') as f:
        json.dump(labeled_data, f, indent=2)
    print(f"\nSaved full labeled dataset to: {output_file}")
    
    # Create filtered dataset (excluding label 2 unless requested)
    if not include_omit:
        filtered_data = [d for d in labeled_data if d['label'] != 2]
    else:
        filtered_data = labeled_data
    
    if filtered_data:
        # Create HuggingFace dataset
        hf_dataset = Dataset.from_list(filtered_data)
        
        # Save as parquet
        parquet_file = output_file.replace('.json', '_filtered.parquet')
        hf_dataset.to_parquet(parquet_file)
        print(f"Saved filtered dataset to: {parquet_file}")
        print(f"  Filtered size: {len(filtered_data)}/{len(labeled_data)}")
        
        # Create training dataset with just question and label
        training_data = [
            {
                'question': d['question'],
                'label': d['label'],
                'correct_answer': d['correct_answer']
            }
            for d in filtered_data
        ]
        
        training_dataset = Dataset.from_list(training_data)
        training_file = output_file.replace('.json', '_training.parquet')
        training_dataset.to_parquet(training_file)
        print(f"Saved training-ready dataset to: {training_file}")
        print(f"  Columns: question, label, correct_answer")
        
        return hf_dataset
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Create difficulty labels from inference outputs"
    )
    parser.add_argument(
        "--slm1_predictions",
        type=str,
        required=True,
        help="Path to first SLM predictions JSON file"
    )
    parser.add_argument(
        "--slm2_predictions",
        type=str,
        required=True,
        help="Path to second SLM predictions JSON file"
    )
    parser.add_argument(
        "--llm_predictions",
        type=str,
        required=True,
        help="Path to LLM predictions JSON file"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=['math', 'medqa'],
        help="Dataset type for correctness checking"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for labeled dataset (JSON)"
    )
    parser.add_argument(
        "--include_omit",
        action='store_true',
        help="Include label 2 (omit) in filtered dataset (default: False)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Creating Difficulty Labels from Predictions")
    print(f"{'='*60}\n")
    
    # Load predictions
    print("Loading predictions...")
    slm1_preds = load_predictions(args.slm1_predictions)
    slm2_preds = load_predictions(args.slm2_predictions)
    llm_preds = load_predictions(args.llm_predictions)
    
    # Evaluate correctness if not already done
    print(f"\nEvaluating correctness for {args.dataset_type}...")
    print("SLM 1:")
    slm1_preds = evaluate_predictions(slm1_preds, args.dataset_type)
    print("SLM 2:")
    slm2_preds = evaluate_predictions(slm2_preds, args.dataset_type)
    print("LLM:")
    llm_preds = evaluate_predictions(llm_preds, args.dataset_type)
    
    # Create difficulty labels
    print(f"\n{'='*60}")
    print("Creating difficulty labels...")
    print(f"{'='*60}")
    labeled_data, label_counts = create_difficulty_labels(
        slm1_preds,
        slm2_preds,
        llm_preds,
        args.dataset_type
    )
    
    # Save labeled dataset
    print(f"\n{'='*60}")
    print("Saving labeled dataset...")
    print(f"{'='*60}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    save_labeled_dataset(labeled_data, args.output, args.include_omit)
    
    print(f"\n{'='*60}")
    print("COMPLETED!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

