"""
Balance difficulty-labeled dataset by converting label 2->1 and creating 1:1 ratio.

Takes labeled dataset and:
1. Converts all label 2 (omit) to label 1 (hard)
2. Balances label 0 (easy) and label 1 (hard) to 1:1 ratio
"""
import json
import argparse
import os
import random
from datasets import Dataset


def load_labeled_data(input_file):
    """Load labeled dataset from JSON."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples from {input_file}")
    return data


def convert_labels(data):
    """Convert all label 2 to label 1."""
    label_changes = 0
    
    for item in data:
        if item['label'] == 2:
            item['label'] = 1
            label_changes += 1
    
    print(f"\nConverted {label_changes} examples from label 2 to label 1")
    return data


def balance_dataset(data, method='downsample', seed=42):
    """
    Balance dataset to 1:1 ratio between label 0 and label 1.
    
    Args:
        data: List of labeled examples
        method: 'downsample' or 'upsample'
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Separate by label
    label_0 = [d for d in data if d['label'] == 0]
    label_1 = [d for d in data if d['label'] == 1]
    
    count_0 = len(label_0)
    count_1 = len(label_1)
    
    print(f"\nOriginal distribution:")
    print(f"  Label 0 (easy): {count_0}")
    print(f"  Label 1 (hard): {count_1}")
    print(f"  Ratio: {count_0}:{count_1}")
    
    if method == 'downsample':
        # Downsample majority class to match minority class
        target_size = min(count_0, count_1)
        
        if count_0 > target_size:
            label_0 = random.sample(label_0, target_size)
        if count_1 > target_size:
            label_1 = random.sample(label_1, target_size)
        
        print(f"\nDownsampling to {target_size} examples per class")
        
    elif method == 'upsample':
        # Upsample minority class to match majority class
        target_size = max(count_0, count_1)
        
        if count_0 < target_size:
            # Repeat samples to reach target size
            label_0 = label_0 + random.choices(label_0, k=target_size - count_0)
        if count_1 < target_size:
            label_1 = label_1 + random.choices(label_1, k=target_size - count_1)
        
        print(f"\nUpsampling to {target_size} examples per class")
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'downsample' or 'upsample'")
    
    # Combine and shuffle
    balanced_data = label_0 + label_1
    random.shuffle(balanced_data)
    
    print(f"\nBalanced distribution:")
    print(f"  Label 0 (easy): {len(label_0)}")
    print(f"  Label 1 (hard): {len(label_1)}")
    print(f"  Total: {len(balanced_data)}")
    print(f"  Ratio: 1:1 ✓")
    
    return balanced_data


def save_balanced_dataset(data, output_file, save_formats=['json', 'parquet', 'training'], 
                          push_to_hub=False, hf_username=None, hf_dataset_name=None):
    """Save balanced dataset in multiple formats and optionally upload to HuggingFace."""
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    training_dataset = None
    
    if 'json' in save_formats:
        # Save full dataset as JSON
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved balanced dataset to: {output_file}")
    
    if 'parquet' in save_formats:
        # Save as parquet
        hf_dataset = Dataset.from_list(data)
        parquet_file = output_file.replace('.json', '.parquet')
        hf_dataset.to_parquet(parquet_file)
        print(f"Saved parquet version to: {parquet_file}")
    
    if 'training' in save_formats:
        # Create training dataset with minimal columns
        training_data = [
            {
                'question': d['question'],
                'label': d['label'],
                'correct_answer': d['correct_answer']
            }
            for d in data
        ]
        
        training_dataset = Dataset.from_list(training_data)
        training_file = output_file.replace('.json', '_training.parquet')
        training_dataset.to_parquet(training_file)
        print(f"Saved training-ready dataset to: {training_file}")
        print(f"  Columns: question, label, correct_answer")
    
    # Upload to HuggingFace Hub if requested
    if push_to_hub:
        if not hf_username or not hf_dataset_name:
            print("\nWarning: --push_to_hub requires --hf_username and --hf_dataset_name")
            return training_dataset
        
        if training_dataset is None:
            # Create training dataset if not already created
            training_data = [
                {
                    'question': d['question'],
                    'label': d['label'],
                    'correct_answer': d['correct_answer']
                }
                for d in data
            ]
            training_dataset = Dataset.from_list(training_data)
        
        print(f"\n{'='*60}")
        print(f"Uploading to HuggingFace Hub...")
        print(f"{'='*60}")
        
        repo_id = f"{hf_username}/{hf_dataset_name}"
        print(f"Repository: {repo_id}")
        
        try:
            training_dataset.push_to_hub(
                repo_id,
                private=False
            )
            print(f"✓ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"✗ Error uploading to HuggingFace: {e}")
            print("\nMake sure you're logged in. Run: huggingface-cli login")
    
    return training_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Balance difficulty-labeled dataset (convert label 2->1, create 1:1 ratio)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input labeled dataset (JSON)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output balanced dataset (JSON)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default='downsample',
        choices=['downsample', 'upsample'],
        help="Balancing method: downsample majority or upsample minority (default: downsample)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--formats",
        nargs='+',
        default=['json', 'parquet', 'training'],
        choices=['json', 'parquet', 'training'],
        help="Output formats to save (default: all)"
    )
    parser.add_argument(
        "--push_to_hub",
        action='store_true',
        help="Upload the training dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        default=None,
        help="HuggingFace username (required if --push_to_hub is set)"
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset name (required if --push_to_hub is set)"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Balancing Difficulty-Labeled Dataset")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading labeled dataset...")
    data = load_labeled_data(args.input)
    
    # Convert label 2 to label 1
    print(f"\n{'='*60}")
    print("Converting labels...")
    print(f"{'='*60}")
    data = convert_labels(data)
    
    # Balance dataset
    print(f"\n{'='*60}")
    print(f"Balancing dataset (method: {args.method})...")
    print(f"{'='*60}")
    balanced_data = balance_dataset(data, method=args.method, seed=args.seed)
    
    # Save balanced dataset
    print(f"\n{'='*60}")
    print("Saving balanced dataset...")
    print(f"{'='*60}")
    save_balanced_dataset(
        balanced_data, 
        args.output, 
        save_formats=args.formats,
        push_to_hub=args.push_to_hub,
        hf_username=args.hf_username,
        hf_dataset_name=args.hf_dataset_name
    )
    
    print(f"\n{'='*60}")
    print("COMPLETED!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

