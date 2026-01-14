import json
import random
import argparse

def split_jsonl(input_file, train_file, val_file, train_ratio=0.9, random_seed=None):
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Shuffle the lines
    random.shuffle(lines)
    
    # Calculate split point
    split_idx = int(len(lines) * train_ratio)
    
    # Write training data
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[:split_idx])
    
    # Write validation data
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[split_idx:])
    
    print(f"Total lines: {len(lines)}")
    print(f"Training lines: {split_idx}")
    print(f"Validation lines: {len(lines) - split_idx}")

def main():
    parser = argparse.ArgumentParser(description='Split JSONL file into training and validation sets')
    parser.add_argument('input_file', help='Input JSONL file')
    parser.add_argument('train_file', help='Output training file')
    parser.add_argument('val_file', help='Output validation file')
    parser.add_argument('--ratio', type=float, default=0.9, help='Training set ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    split_jsonl(args.input_file, args.train_file, args.val_file, args.ratio, args.seed)

if __name__ == "__main__":
    main()