# split_data.py
import random
import sys
from pathlib import Path

def split_jsonl(input_path: str, train_ratio: float = 0.9, seed: int = 42):
    random.seed(seed)
    
    input_file = Path(input_path)
    lines = input_file.read_text().splitlines(keepends=True)
    
    random.shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    
    # New filenames: original_4181_train.jsonl, original_4181_eval.jsonl
    base_name = input_file.stem  # "grpo_problem_algebra"
    suffix = f"_{len(lines)}"
    
    train_file = input_file.parent / f"{base_name}{suffix}_train.jsonl"
    eval_file = input_file.parent / f"{base_name}{suffix}_eval.jsonl"
    
    train_file.write_file(''.join(lines[:split_idx]))
    eval_file.write_file(''.join(lines[split_idx:]))
    
    print(f"✅ Split {len(lines)} lines: {train_file.name}, {eval_file.name}")

if __name__ == '__main__':
    split_jsonl(sys.argv[1])