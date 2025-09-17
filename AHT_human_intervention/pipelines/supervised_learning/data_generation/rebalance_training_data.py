#!/usr/bin/env python3
"""
Rebalance macro training data to fix TAKE_ONION over-representation.

Current distribution (problematic):
- TAKE_ONION: 19.0% (191 samples) - TOO HIGH
- GO_TO_SERVE: 18.1% (182 samples)
- SERVE: 15.0% (151 samples)
- GO_TO_DISH: 12.2% (122 samples)
- GO_TO_POT: 10.8% (108 samples) - should be higher
- PUT_IN_POT: 9.9% (99 samples)
- TAKE_SOUP: 6.0% (60 samples)
- TAKE_DISH: 4.1% (41 samples)
- GO_TO_ONION: 3.5% (35 samples) - should be higher
- WAIT_COOK: 1.5% (15 samples) - should be higher

Target distribution (more balanced):
- All macros should be roughly 8-12% each
- Critical transitions like GO_TO_POT, GO_TO_ONION should be well represented
"""

import json
import random
import collections
from pathlib import Path

def load_samples(shard_path):
    """Load samples from a JSONL file."""
    with open(shard_path, 'r') as f:
        return [json.loads(line) for line in f]

def save_samples(samples, output_path):
    """Save samples to a JSONL file."""
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

def analyze_distribution(samples):
    """Analyze the distribution of macros in samples."""
    macro_counts = collections.Counter()
    for sample in samples:
        macro_counts[sample['macro_id']] += 1
    
    total = len(samples)
    print(f"Total samples: {total}")
    print("Distribution:")
    for macro, count in macro_counts.most_common():
        percentage = (count / total) * 100
        print(f"  {macro:12}: {count:3} samples ({percentage:5.1f}%)")
    
    return macro_counts

def rebalance_samples(samples, target_distribution=None):
    """
    Rebalance samples to achieve target distribution.
    
    Args:
        samples: List of training samples
        target_distribution: Dict of {macro: target_percentage} or None for uniform
    """
    # Group samples by macro
    samples_by_macro = collections.defaultdict(list)
    for sample in samples:
        samples_by_macro[sample['macro_id']].append(sample)
    
    # Define target distribution (roughly uniform with some adjustments)
    if target_distribution is None:
        target_distribution = {
            'GO_TO_ONION': 12.0,    # Increase (was 3.5%)
            'TAKE_ONION': 10.0,     # Decrease (was 19.0%)
            'GO_TO_POT': 12.0,      # Increase (was 10.8%)
            'PUT_IN_POT': 10.0,     # Keep similar (was 9.9%)
            'GO_TO_DISH': 10.0,     # Decrease (was 12.2%)
            'TAKE_DISH': 8.0,       # Increase (was 4.1%)
            'GO_TO_SERVE': 12.0,    # Decrease (was 18.1%)
            'SERVE': 12.0,          # Decrease (was 15.0%)
            'WAIT_COOK': 6.0,       # Increase (was 1.5%)
            'TAKE_SOUP': 8.0,       # Increase (was 6.0%)
        }
    
    # Calculate target counts
    total_target = 1000  # Target total samples
    target_counts = {}
    for macro, percentage in target_distribution.items():
        target_counts[macro] = int(total_target * percentage / 100)
    
    print(f"\nTarget distribution ({total_target} samples):")
    for macro, count in target_counts.items():
        percentage = (count / total_target) * 100
        print(f"  {macro:12}: {count:3} samples ({percentage:5.1f}%)")
    
    # Rebalance samples
    rebalanced_samples = []
    
    for macro, target_count in target_counts.items():
        available_samples = samples_by_macro[macro]
        
        if len(available_samples) >= target_count:
            # Downsample: randomly select target_count samples
            selected = random.sample(available_samples, target_count)
        else:
            # Upsample: repeat samples to reach target_count
            selected = []
            while len(selected) < target_count:
                remaining = target_count - len(selected)
                if remaining >= len(available_samples):
                    selected.extend(available_samples)
                else:
                    selected.extend(random.sample(available_samples, remaining))
        
        rebalanced_samples.extend(selected)
        print(f"  {macro:12}: {len(available_samples):3} -> {len(selected):3} samples")
    
    # Shuffle the rebalanced samples
    random.shuffle(rebalanced_samples)
    
    return rebalanced_samples

def main():
    print("=== REBALANCING MACRO TRAINING DATA ===")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load original training data
    train_path = Path("pretrain_shards/macro_pretrain_00.jsonl")
    val_path = Path("pretrain_shards/macro_pretrain_01.jsonl")
    
    if not train_path.exists():
        print(f"Error: {train_path} not found!")
        return
    
    print("\n--- Original Training Data ---")
    train_samples = load_samples(train_path)
    train_counts = analyze_distribution(train_samples)
    
    if val_path.exists():
        print("\n--- Original Validation Data ---")
        val_samples = load_samples(val_path)
        val_counts = analyze_distribution(val_samples)
    else:
        val_samples = []
        print("\nNo validation data found")
    
    # Rebalance training data
    print("\n--- Rebalancing Training Data ---")
    rebalanced_train = rebalance_samples(train_samples)
    
    # Also rebalance validation data if available
    if val_samples:
        print("\n--- Rebalancing Validation Data ---")
        rebalanced_val = rebalance_samples(val_samples)
    else:
        rebalanced_val = []
    
    # Save rebalanced data
    output_dir = Path("pretrain_shards_balanced")
    output_dir.mkdir(exist_ok=True)
    
    train_output = output_dir / "macro_pretrain_00.jsonl"
    save_samples(rebalanced_train, train_output)
    print(f"\nSaved rebalanced training data: {train_output}")
    
    if rebalanced_val:
        val_output = output_dir / "macro_pretrain_01.jsonl"
        save_samples(rebalanced_val, val_output)
        print(f"Saved rebalanced validation data: {val_output}")
    
    # Verify the new distribution
    print("\n--- Final Training Data Distribution ---")
    final_counts = analyze_distribution(rebalanced_train)
    
    if rebalanced_val:
        print("\n--- Final Validation Data Distribution ---")
        analyze_distribution(rebalanced_val)
    
    print("\n=== REBALANCING COMPLETE ===")
    print(f"Original TAKE_ONION: {train_counts['TAKE_ONION']} samples ({train_counts['TAKE_ONION']/len(train_samples)*100:.1f}%)")
    print(f"Rebalanced TAKE_ONION: {final_counts['TAKE_ONION']} samples ({final_counts['TAKE_ONION']/len(rebalanced_train)*100:.1f}%)")
    print(f"Use these files for training: {output_dir}")

if __name__ == "__main__":
    main()
