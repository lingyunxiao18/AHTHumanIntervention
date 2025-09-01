#!/usr/bin/env python3
"""
Convert trajectory data from PKL to JSON format and demonstrate policy network usage.
"""

import pickle
import json
import os
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

def convert_trajectories_to_json(pkl_file_path: str, json_file_path: str):
    """Convert PKL trajectory data to JSON format."""
    
    print(f"Loading trajectories from {pkl_file_path}...")
    with open(pkl_file_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Converting {len(trajectories)} trajectories to JSON...")
    
    json_trajectories = []
    for traj_idx, trajectory in enumerate(tqdm(trajectories)):
        json_trajectory = {
            'trajectory_id': traj_idx,
            'pair_name': trajectory.get('pair_name', 'unknown'),
            'episode_length': len(trajectory['states']),
            'total_reward': trajectory.get('total_reward', 0.0),
            'steps': []
        }
        
        for step_idx in range(len(trajectory['states'])):
            step_data = {
                'step': step_idx,
                'state_text': trajectory['state_texts'][step_idx],
                'human_command': trajectory['human_commands'][step_idx],
                'actions': trajectory['actions'][step_idx],
                'rewards': trajectory['rewards'][step_idx],
                'done': step_idx == len(trajectory['states']) - 1
            }
            json_trajectory['steps'].append(step_data)
        
        json_trajectories.append(json_trajectory)
    
    print(f"Saving {len(json_trajectories)} trajectories to {json_file_path}...")
    with open(json_file_path, 'w') as f:
        json.dump(json_trajectories, f, indent=2)
    
    print(f"Conversion complete! File size: {os.path.getsize(json_file_path) / (1024*1024):.2f} MB")

def create_policy_training_data(json_file_path: str, output_file_path: str):
    """Create training data format for the policy network."""
    
    print(f"Loading JSON trajectories from {json_file_path}...")
    with open(json_file_path, 'r') as f:
        trajectories = json.load(f)
    
    print("Creating policy training data...")
    
    training_data = []
    for trajectory in tqdm(trajectories):
        for step in trajectory['steps']:
            # Format expected by TextBasedLangConditionedPolicy
            training_example = {
                'state_text': step['state_text'],
                'human_command': step['human_command'],
                'action': step['actions'][0],  # Ego agent action (coordinated agent)
                'reward': step['rewards'],
                'trajectory_id': trajectory['trajectory_id'],
                'step': step['step']
            }
            training_data.append(training_example)
    
    print(f"Saving {len(training_data)} training examples to {output_file_path}...")
    with open(output_file_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Training data created! File size: {os.path.getsize(output_file_path) / (1024*1024):.2f} MB")
    
    # Print some statistics
    print(f"\nTraining Data Statistics:")
    print(f"Total training examples: {len(training_data)}")
    print(f"Unique trajectories: {len(set(ex['trajectory_id'] for ex in training_data))}")
    print(f"Average steps per trajectory: {len(training_data) / len(trajectories):.1f}")
    
    # Show action distribution
    actions = [ex['action'] for ex in training_data]
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print(f"\nAction Distribution:")
    for action, count in sorted(action_counts.items()):
        percentage = (count / len(actions)) * 100
        print(f"  Action {action}: {count} ({percentage:.1f}%)")

def demonstrate_policy_usage(training_data_path: str):
    """Demonstrate how the policy network would use the training data."""
    
    print(f"\nDemonstrating policy network usage with {training_data_path}...")
    
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)
    
    # Show a few examples
    print(f"\nExample training examples:")
    for i in range(min(3, len(training_data))):
        example = training_data[i]
        print(f"\nExample {i+1}:")
        print(f"  State text: {example['state_text'][:100]}...")
        print(f"  Human command: '{example['human_command']}'")
        print(f"  Action: {example['action']}")
        print(f"  Reward: {example['reward']}")
    
    # Show how the policy would process this data
    print(f"\nPolicy Network Processing:")
    print(f"1. State text → DistilBERT encoder → Text embeddings")
    print(f"2. Human command → DistilBERT encoder → Command embeddings") 
    print(f"3. Combine embeddings → Policy head → Action logits")
    print(f"4. Sample action from logits → Action index {example['action']}")

if __name__ == "__main__":
    # Convert all trajectories to JSON
    pkl_file = "generated_data/trajectories_no_commands/all_trajectories.pkl"
    json_file = "generated_data/trajectories_no_commands/all_trajectories.json"
    
    if os.path.exists(pkl_file):
        convert_trajectories_to_json(pkl_file, json_file)
        
        # Create policy training data
        training_data_file = "generated_data/trajectories_no_commands/policy_training_data.json"
        create_policy_training_data(json_file, training_data_file)
        
        # Demonstrate usage
        demonstrate_policy_usage(training_data_file)
    else:
        print(f"PKL file not found: {pkl_file}")
        print("Please run generate_trajectory_data.py first.")
