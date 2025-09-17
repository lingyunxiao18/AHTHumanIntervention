#!/usr/bin/env python3
"""
Convert micro-task trajectories to training data format for the text-based policy.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs', 'overcooked'))

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from shared.utils.state_to_text import describe_state
import pickle
import json
import random

def convert_micro_tasks_to_training_data():
    """Convert micro-task trajectories to training data format."""
    print("🔄 Converting micro-task trajectories to training data format")
    print("=" * 60)
    
    # Load all micro-task trajectories
    micro_task_dir = "generated_data/micro_task_sanity"
    
    # List of all micro-task files
    micro_task_files = [
        "pickup_onion_trajectories.pkl",
        "navigate_to_pot_trajectories.pkl", 
        "interact_with_pot_trajectories.pkl",
        "turn_to_face_onion_trajectories.pkl",
        "turn_to_face_pot_trajectories.pkl",
        "pickup_multiple_onions_trajectories.pkl",
        "pickup_plate_trajectories.pkl",
        "pickup_cooked_soup_trajectories.pkl",
        "deliver_soup_trajectories.pkl"
    ]
    
    all_training_data = []
    mdp = OvercookedGridworld.from_layout_name("random3")
    
    for task_file in micro_task_files:
        print(f"Processing {task_file}...")
        
        try:
            with open(os.path.join(micro_task_dir, task_file), "rb") as f:
                trajectories = pickle.load(f)
            
            print(f"  Loaded {len(trajectories)} trajectories")
            
            # Convert each trajectory to training samples
            for traj_idx, trajectory in enumerate(trajectories):
                states = trajectory['states']
                actions = trajectory['actions']
                
                # Convert each step to a training sample
                for step_idx in range(len(states) - 1):  # -1 because we need next action
                    if step_idx < len(actions):
                        state = states[step_idx]
                        action_pair = actions[step_idx]
                        ego_action = action_pair[0] if action_pair else "STAY"
                        
                        # Convert action string to index
                        action_idx = convert_action_to_index(ego_action)
                        
                        # Generate state text description
                        try:
                            state_text = describe_state(mdp, state, mode="english")
                        except Exception as e:
                            print(f"    Error describing state: {e}")
                            continue
                        
                        # Create training sample
                        training_sample = {
                            'state_text': state_text,
                            'human_command': "",  # Empty command for now
                            'action': action_idx
                        }
                        
                        all_training_data.append(training_sample)
            
            print(f"  Converted to {len(all_training_data)} training samples")
            
        except Exception as e:
            print(f"  Error processing {task_file}: {e}")
            continue
    
    print(f"\nTotal training samples: {len(all_training_data)}")
    
    # Split into train/val (80/20)
    random.shuffle(all_training_data)
    split_idx = int(0.8 * len(all_training_data))
    train_data = all_training_data[:split_idx]
    val_data = all_training_data[split_idx:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Save training data
    output_dir = "generated_data/micro_task_training"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save with commands (empty for now)
    with open(os.path.join(output_dir, "policy_training_data.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    
    # Save without commands
    with open(os.path.join(output_dir, "policy_training_data_no_commands.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    
    # Save validation data
    with open(os.path.join(output_dir, "validation_data.json"), "w") as f:
        json.dump(val_data, f, indent=2)
    
    print(f"\nTraining data saved to: {output_dir}")
    print("Files created:")
    print(f"  - policy_training_data.json ({len(train_data)} samples)")
    print(f"  - policy_training_data_no_commands.json ({len(train_data)} samples)")
    print(f"  - validation_data.json ({len(val_data)} samples)")

def convert_action_to_index(action_str):
    """Convert action string to index."""
    action_mapping = {
        "STAY": 0,
        "MOVE_N": 1,
        "MOVE_S": 2,
        "MOVE_E": 3,
        "MOVE_W": 4,
        "interact": 5,
        "INTERACT": 5
    }
    
    return action_mapping.get(action_str, 0)  # Default to STAY if unknown

def main():
    convert_micro_tasks_to_training_data()

if __name__ == "__main__":
    main()
