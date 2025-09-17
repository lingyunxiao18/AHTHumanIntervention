#!/usr/bin/env python3
"""
Convert existing trajectory data to medium-level action format.
This script takes the existing trajectory data and converts low-level actions to medium-level actions.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from shared.envs.envs.overcooked.overcooked_ai_py.planning.planners import MediumLevelActionManager

def convert_low_level_action_to_medium_level_action(mdp: OvercookedGridworld, state: OvercookedState, 
                                                   player_idx: int, low_level_action: int) -> int:
    """
    Convert a low-level action to the corresponding medium-level action index.
    
    Args:
        mdp: The Overcooked MDP
        state: Current state
        player_idx: Player index (0 or 1)
        low_level_action: Low-level action index (0-5)
    
    Returns:
        Medium-level action index
    """
    # Initialize medium-level action manager
    ml_params = {
        "wait_allowed": True,
        "counter_drop": mdp.get_counter_locations(),
        "counter_pickup": mdp.get_counter_locations(),
        "same_motion_goals": True,
        "start_orientations": False,
        "counter_goals": mdp.get_counter_locations()
    }
    
    ml_action_manager = MediumLevelActionManager(mdp, ml_params)
    
    # Get available medium-level actions for the player
    player = state.players[player_idx]
    available_ml_actions = ml_action_manager.get_medium_level_actions(state, player)
    
    if len(available_ml_actions) == 0:
        return 0  # Default to first action if none available
    
    # Convert low-level action to motion goal
    from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
    
    # Map low-level action to expected motion goal
    if low_level_action == 0:  # STAY
        target_motion_goal = (player.position, player.orientation)
    elif low_level_action == 1:  # NORTH
        target_motion_goal = (player.position, Direction.NORTH)
    elif low_level_action == 2:  # SOUTH
        target_motion_goal = (player.position, Direction.SOUTH)
    elif low_level_action == 3:  # EAST
        target_motion_goal = (player.position, Direction.EAST)
    elif low_level_action == 4:  # WEST
        target_motion_goal = (player.position, Direction.WEST)
    elif low_level_action == 5:  # INTERACT
        # For INTERACT, find the closest interactable feature
        feature_locations = (
            mdp.get_onion_dispenser_locations() +
            mdp.get_dish_dispenser_locations() +
            mdp.get_pot_locations() +
            mdp.get_serving_locations()
        )
        
        # Find closest feature and create motion goal facing it
        min_dist = float('inf')
        target_motion_goal = None
        
        for feature_pos in feature_locations:
            for direction in Direction.ALL_DIRECTIONS:
                adjacent_pos = Action.move_in_direction(feature_pos, direction)
                if adjacent_pos in mdp.get_valid_player_positions():
                    dist = abs(adjacent_pos[0] - player.position[0]) + abs(adjacent_pos[1] - player.position[1])
                    if dist < min_dist:
                        min_dist = dist
                        target_motion_goal = (adjacent_pos, Direction.OPPOSITE_DIRECTIONS[direction])
        
        if target_motion_goal is None:
            target_motion_goal = (player.position, player.orientation)
    
    # Find the index of the target motion goal in available actions
    try:
        ml_action_index = available_ml_actions.index(target_motion_goal)
    except ValueError:
        # If exact match not found, find closest match
        ml_action_index = 0
        min_dist = float('inf')
        
        for i, ml_action in enumerate(available_ml_actions):
            ml_pos, ml_orient = ml_action
            target_pos, target_orient = target_motion_goal
            
            # Calculate distance between positions
            dist = abs(ml_pos[0] - target_pos[0]) + abs(ml_pos[1] - target_pos[1])
            if dist < min_dist:
                min_dist = dist
                ml_action_index = i
    
    return ml_action_index

def convert_trajectory_data(input_file: str, output_file: str, mdp: OvercookedGridworld):
    """
    Convert trajectory data from low-level actions to medium-level actions.
    
    Args:
        input_file: Path to input trajectory file
        output_file: Path to output trajectory file
        mdp: The Overcooked MDP
    """
    print(f"Converting {input_file} to {output_file}...")
    
    # Load input data
    with open(input_file, 'rb') as f:
        trajectories = pickle.load(f)
    
    converted_trajectories = []
    
    for traj_idx, trajectory in enumerate(trajectories):
        print(f"  Converting trajectory {traj_idx + 1}/{len(trajectories)}")
        
        converted_trajectory = {
            'pair_name': trajectory['pair_name'],
            'trajectory_id': trajectory['trajectory_id'],
            'states': trajectory['states'],
            'state_texts': trajectory['state_texts'],
            'low_level_actions': trajectory['actions'],  # Keep original for reference
            'medium_level_actions': [],  # New medium-level actions
            'rewards': trajectory['rewards'],
            'dones': trajectory['dones'],
            'human_commands': trajectory['human_commands'],
            'episode_length': trajectory['episode_length'],
            'total_reward': trajectory['total_reward']
        }
        
        # Convert each step
        for step_idx, (state, low_level_actions) in enumerate(zip(trajectory['states'], trajectory['actions'])):
            # Convert low-level actions to medium-level actions
            medium_level_actions = []
            for player_idx, low_level_action in enumerate(low_level_actions):
                ml_action = convert_low_level_action_to_medium_level_action(
                    mdp, state, player_idx, low_level_action
                )
                medium_level_actions.append(ml_action)
            
            converted_trajectory['medium_level_actions'].append(medium_level_actions)
        
        converted_trajectories.append(converted_trajectory)
    
    # Save converted data
    with open(output_file, 'wb') as f:
        pickle.dump(converted_trajectories, f)
    
    print(f"  Saved {len(converted_trajectories)} converted trajectories")

def create_training_data_json(trajectories: List[Dict], output_file: str):
    """
    Create training data JSON file from trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        output_file: Path to output JSON file
    """
    training_data = []
    
    for trajectory in trajectories:
        for step_idx in range(len(trajectory['states'])):
            # Get data for this step
            state_text = trajectory['state_texts'][step_idx]
            human_command = trajectory['human_commands'][step_idx]
            medium_level_action = trajectory['medium_level_actions'][step_idx][0]  # Use player 0
            
            training_data.append({
                'state_text': state_text,
                'human_command': human_command,
                'medium_level_action': medium_level_action
            })
    
    # Save as JSON
    with open(output_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Created training data JSON with {len(training_data)} samples")

def main():
    """Main function to convert trajectory data."""
    
    print("=== Converting Trajectory Data to Medium-Level Actions ===\n")
    
    # Create MDP
    mdp = OvercookedGridworld.from_layout_name("random3")
    
    # Define input and output directories
    input_dirs = [
        "generated_data/trajectories_no_commands",
        "generated_data/trajectories_with_commands"
    ]
    
    output_dirs = [
        "generated_data/trajectories_medium_level_no_commands",
        "generated_data/trajectories_medium_level_with_commands"
    ]
    
    # Process each directory
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        if not os.path.exists(input_dir):
            print(f"Input directory {input_dir} does not exist, skipping...")
            continue
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing {input_dir} -> {output_dir}")
        
        # Find all trajectory files
        trajectory_files = []
        for filename in os.listdir(input_dir):
            if filename.endswith('_trajectories.pkl'):
                trajectory_files.append(filename)
        
        all_trajectories = []
        
        # Convert each file
        for filename in trajectory_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            convert_trajectory_data(input_path, output_path, mdp)
            
            # Load converted data for training data creation
            with open(output_path, 'rb') as f:
                trajectories = pickle.load(f)
                all_trajectories.extend(trajectories)
        
        # Create training data JSON
        training_data_file = os.path.join(output_dir, "policy_training_data.json")
        create_training_data_json(all_trajectories, training_data_file)
        
        # Create summary
        summary_file = os.path.join(output_dir, "trajectory_summary.json")
        summary = {
            'total_trajectories': len(all_trajectories),
            'total_steps': sum(len(traj['states']) for traj in all_trajectories),
            'pairs': {}
        }
        
        # Group by pair
        for trajectory in all_trajectories:
            pair_name = trajectory['pair_name']
            if pair_name not in summary['pairs']:
                summary['pairs'][pair_name] = {
                    'num_trajectories': 0,
                    'total_reward': 0.0,
                    'avg_length': 0.0
                }
            
            summary['pairs'][pair_name]['num_trajectories'] += 1
            summary['pairs'][pair_name]['total_reward'] += trajectory['total_reward']
            summary['pairs'][pair_name]['avg_length'] += trajectory['episode_length']
        
        # Calculate averages
        for pair_name in summary['pairs']:
            pair_data = summary['pairs'][pair_name]
            num_traj = pair_data['num_trajectories']
            pair_data['avg_reward'] = pair_data['total_reward'] / num_traj
            pair_data['avg_length'] = pair_data['avg_length'] / num_traj
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Created summary: {summary_file}")
    
    print("\n=== Conversion Complete ===")
    print("\nNext steps:")
    print("1. Run generate_trajectory_data_medium_level.py to generate new data")
    print("2. Run train_medium_level_text_policy.py to train the policy")
    print("3. Use llm_medium_level_command_translator.py for command translation")

if __name__ == "__main__":
    main()
