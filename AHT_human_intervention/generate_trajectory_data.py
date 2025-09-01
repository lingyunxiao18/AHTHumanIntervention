#!/usr/bin/env python3
"""
Generate trajectory data with no human commands using coordinated agents and scripted agents.
This script generates 500 trajectories per agent pair with horizon 200, using state-to-text conversion.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedGridworld
from envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from language.coordinated_agent import CoordinatedAgent
from language.state_to_text import describe_state

# Import specialized agents
from specialized_agents_bfs_fixed import OnionCollectorAgent, DishWaiterAgent
from specialized_agents import RandomAgent, PotWatcherAgent, ServingSpecialistAgent

class AgentPair:
    """Wrapper for a pair of agents."""
    
    def __init__(self, agent0, agent1):
        self.agent0 = agent0
        self.agent1 = agent1
        
    def set_mdp(self, mdp):
        # Set MDP for agents that have the set_mdp method
        if hasattr(self.agent0, 'set_mdp'):
            self.agent0.set_mdp(mdp)
        else:
            self.agent0.mdp = mdp
            
        if hasattr(self.agent1, 'set_mdp'):
            self.agent1.set_mdp(mdp)
        else:
            self.agent1.mdp = mdp
        
    def reset(self):
        self.agent0.reset()
        self.agent1.reset()
        
    def get_actions(self, state):
        """Get actions for both agents."""
        action0 = self.agent0.action(state)
        action1 = self.agent1.action(state)
        return [action0, action1]

def create_agent_pairs(mdp: OvercookedGridworld) -> List[Tuple[str, AgentPair]]:
    """Create agent pairs with coordinated agent as ego agent."""
    
    # Create all available partner agents
    partner_agents = {
        'onion_collector': OnionCollectorAgent(1, mdp),
        'dish_waiter': DishWaiterAgent(1, mdp),
        'random': RandomAgent(1, mdp),
        'pot_watcher': PotWatcherAgent(1, mdp),
        'serving_specialist': ServingSpecialistAgent(1, mdp),
        'coordinated': CoordinatedAgent(1, mdp)  # Also include coordinated vs coordinated
    }
    
    agent_pairs = []
    
    # Create pairs with coordinated agent as ego agent (agent 0)
    for partner_name, partner_agent in partner_agents.items():
        pair_name = f"coordinated_vs_{partner_name}"
        
        # Create new instances
        ego_agent = CoordinatedAgent(0, mdp)
        partner_agent_instance = type(partner_agent)(1, mdp)
        
        agent_pair = AgentPair(ego_agent, partner_agent_instance)
        agent_pairs.append((pair_name, agent_pair))
    
    return agent_pairs

def convert_action_to_environment_format(action):
    """Convert action to the format expected by the environment."""
    from envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
    
    if isinstance(action, str):
        if action == "interact":
            return Action.INTERACT
        elif action == "STAY":
            return Action.STAY
        elif action == "MOVE_N":
            return Direction.NORTH
        elif action == "MOVE_S":
            return Direction.SOUTH
        elif action == "MOVE_E":
            return Direction.EAST
        elif action == "MOVE_W":
            return Direction.WEST
        else:
            return Action.STAY  # Default to STAY
    elif isinstance(action, tuple):
        # Handle tuple actions like (0, 1) for movement
        if action == (0, 0):
            return Action.STAY
        elif action == (0, -1):
            return Direction.NORTH
        elif action == (0, 1):
            return Direction.SOUTH
        elif action == (1, 0):
            return Direction.EAST
        elif action == (-1, 0):
            return Direction.WEST
        else:
            return Action.STAY  # Default to STAY
    elif isinstance(action, int):
        # Handle integer actions (0-5)
        action_mapping = {
            0: Action.STAY,
            1: Direction.NORTH,
            2: Direction.SOUTH,
            3: Direction.EAST,
            4: Direction.WEST,
            5: Action.INTERACT
        }
        return action_mapping.get(action, Action.STAY)
    else:
        return Action.STAY  # Default to STAY

def convert_action_to_index(action) -> int:
    """Convert action to index for storage."""
    from envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
    
    if isinstance(action, str):
        if action == "interact":
            return 5
        elif action == "STAY":
            return 0
        elif action == "MOVE_N":
            return 1
        elif action == "MOVE_S":
            return 2
        elif action == "MOVE_E":
            return 3
        elif action == "MOVE_W":
            return 4
        else:
            return 0
    elif isinstance(action, tuple):
        if action == (0, 0):
            return 0  # STAY
        elif action == (0, -1):
            return 1  # NORTH
        elif action == (0, 1):
            return 2  # SOUTH
        elif action == (1, 0):
            return 3  # EAST
        elif action == (-1, 0):
            return 4  # WEST
        else:
            return 0
    elif isinstance(action, int):
        return action if 0 <= action <= 5 else 0
    else:
        return 0

def generate_trajectories_for_pair(env: OvercookedEnv, agent_pair: AgentPair, 
                                 pair_name: str, num_trajectories: int = 500, 
                                 horizon: int = 200) -> List[Dict[str, Any]]:
    """Generate trajectories for a specific agent pair."""
    
    trajectories = []
    
    print(f"Generating {num_trajectories} trajectories for {pair_name}...")
    
    for traj_idx in tqdm(range(num_trajectories), desc=pair_name):
        # Reset environment and agents
        env.reset()
        agent_pair.set_mdp(env.mdp)
        agent_pair.reset()
        
        trajectory_data = {
            'pair_name': pair_name,
            'trajectory_id': traj_idx,
            'states': [],
            'state_texts': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'human_commands': [],  # Empty commands as requested
            'episode_length': 0,
            'total_reward': 0.0
        }
        
        state = env.state
        done = False
        step = 0
        
        while not done and step < horizon:
            # Get state text description
            state_text = describe_state(env.mdp, state, kind="english")
            
            # Get actions from both agents
            actions = agent_pair.get_actions(state)
            
            # Store trajectory data
            trajectory_data['states'].append(state)
            trajectory_data['state_texts'].append(state_text)
            trajectory_data['actions'].append([
                convert_action_to_index(actions[0]),
                convert_action_to_index(actions[1])
            ])
            trajectory_data['human_commands'].append("")  # No human commands
            
            # Take step in environment
            try:
                # Convert actions to environment format
                env_actions = []
                for action in actions:
                    env_action = convert_action_to_environment_format(action)
                    env_actions.append(env_action)
                
                # Execute joint action
                state, reward, done, info = env.step(env_actions)
                
                trajectory_data['rewards'].append(reward)
                trajectory_data['dones'].append(done)
                trajectory_data['total_reward'] += reward
                
            except Exception as e:
                print(f"Error in trajectory {traj_idx}, step {step}: {e}")
                break
            
            step += 1
        
        trajectory_data['episode_length'] = step
        trajectories.append(trajectory_data)
    
    return trajectories

def main():
    """Main function to generate all trajectory data."""
    
    # Configuration
    num_trajectories_per_pair = 500
    horizon = 200
    layout_name = "random3"  # You can change this to other layouts
    
    # Create output directory
    output_dir = "generated_data/trajectories_no_commands"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating trajectory data with {num_trajectories_per_pair} trajectories per pair")
    print(f"Horizon: {horizon}")
    print(f"Layout: {layout_name}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create environment with random starting positions
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    
    # Create a random start state function
    def random_start_state_fn():
        valid_positions = mdp.get_valid_joint_player_positions()
        start_pos = valid_positions[np.random.choice(len(valid_positions))]
        return OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
    
    env = OvercookedEnv(mdp, start_state_fn=random_start_state_fn, horizon=horizon)
    
    # Create all agent pairs
    agent_pairs = create_agent_pairs(mdp)
    
    print(f"Created {len(agent_pairs)} agent pairs:")
    for pair_name, _ in agent_pairs:
        print(f"  - {pair_name}")
    print()
    
    # Generate trajectories for each pair
    all_trajectories = []
    
    for pair_name, agent_pair in agent_pairs:
        try:
            trajectories = generate_trajectories_for_pair(
                env, agent_pair, pair_name, 
                num_trajectories_per_pair, horizon
            )
            all_trajectories.extend(trajectories)
            
            # Save trajectories for this pair
            pair_filename = f"{output_dir}/{pair_name}_trajectories.pkl"
            with open(pair_filename, 'wb') as f:
                pickle.dump(trajectories, f)
            
            print(f"Saved {len(trajectories)} trajectories to {pair_filename}")
            
        except Exception as e:
            print(f"Error generating trajectories for {pair_name}: {e}")
            continue
    
    # Save all trajectories together
    all_trajectories_filename = f"{output_dir}/all_trajectories.pkl"
    with open(all_trajectories_filename, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    # Save metadata
    metadata = {
        'num_trajectories_per_pair': num_trajectories_per_pair,
        'total_trajectories': len(all_trajectories),
        'horizon': horizon,
        'layout_name': layout_name,
        'agent_pairs': [pair_name for pair_name, _ in agent_pairs],
        'description': 'Trajectory data with no human commands, using state-to-text conversion'
    }
    
    metadata_filename = f"{output_dir}/metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGeneration complete!")
    print(f"Total trajectories generated: {len(all_trajectories)}")
    print(f"All trajectories saved to: {all_trajectories_filename}")
    print(f"Metadata saved to: {metadata_filename}")
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    episode_lengths = [traj['episode_length'] for traj in all_trajectories]
    total_rewards = [traj['total_reward'] for traj in all_trajectories]
    
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Min episode length: {min(episode_lengths)}")
    print(f"Max episode length: {max(episode_lengths)}")
    print(f"Min total reward: {min(total_rewards):.2f}")
    print(f"Max total reward: {max(total_rewards):.2f}")

if __name__ == "__main__":
    main()
