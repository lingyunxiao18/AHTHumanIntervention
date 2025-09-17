#!/usr/bin/env python3
"""
Generate trajectory data WITH human commands using LLM command translator
"""

import json
import pickle
import os
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

# Add the project root to the path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from language.llm_command_translator import LLMCommandTranslator, load_commands
from shared.utils.state_to_text import describe_state
from language.coordinated_agent import CoordinatedAgent
from specialized_agents_bfs_fixed import OnionCollectorAgent, DishWaiterAgent, RandomAgent, PotWatcherAgent, ServingSpecialistAgent
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedGridworld
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Direction, Action

class AgentPair:
    """Wrapper for a pair of agents."""
    
    def __init__(self, agent0, agent1, name):
        self.agent0 = agent0
        self.agent1 = agent1
        self.name = name
    
    def set_mdp(self, mdp):
        """Set MDP for both agents."""
        if hasattr(self.agent0, 'set_mdp'):
            self.agent0.set_mdp(mdp)
        else:
            self.agent0.mdp = mdp
            
        if hasattr(self.agent1, 'set_mdp'):
            self.agent1.set_mdp(mdp)
        else:
            self.agent1.mdp = mdp

def create_agent_pairs():
    """Create agent pairs with CoordinatedAgent as Player0 (ego agent)."""
    # Create MDP for agent initialization
    mdp = OvercookedGridworld.from_layout_name("random3")
    
    # Player0 is always CoordinatedAgent
    player0_agent = CoordinatedAgent(agent_index=0, mdp=mdp)
    
    # Player1 agents (partners)
    player1_agents = [
        OnionCollectorAgent(agent_index=1, mdp=mdp),
        DishWaiterAgent(agent_index=1, mdp=mdp), 
        RandomAgent(agent_index=1, mdp=mdp),
        PotWatcherAgent(agent_index=1, mdp=mdp),
        ServingSpecialistAgent(agent_index=1, mdp=mdp),
        CoordinatedAgent(agent_index=1, mdp=mdp)  # Also test coordinated vs coordinated
    ]
    
    agent_names = [
        "onion_collector", 
        "dish_waiter",
        "random",
        "pot_watcher",
        "serving_specialist",
        "coordinated"
    ]
    
    pairs = []
    
    for i, player1_agent in enumerate(player1_agents):
        pair_name = f"coordinated_vs_{agent_names[i]}"
        pairs.append(AgentPair(player0_agent, player1_agent, pair_name))
    
    return pairs

def convert_action_to_environment_format(action):
    """Convert agent action to environment format."""
    if isinstance(action, int):
        if action == 0:
            return Action.STAY
        elif action == 1:
            return Direction.NORTH
        elif action == 2:
            return Direction.SOUTH
        elif action == 3:
            return Direction.EAST
        elif action == 4:
            return Direction.WEST
        elif action == 5:
            return Action.INTERACT
    elif isinstance(action, str):
        if action == "STAY":
            return Action.STAY
        elif action == "NORTH":
            return Direction.NORTH
        elif action == "SOUTH":
            return Direction.SOUTH
        elif action == "EAST":
            return Direction.EAST
        elif action == "WEST":
            return Direction.WEST
        elif action == "INTERACT":
            return Action.INTERACT
    elif isinstance(action, tuple):
        return action
    
    # Default fallback
    return Action.STAY

def convert_action_to_index(action):
    """Convert action to index for consistency."""
    if action == Action.STAY:
        return 0
    elif action == Direction.NORTH:
        return 1
    elif action == Direction.SOUTH:
        return 2
    elif action == Direction.EAST:
        return 3
    elif action == Direction.WEST:
        return 4
    elif action == Action.INTERACT:
        return 5
    else:
        return 0  # Default to STAY

def check_command_feasibility(command: str, state: OvercookedState, mdp: OvercookedGridworld) -> str:
    """
    Check if a command is feasible in the current state and potentially repair it.
    Returns the feasible command or empty string if not feasible.
    """
    command_lower = command.lower()
    player = state.players[0]  # Player0 (ego agent)
    player_pos = player.position
    player_holding = player.get_object().name if player.has_object() else None
    
    # Get game state information
    pots_state = []
    for pot_pos in mdp.get_pot_locations():
        # Check if there's a soup in this pot
        soup_obj = None
        for obj_pos, obj in state.objects.items():
            if obj_pos == pot_pos and obj.name == "soup":
                soup_obj = obj
                break
        
        if soup_obj:
            soup_type, num_items, cook_time = soup_obj.state
            pots_state.append({
                'position': pot_pos,
                'type': soup_type,
                'items': num_items,
                'cook_time': cook_time,
                'ready': cook_time == 0 and num_items >= 3
            })
        else:
            pots_state.append({
                'position': pot_pos,
                'type': None,
                'items': 0,
                'cook_time': -1,
                'ready': False
            })
    
    # Check for ready soups
    ready_soups = [pot for pot in pots_state if pot['ready']]
    cooking_soups = [pot for pot in pots_state if pot['cook_time'] >= 0 and not pot['ready']]
    empty_pots = [pot for pot in pots_state if pot['cook_time'] == -1]
    
    # Feasibility checks based on command content
    if any(word in command_lower for word in ["serve", "deliver", "bring soup", "take soup"]):
        # Check if player is holding soup
        if player_holding != "soup":
            # Check if there are ready soups to pick up
            if not ready_soups:
                return ""  # No soup to serve
            else:
                # Repair: change to "pick up soup"
                return "Pick up the soup from the pot"
        else:
            # Player is holding soup, check if near serving area
            serving_locations = mdp.get_serving_locations()
            if not serving_locations:
                return ""  # No serving areas
            return command  # Feasible
    
    elif any(word in command_lower for word in ["pick up soup", "get soup", "grab soup"]):
        if not ready_soups:
            return ""  # No soup ready to pick up
        return command  # Feasible
    
    elif any(word in command_lower for word in ["cook", "put onion", "add onion"]):
        # Check if player is holding onion
        if player_holding != "onion":
            return ""  # Not holding onion
        # Check if there are empty pots
        if not empty_pots:
            return ""  # No empty pots
        return command  # Feasible
    
    elif any(word in command_lower for word in ["pick up onion", "get onion", "grab onion"]):
        # Always feasible (unlimited supply)
        return command
    
    elif any(word in command_lower for word in ["pick up dish", "get dish", "grab dish", "pick up plate"]):
        # Always feasible (unlimited supply)
        return command
    
    elif any(word in command_lower for word in ["finish", "complete", "ready"]):
        # Check if there are cooking soups
        if not cooking_soups:
            return ""  # No soups cooking
        # Repair: change to "wait for soup to finish"
        return "Wait for the soup to finish cooking"
    
    elif any(word in command_lower for word in ["move", "go", "turn", "face", "stay", "wait"]):
        # Movement commands are always feasible
        return command
    
    else:
        # Unknown command, assume feasible
        return command

def generate_trajectories_with_commands(pairs: List[AgentPair], num_trajectories_per_pair: int = 667, 
                                      horizon: int = 200, layout_name: str = "random3",
                                      output_dir: str = "generated_data/trajectories_with_commands"):
    """Generate trajectories with human commands."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load human commands
    print("Loading human commands...")
    commands = load_commands()
    print(f"Loaded {len(commands)} commands")
    
    # Create MDP
    print(f"Creating MDP with layout: {layout_name}")
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    
    # Create LLM command translator
    translator = LLMCommandTranslator(mdp)
    
    # Create random start state function
    def random_start_state_fn():
        """Create a random starting state."""
        # Get valid positions
        valid_positions = []
        for y in range(len(mdp.terrain_mtx)):
            for x in range(len(mdp.terrain_mtx[0])):
                if mdp.terrain_mtx[y][x] == " ":
                    valid_positions.append((x, y))
        
        # Randomly select two positions
        if len(valid_positions) >= 2:
            pos1, pos2 = random.sample(valid_positions, 2)
            return OvercookedState.from_player_positions([pos1, pos2], order_list=mdp.start_order_list)
        else:
            return mdp.get_standard_start_state()
    
    all_trajectories = []
    all_training_examples = []
    
    print(f"\nGenerating {num_trajectories_per_pair} trajectories per pair with human commands")
    print(f"Horizon: {horizon}")
    print(f"Layout: {layout_name}")
    print(f"Output directory: {output_dir}")
    
    # Create agent pairs
    agent_pairs = create_agent_pairs()
    print(f"\nCreated {len(agent_pairs)} agent pairs:")
    for pair in agent_pairs:
        print(f"  - {pair.name}")
    
    total_examples = 0
    max_tokens = 0
    token_counts = []
    
    for pair in agent_pairs:
        print(f"\nGenerating {num_trajectories_per_pair} trajectories for {pair.name}...")
        
        # Set MDP for agents
        pair.set_mdp(mdp)
        
        trajectories = []
        
        for traj_idx in tqdm(range(num_trajectories_per_pair), desc=pair.name):
            # Create environment with random starting positions
            env = OvercookedEnv(mdp, start_state_fn=random_start_state_fn)
            state = env.state
            
            # Initialize command chance for this trajectory
            command_chance = 0.3  # 30% chance initially
            
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'human_commands': [],
                'state_texts': [],
                'agent_pair': pair.name,
                'trajectory_id': len(all_trajectories) + traj_idx
            }
            
            # Generate trajectory
            for step in range(horizon):
                # Get agent actions
                action0 = pair.agent0.action(state)  # Player0 (CoordinatedAgent)
                action1 = pair.agent1.action(state)  # Player1 (partner agent)
                
                # Convert actions to environment format
                env_action0 = convert_action_to_environment_format(action0)
                env_action1 = convert_action_to_environment_format(action1)
                
                # Randomly decide if this step should have a human command for Player0
                # We want about 30% of steps to have commands, but at most 1 per trajectory
                has_command = random.random() < command_chance
                
                if has_command:
                    # Select a random human command
                    human_command = random.choice(commands)
                    
                    # Check feasibility and potentially repair the command
                    feasible_command = check_command_feasibility(human_command, state, mdp)
                    
                    if feasible_command:
                        # Translate feasible command to action plan for Player0 only
                        try:
                            action_plan = translator.translate_command(feasible_command, state, agent_idx=0)
                            
                            # Use the translated action instead of Player0's action
                            if action_plan.low_level_actions:
                                # Use the first action from the plan
                                translated_action = action_plan.low_level_actions[0]
                                env_action0 = convert_action_to_environment_format(translated_action)
                            
                            # Set command chance to 0 for the rest of this trajectory
                            command_chance = 0.0
                        except Exception as e:
                            # If translation fails, use original action
                            human_command = ""
                            print(f"Translation failed: {e}")
                    else:
                        # Command is not feasible, use empty command
                        human_command = ""
                else:
                    human_command = ""
                
                # Step environment with final actions (Player0 may be modified by human command)
                joint_action = (env_action0, env_action1)
                next_state, reward, done, info = env.step(joint_action)
                
                # Get state text (using ctx mode to save space)
                state_text = describe_state(mdp, state, mode="ctx")
                
                # Check token count
                combined_text = state_text + " " + human_command
                token_count = len(combined_text.split())
                token_counts.append(token_count)
                max_tokens = max(max_tokens, token_count)
                
                # Store trajectory data
                trajectory['states'].append(state)
                trajectory['actions'].append(convert_action_to_index(env_action0))
                trajectory['rewards'].append(reward)
                trajectory['human_commands'].append(human_command)
                trajectory['state_texts'].append(state_text)
                
                # Create training example
                training_example = {
                    'state_text': state_text,
                    'human_command': human_command,
                    'action': convert_action_to_index(env_action0),
                    'reward': reward,
                    'trajectory_id': trajectory['trajectory_id'],
                    'step': step,
                    'agent_pair': pair.name
                }
                all_training_examples.append(training_example)
                total_examples += 1
                
                state = next_state
                
                if done:
                    break
            
            trajectories.append(trajectory)
        
        # Save individual pair trajectories
        pair_filename = f"{pair.name}_trajectories.pkl"
        pair_path = os.path.join(output_dir, pair_filename)
        with open(pair_path, 'wb') as f:
            pickle.dump(trajectories, f)
        print(f"Saved {len(trajectories)} trajectories to {pair_path}")
        
        all_trajectories.extend(trajectories)
    
    # Save all trajectories
    all_trajectories_path = os.path.join(output_dir, "all_trajectories.pkl")
    with open(all_trajectories_path, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    # Save training data
    training_data_path = os.path.join(output_dir, "policy_training_data.json")
    with open(training_data_path, 'w') as f:
        json.dump(all_training_examples, f, indent=2)
    
    # Save metadata
    metadata = {
        'num_trajectories_per_pair': num_trajectories_per_pair,
        'total_trajectories': len(all_trajectories),
        'total_training_examples': len(all_training_examples),
        'horizon': horizon,
        'layout_name': layout_name,
        'agent_pairs': [pair.name for pair in agent_pairs],
        'max_tokens': max_tokens,
        'avg_tokens': np.mean(token_counts),
        'std_tokens': np.std(token_counts),
        'description': 'Trajectory data WITH human commands, using LLM command translator'
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nGeneration complete!")
    print(f"Total trajectories generated: {len(all_trajectories)}")
    print(f"Total training examples: {len(all_training_examples)}")
    print(f"All trajectories saved to: {all_trajectories_path}")
    print(f"Training data saved to: {training_data_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    print(f"\nToken statistics:")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Average tokens: {np.mean(token_counts):.1f}")
    print(f"  Std tokens: {np.std(token_counts):.1f}")
    
    return all_trajectories, all_training_examples, metadata

if __name__ == "__main__":
    # Generate 400,000 examples (667 trajectories per pair * 6 pairs * 200 steps * 0.3 command rate ≈ 400K)
    # With at most 1 command per trajectory, we expect ~400K examples
    agent_pairs = create_agent_pairs()
    trajectories, training_examples, metadata = generate_trajectories_with_commands(
        pairs=agent_pairs,
        num_trajectories_per_pair=667,
        horizon=200,
        layout_name="random3"
    )
