#!/usr/bin/env python3
"""
No-Command Trajectory Generator Using Coordinated Agents

This generator creates trajectories using the improved CoordinatedAgent class
for both Player 0 and Player 1, allowing them to play through the environment
naturally without any commands. This generates baseline trajectories for pretraining.
"""

import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

# Import our improved coordinated agent
from language.coordinated_agent import CoordinatedAgent

@dataclass
class NoCommandExample:
    """Training example without command."""
    state_features: List[float]
    action: int
    episode_number: int
    step_number: int
    player_index: int
    reward: float
    done: bool

class NoCommandTrajectoryGenerator:
    """Generates no-command trajectories using coordinated agents."""
    
    def __init__(self, layout_name: str = "random3"):
        self.layout_name = layout_name
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.env = OvercookedEnv(self.mdp, horizon=400)
        
        # Get layout dimensions
        self.height = len(self.mdp.terrain_mtx)
        self.width = len(self.mdp.terrain_mtx[0]) if self.mdp.terrain_mtx else 0
        
        # Get key locations
        self.onion_locations = self.mdp.get_onion_dispenser_locations()
        self.dish_locations = self.mdp.get_dish_dispenser_locations()
        self.pot_locations = self.mdp.get_pot_locations()
        self.serving_locations = self.mdp.get_serving_locations()
        self.counter_locations = self.mdp.get_counter_locations()
        
        print(f"✅ No-Command Trajectory Generator initialized for layout: {layout_name}")
        print(f"   Layout size: {self.width}x{self.height}")
        print(f"   Onion dispensers: {len(self.onion_locations)}")
        print(f"   Pots: {len(self.pot_locations)}")
        print(f"   Serving areas: {len(self.serving_locations)}")
        print(f"   Dish dispensers: {len(self.dish_locations)}")
    
    def _extract_state_features(self, state: OvercookedState, player_index: int) -> List[float]:
        """Extract state features for the given player."""
        player = state.players[player_index]
        other_player = state.players[1 - player_index]  # Other player
        
        features = []
        
        # Player position (normalized)
        features.extend([
            player.position[0] / self.width,
            player.position[1] / self.height
        ])
        
        # Player orientation (one-hot encoded)
        orientation = player.orientation
        if orientation == (0, -1):  # North
            features.extend([1, 0, 0, 0])
        elif orientation == (0, 1):  # South
            features.extend([0, 1, 0, 0])
        elif orientation == (1, 0):  # East
            features.extend([0, 0, 1, 0])
        elif orientation == (-1, 0):  # West
            features.extend([0, 0, 0, 1])
        else:
            features.extend([0, 0, 0, 0])
        
        # Held object (one-hot encoded)
        if player.has_object():
            obj = player.get_object()
            if obj.name == "onion":
                features.extend([1, 0, 0, 0])
            elif obj.name == "dish":
                features.extend([0, 1, 0, 0])
            elif obj.name == "soup":
                features.extend([0, 0, 1, 0])
                # Add soup state if available
                if hasattr(obj, 'state') and obj.state:
                    soup_type, num_items, cook_time = obj.state
                    features.extend([num_items / 3, cook_time / 25])  # Normalized
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0, 0, 1, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Other player position (normalized)
        features.extend([
            other_player.position[0] / self.width,
            other_player.position[1] / self.height
        ])
        
        # Other player orientation (one-hot encoded)
        other_orientation = other_player.orientation
        if other_orientation == (0, -1):  # North
            features.extend([1, 0, 0, 0])
        elif other_orientation == (0, 1):  # South
            features.extend([0, 1, 0, 0])
        elif other_orientation == (1, 0):  # East
            features.extend([0, 0, 1, 0])
        elif other_orientation == (-1, 0):  # West
            features.extend([0, 0, 0, 1])
        else:
            features.extend([0, 0, 0, 0])
        
        # Other player held object (one-hot encoded)
        if other_player.has_object():
            obj = other_player.get_object()
            if obj.name == "onion":
                features.extend([1, 0, 0, 0])
            elif obj.name == "dish":
                features.extend([0, 1, 0, 0])
            elif obj.name == "soup":
                features.extend([0, 0, 1, 0])
            else:
                features.extend([0, 0, 0, 1])
        else:
            features.extend([0, 0, 0, 0])
        
        # Object counts in environment
        onion_count = sum(1 for obj in state.objects.values() if obj.name == "onion")
        dish_count = sum(1 for obj in state.objects.values() if obj.name == "dish")
        soup_count = sum(1 for obj in state.objects.values() if obj.name == "soup")
        
        features.extend([
            onion_count / 10,  # Normalized
            dish_count / 10,
            soup_count / 10
        ])
        
        # Timestep (normalized)
        features.append(state.timestep / 400)  # Normalized by max horizon
        
        return features
    
    def _convert_action_to_index(self, action) -> int:
        """Convert action to index."""
        if isinstance(action, tuple):
            if action == (0, 0):
                return 0  # STAY
            elif action == (0, -1):
                return 1  # UP
            elif action == (0, 1):
                return 2  # DOWN
            elif action == (-1, 0):
                return 3  # LEFT
            elif action == (1, 0):
                return 4  # RIGHT
        elif action == "interact":
            return 5  # INTERACT
        else:
            return 0  # STAY
    
    def generate_single_episode(self, episode_number: int) -> List[NoCommandExample]:
        """Generate a single episode using coordinated agents."""
        # Reset environment and get initial state
        try:
            self.env.reset()  # Don't capture return value
            state = self.env.state  # Access state directly
        except Exception as e:
            print(f"⚠️  Episode {episode_number}: Environment reset failed: {e}")
            return []
        
        # Check if we have a valid state
        if state is None or not hasattr(state, 'players'):
            print(f"⚠️  Episode {episode_number}: Invalid state, skipping...")
            return []
        
        # Create coordinated agents
        agent0 = CoordinatedAgent(0, self.mdp)
        agent1 = CoordinatedAgent(1, self.mdp)
        
        # Reset agents
        agent0.reset()
        agent1.reset()
        
        examples = []
        step_number = 0
        
        # Run episode
        while True:
            # Get actions from both agents
            action0 = agent0.action(state)
            action1 = agent1.action(state)
            
            # Convert actions to indices
            action0_idx = self._convert_action_to_index(action0)
            action1_idx = self._convert_action_to_index(action1)
            
            # Extract state features for both players
            features0 = self._extract_state_features(state, 0)
            features1 = self._extract_state_features(state, 1)
            
            # Step environment
            try:
                next_state, reward, done, info = self.env.step([action0, action1])
            except Exception as e:
                print(f"⚠️  Episode {episode_number}, Step {step_number}: Environment step failed: {e}")
                break
            
            # Convert actions to indices for training examples
            action0_idx = self._convert_action_to_index(action0)
            action1_idx = self._convert_action_to_index(action1)
            
            # Create examples for both players
            example0 = NoCommandExample(
                state_features=features0,
                action=action0_idx,
                episode_number=episode_number,
                step_number=step_number,
                player_index=0,
                reward=reward,
                done=done
            )
            
            example1 = NoCommandExample(
                state_features=features1,
                action=action1_idx,
                episode_number=episode_number,
                step_number=step_number,
                player_index=1,
                reward=reward,
                done=done
            )
            
            examples.extend([example0, example1])
            
            # Update state
            state = next_state
            step_number += 1
            
            # Check if episode is done
            if done or step_number >= 400:
                break
        
        return examples
    
    def generate_dataset(self, target_steps: int = 120000) -> List[NoCommandExample]:
        """Generate dataset with target number of steps."""
        print(f"🎯 Generating no-command dataset with target {target_steps} steps...")
        
        all_examples = []
        episode_number = 0
        total_steps = 0
        
        while total_steps < target_steps:
            episode_number += 1
            
            if episode_number % 10 == 0:
                print(f"   Generated {episode_number} episodes, {total_steps} steps...")
            
            # Generate single episode
            episode_examples = self.generate_single_episode(episode_number)
            
            # Add to total
            all_examples.extend(episode_examples)
            total_steps += len(episode_examples) // 2  # Divide by 2 since each step has 2 players
            
            # Safety check to avoid infinite loop
            if episode_number > 1000:
                print(f"⚠️  Reached maximum episodes ({episode_number}), stopping generation")
                break
        
        print(f"✅ Generated {len(all_examples)} examples from {episode_number} episodes")
        print(f"📈 Target steps: {target_steps}, Actual steps: {total_steps}")
        
        # Analyze dataset
        self._analyze_dataset(all_examples)
        
        return all_examples
    
    def _analyze_dataset(self, examples: List[NoCommandExample]):
        """Analyze the generated dataset."""
        print(f"\n📊 Dataset Analysis:")
        
        # Episode statistics
        episodes = set(ex.episode_number for ex in examples)
        print(f"   Total episodes: {len(episodes)}")
        print(f"   Total examples: {len(examples)}")
        print(f"   Examples per episode: {len(examples) / len(episodes):.1f}")
        
        # Player distribution
        player0_count = sum(1 for ex in examples if ex.player_index == 0)
        player1_count = sum(1 for ex in examples if ex.player_index == 1)
        print(f"   Player 0 examples: {player0_count}")
        print(f"   Player 1 examples: {player1_count}")
        
        # Action distribution
        actions = [ex.action for ex in examples]
        action_dist = {}
        for action in actions:
            action_dist[action] = action_dist.get(action, 0) + 1
        
        print(f"   Action distribution:")
        action_names = {0: 'STAY', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'INTERACT'}
        for action, freq in sorted(action_dist.items()):
            action_name = action_names.get(action, f"Unknown_{action}")
            percentage = freq / len(examples) * 100
            print(f"     {action_name}: {freq} times ({percentage:.1f}%)")
        
        # Reward statistics
        rewards = [ex.reward for ex in examples]
        positive_rewards = [r for r in rewards if r > 0]
        print(f"   Total rewards: {sum(rewards):.3f}")
        print(f"   Positive rewards: {len(positive_rewards)} ({len(positive_rewards)/len(rewards)*100:.1f}%)")
        if positive_rewards:
            print(f"   Average positive reward: {np.mean(positive_rewards):.3f}")
        
        # Feature statistics
        if examples:
            feature_length = len(examples[0].state_features)
            print(f"   State feature length: {feature_length}")
            
            # Check feature variance (only if all features have same length)
            try:
                features_array = np.array([ex.state_features for ex in examples])
                feature_variance = np.var(features_array, axis=0)
                print(f"   Average feature variance: {np.mean(feature_variance):.4f}")
            except ValueError:
                print(f"   Feature variance calculation skipped (inconsistent feature lengths)")

def main():
    """Main function to generate no-command training data."""
    print("🚀 NO-COMMAND TRAJECTORY GENERATOR USING COORDINATED AGENTS")
    print("=" * 70)
    print("This generator uses the improved CoordinatedAgent class for both players")
    print("to generate natural gameplay trajectories without any commands.")
    print("=" * 70)
    
    # Initialize generator
    generator = NoCommandTrajectoryGenerator(layout_name="random3")
    
    # Generate dataset
    target_steps = 120000  # 120k steps as requested
    examples = generator.generate_dataset(target_steps)
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("pretraining_data")
    output_dir.mkdir(exist_ok=True)
    
    # Convert to JSON-serializable format
    json_examples = []
    for ex in examples:
        json_examples.append({
            'state_features': ex.state_features,
            'action': ex.action,
            'episode_number': ex.episode_number,
            'step_number': ex.step_number,
            'player_index': ex.player_index,
            'reward': ex.reward,
            'done': ex.done
        })
    
    # Save to file
    filename = output_dir / f"no_command_training_data_coordinated_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(json_examples, f, indent=2)
    
    # Save statistics
    stats = {
        'total_examples': len(examples),
        'target_steps': target_steps,
        'actual_steps': len(examples) // 2,  # Divide by 2 since each step has 2 players
        'episodes': len(set(ex.episode_number for ex in examples)),
        'average_steps_per_episode': len(examples) / len(set(ex.episode_number for ex in examples)),
        'player0_examples': sum(1 for ex in examples if ex.player_index == 0),
        'player1_examples': sum(1 for ex in examples if ex.player_index == 1),
        'total_reward': sum(ex.reward for ex in examples),
        'positive_rewards': sum(1 for ex in examples if ex.reward > 0),
        'timestamp': timestamp,
        'layout_name': generator.layout_name,
        'agent_type': 'CoordinatedAgent'
    }
    
    # Action distribution
    actions = [ex.action for ex in examples]
    action_dist = {}
    for action in actions:
        action_dist[action] = action_dist.get(action, 0) + 1
    
    action_names = {0: 'STAY', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'INTERACT'}
    stats['action_distribution'] = {}
    for action, freq in sorted(action_dist.items()):
        action_name = action_names.get(action, f"Unknown_{action}")
        stats['action_distribution'][action_name] = {
            'count': freq,
            'percentage': freq/len(examples)*100
        }
    
    stats_filename = output_dir / f"no_command_training_stats_coordinated_{timestamp}.json"
    with open(stats_filename, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n💾 Saved dataset to: {filename}")
    print(f"📊 Saved statistics to: {stats_filename}")
    print(f"📈 Final dataset statistics:")
    print(f"   Total examples: {len(examples)}")
    print(f"   Episodes: {stats['episodes']}")
    print(f"   Average steps per episode: {stats['average_steps_per_episode']:.1f}")
    print(f"   Player 0 examples: {stats['player0_examples']}")
    print(f"   Player 1 examples: {stats['player1_examples']}")
    print(f"   Total reward: {stats['total_reward']:.3f}")
    print(f"   Positive rewards: {stats['positive_rewards']} ({stats['positive_rewards']/len(examples)*100:.1f}%)")

if __name__ == "__main__":
    main()
