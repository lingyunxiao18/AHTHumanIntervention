#!/usr/bin/env python3
"""
Trajectory Visualizer

This script generates diverse trajectories with different starting positions
and visualizes each step with ASCII art representation of the Overcooked environment.
"""

import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import sys
import time
import itertools

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

# Import our coordinated agent
from language.coordinated_agent import CoordinatedAgent

@dataclass
class TrajectoryStep:
    """A single step in the trajectory with visualization data."""
    step_number: int
    state: OvercookedState
    action0: str
    action1: str
    reward: float
    done: bool
    layout_visualization: str
    player0_pos: Tuple[int, int]
    player1_pos: Tuple[int, int]
    player0_holding: str
    player1_holding: str
    player0_orientation: str
    player1_orientation: str

@dataclass
class TrajectoryResult:
    """Result of a complete trajectory."""
    trajectory_steps: List[TrajectoryStep]
    starting_positions: Tuple[Tuple[int, int], Tuple[int, int]]
    total_reward: float
    total_steps: int
    final_done: bool
    action_distribution: Dict[str, Dict[str, int]]

class TrajectoryVisualizer:
    """Visualizes trajectories step by step with diverse starting positions."""
    
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
        
        # Get all valid starting positions
        self.valid_positions = self.mdp.get_valid_player_positions()
        self.valid_joint_positions = self.mdp.get_valid_joint_player_positions()
        
        print(f"🎨 Trajectory Visualizer initialized for layout: {layout_name}")
        print(f"   Layout size: {self.width}x{self.height}")
        print(f"   Valid positions: {len(self.valid_positions)}")
        print(f"   Valid joint positions: {len(self.valid_joint_positions)}")
    
    def _get_orientation_symbol(self, orientation: Tuple[int, int]) -> str:
        """Convert orientation to arrow symbol."""
        if orientation == (0, -1):  # North
            return "↑"
        elif orientation == (0, 1):  # South
            return "↓"
        elif orientation == (1, 0):  # East
            return "→"
        elif orientation == (-1, 0):  # West
            return "←"
        else:
            return "·"
    
    def _get_terrain_symbol(self, terrain: str) -> str:
        """Convert terrain to symbol."""
        if terrain == 'X':  # Wall
            return "█"
        elif terrain == ' ':  # Floor
            return " "
        elif terrain == 'O':  # Onion dispenser
            return "🧅"
        elif terrain == 'D':  # Dish dispenser
            return "🍽️"
        elif terrain == 'P':  # Pot
            return "🍳"
        elif terrain == 'S':  # Serving area
            return "🍽️"
        else:
            return terrain
    
    def _get_object_symbol(self, obj_name: str) -> str:
        """Convert object name to symbol."""
        if obj_name == "onion":
            return "🧅"
        elif obj_name == "dish":
            return "🍽️"
        elif obj_name == "soup":
            return "🥘"
        else:
            return "?"
    
    def _get_soup_info(self, obj) -> str:
        """Get detailed soup information."""
        if obj.name == "soup" and hasattr(obj, 'state') and obj.state:
            soup_type, num_items, cook_time = obj.state
            return f"🥘({num_items}/{3},{cook_time}/{20})"
        return "🥘"
    
    def _create_layout_visualization(self, state: OvercookedState) -> str:
        """Create ASCII visualization of the layout with players and objects."""
        visualization = []
        
        # Create the base layout
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                terrain = self.mdp.terrain_mtx[y][x]
                terrain_symbol = self._get_terrain_symbol(terrain)
                row += terrain_symbol
            visualization.append(list(row))
        
        # Add objects
        for obj in state.objects.values():
            x, y = obj.position
            if 0 <= x < self.width and 0 <= y < self.height:
                if obj.name == "soup":
                    obj_symbol = self._get_soup_info(obj)
                else:
                    obj_symbol = self._get_object_symbol(obj.name)
                visualization[y][x] = obj_symbol
        
        # Add players
        for i, player in enumerate(state.players):
            x, y = player.position
            if 0 <= x < self.width and 0 <= y < self.height:
                orientation_symbol = self._get_orientation_symbol(player.orientation)
                if i == 0:
                    visualization[y][x] = "👨"  # Player 0
                else:
                    visualization[y][x] = "👩"  # Player 1
        
        # Convert to string
        result = ""
        for row in visualization:
            result += "".join(row) + "\n"
        
        return result
    
    def _convert_action_to_string(self, action) -> str:
        """Convert action to readable string."""
        if isinstance(action, tuple):
            if action == (0, 0):
                return "STAY"
            elif action == (0, -1):
                return "UP"
            elif action == (0, 1):
                return "DOWN"
            elif action == (-1, 0):
                return "LEFT"
            elif action == (1, 0):
                return "RIGHT"
        elif action == "interact":
            return "INTERACT"
        else:
            return "UNKNOWN"
    
    def _get_holding_status(self, player: PlayerState) -> str:
        """Get what the player is holding."""
        if player.has_object():
            obj = player.get_object()
            return obj.name
        else:
            return "nothing"
    
    def _create_custom_start_state(self, player0_pos: Tuple[int, int], player1_pos: Tuple[int, int]) -> OvercookedState:
        """Create a custom starting state with specified player positions."""
        # Create new player states
        player0 = PlayerState(position=player0_pos, orientation=(0, -1), held_object=None)
        player1 = PlayerState(position=player1_pos, orientation=(0, -1), held_object=None)
        
        # Create empty state with no objects
        state = OvercookedState(
            players=[player0, player1],
            objects={},
            order_list=None,
            timestep=0
        )
        
        return state
    
    def generate_trajectory_with_starting_positions(self, player0_pos: Tuple[int, int], player1_pos: Tuple[int, int], max_steps: int = 100) -> TrajectoryResult:
        """Generate a trajectory with specific starting positions."""
        print(f"🎬 Generating trajectory with starting positions: P0={player0_pos}, P1={player1_pos}")
        
        # Create custom starting state
        start_state = self._create_custom_start_state(player0_pos, player1_pos)
        
        # Reset environment with custom state
        self.env.reset()
        self.env.state = start_state
        state = self.env.state
        
        # Create coordinated agents
        agent0 = CoordinatedAgent(0, self.mdp)
        agent1 = CoordinatedAgent(1, self.mdp)
        
        # Reset agents
        agent0.reset()
        agent1.reset()
        
        trajectory_steps = []
        step_number = 0
        
        # Run episode
        while step_number < max_steps:
            # Get actions from both agents
            action0 = agent0.action(state)
            action1 = agent1.action(state)
            
            # Convert actions to strings
            action0_str = self._convert_action_to_string(action0)
            action1_str = self._convert_action_to_string(action1)
            
            # Create layout visualization
            layout_viz = self._create_layout_visualization(state)
            
            # Get player information
            player0 = state.players[0]
            player1 = state.players[1]
            
            player0_holding = self._get_holding_status(player0)
            player1_holding = self._get_holding_status(player1)
            player0_orientation = self._get_orientation_symbol(player0.orientation)
            player1_orientation = self._get_orientation_symbol(player1.orientation)
            
            # Step environment
            try:
                next_state, reward, done, info = self.env.step([action0, action1])
            except Exception as e:
                print(f"⚠️  Step {step_number}: Environment step failed: {e}")
                break
            
            # Create trajectory step
            step = TrajectoryStep(
                step_number=step_number,
                state=state,
                action0=action0_str,
                action1=action1_str,
                reward=reward,
                done=done,
                layout_visualization=layout_viz,
                player0_pos=player0.position,
                player1_pos=player1.position,
                player0_holding=player0_holding,
                player1_holding=player1_holding,
                player0_orientation=player0_orientation,
                player1_orientation=player1_orientation
            )
            
            trajectory_steps.append(step)
            
            # Update state
            state = next_state
            step_number += 1
            
            # Check if episode is done
            if done:
                break
        
        # Calculate action distribution
        action0_counts = {}
        action1_counts = {}
        for step in trajectory_steps:
            action0_counts[step.action0] = action0_counts.get(step.action0, 0) + 1
            action1_counts[step.action1] = action1_counts.get(step.action1, 0) + 1
        
        action_distribution = {
            "player0": action0_counts,
            "player1": action1_counts
        }
        
        result = TrajectoryResult(
            trajectory_steps=trajectory_steps,
            starting_positions=(player0_pos, player1_pos),
            total_reward=sum(step.reward for step in trajectory_steps),
            total_steps=len(trajectory_steps),
            final_done=trajectory_steps[-1].done if trajectory_steps else False,
            action_distribution=action_distribution
        )
        
        print(f"✅ Generated trajectory with {len(trajectory_steps)} steps, total reward: {result.total_reward:.3f}")
        return result
    
    def generate_diverse_trajectories(self, max_trajectories: int = None, max_steps_per_trajectory: int = 100) -> List[TrajectoryResult]:
        """Generate diverse trajectories by exploring different starting positions."""
        print(f"🎭 Generating diverse trajectories...")
        print(f"   Available joint positions: {len(self.valid_joint_positions)}")
        
        # Determine how many trajectories to generate
        if max_trajectories is None:
            max_trajectories = len(self.valid_joint_positions)
        
        num_trajectories = min(max_trajectories, len(self.valid_joint_positions))
        print(f"   Will generate: {num_trajectories} trajectories")
        
        # Sample starting positions (either all or a subset)
        if num_trajectories == len(self.valid_joint_positions):
            starting_positions = self.valid_joint_positions
        else:
            starting_positions = random.sample(self.valid_joint_positions, num_trajectories)
        
        trajectories = []
        
        for i, (player0_pos, player1_pos) in enumerate(starting_positions):
            print(f"\n🎬 Trajectory {i+1}/{num_trajectories}: P0={player0_pos}, P1={player1_pos}")
            
            try:
                trajectory_result = self.generate_trajectory_with_starting_positions(
                    player0_pos, player1_pos, max_steps_per_trajectory
                )
                trajectories.append(trajectory_result)
            except Exception as e:
                print(f"⚠️  Failed to generate trajectory {i+1}: {e}")
                continue
        
        print(f"\n🎉 Successfully generated {len(trajectories)} diverse trajectories!")
        return trajectories
    
    def display_trajectory(self, trajectory_steps: List[TrajectoryStep], delay: float = 1.0):
        """Display the trajectory step by step with animation."""
        print(f"\n🎭 TRAJECTORY VISUALIZATION")
        print("=" * 60)
        print(f"Layout: {self.layout_name}")
        print(f"Total steps: {len(trajectory_steps)}")
        print("=" * 60)
        
        for step in trajectory_steps:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            print(f"🎮 STEP {step.step_number}")
            print("=" * 40)
            
            # Display layout
            print("📋 LAYOUT:")
            print(step.layout_visualization)
            
            # Display player information
            print(f"👨 Player 0: {step.player0_pos} {step.player0_orientation} holding {step.player0_holding}")
            print(f"👩 Player 1: {step.player1_pos} {step.player1_orientation} holding {step.player1_holding}")
            
            # Display actions
            print(f"🎯 Actions: P0={step.action0}, P1={step.action1}")
            print(f"💰 Reward: {step.reward}")
            print(f"🏁 Done: {step.done}")
            
            # Display key locations
            print(f"📍 Key Locations:")
            print(f"   Onion dispensers: {self.onion_locations}")
            print(f"   Pots: {self.pot_locations}")
            print(f"   Serving areas: {self.serving_locations}")
            print(f"   Dish dispensers: {self.dish_locations}")
            
            # Display soup cooking progress
            print(f"🍳 Soup Status:")
            for obj in step.state.objects.values():
                if obj.name == "soup":
                    if hasattr(obj, 'state') and obj.state:
                        soup_type, num_items, cook_time = obj.state
                        status = "READY" if num_items == 3 and cook_time >= 20 else "COOKING"
                        print(f"   Soup at {obj.position}: {soup_type} soup ({num_items}/3 onions, {cook_time}/20 steps) - {status}")
                    else:
                        print(f"   Soup at {obj.position}: Unknown state")
            
            # Wait before next step
            time.sleep(delay)
        
        print("\n🎉 Trajectory visualization complete!")
    
    def save_trajectory_visualization(self, trajectory_steps: List[TrajectoryStep], filename: str = None):
        """Save trajectory visualization to a text file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_data/visualizations/trajectory_visualization_{timestamp}.txt"
        else:
            # Ensure the file is saved in the visualizations directory
            if not filename.startswith("generated_data/visualizations/"):
                filename = f"generated_data/visualizations/{filename}"
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write("🎭 TRAJECTORY VISUALIZATION\n")
            f.write("=" * 60 + "\n")
            f.write(f"Layout: {self.layout_name}\n")
            f.write(f"Total steps: {len(trajectory_steps)}\n")
            f.write("=" * 60 + "\n\n")
            
            for step in trajectory_steps:
                f.write(f"🎮 STEP {step.step_number}\n")
                f.write("-" * 40 + "\n")
                f.write("📋 LAYOUT:\n")
                f.write(step.layout_visualization)
                f.write(f"👨 Player 0: {step.player0_pos} {step.player0_orientation} holding {step.player0_holding}\n")
                f.write(f"👩 Player 1: {step.player1_pos} {step.player1_orientation} holding {step.player1_holding}\n")
                f.write(f"🎯 Actions: P0={step.action0}, P1={step.action1}\n")
                f.write(f"💰 Reward: {step.reward}\n")
                f.write(f"🏁 Done: {step.done}\n")
                f.write(f"📍 Key Locations:\n")
                f.write(f"   Onion dispensers: {self.onion_locations}\n")
                f.write(f"   Pots: {self.pot_locations}\n")
                f.write(f"   Serving areas: {self.serving_locations}\n")
                f.write(f"   Dish dispensers: {self.dish_locations}\n")
                f.write("\n" + "=" * 60 + "\n\n")
        
        print(f"💾 Saved trajectory visualization to: {filename}")
    
    def save_diverse_trajectories_summary(self, trajectories: List[TrajectoryResult], filename: str = None):
        """Save a summary of diverse trajectories to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_data/summaries/diverse_trajectories_summary_{timestamp}.json"
        else:
            # Ensure the file is saved in the summaries directory
            if not filename.startswith("generated_data/summaries/"):
                filename = f"generated_data/summaries/{filename}"
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        summary = {
            "layout_name": self.layout_name,
            "total_trajectories": len(trajectories),
            "layout_info": {
                "width": self.width,
                "height": self.height,
                "valid_positions": len(self.valid_positions),
                "valid_joint_positions": len(self.valid_joint_positions),
                "onion_locations": self.onion_locations,
                "pot_locations": self.pot_locations,
                "serving_locations": self.serving_locations,
                "dish_locations": self.dish_locations
            },
            "trajectories": []
        }
        
        for i, trajectory in enumerate(trajectories):
            trajectory_summary = {
                "trajectory_id": i,
                "starting_positions": {
                    "player0": trajectory.starting_positions[0],
                    "player1": trajectory.starting_positions[1]
                },
                "total_reward": trajectory.total_reward,
                "total_steps": trajectory.total_steps,
                "final_done": trajectory.final_done,
                "action_distribution": trajectory.action_distribution
            }
            summary["trajectories"].append(trajectory_summary)
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Saved diverse trajectories summary to: {filename}")

def main():
    """Main function to generate and visualize diverse trajectories."""
    print("🎨 DIVERSE TRAJECTORY VISUALIZER")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = TrajectoryVisualizer(layout_name="random3")
    
    # Generate diverse trajectories
    print(f"\n📊 Starting Position Analysis:")
    print(f"   Total valid positions: {len(visualizer.valid_positions)}")
    print(f"   Total valid joint positions: {len(visualizer.valid_joint_positions)}")
    
    # Show some example starting positions
    print(f"\n📍 Example starting positions:")
    for i, (pos0, pos1) in enumerate(visualizer.valid_joint_positions[:5]):
        print(f"   {i+1}. P0={pos0}, P1={pos1}")
    
    # Generate diverse trajectories (limit to first 5 for demonstration)
    max_trajectories = min(5, len(visualizer.valid_joint_positions))
    trajectories = visualizer.generate_diverse_trajectories(max_trajectories=max_trajectories, max_steps_per_trajectory=100)
    
    # Save summary
    visualizer.save_diverse_trajectories_summary(trajectories)
    
    # Display one example trajectory
    if trajectories:
        print(f"\n🎭 Displaying example trajectory (first one):")
        visualizer.display_trajectory(trajectories[0].trajectory_steps, delay=0.3)
    
    # Print overall summary
    print(f"\n📊 DIVERSE TRAJECTORIES SUMMARY:")
    print(f"   Total trajectories generated: {len(trajectories)}")
    print(f"   Average steps per trajectory: {np.mean([t.total_steps for t in trajectories]):.1f}")
    print(f"   Average reward per trajectory: {np.mean([t.total_reward for t in trajectories]):.3f}")
    print(f"   Trajectories that completed: {sum(1 for t in trajectories if t.final_done)}")
    
    # Show reward distribution
    rewards = [t.total_reward for t in trajectories]
    print(f"   Reward range: {min(rewards):.3f} to {max(rewards):.3f}")
    print(f"   Reward std: {np.std(rewards):.3f}")

if __name__ == "__main__":
    main()
