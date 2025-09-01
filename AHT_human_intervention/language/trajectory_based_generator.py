#!/usr/bin/env python3
"""
Trajectory-Based Training Data Generator for Language-Conditioned Policies

This generator creates diverse training data by:
1. Generating full trajectories using A* pathfinding
2. Sampling multiple states along each trajectory path
3. Creating semantic variations of human commands for each trajectory
4. Focusing on "direct command" intervention type
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import torch
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState, ObjectState

@dataclass
class TrajectoryPoint:
    """A single point along a trajectory with state and action."""
    state: OvercookedState
    action: int
    position: Tuple[int, int]
    command: str
    state_features: List[float]

@dataclass
class TrainingExample:
    """Training example with state, command, and action."""
    state_features: List[float]
    command: str
    action: int
    standardized_intervention: Dict[str, Any]
    macro_actions: List[Dict[str, Any]]

class TrajectoryBasedGenerator:
    """Generates diverse training data using full trajectories."""
    
    def __init__(self, layout_name: str = "random3"):
        self.layout_name = layout_name
        self.overcooked_mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.layout = self.overcooked_mdp.terrain_mtx
        
        # Get layout dimensions
        self.height = len(self.layout)
        self.width = len(self.layout[0]) if self.layout else 0
        
        # Get key locations
        self.onion_locations = self.overcooked_mdp.get_onion_dispenser_locations()
        self.dish_locations = self.overcooked_mdp.get_dish_dispenser_locations()
        self.pot_locations = self.overcooked_mdp.get_pot_locations()
        self.serving_locations = self.overcooked_mdp.get_serving_locations()
        self.counter_locations = self.overcooked_mdp.get_counter_locations()
        
        print(f"✅ Trajectory generator initialized for layout: {layout_name}")
        print(f"   Layout size: {self.width}x{self.height}")
        print(f"   Onion dispensers: {len(self.onion_locations)}")
        print(f"   Pots: {len(self.pot_locations)}")
        print(f"   Serving areas: {len(self.serving_locations)}")
    
    def generate_trajectory_data(self, num_trajectories: int = 100, states_per_trajectory: int = 8) -> List[TrainingExample]:
        """Generate diverse training data using trajectories."""
        print(f"🎯 Generating {num_trajectories} trajectories with {states_per_trajectory} states each...")
        
        all_examples = []
        
        for i in range(num_trajectories):
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{num_trajectories} trajectories...")
            
            # Generate a single trajectory
            trajectory_examples = self._generate_single_trajectory(states_per_trajectory)
            all_examples.extend(trajectory_examples)
        
        print(f"✅ Generated {len(all_examples)} total training examples")
        return all_examples
    
    def _generate_single_trajectory(self, num_states: int) -> List[TrainingExample]:
        """Generate a single trajectory and sample multiple states along it."""
        
        # 1. Create a diverse starting scenario
        start_state = self._create_diverse_start_state()
        
        # 2. Choose a goal (onion dispenser, pot, or serving area)
        goal_type, goal_location = self._choose_goal(start_state)
        
        # 3. Generate A* path from start to goal
        path = self._astar_path(start_state.players[0].position, goal_location)
        
        if not path or len(path) < 2:
            # Fallback: create simple examples
            return self._create_fallback_examples(start_state, num_states)
        
        # 4. Sample states along the path with command diversity
        trajectory_points = self._sample_trajectory_states(start_state, path, goal_type, num_states)
        
        # 5. Convert to training examples
        examples = []
        for point in trajectory_points:
            example = self._create_training_example(point)
            if example:
                examples.append(example)
        
        return examples
    
    def _create_diverse_start_state(self) -> OvercookedState:
        """Create a diverse starting state with varied player positions and object configurations."""
        
        # Randomize player 0 position (avoid walls and objects)
        valid_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.layout[y][x] == ' ':  # Empty space
                    valid_positions.append((x, y))
        
        # Use more diverse starting positions - avoid clustering in center
        if len(valid_positions) > 20:
            # Prioritize edge and corner positions for more diversity
            edge_positions = []
            center_positions = []
            
            for pos in valid_positions:
                x, y = pos
                # Check if position is near edges or corners
                if (x <= 1 or x >= self.width - 2 or y <= 1 or y >= self.height - 2):
                    edge_positions.append(pos)
                else:
                    center_positions.append(pos)
            
            # 70% chance of using edge positions for more diversity
            if edge_positions and random.random() < 0.7:
                player0_pos = random.choice(edge_positions)
            else:
                player0_pos = random.choice(valid_positions)
        else:
            player0_pos = random.choice(valid_positions)
        
        # Randomize player 1 position (different from player 0, prefer opposite side)
        remaining_positions = [p for p in valid_positions if p != player0_pos]
        
        if len(remaining_positions) > 10:
            # Try to place player 1 on opposite side for more interesting scenarios
            p0_x, p0_y = player0_pos
            opposite_positions = []
            same_side_positions = []
            
            for pos in remaining_positions:
                x, y = pos
                # Check if on opposite side
                if (p0_x <= self.width // 2 and x > self.width // 2) or \
                   (p0_x > self.width // 2 and x <= self.width // 2):
                    opposite_positions.append(pos)
                else:
                    same_side_positions.append(pos)
            
            # 60% chance of placing on opposite side
            if opposite_positions and random.random() < 0.6:
                player1_pos = random.choice(opposite_positions)
            else:
                player1_pos = random.choice(remaining_positions)
        else:
            player1_pos = random.choice(remaining_positions)
        
        # Randomize held objects with more variety
        player0_held = self._random_held_object(player0_pos)
        player1_held = self._random_held_object(player1_pos)
        
        # Randomize object positions with more variety
        objects = self._randomize_objects()
        
        # Create player states with more varied orientations
        orientations = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        player0 = PlayerState(
            position=player0_pos,
            orientation=random.choice(orientations),
            held_object=player0_held
        )
        
        player1 = PlayerState(
            position=player1_pos,
            orientation=random.choice(orientations),
            held_object=player1_held
        )
        
        # Create state with more varied timesteps
        state = OvercookedState(
            players=[player0, player1],
            objects=objects,
            order_list=["onion"],
            timestep=random.randint(0, 200)  # Increased range
        )
        
        return state
    
    def _random_held_object(self, player_pos: Tuple[int, int]) -> ObjectState:
        """Randomly choose what the player is holding."""
        choices = [None, "onion", "dish", "soup"]
        weights = [0.6, 0.2, 0.15, 0.05]  # More likely to hold nothing
        
        choice = random.choices(choices, weights=weights)[0]
        if choice is None:
            return None
        
        if choice == "soup":
            # Soup needs state: (soup_type, num_items, cook_time)
            return ObjectState(
                name=choice,
                position=player_pos,  # Set to player position
                state=("onion", 1, 10)  # onion soup, 1 item, cooked for 10 timesteps
            )
        else:
            return ObjectState(
                name=choice,
                position=player_pos  # Set to player position
            )
    
    def _randomize_objects(self) -> Dict[str, ObjectState]:
        """Randomize object positions and states with more variety."""
        objects = {}
        
        # Randomize pot states with more variety
        for i, pot_pos in enumerate(self.pot_locations):
            if random.random() < 0.4:  # Increased chance of having soup
                # More varied soup states
                soup_types = ["onion"]  # For now, just use onion
                soup_type = random.choice(soup_types)
                num_items = random.choice([1, 2, 3])  # Vary number of items
                
                if random.random() < 0.8:  # 80% chance of being ready
                    cook_time = random.randint(10, 20)  # Ready soup
                else:
                    cook_time = random.randint(1, 9)   # Cooking soup
                
                objects[pot_pos] = ObjectState(
                    name="soup",
                    position=pot_pos,
                    state=(soup_type, num_items, cook_time)
                )
        
        # Randomize counter objects with more variety
        counter_sample_size = min(random.randint(4, 8), len(self.counter_locations))
        for counter_pos in random.sample(self.counter_locations, counter_sample_size):
            if random.random() < 0.5:  # 50% chance of having object
                obj_type = random.choice(["onion", "dish", "soup"])
                
                if obj_type == "soup":
                    soup_types = ["onion"]  # For now, just use onion
                    soup_type = random.choice(soup_types)
                    num_items = random.choice([1, 2])
                    cook_time = random.randint(5, 15)
                    objects[counter_pos] = ObjectState(
                        name=obj_type,
                        position=counter_pos,
                        state=(soup_type, num_items, cook_time)
                    )
                else:
                    objects[counter_pos] = ObjectState(
                        name=obj_type,
                        position=counter_pos
                    )
        
        return objects
    
    def _choose_goal(self, state: OvercookedState) -> Tuple[str, Tuple[int, int]]:
        """Choose a goal based on current state with more diversity."""
        player = state.players[0]
        pos = player.position
        
        # Add randomness to goal selection for more diversity
        if random.random() < 0.3:  # 30% chance of random goal regardless of state
            goal_types = ["onion", "pot", "serving"]
            goal_type = random.choice(goal_types)
            
            if goal_type == "onion":
                return "onion", random.choice(self.onion_locations)
            elif goal_type == "pot":
                return "pot", random.choice(self.pot_locations)
            else:  # serving
                return "serving", random.choice(self.serving_locations)
        
        # 70% chance of logical goal based on what player is holding
        if player.has_object() and player.get_object().name == "onion":
            return "pot", random.choice(self.pot_locations)
        
        if player.has_object() and player.get_object().name == "soup":
            return "serving", random.choice(self.serving_locations)
        
        # If not holding anything, go to onion dispenser
        return "onion", random.choice(self.onion_locations)
    
    def _astar_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding from start to goal (ending adjacent to goal, not on it)."""
        try:
            # Simple A* implementation
            open_set = {start}
            came_from = {}
            g_score = {start: 0}
            f_score = {start: self._manhattan_distance(start, goal)}
            
            while open_set:
                current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
                
                # Check if we're adjacent to the goal (not on it)
                if self._manhattan_distance(current, goal) == 1:
                    # Reconstruct path
                    path = []
                    while current in came_from:
                        path.append(current)
                        current = came_from[current]
                    path.append(start)
                    path.reverse()
                    return path
                
                open_set.remove(current)
                
                # Check neighbors
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    if not self._is_valid_position(neighbor):
                        continue
                    
                    tentative_g = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self._manhattan_distance(neighbor, goal)
                        open_set.add(neighbor)
            
            return []
            
        except Exception as e:
            print(f"A* pathfinding failed: {e}")
            return []
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not a wall)."""
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        # Allow empty spaces and goal locations (O, P, S)
        return self.layout[y][x] in [' ', 'O', 'P', 'S']
    
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _sample_trajectory_states(self, start_state: OvercookedState, path: List[Tuple[int, int]], 
                                 goal_type: str, num_states: int) -> List[TrajectoryPoint]:
        """Sample multiple states along the trajectory path with more diversity."""
        points = []
        
        # Sample states along the path with better distribution
        if len(path) <= num_states:
            # Use all path points
            sampled_indices = list(range(len(path)))
        else:
            # Sample more intelligently - include start, end, and distributed middle points
            sampled_indices = [0]  # Always include start
            
            # Add middle points with some randomness
            middle_points = list(range(1, len(path) - 1))
            if len(middle_points) >= num_states - 2:
                # Sample middle points with some clustering around key positions
                step = len(middle_points) // (num_states - 2)
                for i in range(1, num_states - 1):
                    base_idx = i * step
                    # Add some randomness around the base index
                    random_offset = random.randint(-step//2, step//2)
                    actual_idx = max(1, min(len(path) - 2, base_idx + random_offset))
                    sampled_indices.append(actual_idx)
            else:
                sampled_indices.extend(middle_points)
            
            sampled_indices.append(len(path) - 1)  # Always include end
            sampled_indices = list(set(sampled_indices))  # Remove duplicates
            sampled_indices.sort()
        
        # Generate semantic command variations for the primary goal type
        primary_commands = self._generate_base_commands(goal_type)
        
        # Also generate some commands from other goal types for diversity
        other_goal_types = [g for g in ["onion", "pot", "serving"] if g != goal_type]
        other_commands = []
        for other_goal in other_goal_types:
            other_commands.extend(self._generate_base_commands(other_goal))
        
        # Combine all commands for maximum diversity
        all_commands = primary_commands + other_commands
        
        for i, path_idx in enumerate(sampled_indices):
            if path_idx >= len(path):
                continue
                
            pos = path[path_idx]
            
            # Create state at this position
            state = self._create_state_at_position(start_state, pos, path_idx)
            
            # Determine action (direction to next position)
            if path_idx < len(path) - 1:
                next_pos = path[path_idx + 1]
                action = self._get_direction_action(pos, next_pos)
            else:
                # At goal, interact
                action = 5  # INTERACT
            
            # Extract state features
            state_features = self._extract_state_features(state)
            
            # Create trajectory point with semantic command variation
            # Use completely random command selection for maximum diversity
            # 70% chance of using primary goal commands, 30% chance of other commands
            if random.random() < 0.7:
                command = random.choice(primary_commands)
            else:
                command = random.choice(other_commands)
            
            point = TrajectoryPoint(
                state=state,
                action=action,
                position=pos,
                command=command,
                state_features=state_features
            )
            points.append(point)
        
        return points
    
    def _generate_base_commands(self, goal_type: str) -> List[str]:
        """Generate semantic variations of commands for the goal type."""
        if goal_type == "onion":
            return [
                # Direct commands
                "go to the onion dispenser",
                "head towards the onions",
                "navigate to get onions",
                "move to pick up onions",
                "walk to the onion station",
                "proceed to the onion area",
                "make your way to the onions",
                "travel to the onion dispenser",
                # More variations
                "get yourself to the onion dispenser",
                "steer towards the onions",
                "advance to the onion area",
                "direct yourself to the onions",
                "set course for the onion dispenser",
                "approach the onion station",
                "make a move toward the onions",
                "take a step toward the onion dispenser",
                "shift position to the onions",
                "relocate to the onion area",
                "position yourself at the onion dispenser",
                "find your way to the onions",
                "locate the onion dispenser",
                "seek out the onion station",
                "discover the onion area",
                "explore the path to onions",
                "venture toward the onion dispenser"
            ]
        elif goal_type == "pot":
            return [
                # Direct commands
                "go to the pot",
                "head towards the cooking pot",
                "navigate to the pot",
                "move to the cooking area",
                "walk to the pot",
                "proceed to the pot",
                "make your way to the pot",
                "travel to the cooking pot",
                # More variations
                "get yourself to the cooking pot",
                "steer towards the pot area",
                "advance to the cooking station",
                "direct yourself to the pot",
                "set course for the cooking area",
                "approach the pot location",
                "make a move toward the pot",
                "take a step toward the cooking area",
                "shift position to the pot",
                "relocate to the cooking station",
                "position yourself at the pot",
                "find your way to the cooking area",
                "locate the cooking pot",
                "seek out the pot station",
                "discover the cooking area",
                "explore the path to the pot",
                "venture toward the cooking station"
            ]
        elif goal_type == "serving":
            return [
                # Direct commands
                "go to the serving area",
                "head towards the serving station",
                "navigate to serve",
                "move to the serving area",
                "walk to serve",
                "proceed to serve",
                "make your way to serve",
                "travel to the serving station",
                # More variations
                "get yourself to the serving area",
                "steer towards the serving station",
                "advance to the serving location",
                "direct yourself to serve",
                "set course for the serving area",
                "approach the serving station",
                "make a move toward the serving area",
                "take a step toward the serving station",
                "shift position to the serving area",
                "relocate to the serving station",
                "position yourself at the serving area",
                "find your way to serve",
                "locate the serving station",
                "seek out the serving area",
                "discover the serving location",
                "explore the path to the serving area",
                "venture toward the serving station"
            ]
        else:
            return ["go to the target"]
    
    def _create_state_at_position(self, base_state: OvercookedState, pos: Tuple[int, int], 
                                 path_index: int) -> OvercookedState:
        """Create a state with player at the given position."""
        # Create players list first
        new_players = []
        
        # Update player 0 position
        for i, player in enumerate(base_state.players):
            if i == 0:  # Player 0 (ego agent)
                # Update held object position if it exists
                held_object = None
                if player.held_object:
                    held_object = ObjectState(
                        name=player.held_object.name,
                        position=pos,  # Update to new position
                        state=player.held_object.state if hasattr(player.held_object, 'state') else None
                    )
                
                new_player = PlayerState(
                    position=pos,
                    orientation=player.orientation,
                    held_object=held_object
                )
            else:  # Player 1 (confederate)
                new_player = PlayerState(
                    position=player.position,
                    orientation=player.orientation,
                    held_object=player.held_object
                )
            new_players.append(new_player)
        
        # Create the new state with the players list
        new_state = OvercookedState(
            players=new_players,
            objects=base_state.objects.copy(),
            order_list=base_state.order_list.copy(),
            timestep=base_state.timestep + path_index
        )
        
        return new_state
    
    def _get_direction_action(self, current: Tuple[int, int], next_pos: Tuple[int, int]) -> int:
        """Get action index for moving from current to next position."""
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        
        if dx > 0:
            return 4  # RIGHT
        elif dx < 0:
            return 3  # LEFT
        elif dy > 0:
            return 2  # DOWN
        elif dy < 0:
            return 1  # UP
        else:
            return 0  # STAY
    
    def _extract_state_features(self, state: OvercookedState) -> List[float]:
        """Extract state features similar to training data."""
        features = []
        
        # Player positions and held objects
        for player in state.players:
            features.extend([float(player.position[0]), float(player.position[1])])
            features.append(1.0 if player.held_object else 0.0)
        
        # Object counts
        object_counts = {'onion': 0, 'dish': 0, 'soup': 0}
        for obj in state.objects.values():
            if obj.name in object_counts:
                object_counts[obj.name] += 1
        
        for obj_type in ['onion', 'dish', 'soup']:
            features.append(float(object_counts[obj_type]))
        
        # Layout features
        features.append(float(len(self.pot_locations)))
        features.append(float(len(self.counter_locations)))
        features.append(float(len(self.onion_locations)))
        features.append(float(len(self.dish_locations)))
        features.append(float(len(self.serving_locations)))
        
        # Timestep (normalized)
        features.append(float(state.timestep) / 400.0)
        
        # Pad to 20 features
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def _create_training_example(self, point: TrajectoryPoint) -> TrainingExample:
        """Convert trajectory point to training example."""
        # Create standardized intervention
        standardized_intervention = {
            "trigger": "agent_performance_correction",
            "intervention_type": "direct_command",
            "command": point.command,
            "target_agent": 0
        }
        
        # Create macro actions
        macro_actions = [{
            "action_type": "movement",
            "target": point.position,
            "description": f"Move to {point.position}"
        }]
        
        return TrainingExample(
            state_features=point.state_features,
            command=point.command,
            action=point.action,
            standardized_intervention=standardized_intervention,
            macro_actions=macro_actions
        )
    
    def _create_fallback_examples(self, state: OvercookedState, num_states: int) -> List[TrainingExample]:
        """Create fallback examples when trajectory generation fails."""
        examples = []
        
        for i in range(num_states):
            # Random position
            pos = random.choice([(2, 2), (3, 2), (2, 3), (3, 3)])
            
            # Random action
            action = random.randint(0, 5)
            
            # Random command
            commands = ["go to the onion dispenser", "move to the pot", "serve the soup"]
            command = random.choice(commands)
            
            # Extract features
            state_features = self._extract_state_features(state)
            
            # Create example
            example = TrainingExample(
                state_features=state_features,
                command=command,
                action=action,
                standardized_intervention={
                    "trigger": "agent_performance_correction",
                    "intervention_type": "direct_command",
                    "command": command,
                    "target_agent": 0
                },
                macro_actions=[{
                    "action_type": "movement",
                    "target": pos,
                    "description": f"Move to {pos}"
                }]
            )
            examples.append(example)
        
        return examples

def main():
    """Main function to generate trajectory-based training data."""
    print("🚀 TRAJECTORY-BASED TRAINING DATA GENERATION")
    print("=" * 60)
    
    # Initialize generator
    generator = TrajectoryBasedGenerator(layout_name="random3")
    
    # Generate data
    examples = generator.generate_trajectory_data(
        num_trajectories=100,      # 100 different trajectories
        states_per_trajectory=8    # 8 states per trajectory
    )
    
    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("pretraining_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save examples
    training_file = output_dir / f"trajectory_direct_command_{timestamp}.json"
    with open(training_file, 'w') as f:
        json.dump([asdict(example) for example in examples], f, indent=2)
    
    # Save statistics
    stats = {
        "total_examples": len(examples),
        "trajectories": 100,
        "states_per_trajectory": 8,
        "intervention_types": ["agent_performance_correction - direct_command"],
        "action_distribution": {},
        "timestamp": timestamp
    }
    
    # Calculate action distribution
    for example in examples:
        action = example.action
        stats["action_distribution"][action] = stats["action_distribution"].get(action, 0) + 1
    
    stats_file = output_dir / f"trajectory_direct_command_stats_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Trajectory-based data generation completed!")
    print(f"   Examples: {training_file}")
    print(f"   Stats: {stats_file}")
    print(f"   Total examples: {len(examples)}")
    print(f"   Action distribution: {stats['action_distribution']}")

if __name__ == "__main__":
    main()
