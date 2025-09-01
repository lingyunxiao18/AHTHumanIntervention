#!/usr/bin/env python3
"""
Enhanced Trajectory-Based Training Data Generator

This script generates highly diverse training data for language-conditioned policies
with expanded command variations, more state diversity, and additional scenarios.
Improved version with better orientation handling, soup readiness checks, and diverse starting conditions.
"""

import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from datetime import datetime

# Import from existing trajectory generator
from trajectory_based_generator import TrajectoryBasedGenerator, TrajectoryPoint, TrainingExample

class EnhancedTrajectoryGenerator(TrajectoryBasedGenerator):
    """Enhanced generator for diverse trajectory-based training data."""
    
    def __init__(self, layout_name: str = "random3"):
        super().__init__(layout_name)
        
        # Enhanced command templates with much more variety
        self.enhanced_command_templates = {
            "onion": [
                # Direct commands (original)
                "go to the onion dispenser", "head towards the onions", "navigate to get onions",
                "move to pick up onions", "walk to the onion station", "proceed to the onion area",
                "make your way to the onions", "travel to the onion dispenser",
                "get yourself to the onion dispenser", "steer towards the onions", "advance to the onion area",
                "direct yourself to the onions", "set course for the onion dispenser", "approach the onion station",
                "make a move toward the onions", "take a step toward the onion dispenser",
                "shift position to the onions", "relocate to the onion area",
                "position yourself at the onion dispenser", "find your way to the onions",
                "locate the onion dispenser", "seek out the onion station",
                "discover the onion area", "explore the path to onions", "venture toward the onion dispenser",
                # More creative variations
                "head over to grab some onions", "make your way to the onion supply", "navigate toward the onion source",
                "move toward the onion pickup area", "walk over to the onion station", "proceed to collect onions",
                "travel to the onion collection point", "get to the onion dispenser", "steer toward the onion area",
                "advance toward the onion station", "direct yourself toward onions", "set course for onion pickup",
                "approach the onion collection area", "make a move toward onion pickup", "take a step toward onions",
                "shift toward the onion dispenser", "relocate to the onion pickup", "position at the onion station",
                "find the onion dispenser", "locate the onion pickup area", "seek the onion station",
                "discover the onion pickup point", "explore the onion area", "venture to the onion dispenser",
                # Action-oriented commands
                "collect onions from the dispenser", "pick up onions from the station", "gather onions from the area",
                "retrieve onions from the dispenser", "obtain onions from the station", "acquire onions from the area",
                "fetch onions from the dispenser", "secure onions from the station", "procure onions from the area",
                "harvest onions from the dispenser", "gather onions from the station", "collect onions from the area"
            ],
            "pot": [
                # Direct commands (original)
                "go to the cooking pot", "head towards the pot", "navigate to the cooking area",
                "move to the pot", "walk to the cooking station", "proceed to the pot area",
                "make your way to the pot", "travel to the cooking area", "get yourself to the pot",
                "steer towards the cooking area", "advance to the pot", "direct yourself to the cooking station",
                "set course for the pot", "approach the cooking area", "make a move toward the pot",
                "take a step toward the cooking station", "shift position to the pot", "relocate to the cooking area",
                "position yourself at the pot", "find your way to the cooking station",
                "locate the pot", "seek out the cooking area", "discover the pot area",
                "explore the path to the cooking station", "venture toward the pot",
                # More creative variations
                "head over to the cooking pot", "make your way to the cooking area", "navigate toward the pot",
                "move toward the cooking station", "walk over to the pot", "proceed to the cooking area",
                "travel to the cooking station", "get to the pot", "steer toward the cooking area",
                "advance toward the pot", "direct yourself toward the cooking station", "set course for the pot",
                "approach the cooking area", "make a move toward the cooking station", "take a step toward the pot",
                "shift toward the cooking area", "relocate to the pot", "position at the cooking station",
                "find the pot", "locate the cooking area", "seek the cooking station",
                "discover the pot area", "explore the cooking station", "venture to the cooking area",
                # Action-oriented commands
                "cook at the pot", "use the cooking station", "work at the pot",
                "operate the cooking area", "utilize the pot", "employ the cooking station",
                "engage with the pot", "interact with the cooking area", "work with the pot",
                "cook in the pot", "prepare food at the station", "cook at the area"
            ],
            "serving": [
                # Direct commands (original)
                "go to the serving area", "head towards the serving station", "navigate to the serving area",
                "move to the serving station", "walk to the serving area", "proceed to the serving station",
                "make your way to the serving area", "travel to the serving station", "get yourself to the serving area",
                "steer towards the serving station", "advance to the serving area", "direct yourself to the serving station",
                "set course for the serving area", "approach the serving station", "make a move toward the serving area",
                "take a step toward the serving station", "shift position to the serving area", "relocate to the serving station",
                "position yourself at the serving area", "find your way to the serving station",
                "locate the serving area", "seek out the serving station", "discover the serving area",
                "explore the path to the serving station", "venture toward the serving area",
                # More creative variations
                "head over to the serving area", "make your way to the serving station", "navigate toward the serving area",
                "move toward the serving station", "walk over to the serving area", "proceed to the serving station",
                "travel to the serving area", "get to the serving station", "steer toward the serving area",
                "advance toward the serving station", "direct yourself toward the serving area", "set course for the serving station",
                "approach the serving area", "make a move toward the serving station", "take a step toward the serving area",
                "shift toward the serving station", "relocate to the serving area", "position at the serving station",
                "find the serving area", "locate the serving station", "seek the serving area",
                "discover the serving station", "explore the serving area", "venture to the serving station",
                # Action-oriented commands
                "serve at the serving area", "deliver to the serving station", "serve at the station",
                "deliver food to the area", "serve meals at the station", "deliver to the serving area",
                "serve at the area", "deliver to the station", "serve food at the area",
                "deliver meals to the station", "serve at the serving station", "deliver to the area"
            ]
        }
        
        # Additional command variations for even more diversity
        self.additional_variations = {
            "urgency": ["quickly", "hurry", "fast", "immediately", "right now", "asap", "urgently"],
            "manner": ["carefully", "gently", "precisely", "exactly", "directly", "straight", "efficiently"],
            "emphasis": ["please", "kindly", "if you could", "would you mind", "can you", "try to"]
        }
        
        # Diverse starting scenarios for pretraining data
        self.starting_scenarios = [
            "empty_hands",           # Start with nothing
            "holding_onion",         # Start holding an onion
            "holding_dish",          # Start holding a dish
            "holding_soup",          # Start holding ready soup
            "near_onion_dispenser", # Start near onion dispenser
            "near_pot",             # Start near pot
            "near_serving",         # Start near serving area
            "random_position",       # Start at random position
            "with_partial_soup",    # Start with soup in pot (not ready)
            "with_ready_soup",      # Start with ready soup in pot
        ]
        
        print(f"🎯 Enhanced Trajectory Generator initialized for layout: {layout_name}")
        print(f"   Layout size: {self.width}x{self.height}")
        print(f"   Enhanced command templates: {sum(len(cmds) for cmds in self.enhanced_command_templates.values())} base commands")
        print(f"   Starting scenarios: {len(self.starting_scenarios)} different scenarios")
    
    def _generate_enhanced_commands(self, goal_type: str) -> List[str]:
        """Generate enhanced commands with more variations."""
        base_commands = self.enhanced_command_templates[goal_type]
        
        # Add urgency variations
        urgency_commands = []
        for cmd in base_commands[:20]:  # Use first 20 for variations
            for urgency in self.additional_variations["urgency"]:
                urgency_commands.append(f"{urgency} {cmd}")
        
        # Add manner variations
        manner_commands = []
        for cmd in base_commands[:15]:  # Use first 15 for variations
            for manner in self.additional_variations["manner"]:
                manner_commands.append(f"{manner} {cmd}")
        
        # Add emphasis variations
        emphasis_commands = []
        for cmd in base_commands[:10]:  # Use first 10 for variations
            for emphasis in self.additional_variations["emphasis"]:
                emphasis_commands.append(f"{emphasis} {cmd}")
        
        # Combine all variations
        all_commands = base_commands + urgency_commands + manner_commands + emphasis_commands
        
        # Shuffle for more randomness
        random.shuffle(all_commands)
        
        return all_commands
    
    def _create_diverse_start_state_enhanced(self):
        """Create a highly diverse starting state with more randomization."""
        # Choose a random starting scenario
        scenario = random.choice(self.starting_scenarios)
        
        # Use parent method as base
        state = super()._create_diverse_start_state()
        
        # Apply scenario-specific modifications
        if scenario == "holding_onion":
            state = self._modify_state_to_hold_object(state, "onion")
        elif scenario == "holding_dish":
            state = self._modify_state_to_hold_object(state, "dish")
        elif scenario == "holding_soup":
            state = self._modify_state_to_hold_object(state, "soup")
        elif scenario == "near_onion_dispenser":
            state = self._modify_state_position(state, self._find_position_near_dispenser("onion"))
        elif scenario == "near_pot":
            state = self._modify_state_position(state, self._find_position_near_dispenser("pot"))
        elif scenario == "near_serving":
            state = self._modify_state_position(state, self._find_position_near_dispenser("serving"))
        elif scenario == "random_position":
            state = self._modify_state_position(state, self._find_random_valid_position())
        elif scenario == "with_partial_soup":
            state = self._add_partial_soup_to_pot(state)
        elif scenario == "with_ready_soup":
            state = self._add_ready_soup_to_pot(state)
        
        # Enhanced object randomization
        new_objects_list = self._randomize_objects_enhanced(state.objects)
        
        # Convert list back to dictionary for OvercookedState
        new_objects_dict = {}
        for obj in new_objects_list:
            new_objects_dict[obj.position] = obj
        
        # Randomize timestep more broadly
        new_timestep = random.randint(0, 300)
        
        # Create new state
        new_state = type(state)(
            players=state.players,
            objects=new_objects_dict,
            order_list=state.order_list,
            timestep=new_timestep
        )
        
        return new_state
    
    def _modify_state_to_hold_object(self, state, object_type: str):
        """Modify state so that player 0 holds the specified object."""
        # Remove object from player's hands first
        player0 = state.players[0]
        if player0.has_object():
            # Remove the held object from the state
            new_objects = {pos: obj for pos, obj in state.objects.items() 
                         if not (obj.name == player0.get_object().name and obj.position == player0.position)}
        else:
            new_objects = state.objects.copy()
        
        # Create the new object for the player to hold
        if object_type == "onion":
            new_obj = type(state.objects[list(state.objects.keys())[0]])(
                name="onion",
                position=player0.position
            )
        elif object_type == "dish":
            new_obj = type(state.objects[list(state.objects.keys())[0]])(
                name="dish",
                position=player0.position
            )
        elif object_type == "soup":
            new_obj = type(state.objects[list(state.objects.keys())[0]])(
                name="soup",
                position=player0.position,
                state=("onion", 3, 25)  # Ready soup
            )
        
        # Update player's held object
        new_player0 = type(player0)(
            position=player0.position,
            orientation=player0.orientation,
            held_object=new_obj
        )
        
        # Create new state
        new_players = [new_player0]
        if len(state.players) > 1:
            new_players.extend(state.players[1:])
        
        return type(state)(
            players=new_players,
            objects=new_objects,
            order_list=state.order_list,
            timestep=state.timestep
        )
    
    def _modify_state_position(self, state, new_position: Tuple[int, int]):
        """Modify state so that player 0 is at the new position."""
        if not new_position:
            return state
        
        player0 = state.players[0]
        
        # Update held object position if player is holding something
        new_held_object = None
        if player0.has_object():
            held_obj = player0.get_object()
            new_held_object = type(held_obj)(
                name=held_obj.name,
                position=new_position,
                state=getattr(held_obj, 'state', None)
            )
        
        new_player0 = type(player0)(
            position=new_position,
            orientation=player0.orientation,
            held_object=new_held_object
        )
        
        # Create new state
        new_players = [new_player0]
        if len(state.players) > 1:
            new_players.extend(state.players[1:])
        
        return type(state)(
            players=new_players,
            objects=state.objects,
            order_list=state.order_list,
            timestep=state.timestep
        )
    
    def _find_position_near_dispenser(self, dispenser_type: str) -> Tuple[int, int]:
        """Find a position near the specified dispenser."""
        if dispenser_type == "onion":
            locations = self.onion_locations
        elif dispenser_type == "pot":
            locations = self.pot_locations
        elif dispenser_type == "serving":
            locations = self.serving_locations
        else:
            return self._find_random_valid_position()
        
        if not locations:
            return self._find_random_valid_position()
        
        # Choose a random dispenser
        dispenser_pos = random.choice(locations)
        
        # Find adjacent positions
        adjacent_positions = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (dispenser_pos[0] + dx, dispenser_pos[1] + dy)
            if self._is_valid_position(new_pos):
                adjacent_positions.append(new_pos)
        
        return random.choice(adjacent_positions) if adjacent_positions else self._find_random_valid_position()
    
    def _add_partial_soup_to_pot(self, state):
        """Add a partially cooked soup to a pot."""
        if not self.pot_locations:
            return state
        
        pot_pos = random.choice(self.pot_locations)
        soup_obj = type(state.objects[list(state.objects.keys())[0]])(
            name="soup",
            position=pot_pos,
            state=("onion", 2, 10)  # Partially cooked soup
        )
        
        new_objects = state.objects.copy()
        new_objects[pot_pos] = soup_obj
        
        return type(state)(
            players=state.players,
            objects=new_objects,
            order_list=state.order_list,
            timestep=state.timestep
        )
    
    def _add_ready_soup_to_pot(self, state):
        """Add a ready soup to a pot."""
        if not self.pot_locations:
            return state
        
        pot_pos = random.choice(self.pot_locations)
        soup_obj = type(state.objects[list(state.objects.keys())[0]])(
            name="soup",
            position=pot_pos,
            state=("onion", 3, 25)  # Ready soup
        )
        
        new_objects = state.objects.copy()
        new_objects[pot_pos] = soup_obj
        
        return type(state)(
            players=state.players,
            objects=new_objects,
            order_list=state.order_list,
            timestep=state.timestep
        )
    
    def _find_random_valid_position(self) -> Tuple[int, int]:
        """Find a random valid position."""
        valid_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.layout[y][x] == ' ':  # Empty space
                    valid_positions.append((x, y))
        return random.choice(valid_positions) if valid_positions else (2, 2)
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid."""
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.layout[y][x] == ' '  # Empty space
    
    def _randomize_objects_enhanced(self, objects) -> List:
        """Enhanced object randomization with more variety."""
        new_objects = []
        
        # Handle objects as dictionary (convert to list)
        if isinstance(objects, dict):
            objects_list = list(objects.values())
        else:
            objects_list = objects
        
        # Get object type from existing objects or use default
        if objects_list and len(objects_list) > 0:
            obj_type = type(objects_list[0])
        else:
            # Import ObjectState if needed
            import sys
            sys.path.append('../envs/overcooked_new/src')
            from overcooked_ai_py.mdp.overcooked_mdp import ObjectState
            obj_type = ObjectState
        
        # More soup variety with different cook states
        soup_count = random.randint(1, 4)  # 1-4 soups
        for _ in range(soup_count):
            soup_pos = self._find_random_position()
            if soup_pos:
                soup_type = "onion"  # For now, just use onion
                num_items = random.choice([1, 2, 3])
                # More diverse cook times: raw, partially cooked, ready
                cook_time = random.choice([0, 5, 10, 15, 20, 25])
                soup_obj = obj_type(
                    name="soup",
                    position=soup_pos,
                    state=(soup_type, num_items, cook_time)
                )
                new_objects.append(soup_obj)
        
        # More counter objects with variety
        counter_count = random.randint(3, 10)  # 3-10 counter objects
        for _ in range(counter_count):
            if random.random() < 0.6:  # 60% chance of object
                obj_pos = self._find_random_position()
                if obj_pos:
                    obj_types = ["onion", "dish", "soup"]
                    obj_type_name = random.choice(obj_types)
                    
                    if obj_type_name == "soup":
                        soup_type = "onion"
                        num_items = random.choice([1, 2])
                        cook_time = random.randint(5, 25)  # Various cook states
                        obj = obj_type(
                            name=obj_type_name,
                            position=obj_pos,
                            state=(soup_type, num_items, cook_time)
                        )
                    else:
                        obj = obj_type(
                            name=obj_type_name,
                            position=obj_pos
                        )
                    new_objects.append(obj)
        
        return new_objects
    
    def _create_fallback_trajectory_points(self, state, num_states: int) -> List[TrajectoryPoint]:
        """Create fallback trajectory points when pathfinding fails."""
        points = []
        
        for i in range(num_states):
            # Random position
            pos = self._find_random_valid_position()
            
            # Random action
            action = random.randint(0, 5)
            
            # Random command
            commands = ["go to the onion dispenser", "move to the pot", "serve the soup"]
            command = random.choice(commands)
            
            # Create state at position
            state_at_pos = self._create_state_at_position(state, pos, i)
            
            # Extract features
            state_features = self._extract_state_features(state_at_pos)
            
            # Create trajectory point
            point = TrajectoryPoint(
                state=state_at_pos,
                action=action,
                position=pos,
                command=command,
                state_features=state_features
            )
            points.append(point)
        
        return points
    
    def _choose_goal_simple(self, held_object) -> str:
        """Simple goal selection based on held object."""
        # 40% chance of random goal regardless of held object
        if random.random() < 0.4:
            return random.choice(["onion", "pot", "serving"])
        
        # 60% chance of logical goal based on held object
        if held_object is None:
            return random.choice(["onion", "pot"])
        elif held_object.name == "onion":
            return random.choice(["pot", "serving"])  # Can go to pot or serving
        elif held_object.name == "dish":
            return "pot"  # Need soup
        elif held_object.name == "soup":
            return "serving"  # Need to serve
        else:
            return random.choice(["onion", "pot", "serving"])
    
    def _find_random_position(self) -> Tuple[int, int]:
        """Find a random valid position for objects."""
        positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.layout[y][x] == ' ':  # Empty space
                    positions.append((x, y))
        return random.choice(positions) if positions else None
    
    def generate_trajectory_enhanced(self) -> List[TrajectoryPoint]:
        """Generate a single trajectory with enhanced diversity."""
        # Create diverse start state
        start_state = self._create_diverse_start_state_enhanced()
        
        # Choose goal
        player0 = start_state.players[0]
        goal_type = self._choose_goal_simple(player0.held_object)
        
        # Find goal position
        goal_pos = None
        for y in range(self.height):
            for x in range(self.width):
                if goal_type == "onion" and self.layout[y][x] == 'O':
                    goal_pos = (x, y)
                elif goal_type == "pot" and self.layout[y][x] == 'P':
                    goal_pos = (x, y)
                elif goal_type == "serving" and self.layout[y][x] == 'S':
                    goal_pos = (x, y)
        
        if not goal_pos:
            return self._create_fallback_trajectory_points(start_state, 3)
        
        # Generate A* path
        start_pos = player0.position
        path = self._astar_path(start_pos, goal_pos)
        
        if len(path) <= 1:
            return self._create_fallback_trajectory_points(start_state, 3)
        
        # Generate enhanced commands
        enhanced_commands = self._generate_enhanced_commands(goal_type)
        
        # Sample trajectory states with more diversity
        trajectory_points = []
        
        # Always include start and end
        sampled_indices = [0, len(path) - 1]
        
        # Add more middle points for diversity
        if len(path) > 2:
            middle_points = random.sample(range(1, len(path) - 1), min(3, len(path) - 2))
            sampled_indices.extend(middle_points)
        
        sampled_indices.sort()
        
        # Create trajectory points
        for i, path_idx in enumerate(sampled_indices):
            pos = path[path_idx]
            
            # Create state at this position
            state_at_pos = self._create_state_at_position(start_state, pos, path_idx)
            
            # Select command with more diversity
            if i == 0:  # Start
                command = random.choice(enhanced_commands[:len(enhanced_commands)//3])  # First third
            elif i == len(sampled_indices) - 1:  # End
                command = random.choice(enhanced_commands[2*len(enhanced_commands)//3:])  # Last third
            else:  # Middle
                command = random.choice(enhanced_commands)  # Any command
            
            # Determine action
            if path_idx < len(path) - 1:
                action = self._get_direction_action(pos, path[path_idx + 1])
            else:
                action = 5  # INTERACT when at goal
            
            # Extract state features
            state_features = self._extract_state_features(state_at_pos)
            
            trajectory_point = TrajectoryPoint(
                state=state_at_pos,
                action=action,
                position=pos,
                command=command,
                state_features=state_features
            )
            trajectory_points.append(trajectory_point)
        
        return trajectory_points
    
    def generate_dataset_enhanced(self, num_trajectories: int = 200) -> List[TrainingExample]:
        """Generate a complete dataset with enhanced diversity."""
        print(f"🎯 Generating enhanced dataset with {num_trajectories} trajectories...")
        
        all_examples = []
        
        for i in range(num_trajectories):
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{num_trajectories} trajectories")
            
            trajectory = self.generate_trajectory_enhanced()
            
            for point in trajectory:
                # Get position from the state
                player_pos = point.state.players[0].position
                example = TrainingExample(
                    command=point.command,
                    state_features=point.state_features,
                    action=point.action,
                    standardized_intervention={
                        "trigger": "agent_performance_correction",
                        "intervention_type": "direct_command",
                        "command": point.command,
                        "target_agent": 0
                    },
                    macro_actions=[{
                        "action_type": "movement",
                        "target": player_pos,
                        "description": f"Move to position {player_pos}"
                    }]
                )
                all_examples.append(example)
        
        print(f"✅ Generated {len(all_examples)} training examples")
        
        # Analyze diversity
        commands = [ex.command for ex in all_examples]
        unique_commands = len(set(commands))
        print(f"💬 Command diversity: {unique_commands} unique commands out of {len(all_examples)} examples")
        print(f"   Command diversity ratio: {unique_commands/len(all_examples)*100:.1f}%")
        
        # Action distribution
        actions = [ex.action for ex in all_examples]
        action_dist = {}
        for action in actions:
            action_dist[action] = action_dist.get(action, 0) + 1
        
        print(f"🎮 Action distribution:")
        action_names = {0: 'STAY', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'INTERACT'}
        for action, freq in sorted(action_dist.items()):
            action_name = action_names.get(action, f"Unknown_{action}")
            print(f"   {action_name}: {freq} times ({freq/len(all_examples)*100:.1f}%)")
        
        return all_examples

def main():
    """Main function to generate enhanced training data."""
    print("🚀 ENHANCED TRAJECTORY-BASED TRAINING DATA GENERATOR")
    print("=" * 70)
    
    # Initialize generator
    generator = EnhancedTrajectoryGenerator(layout_name="random3")
    
    # Generate dataset
    num_trajectories = 200  # More trajectories for more diversity
    examples = generator.generate_dataset_enhanced(num_trajectories)
    
    # Save dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pretraining_data/enhanced_trajectory_direct_command_{timestamp}.json"
    
    # Convert to JSON-serializable format
    json_examples = []
    for ex in examples:
        json_examples.append({
            'command': ex.command,
            'state_features': ex.state_features,
            'action': ex.action
        })
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(json_examples, f, indent=2)
    
    print(f"\n💾 Saved enhanced dataset to: {filename}")
    print(f"📊 Dataset statistics:")
    print(f"   Total examples: {len(examples)}")
    print(f"   Unique commands: {len(set(ex.command for ex in examples))}")
    print(f"   Command diversity: {len(set(ex.command for ex in examples))/len(examples)*100:.1f}%")
    
    # Save statistics
    stats_filename = f"pretraining_data/enhanced_trajectory_direct_command_{timestamp}_stats.json"
    stats = {
        'total_examples': len(examples),
        'unique_commands': len(set(ex.command for ex in examples)),
        'command_diversity_ratio': len(set(ex.command for ex in examples))/len(examples)*100,
        'action_distribution': {},
        'sample_commands': list(set(ex.command for ex in examples))[:20]
    }
    
    # Action distribution
    actions = [ex.action for ex in examples]
    action_dist = {}
    for action in actions:
        action_dist[action] = action_dist.get(action, 0) + 1
    
    action_names = {0: 'STAY', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'INTERACT'}
    for action, freq in sorted(action_dist.items()):
        action_name = action_names.get(action, f"Unknown_{action}")
        stats['action_distribution'][action_name] = {
            'count': freq,
            'percentage': freq/len(examples)*100
        }
    
    with open(stats_filename, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📈 Saved statistics to: {stats_filename}")

if __name__ == "__main__":
    main()
