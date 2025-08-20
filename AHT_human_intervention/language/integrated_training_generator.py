#!/usr/bin/env python
"""
Integrated Training Data Generator for Language-Conditioned Policy
Combines:
1. Generated human intervention commands (3x3 framework)
2. Object-centric state text descriptions
3. Computed reasonable actions (using A* or planning)
"""

import random
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

from llm_enhanced_command_generator import LLMEnhancedCommandGenerator
from simple_state_converter import SimpleStateConverter

@dataclass
class TrainingExample:
    """A complete training example for the language-conditioned policy."""
    state_text: str
    command: str
    action: int
    intervention_type: str
    intervention_category: str
    state_features: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class IntegratedTrainingGenerator:
    """Generates comprehensive training data for language-conditioned policy training."""
    
    def __init__(self, 
                 command_generator: LLMEnhancedCommandGenerator,
                 state_converter: SimpleStateConverter,
                 action_space_size: int = 6):
        self.command_generator = command_generator
        self.state_converter = state_converter
        self.action_space_size = action_space_size
        
        # Overcooked action mapping (adjust based on your environment)
        self.action_mapping = {
            'STAY': 0,
            'UP': 1, 
            'DOWN': 2,
            'LEFT': 3,
            'RIGHT': 4,
            'INTERACT': 5
        }
        
        # Reverse mapping for action names
        self.action_names = {v: k for k, v in self.action_mapping.items()}
        
        # Training data storage
        self.training_data: List[TrainingExample] = []
        
    def generate_state_command_pairs(self, 
                                   num_states: int = 100,
                                   commands_per_state: int = 3) -> List[TrainingExample]:
        """Generate state-command pairs with computed actions."""
        examples = []
        
        # Generate diverse states (you can replace this with actual Overcooked states)
        states = self._generate_diverse_states(num_states)
        
        for state in states:
            # Convert state to object-centric representation and text
            state_text = self.state_converter.state_to_text(state)
            
            # Generate commands for each intervention type
            for intervention_type in self.command_generator.get_intervention_types():
                for intervention_category in self.command_generator.get_intervention_categories():
                    # Generate commands for this intervention type
                    commands = self.command_generator.generate_intervention_command(
                        intervention_type, intervention_category, commands_per_state
                    )
                    
                    for command in commands:
                        # Compute reasonable action based on command and state
                        action = self._compute_reasonable_action(command, state)
                        
                        # Create training example
                        example = TrainingExample(
                            state_text=state_text,
                            command=command,
                            action=action,
                            intervention_type=intervention_type,
                            intervention_category=intervention_category,
                            state_features=self._extract_state_features(state),
                            metadata={
                                'state_id': id(state),
                                'command_length': len(command),
                                'action_name': self.action_names.get(action, 'UNKNOWN')
                            }
                        )
                        examples.append(example)
        
        self.training_data.extend(examples)
        return examples
    
    def _generate_diverse_states(self, num_states: int) -> List[Dict]:
        """Generate diverse Overcooked game states for training."""
        states = []
        
        # Define different game scenarios
        scenarios = [
            'agent_holding_ingredient',
            'agent_near_pot',
            'agent_near_counter',
            'agent_near_stove',
            'agent_empty_hands',
            'pot_ready_to_cook',
            'pot_cooking',
            'pot_ready_to_serve',
            'counter_with_ingredients',
            'counter_with_dishes',
            'teammate_coordination',
            'emergency_situation'
        ]
        
        for i in range(num_states):
            scenario = random.choice(scenarios)
            state = self._create_scenario_state(scenario)
            states.append(state)
        
        return states
    
    def _create_scenario_state(self, scenario: str) -> Dict:
        """Create a specific scenario state."""
        base_state = {
            'agents': [
                {'id': 1, 'pos': (1, 1), 'holding': None, 'facing': 'UP'},
                {'id': 2, 'pos': (2, 2), 'holding': None, 'facing': 'DOWN'}
            ],
            'objects': [],
            'layout': 'random3',
            'orders': [],
            'time_remaining': 300
        }
        
        if scenario == 'agent_holding_ingredient':
            base_state['agents'][0]['holding'] = 'onion'
            base_state['objects'].extend([
                {'type': 'pot', 'pos': (1, 2), 'status': 'empty'},
                {'type': 'onion', 'pos': (0, 0), 'status': 'available'}
            ])
            
        elif scenario == 'agent_near_pot':
            base_state['agents'][0]['pos'] = (1, 1)
            base_state['objects'].extend([
                {'type': 'pot', 'pos': (1, 2), 'status': 'empty'},
                {'type': 'onion', 'pos': (0, 0), 'status': 'available'}
            ])
            
        elif scenario == 'pot_ready_to_cook':
            base_state['agents'][0]['holding'] = 'onion'
            base_state['agents'][0]['pos'] = (1, 1)
            base_state['objects'].extend([
                {'type': 'pot', 'pos': (1, 2), 'status': 'empty'},
                {'type': 'stove', 'pos': (1, 3), 'status': 'available'}
            ])
            
        elif scenario == 'teammate_coordination':
            base_state['agents'][0]['pos'] = (0, 1)
            base_state['agents'][1]['pos'] = (2, 1)
            base_state['agents'][0]['holding'] = 'onion'
            base_state['agents'][1]['holding'] = 'dish'
            base_state['objects'].extend([
                {'type': 'pot', 'pos': (1, 1), 'status': 'cooking'},
                {'type': 'counter', 'pos': (1, 0), 'status': 'available'}
            ])
            
        elif scenario == 'emergency_situation':
            base_state['objects'].extend([
                {'type': 'pot', 'pos': (1, 2), 'status': 'burning'},
                {'type': 'fire', 'pos': (1, 2), 'status': 'active'}
            ])
        
        return base_state
    
    def _compute_reasonable_action(self, command: str, state: Dict) -> int:
        """Compute a reasonable action based on the command and state."""
        command_lower = command.lower()
        
        # Extract agent and object information
        agent = state['agents'][0]  # Focus on first agent
        agent_pos = agent['pos']
        agent_holding = agent.get('holding')
        
        # Find relevant objects
        pot_pos = None
        onion_pos = None
        counter_pos = None
        stove_pos = None
        
        for obj in state['objects']:
            if obj['type'] == 'pot':
                pot_pos = obj['pos']
            elif obj['type'] == 'onion':
                onion_pos = obj['pos']
            elif obj['type'] == 'counter':
                counter_pos = obj['pos']
            elif obj['type'] == 'stove':
                stove_pos = obj['pos']
        
        # Action logic based on command content
        if any(word in command_lower for word in ['go', 'move', 'head', 'navigate']):
            if 'onion' in command_lower and onion_pos:
                return self._get_movement_action(agent_pos, onion_pos)
            elif 'pot' in command_lower and pot_pos:
                return self._get_movement_action(agent_pos, pot_pos)
            elif 'counter' in command_lower and counter_pos:
                return self._get_movement_action(agent_pos, counter_pos)
            elif 'stove' in command_lower and stove_pos:
                return self._get_movement_action(agent_pos, stove_pos)
            else:
                # Random movement
                return random.choice([1, 2, 3, 4])  # UP, DOWN, LEFT, RIGHT
                
        elif any(word in command_lower for word in ['pick', 'grab', 'take', 'get']):
            if 'onion' in command_lower and onion_pos:
                # Move to onion first, then interact
                if self._is_adjacent(agent_pos, onion_pos):
                    return self.action_mapping['INTERACT']
                else:
                    return self._get_movement_action(agent_pos, onion_pos)
            else:
                return random.choice([1, 2, 3, 4])  # Random movement
                
        elif any(word in command_lower for word in ['put', 'drop', 'place', 'add']):
            if 'pot' in command_lower and pot_pos and agent_holding:
                if self._is_adjacent(agent_pos, pot_pos):
                    return self.action_mapping['INTERACT']
                else:
                    return self._get_movement_action(agent_pos, pot_pos)
            elif 'counter' in command_lower and counter_pos and agent_holding:
                if self._is_adjacent(agent_pos, counter_pos):
                    return self.action_mapping['INTERACT']
                else:
                    return self._get_movement_action(agent_pos, counter_pos)
            else:
                return random.choice([1, 2, 3, 4])  # Random movement
                
        elif any(word in command_lower for word in ['cook', 'prepare', 'process']):
            if pot_pos and agent_holding:
                if self._is_adjacent(agent_pos, pot_pos):
                    return self.action_mapping['INTERACT']
                else:
                    return self._get_movement_action(agent_pos, pot_pos)
            else:
                return random.choice([1, 2, 3, 4])  # Random movement
                
        elif any(word in command_lower for word in ['serve', 'deliver', 'bring']):
            if counter_pos and agent_holding:
                if self._is_adjacent(agent_pos, counter_pos):
                    return self.action_mapping['INTERACT']
                else:
                    return self._get_movement_action(agent_pos, counter_pos)
            else:
                return random.choice([1, 2, 3, 4])  # Random movement
                
        elif any(word in command_lower for word in ['stop', 'wait', 'stay']):
            return self.action_mapping['STAY']
            
        else:
            # Default: random action
            return random.randint(0, self.action_space_size - 1)
    
    def _get_movement_action(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> int:
        """Get the best movement action to reach target position."""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Prioritize horizontal movement
        if abs(dx) > abs(dy):
            if dx > 0:
                return self.action_mapping['RIGHT']
            else:
                return self.action_mapping['LEFT']
        else:
            if dy > 0:
                return self.action_mapping['DOWN']
            else:
                return self.action_mapping['UP']
    
    def _is_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
    
    def _extract_state_features(self, state: Dict) -> List[float]:
        """Extract numerical features from state for potential use."""
        features = []
        
        # Agent positions
        for agent in state['agents']:
            features.extend([float(agent['pos'][0]), float(agent['pos'][1])])
            features.append(1.0 if agent['holding'] else 0.0)
        
        # Object counts by type
        object_counts = {}
        for obj in state['objects']:
            obj_type = obj['type']
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Add object counts to features
        for obj_type in ['onion', 'pot', 'counter', 'stove', 'dish']:
            features.append(float(object_counts.get(obj_type, 0)))
        
        # Time remaining (normalized)
        features.append(float(state.get('time_remaining', 300)) / 300.0)
        
        return features
    
    def generate_balanced_dataset(self, 
                                 examples_per_category: int = 50,
                                 balance_interventions: bool = True) -> List[TrainingExample]:
        """Generate a balanced dataset across intervention types."""
        if balance_interventions:
            # Generate equal numbers for each intervention type
            total_examples = examples_per_category * 9  # 3x3 framework
            examples = []
            
            for intervention_type in self.command_generator.get_intervention_types():
                for intervention_category in self.command_generator.get_intervention_categories():
                    category_examples = self.generate_state_command_pairs(
                        num_states=examples_per_category,
                        commands_per_state=1
                    )
                    # Filter for this specific category
                    filtered_examples = [
                        ex for ex in category_examples 
                        if ex.intervention_type == intervention_type and 
                           ex.intervention_category == intervention_category
                    ]
                    examples.extend(filtered_examples[:examples_per_category])
            
            return examples
        else:
            # Generate without balancing
            return self.generate_state_command_pairs(
                num_states=examples_per_category * 9,
                commands_per_state=1
            )
    
    def save_training_data(self, filename: str, format: str = 'json'):
        """Save training data to file."""
        if format == 'json':
            data = [asdict(example) for example in self.training_data]
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'txt':
            with open(filename, 'w') as f:
                for i, example in enumerate(self.training_data):
                    f.write(f"Example {i+1}:\n")
                    f.write(f"State: {example.state_text}\n")
                    f.write(f"Command: {example.command}\n")
                    f.write(f"Action: {example.action} ({self.action_names.get(example.action, 'UNKNOWN')})\n")
                    f.write(f"Intervention: {example.intervention_type} - {example.intervention_category}\n")
                    f.write("-" * 50 + "\n")
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generated dataset."""
        if not self.training_data:
            return {}
        
        stats = {
            'total_examples': len(self.training_data),
            'intervention_type_distribution': {},
            'intervention_category_distribution': {},
            'action_distribution': {},
            'command_length_stats': {},
            'avg_command_length': 0.0
        }
        
        # Count distributions
        for example in self.training_data:
            # Intervention type
            stats['intervention_type_distribution'][example.intervention_type] = \
                stats['intervention_type_distribution'].get(example.intervention_type, 0) + 1
            
            # Intervention category
            stats['intervention_category_distribution'][example.intervention_category] = \
                stats['intervention_category_distribution'].get(example.intervention_category, 0) + 1
            
            # Action
            stats['action_distribution'][example.action] = \
                stats['action_distribution'].get(example.action, 0) + 1
        
        # Command length statistics
        command_lengths = [len(ex.command) for ex in self.training_data]
        stats['command_length_stats'] = {
            'min': min(command_lengths),
            'max': max(command_lengths),
            'mean': np.mean(command_lengths),
            'std': np.std(command_lengths)
        }
        stats['avg_command_length'] = np.mean(command_lengths)
        
        return stats

def main():
    """Demo the integrated training data generator."""
    print("=== Integrated Training Data Generator Demo ===\n")
    
    # Initialize components
    command_generator = LLMEnhancedCommandGenerator()
    state_converter = SimpleStateConverter()
    training_generator = IntegratedTrainingGenerator(command_generator, state_converter)
    
    # Generate balanced dataset
    print("Generating balanced training dataset...")
    examples = training_generator.generate_balanced_dataset(examples_per_category=10)
    
    print(f"Generated {len(examples)} training examples\n")
    
    # Show some examples
    print("Sample Training Examples:")
    for i, example in enumerate(examples[:5]):
        print(f"\nExample {i+1}:")
        print(f"State: {example.state_text[:100]}...")
        print(f"Command: {example.command}")
        print(f"Action: {example.action} ({training_generator.action_names.get(example.action, 'UNKNOWN')})")
        print(f"Intervention: {example.intervention_type} - {example.intervention_category}")
    
    # Show dataset statistics
    print("\n" + "="*50)
    print("Dataset Statistics:")
    stats = training_generator.get_dataset_statistics()
    for key, value in stats.items():
        if key != 'command_length_stats':
            print(f"{key}: {value}")
    
    print("\nCommand Length Statistics:")
    for key, value in stats['command_length_stats'].items():
        print(f"  {key}: {value:.2f}")
    
    # Save training data
    print("\nSaving training data...")
    training_generator.save_training_data('training_data.json', 'json')
    training_generator.save_training_data('training_data.txt', 'txt')
    print("Training data saved to 'training_data.json' and 'training_data.txt'")

if __name__ == "__main__":
    main() 