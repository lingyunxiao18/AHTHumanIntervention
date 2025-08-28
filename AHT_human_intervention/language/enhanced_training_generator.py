#!/usr/bin/env python
"""
Enhanced Training Data Generator for Language-Conditioned Policy
Addresses all identified issues:
1. Fixed starting positions from layout
2. Enhanced scenario states with proper player configurations
3. A* pathfinding for movement actions
4. LLM-based command parsing and categorization
5. Refined prompts targeting specific Overcooked layouts
6. Improved state text with coordinates
7. Standardized intervention format with macro action decomposition
"""

import random
import json
import os
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import openai
import time

from enhanced_llm_command_generator import EnhancedLLMCommandGenerator
# Import Overcooked components for proper state generation
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'envs', 'overcooked', 'overcooked_ai_py', 'mdp'))
from overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from overcooked_state_converter import create_state_converter

# Import A* pathfinding
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'envs', 'overcooked', 'overcooked_ai_py', 'planning'))
from search import SearchTree, SearchNode
from planners import MediumLevelPlanner

@dataclass
class MacroAction:
    """Represents a macro action decomposed from a human command."""
    action_type: str  # 'movement', 'interaction', 'wait', 'coordination'
    target: Optional[str] = None  # target object or location
    description: str = ""
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class StandardizedIntervention:
    """Standardized format for human interventions."""
    intervention_type: str  # 'direct_command', 'factual_information', 'general_instruction'
    trigger: str  # 'agent_performance', 'environmental_update', 'teammate_coordination'
    macro_actions: List[MacroAction]
    urgency: str = "normal"  # 'urgent', 'normal', 'low'
    context: str = ""

@dataclass
class TrainingExample:
    """A complete training example for the language-conditioned policy."""
    state_text: str
    command: str
    action: int
    intervention_type: str
    intervention_category: str
    standardized_intervention: Optional[StandardizedIntervention] = None
    macro_actions: Optional[List[MacroAction]] = None
    state_features: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class EnhancedTrainingGenerator:
    """Enhanced training data generator with all improvements."""
    
    def __init__(self, 
                 command_generator: EnhancedLLMCommandGenerator,
                 overcooked_mdp: OvercookedGridworld,
                 action_space_size: int = 6,
                 use_llm_parsing: bool = True):
        self.command_generator = command_generator
        self.overcooked_mdp = overcooked_mdp
        self.state_converter = create_state_converter(overcooked_mdp)
        self.action_space_size = action_space_size
        self.use_llm_parsing = use_llm_parsing
        
        # Get actual start positions from layout
        self.start_positions = self._get_start_positions_from_layout()
        
        # Overcooked action mapping
        self.action_mapping = {
            'STAY': 0,
            'UP': 1, 
            'DOWN': 2,
            'LEFT': 3,
            'RIGHT': 4,
            'INTERACT': 5
        }
        self.action_names = {v: k for k, v in self.action_mapping.items()}
        
        # Initialize A* pathfinding components
        self.mlp = None
        self._initialize_pathfinding()
        
        # Training data storage
        self.training_data: List[TrainingExample] = []
        
        # Layout-specific information for better prompts
        self.layout_info = self._analyze_layout()
        
    def _get_start_positions_from_layout(self) -> List[Tuple[int, int]]:
        """Extract actual start positions from the layout file."""
        # The MDP already processes the layout and provides correct start positions
        # The '1' and '2' in the layout file are converted to actual coordinates
        return list(self.overcooked_mdp.start_player_positions)
    
    def _analyze_layout(self) -> Dict[str, Any]:
        """Analyze the layout to provide context for LLM prompts."""
        layout_info = {
            'name': getattr(self.overcooked_mdp, 'layout_name', 'unknown'),
            'shape': self.overcooked_mdp.shape,
            'pot_locations': self.overcooked_mdp.get_pot_locations(),
            'counter_locations': self.overcooked_mdp.get_counter_locations(),
            'onion_dispenser_locations': self.overcooked_mdp.get_onion_dispenser_locations(),
            'dish_dispenser_locations': self.overcooked_mdp.get_dish_dispenser_locations(),
            'serving_locations': self.overcooked_mdp.get_serving_locations(),
            'start_positions': self.start_positions
        }
        return layout_info
    
    def _create_mlp_params(self):
        """Create MLP parameters for the current layout."""
        # Get counter positions (X terrain)
        counter_positions = self.overcooked_mdp.terrain_pos_dict.get("X", [])
        
        # Create MLP parameters based on the layout
        mlp_params = {
            "start_orientations": False,  # Don't store all orientations for efficiency
            "wait_allowed": False,        # Don't allow waiting actions
            "counter_goals": counter_positions,  # Use all counter positions as goals
            "counter_drop": counter_positions[:2] if len(counter_positions) >= 2 else counter_positions,  # Use first 2 counters for dropping
            "counter_pickup": [],         # No specific pickup counters
            "same_motion_goals": True,    # Allow same motion goals for efficiency
        }
        
        return mlp_params
    
    def _initialize_pathfinding(self):
        """Initialize A* pathfinding components."""
        try:
            # For now, we'll use a simplified A* implementation
            # that doesn't require the complex MLP initialization
            self.mlp = "astar_available"  # Flag that A* is available
            print(f"✅ A* pathfinding initialized for layout: {self.overcooked_mdp.layout_name}")
            
        except Exception as e:
            print(f"Warning: Could not initialize A* pathfinding: {e}")
            print("Falling back to simple pathfinding...")
            self.mlp = None
    
    def _get_astar_path(self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> List[int]:
        """Get optimal path using A* search."""
        if not self.mlp:
            return self._get_simple_path(start_pos, goal_pos)
        
        try:
            # Use a simplified A* implementation that works with the Overcooked environment
            # Since the complex state transitions are failing, we'll use a grid-based approach
            
            # Get the grid dimensions
            height, width = self.overcooked_mdp.shape
            
            # Define valid moves (up, down, left, right)
            moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
            move_actions = [1, 2, 3, 4]  # corresponding actions
            
            # A* search on the grid
            from collections import deque
            import heapq
            
            # Priority queue: (f_cost, g_cost, position, path)
            open_set = [(0, 0, start_pos, [])]
            closed_set = set()
            
            while open_set:
                f_cost, g_cost, current_pos, path = heapq.heappop(open_set)
                
                if current_pos in closed_set:
                    continue
                    
                closed_set.add(current_pos)
                
                # Check if we reached the goal
                if current_pos == goal_pos:
                    return path if path else [0]  # Return path or stay action
                
                # Explore neighbors
                for i, (dx, dy) in enumerate(moves):
                    new_x = current_pos[0] + dx
                    new_y = current_pos[1] + dy
                    new_pos = (new_x, new_y)
                    
                    # Check bounds
                    if (0 <= new_x < width and 0 <= new_y < height and 
                        new_pos not in closed_set):
                        
                        # Check if the position is walkable (not a wall)
                        terrain = self.overcooked_mdp.terrain_mtx[new_y][new_x]
                        if terrain not in ['X', 'P']:  # Not a wall or pot
                            
                            new_g_cost = g_cost + 1
                            h_cost = abs(new_x - goal_pos[0]) + abs(new_y - goal_pos[1])
                            f_cost = new_g_cost + h_cost
                            
                            new_path = path + [move_actions[i]]
                            heapq.heappush(open_set, (f_cost, new_g_cost, new_pos, new_path))
            
            # If no path found, return simple path
            return self._get_simple_path(start_pos, goal_pos)
            
        except Exception as e:
            print(f"A* pathfinding failed: {e}, using simple pathfinding")
            return self._get_simple_path(start_pos, goal_pos)
    
    def _get_simple_path(self, start_pos: Tuple[int, int], goal_pos: Tuple[int, int]) -> List[int]:
        """Fallback simple pathfinding."""
        dx = goal_pos[0] - start_pos[0]
        dy = goal_pos[1] - start_pos[1]
        
        actions = []
        # Prioritize horizontal movement
        if abs(dx) > abs(dy):
            if dx > 0:
                actions.append(self.action_mapping['RIGHT'])
            elif dx < 0:
                actions.append(self.action_mapping['LEFT'])
        else:
            if dy > 0:
                actions.append(self.action_mapping['DOWN'])
            elif dy < 0:
                actions.append(self.action_mapping['UP'])
        
        return actions if actions else [self.action_mapping['STAY']]
    
    def _parse_command_with_llm(self, command: str, state: OvercookedState) -> StandardizedIntervention:
        """Use LLM to parse and standardize human commands."""
        if not self.use_llm_parsing:
            return self._parse_command_simple(command, state)
        
        try:
            # Create context about the current state
            state_context = self._create_state_context_for_llm(state)
            
            prompt = f"""
You are an expert at parsing human intervention commands for AI agents in the Overcooked cooking game.

CURRENT GAME STATE:
{state_context}

LAYOUT INFORMATION:
- Layout: {self.layout_info['name']}
- Grid size: {self.layout_info['shape']}
- Available objects: pots, counters, onion dispensers, dish dispensers, serving areas

HUMAN COMMAND: "{command}"

TASK: Parse this command and return a standardized intervention format.

INTERVENTION TYPES:
1. direct_command: Specific action instructions (e.g., "go to onion", "pick up dish")
2. factual_information: State information sharing (e.g., "the pot is ready", "you're holding an onion")
3. general_instruction: High-level guidance (e.g., "prioritize cooking", "work faster")

TRIGGERS:
1. agent_performance: Correcting agent mistakes or inefficiencies
2. environmental_update: Sharing information about the environment
3. teammate_coordination: Coordinating with or about teammates

MACRO ACTIONS (if command requires multiple steps):
- movement: Moving to a location
- interaction: Picking up, putting down, or using objects
- wait: Waiting or staying in place
- coordination: Working with teammates

URGENCY LEVELS:
- urgent: Immediate action required
- normal: Standard priority
- low: Can be done when convenient

Return ONLY a JSON object with this exact structure:
{{
    "intervention_type": "direct_command|factual_information|general_instruction",
    "trigger": "agent_performance|environmental_update|teammate_coordination",
    "macro_actions": [
        {{
            "action_type": "movement|interaction|wait|coordination",
            "target": "specific object or location",
            "description": "what this action does",
            "priority": 1
        }}
    ],
    "urgency": "urgent|normal|low",
    "context": "brief context about why this intervention is needed"
}}

Focus on Overcooked-specific actions and objects. Avoid nonsensical terms like "heavy lifting", "shelves", etc.
"""

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an expert at parsing human intervention commands for AI agents in cooking games. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content.strip())
            
            # Convert to StandardizedIntervention
            macro_actions = [MacroAction(**ma) for ma in result.get('macro_actions', [])]
            
            return StandardizedIntervention(
                intervention_type=result.get('intervention_type', 'direct_command'),
                trigger=result.get('trigger', 'agent_performance'),
                macro_actions=macro_actions,
                urgency=result.get('urgency', 'normal'),
                context=result.get('context', '')
            )
            
        except Exception as e:
            print(f"LLM parsing failed: {e}, using simple parsing")
            return self._parse_command_simple(command, state)
    
    def _create_state_context_for_llm(self, state: OvercookedState) -> str:
        """Create a concise context description for LLM parsing."""
        context_parts = []
        
        # Player information
        for i, player in enumerate(state.players):
            pos = player.position
            holding = player.get_object().name if player.has_object() else "nothing"
            context_parts.append(f"Player {i}: at ({pos[0]}, {pos[1]}), holding {holding}")
        
        # Object information
        if state.objects:
            obj_info = []
            for pos, obj in state.objects.items():
                if obj.name == 'soup':
                    soup_type, num_items, cook_time = obj.state
                    obj_info.append(f"{num_items} {soup_type} soup at ({pos[0]}, {pos[1]}) - cooking {cook_time} steps")
                else:
                    obj_info.append(f"{obj.name} at ({pos[0]}, {pos[1]})")
            context_parts.append(f"Objects: {', '.join(obj_info)}")
        
        # Orders
        if state.order_list:
            context_parts.append(f"Orders: {', '.join(state.order_list)}")
        
        return ". ".join(context_parts)
    
    def _parse_command_simple(self, command: str, state: OvercookedState) -> StandardizedIntervention:
        """Fallback simple command parsing."""
        command_lower = command.lower()
        
        # Determine intervention type
        if any(word in command_lower for word in ['go', 'move', 'pick', 'put', 'drop']):
            intervention_type = 'direct_command'
        elif any(word in command_lower for word in ['is', 'are', 'has', 'have', 'there']):
            intervention_type = 'factual_information'
        else:
            intervention_type = 'general_instruction'
        
        # Determine trigger
        if any(word in command_lower for word in ['wrong', 'mistake', 'should', 'need to']):
            trigger = 'agent_performance'
        elif any(word in command_lower for word in ['teammate', 'partner', 'help']):
            trigger = 'teammate_coordination'
        else:
            trigger = 'environmental_update'
        
        # Create simple macro action
        macro_action = MacroAction(
            action_type='movement' if 'go' in command_lower or 'move' in command_lower else 'interaction',
            target='onion' if 'onion' in command_lower else 'pot' if 'pot' in command_lower else None,
            description=command,
            priority=1
        )
        
        return StandardizedIntervention(
            intervention_type=intervention_type,
            trigger=trigger,
            macro_actions=[macro_action],
            urgency='normal',
            context=''
        )
    
    def _compute_reasonable_action(self, command: str, state: OvercookedState) -> int:
        """Compute reasonable action using enhanced parsing and A* pathfinding."""
        # Parse command with LLM
        standardized_intervention = self._parse_command_with_llm(command, state)
        
        # Get the highest priority macro action
        if not standardized_intervention.macro_actions:
            return self.action_mapping['STAY']
        
        primary_action = min(standardized_intervention.macro_actions, key=lambda x: x.priority)
        
        # Execute the macro action
        return self._execute_macro_action(primary_action, state)
    
    def _execute_macro_action(self, macro_action: MacroAction, state: OvercookedState) -> int:
        """Execute a macro action and return the appropriate low-level action."""
        agent = state.players[0]
        agent_pos = agent.position
        
        if macro_action.action_type == 'movement':
            target_pos = self._resolve_target_position(macro_action.target, state)
            if target_pos:
                # Use A* pathfinding
                path = self._get_astar_path(agent_pos, target_pos)
                return path[0] if path else self.action_mapping['STAY']
            else:
                return self.action_mapping['STAY']
        
        elif macro_action.action_type == 'interaction':
            target_pos = self._resolve_target_position(macro_action.target, state)
            if target_pos and self._is_adjacent(agent_pos, target_pos):
                return self.action_mapping['INTERACT']
            elif target_pos:
                # Move towards target first
                path = self._get_astar_path(agent_pos, target_pos)
                return path[0] if path else self.action_mapping['STAY']
            else:
                return self.action_mapping['STAY']
        
        elif macro_action.action_type == 'wait':
            return self.action_mapping['STAY']
        
        else:  # coordination
            return self.action_mapping['STAY']
    
    def _resolve_target_position(self, target: Optional[str], state: OvercookedState) -> Optional[Tuple[int, int]]:
        """Resolve a target string to a specific position."""
        if not target:
            return None
        
        target_lower = target.lower()
        
        # Check for objects in state
        for pos, obj in state.objects.items():
            if target_lower in obj.name.lower():
                return pos
        
        # Check layout features
        if 'onion' in target_lower:
            onion_dispensers = self.overcooked_mdp.get_onion_dispenser_locations()
            return onion_dispensers[0] if onion_dispensers else None
        elif 'dish' in target_lower:
            dish_dispensers = self.overcooked_mdp.get_dish_dispenser_locations()
            return dish_dispensers[0] if dish_dispensers else None
        elif 'pot' in target_lower:
            pot_locations = self.overcooked_mdp.get_pot_locations()
            return pot_locations[0] if pot_locations else None
        elif 'serve' in target_lower or 'serving' in target_lower:
            serving_locations = self.overcooked_mdp.get_serving_locations()
            return serving_locations[0] if serving_locations else None
        
        return None
    
    def _is_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
    
    def _create_enhanced_scenario_state(self, scenario: str, timestep: int = 0) -> OvercookedState:
        """Create enhanced scenario states with proper player configurations."""
        # Use actual start positions from layout
        player1_pos = self.start_positions[0]
        player2_pos = self.start_positions[1]
        
        # Create base players with proper orientations
        orientations = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        player1_orientation = random.choice(orientations)
        player2_orientation = random.choice(orientations)
        
        # Base player configurations
        player1 = PlayerState(position=player1_pos, orientation=player1_orientation, held_object=None)
        player2 = PlayerState(position=player2_pos, orientation=player2_orientation, held_object=None)
        
        # Base objects and orders
        objects = {}
        order_list = ["onion"]  # Default order
        
        # Enhanced scenario configurations
        if scenario == 'agent_holding_ingredient':
            # Player 0 holding onion, Player 1 in various states
            player1 = PlayerState(
                position=player1_pos, 
                orientation=player1_orientation,
                held_object=ObjectState("onion", player1_pos)
            )
            # Vary Player 1's state
            player1_variations = [
                PlayerState(position=player2_pos, orientation=player2_orientation, held_object=None),
                PlayerState(position=player2_pos, orientation=player2_orientation, held_object=ObjectState("dish", player2_pos)),
                PlayerState(position=player2_pos, orientation=player2_orientation, held_object=ObjectState("onion", player2_pos))
            ]
            player2 = random.choice(player1_variations)
            
        elif scenario == 'agent_near_pot':
            # Position player 0 near a pot
            pot_locs = self.overcooked_mdp.get_pot_locations()
            if pot_locs:
                pot_pos = random.choice(pot_locs)
                # Find adjacent position
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_pos = (pot_pos[0] + dx, pot_pos[1] + dy)
                    if (0 <= new_pos[0] < self.overcooked_mdp.shape[0] and 
                        0 <= new_pos[1] < self.overcooked_mdp.shape[1] and
                        self.overcooked_mdp.terrain_mtx[new_pos[1]][new_pos[0]] not in ['X', 'P']):
                        player1 = PlayerState(position=new_pos, orientation=player1_orientation, held_object=None)
                        break
            
            # Vary Player 1's state
            player1_variations = [
                PlayerState(position=player2_pos, orientation=player2_orientation, held_object=None),
                PlayerState(position=player2_pos, orientation=player2_orientation, held_object=ObjectState("onion", player2_pos)),
                PlayerState(position=player2_pos, orientation=player2_orientation, held_object=ObjectState("dish", player2_pos))
            ]
            player2 = random.choice(player1_variations)
            
        elif scenario == 'pot_ready_to_cook':
            # Player 0 holding onion, positioned near pot, pot has some ingredients
            player1 = PlayerState(
                position=player1_pos, 
                orientation=player1_orientation,
                held_object=ObjectState("onion", player1_pos)
            )
            
            # Add soup in pot (partially filled)
            pot_locs = self.overcooked_mdp.get_pot_locations()
            if pot_locs:
                selected_pot = random.choice(pot_locs)
                objects[selected_pot] = ObjectState("soup", selected_pot, ("onion", 2, 10))
            
            # Vary Player 1's state
            player1_variations = [
                PlayerState(position=player2_pos, orientation=player2_orientation, held_object=None),
                PlayerState(position=player2_pos, orientation=player2_orientation, held_object=ObjectState("onion", player2_pos)),
                PlayerState(position=player2_pos, orientation=player2_orientation, held_object=ObjectState("dish", player2_pos))
            ]
            player2 = random.choice(player1_variations)
            
        elif scenario == 'pot_cooking':
            # Soup cooking in pot
            pot_locs = self.overcooked_mdp.get_pot_locations()
            if pot_locs:
                selected_pot = random.choice(pot_locs)
                objects[selected_pot] = ObjectState("soup", selected_pot, ("onion", 3, 15))
            
            # Vary both players' states
            player_variations = [
                (None, None),
                (ObjectState("onion", player1_pos), None),
                (None, ObjectState("dish", player2_pos)),
                (ObjectState("onion", player1_pos), ObjectState("dish", player2_pos))
            ]
            p1_obj, p2_obj = random.choice(player_variations)
            player1 = PlayerState(position=player1_pos, orientation=player1_orientation, held_object=p1_obj)
            player2 = PlayerState(position=player2_pos, orientation=player2_orientation, held_object=p2_obj)
            
        elif scenario == 'pot_ready_to_serve':
            # Soup ready to serve
            pot_locs = self.overcooked_mdp.get_pot_locations()
            if pot_locs:
                selected_pot = random.choice(pot_locs)
                objects[selected_pot] = ObjectState("soup", selected_pot, ("onion", 3, 20))
            
            # Vary both players' states
            player_variations = [
                (None, None),
                (ObjectState("dish", player1_pos), None),
                (None, ObjectState("onion", player2_pos)),
                (ObjectState("dish", player1_pos), ObjectState("onion", player2_pos))
            ]
            p1_obj, p2_obj = random.choice(player_variations)
            player1 = PlayerState(position=player1_pos, orientation=player1_orientation, held_object=p1_obj)
            player2 = PlayerState(position=player2_pos, orientation=player2_orientation, held_object=p2_obj)
            
        elif scenario == 'teammate_coordination':
            # Complex coordination scenario
            player1 = PlayerState(
                position=player1_pos, 
                orientation=player1_orientation,
                held_object=ObjectState("onion", player1_pos)
            )
            player2 = PlayerState(
                position=player2_pos, 
                orientation=player2_orientation,
                held_object=ObjectState("dish", player2_pos)
            )
            
            # Add cooking soup in pot
            pot_locs = self.overcooked_mdp.get_pot_locations()
            if pot_locs:
                selected_pot = random.choice(pot_locs)
                objects[selected_pot] = ObjectState("soup", selected_pot, ("onion", 3, 18))
        
        # Create and return the OvercookedState
        return OvercookedState(
            players=[player1, player2],
            objects=objects,
            order_list=order_list,
            timestep=timestep
        )
    
    def _generate_diverse_states(self, num_states: int) -> List[OvercookedState]:
        """Generate diverse Overcooked game states for training."""
        states = []
        
        # Enhanced scenarios with more variety
        scenarios = [
            'agent_holding_ingredient',
            'agent_near_pot', 
            'pot_ready_to_cook',
            'pot_cooking',
            'pot_ready_to_serve',
            'teammate_coordination'
        ]
        
        for i in range(num_states):
            scenario = random.choice(scenarios)
            state = self._create_enhanced_scenario_state(scenario, timestep=i)
            states.append(state)
        
        return states
    
    def generate_enhanced_training_data(self, 
                                      num_states: int = 100,
                                      commands_per_state: int = 3) -> List[TrainingExample]:
        """Generate enhanced training data with all improvements."""
        examples = []
        
        # Generate diverse states
        states = self._generate_diverse_states(num_states)
        
        for state in states:
            # Convert state to enhanced text description
            state_text = self._create_enhanced_state_text(state)
            
            # Generate commands for each intervention type
            for intervention_type in self.command_generator.get_intervention_types():
                for intervention_category in self.command_generator.get_intervention_categories():
                    # Get commands from the framework
                    if (intervention_type in self.command_generator.intervention_framework and 
                        intervention_category in self.command_generator.intervention_framework[intervention_type]):
                        intervention_obj = self.command_generator.intervention_framework[intervention_type][intervention_category]
                        commands = intervention_obj.examples[:commands_per_state]
                    else:
                        commands = ["Complete the cooking task efficiently"]
                    
                    for command in commands:
                        # Parse command with LLM
                        standardized_intervention = self._parse_command_with_llm(command, state)
                        
                        # Compute reasonable action using enhanced methods
                        action = self._compute_reasonable_action(command, state)
                        
                        # Create training example
                        example = TrainingExample(
                            state_text=state_text,
                            command=command,
                            action=action,
                            intervention_type=intervention_type,
                            intervention_category=intervention_category,
                            standardized_intervention=standardized_intervention,
                            macro_actions=standardized_intervention.macro_actions,
                            state_features=self._extract_state_features(state),
                            metadata={
                                'state_id': id(state),
                                'command_length': len(command),
                                'action_name': self.action_names.get(action, 'UNKNOWN'),
                                'layout': self.layout_info['name'],
                                'start_positions': self.start_positions
                            }
                        )
                        examples.append(example)
        
        self.training_data.extend(examples)
        return examples
    
    def generate_single_example(self, trigger: str, intervention_type: str) -> TrainingExample:
        """Generate a single training example for a specific trigger and intervention type."""
        # Generate a random state
        states = self._generate_diverse_states(1)
        state = states[0]
        
        # Generate a command for this specific trigger and intervention type
        command = self.command_generator.generate_layout_specific_commands(
            trigger, intervention_type, num_variations=1
        )[0]
        
        # Parse command with LLM to get standardized intervention and macro actions
        standardized_intervention = self._parse_command_with_llm(command, state)
        
        # Compute reasonable action
        action = self._compute_reasonable_action(command, state)
        
        # Create state text
        state_text = self._create_enhanced_state_text(state)
        
        # Extract real state features
        features = self._extract_state_features(state)
        
        return TrainingExample(
            state_text=state_text,
            command=command,
            action=action,
            intervention_type=trigger,
            intervention_category=intervention_type,
            standardized_intervention=standardized_intervention,
            macro_actions=standardized_intervention.macro_actions if standardized_intervention else None,
            state_features=features
        )
    
    def _create_enhanced_state_text(self, state: OvercookedState) -> str:
        """Create enhanced state text with coordinates and better descriptions."""
        text_parts = []
        
        # Layout information with coordinates
        text_parts.append(f"The kitchen layout is '{self.layout_info['name']}' with grid size {self.layout_info['shape']}")
        
        # Player descriptions with coordinates
        for i, player in enumerate(state.players):
            pos = player.position
            orientation_desc = self._get_orientation_description(player.orientation)
            
            if player.has_object():
                held_obj = player.get_object()
                if held_obj.name == 'soup':
                    soup_type, num_items, cook_time = held_obj.state
                    holding_desc = f"holding {num_items} {soup_type} soup (cooking {cook_time} steps)"
                else:
                    holding_desc = f"holding a {held_obj.name}"
            else:
                holding_desc = "not holding anything"
            
            player_desc = f"Player {i} is at coordinates ({pos[0]}, {pos[1]}), facing {orientation_desc}, and {holding_desc}"
            text_parts.append(player_desc)
        
        # Object descriptions with coordinates
        if state.objects:
            obj_descriptions = []
            for pos, obj in state.objects.items():
                if obj.name == 'soup':
                    soup_type, num_items, cook_time = obj.state
                    obj_descriptions.append(f"{num_items} {soup_type} soup at ({pos[0]}, {pos[1]}) - cooking {cook_time} steps")
                else:
                    obj_descriptions.append(f"{obj.name} at ({pos[0]}, {pos[1]})")
            text_parts.append(f"Objects: {', '.join(obj_descriptions)}")
        
        # Layout features with coordinates
        feature_descriptions = []
        for feature_type, locations in [
            ('pots', self.layout_info['pot_locations']),
            ('counters', self.layout_info['counter_locations']),
            ('onion dispensers', self.layout_info['onion_dispenser_locations']),
            ('dish dispensers', self.layout_info['dish_dispenser_locations']),
            ('serving areas', self.layout_info['serving_locations'])
        ]:
            if locations:
                coord_list = [f"({pos[0]}, {pos[1]})" for pos in locations]
                feature_descriptions.append(f"{feature_type} at {', '.join(coord_list)}")
        
        if feature_descriptions:
            text_parts.append(f"Kitchen features: {', '.join(feature_descriptions)}")
        
        # Order information
        if state.order_list:
            text_parts.append(f"Active orders: {', '.join(state.order_list)}")
        
        # Timestep
        text_parts.append(f"Current timestep: {state.timestep}")
        
        return ". ".join(text_parts) + "."
    
    def _get_orientation_description(self, orientation: Tuple[int, int]) -> str:
        """Get human-readable orientation description."""
        orientation_map = {
            (0, -1): "north (up)",
            (0, 1): "south (down)", 
            (-1, 0): "west (left)",
            (1, 0): "east (right)"
        }
        return orientation_map.get(orientation, "unknown direction")
    
    def _extract_state_features(self, state: OvercookedState) -> List[float]:
        """Extract numerical features from state."""
        features = []
        
        # Agent positions and held objects
        for player in state.players:
            features.extend([float(player.position[0]), float(player.position[1])])
            features.append(1.0 if player.held_object else 0.0)
        
        # Object counts by type
        object_counts = {'onion': 0, 'dish': 0, 'soup': 0}
        for obj in state.objects.values():
            if obj.name in object_counts:
                object_counts[obj.name] += 1
        
        # Add object counts to features
        for obj_type in ['onion', 'dish', 'soup']:
            features.append(float(object_counts[obj_type]))
        
        # Layout features (static)
        features.append(float(len(self.layout_info['pot_locations'])))
        features.append(float(len(self.layout_info['counter_locations'])))
        features.append(float(len(self.layout_info['onion_dispenser_locations'])))
        features.append(float(len(self.layout_info['dish_dispenser_locations'])))
        features.append(float(len(self.layout_info['serving_locations'])))
        
        # Timestep (normalized)
        features.append(float(state.timestep) / 400.0)
        
        # Pad features to ensure consistent size (20 features)
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def save_training_data(self, filename: str, format: str = 'json', examples: List[TrainingExample] = None):
        """Save training data to file."""
        data_to_save = examples if examples is not None else self.training_data
        
        if format == 'json':
            # Convert to serializable format
            data = []
            for example in data_to_save:
                example_dict = asdict(example)
                # Convert macro actions and standardized intervention
                if example.standardized_intervention:
                    example_dict['standardized_intervention'] = asdict(example.standardized_intervention)
                if example.macro_actions:
                    example_dict['macro_actions'] = [asdict(ma) for ma in example.macro_actions]
                data.append(example_dict)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'txt':
            with open(filename, 'w') as f:
                for i, example in enumerate(data_to_save):
                    f.write(f"Example {i+1}:\n")
                    f.write(f"State: {example.state_text}\n")
                    f.write(f"Command: {example.command}\n")
                    f.write(f"Action: {example.action} ({self.action_names.get(example.action, 'UNKNOWN')})\n")
                    f.write(f"Intervention: {example.intervention_type} - {example.intervention_category}\n")
                    if example.standardized_intervention:
                        f.write(f"Standardized: {example.standardized_intervention.intervention_type} - {example.standardized_intervention.trigger}\n")
                        f.write(f"Urgency: {example.standardized_intervention.urgency}\n")
                        f.write(f"Context: {example.standardized_intervention.context}\n")
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
            'avg_command_length': 0.0,
            'layout_info': self.layout_info,
            'start_positions': self.start_positions
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
    """Demo the enhanced training data generator."""
    print("=== Enhanced Training Data Generator Demo ===\n")
    
    # Initialize components
    command_generator = LLMEnhancedCommandGenerator()
    
    # Create Overcooked MDP with random3 layout
    print("Creating Overcooked MDP with random3 layout...")
    overcooked_mdp = OvercookedGridworld.from_layout_name("random3")
    print(f"Using layout: {overcooked_mdp.layout_name}")
    print(f"Grid shape: {overcooked_mdp.shape}")
    
    # Initialize enhanced generator
    training_generator = EnhancedTrainingGenerator(
        command_generator, 
        overcooked_mdp,
        use_llm_parsing=True
    )
    
    print(f"Start positions: {training_generator.start_positions}")
    print(f"Layout info: {training_generator.layout_info}")
    
    # Generate enhanced training data
    print("Generating enhanced training data...")
    examples = training_generator.generate_enhanced_training_data(num_states=20, commands_per_state=2)
    
    print(f"Generated {len(examples)} training examples\n")
    
    # Show some examples
    print("Sample Enhanced Training Examples:")
    for i, example in enumerate(examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"State: {example.state_text[:150]}...")
        print(f"Command: {example.command}")
        print(f"Action: {example.action} ({training_generator.action_names.get(example.action, 'UNKNOWN')})")
        print(f"Intervention: {example.intervention_type} - {example.intervention_category}")
        if example.standardized_intervention:
            print(f"Standardized: {example.standardized_intervention.intervention_type} - {example.standardized_intervention.trigger}")
            print(f"Urgency: {example.standardized_intervention.urgency}")
            print(f"Macro Actions: {len(example.standardized_intervention.macro_actions)}")
    
    # Show dataset statistics
    print("\n" + "="*50)
    print("Enhanced Dataset Statistics:")
    stats = training_generator.get_dataset_statistics()
    for key, value in stats.items():
        if key not in ['command_length_stats', 'layout_info', 'start_positions']:
            print(f"{key}: {value}")
    
    print(f"\nLayout: {stats['layout_info']['name']}")
    print(f"Start positions: {stats['start_positions']}")
    
    print("\nCommand Length Statistics:")
    for key, value in stats['command_length_stats'].items():
        print(f"  {key}: {value:.2f}")
    
    # Save training data
    print("\nSaving enhanced training data...")
    training_generator.save_training_data('enhanced_training_data.json', 'json')
    training_generator.save_training_data('enhanced_training_data.txt', 'txt')
    print("Enhanced training data saved to 'enhanced_training_data.json' and 'enhanced_training_data.txt'")

if __name__ == "__main__":
    main()
