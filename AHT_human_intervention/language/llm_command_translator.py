#!/usr/bin/env python3
"""
LLM-based Command Translator: Uses sophisticated prompts to translate human commands into low-level actions and macro actions.
"""

import json
import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import openai
from dataclasses import dataclass
from enum import Enum

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld
from envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from language.state_to_text import describe_state

class MacroAction(Enum):
    """Macro actions that can be performed."""
    PICKUP_ONION = "pickup_onion"
    PICKUP_PLATE = "pickup_plate"
    PUT_ONION_IN_POT = "put_onion_in_pot"
    PICKUP_SOUP = "pickup_soup"
    DELIVER_SOUP = "deliver_soup"

@dataclass
class ActionPlan:
    """Represents a plan of actions to execute."""
    macro_action: MacroAction
    low_level_actions: List[int]  # List of action indices (0-5)
    target_position: Optional[Tuple[int, int]] = None
    description: str = ""

class LLMCommandTranslator:
    """Translates human commands into action plans using LLM."""
    
    def __init__(self, mdp: OvercookedGridworld, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.mdp = mdp
        self.model = model
        
        # Set up OpenAI API
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        elif 'OPENAI_API_KEY' in os.environ:
            self.client = openai.OpenAI()
        else:
            print("Warning: No OpenAI API key found. Will use fallback translation.")
            self.client = None
        
        # Define target locations
        self.pot_locations = mdp.get_pot_locations()
        self.onion_dispenser_locations = mdp.get_onion_dispenser_locations()
        self.dish_dispenser_locations = mdp.get_dish_dispenser_locations()
        self.serving_locations = mdp.get_serving_locations()
        
        # Create the sophisticated prompt
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> str:
        """Create a sophisticated prompt for command translation."""
        
        prompt = f"""
You are an expert command translator for the Overcooked game environment. Your task is to translate human commands into specific action plans.

GAME ENVIRONMENT:
- Layout: random3
- Available locations:
  * Pots: {self.pot_locations}
  * Onion dispensers: {self.onion_dispenser_locations}
  * Dish dispensers: {self.dish_dispenser_locations}
  * Serving counters: {self.serving_locations}

AVAILABLE ACTIONS:
- Action 0: STAY (0, 0) - stay in place
- Action 1: NORTH (0, -1) - move up/turn north
- Action 2: SOUTH (0, 1) - move down/turn south  
- Action 3: EAST (1, 0) - move right/turn east
- Action 4: WEST (-1, 0) - move left/turn west
- Action 5: INTERACT - pick up, put down, or interact with objects

MACRO ACTIONS (for complex tasks):
- pickup_onion: Pick up an onion from dispenser
- pickup_plate: Pick up a plate from dispenser
- put_onion_in_pot: Put onion in pot to start cooking
- pickup_soup: Pick up cooked soup from pot
- deliver_soup: Deliver soup to serving counter

COMMAND TRANSLATION RULES:
1. For simple movement commands (e.g., "turn left", "move right", "go up"):
   - Extract the direct low-level action (0-5)
   - No macro action needed (set to null)
   - Examples: "turn left" → action 4, "move right" → action 3

2. For interaction commands (e.g., "pick up onion", "grab plate"):
   - Use action 5 (INTERACT)
   - Set appropriate macro action (pickup_onion, pickup_plate, etc.)

3. For complex commands (e.g., "go to pot and put onion in"):
   - Focus on the immediate action first
   - Use macro action for the overall goal
   - Extract any immediate low-level actions

4. For stay/wait commands (e.g., "stay", "wait", "stop"):
   - Use action 0 (STAY)
   - No macro action needed

5. For directional commands (e.g., "face north", "look left"):
   - Use corresponding action (1-4)
   - No macro action needed

CURRENT STATE CONTEXT:
The agent's current state will be provided in the format:
"Player at (x, y) facing direction, holding object. Teammate at (x, y) facing direction, holding object. Pot states. Available dispensers."

RESPONSE FORMAT:
Return a JSON object with:
{{
    "macro_action": "macro_action_name" or null,
    "low_level_actions": [list_of_action_indices_0_to_5],
    "target_position": [x, y] or null,
    "description": "explanation_of_what_this_plan_does",
    "reasoning": "step_by_step_reasoning_for_the_translation"
}}

EXAMPLES:

Command: "Turn left"
Response: {{
    "macro_action": null,
    "low_level_actions": [4],
    "target_position": null,
    "description": "Turning left (facing west)",
    "reasoning": "Simple directional command. Action 4 is WEST, which is turning left."
}}

Command: "Pick up the onion"
Response: {{
    "macro_action": "pickup_onion",
    "low_level_actions": [5],
    "target_position": null,
    "description": "Picking up onion from dispenser",
    "reasoning": "Interaction command to pick up onion. Using action 5 (INTERACT) with pickup_onion macro."
}}

Command: "Stay here"
Response: {{
    "macro_action": null,
    "low_level_actions": [0],
    "target_position": null,
    "description": "Staying in place",
    "reasoning": "Simple stay command. Using action 0 (STAY)."
}}

Command: "Move right"
Response: {{
    "macro_action": null,
    "low_level_actions": [3],
    "target_position": null,
    "description": "Moving right",
    "reasoning": "Simple movement command. Action 3 is EAST (right)."
}}

Now translate the following command:
"""
        return prompt
    
    def translate_command(self, command: str, current_state: OvercookedState, agent_idx: int = 0) -> ActionPlan:
        """Translate a human command into an action plan using LLM."""
        
        if not self.client:
            print("No OpenAI client available. Using fallback translation.")
            return self._fallback_translation(command, current_state, agent_idx)
        
        # Get current agent position and orientation
        agent_pos = current_state.players[agent_idx].position
        agent_ori = current_state.players[agent_idx].orientation
        
        # Create the full prompt with current state
        state_description = describe_state(self.mdp, current_state, mode="full")
        full_prompt = self.prompt_template + f"\nCommand: \"{command}\"\nCurrent State: {state_description}\nResponse:"
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert command translator for the Overcooked game."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
            else:
                raise ValueError(f"Could not extract JSON from response: {response_text}")
            
            # Parse JSON
            result = json.loads(json_str)
            
            # Convert to ActionPlan
            macro_action = MacroAction(result["macro_action"]) if result["macro_action"] else None
            target_position = tuple(result["target_position"]) if result["target_position"] else None
            description = result["description"]
            
            # Fix target position for macro actions if not set by LLM
            if macro_action and not target_position:
                if macro_action == MacroAction.PICKUP_ONION and self.onion_dispenser_locations:
                    target_position = self.onion_dispenser_locations[0]
                elif macro_action == MacroAction.PICKUP_PLATE and self.dish_dispenser_locations:
                    target_position = self.dish_dispenser_locations[0]
                elif macro_action == MacroAction.DELIVER_SOUP and self.serving_locations:
                    target_position = self.serving_locations[0]
            
            # Generate low-level actions using pathfinding
            low_level_actions = self._generate_low_level_actions(
                agent_pos, agent_ori, target_position, macro_action, current_state
            )
            
            return ActionPlan(
                macro_action=macro_action,
                low_level_actions=low_level_actions,
                target_position=target_position,
                description=description
            )
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            # Fallback to simple rule-based translation
            return self._fallback_translation(command, current_state, agent_idx)
    
    def _generate_low_level_actions(self, agent_pos, agent_ori, target_position, macro_action, current_state):
        """Generate low-level actions using pathfinding and current game state."""
        actions = []
        current_pos = agent_pos
        current_ori = agent_ori
        
        # Get other agent positions to avoid collisions
        other_agent_positions = []
        for i, player in enumerate(current_state.players):
            if i != 0:  # Skip the current agent (agent_idx=0)
                other_agent_positions.append(player.position)
        
        # If we have a target position and it's different from current position
        if target_position and target_position != current_pos:
            # Use pathfinder to get path to target, avoiding other agents
            pathfinder = AStarPathfinder(self.mdp)
            path = pathfinder.find_path(current_pos, target_position, blocked_positions=other_agent_positions)
            
            if path:
                # Convert path to actions
                for i, next_pos in enumerate(path[1:], 1):  # Skip first position (current)
                    # Determine direction to move
                    dx = next_pos[0] - current_pos[0]
                    dy = next_pos[1] - current_pos[1]
                    
                    if dx == 1:  # Move east
                        action = 3
                    elif dx == -1:  # Move west
                        action = 4
                    elif dy == 1:  # Move south
                        action = 2
                    elif dy == -1:  # Move north
                        action = 1
                    else:
                        continue  # Skip if no movement
                    
                    actions.append(action)
                    current_pos = next_pos
            else:
                # If no path found (due to blocking), try alternative approach
                # Add a wait action and try again with a different path
                actions.append(0)  # STAY action
                # Could implement more sophisticated re-planning here
        
        # Add orientation actions if needed
        if macro_action:
            if macro_action == MacroAction.PICKUP_ONION:
                # Need to face the onion dispenser
                onion_dispensers = self.onion_dispenser_locations
                if onion_dispensers:
                    target_dispenser = onion_dispensers[0]  # Use first one
                    # Add turn action to face the dispenser
                    if current_pos[0] < target_dispenser[0]:  # Need to face east
                        if current_ori != (1, 0):
                            actions.append(3)  # Turn east
                    elif current_pos[0] > target_dispenser[0]:  # Need to face west
                        if current_ori != (-1, 0):
                            actions.append(4)  # Turn west
                    elif current_pos[1] < target_dispenser[1]:  # Need to face south
                        if current_ori != (0, 1):
                            actions.append(2)  # Turn south
                    elif current_pos[1] > target_dispenser[1]:  # Need to face north
                        if current_ori != (0, -1):
                            actions.append(1)  # Turn north
                    actions.append(5)  # INTERACT
                    
            elif macro_action == MacroAction.PICKUP_PLATE:
                # Need to face the dish dispenser
                dish_dispensers = self.dish_dispenser_locations
                if dish_dispensers:
                    target_dispenser = dish_dispensers[0]  # Use first one
                    # Add turn action to face the dispenser
                    if current_pos[0] < target_dispenser[0]:  # Need to face east
                        if current_ori != (1, 0):
                            actions.append(3)  # Turn east
                    elif current_pos[0] > target_dispenser[0]:  # Need to face west
                        if current_ori != (-1, 0):
                            actions.append(4)  # Turn west
                    elif current_pos[1] < target_dispenser[1]:  # Need to face south
                        if current_ori != (0, 1):
                            actions.append(2)  # Turn south
                    elif current_pos[1] > target_dispenser[1]:  # Need to face north
                        if current_ori != (0, -1):
                            actions.append(1)  # Turn north
                    actions.append(5)  # INTERACT
                    
            elif macro_action == MacroAction.DELIVER_SOUP:
                # Need to face the serving area
                serving_locations = self.serving_locations
                if serving_locations:
                    target_serving = serving_locations[0]  # Use first one
                    # Add turn action to face the serving area
                    if current_pos[0] < target_serving[0]:  # Need to face east
                        if current_ori != (1, 0):
                            actions.append(3)  # Turn east
                    elif current_pos[0] > target_serving[0]:  # Need to face west
                        if current_ori != (-1, 0):
                            actions.append(4)  # Turn west
                    elif current_pos[1] < target_serving[1]:  # Need to face south
                        if current_ori != (0, 1):
                            actions.append(2)  # Turn south
                    elif current_pos[1] > target_serving[1]:  # Need to face north
                        if current_ori != (0, -1):
                            actions.append(1)  # Turn north
                    actions.append(5)  # INTERACT
        
        return actions

class AStarPathfinder:
    """Simple BFS-based pathfinder for navigation in Overcooked."""
    
    def __init__(self, mdp: OvercookedGridworld):
        self.mdp = mdp
        self.terrain_mtx = mdp.terrain_mtx
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  blocked_positions: Optional[List[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        """Find shortest path from start to goal using BFS, avoiding blocked positions."""
        if start == goal:
            return [start]
        
        # Convert blocked positions to set for faster lookup
        blocked = set(blocked_positions) if blocked_positions else set()
        
        # BFS implementation
        queue = [(start, [start])]
        visited = set([start])
        
        while queue:
            current, path = queue.pop(0)
            
            # Check all 4 directions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_x = current[0] + dx
                next_y = current[1] + dy
                next_pos = (next_x, next_y)
                
                # Check bounds
                if (next_x < 0 or next_x >= len(self.terrain_mtx[0]) or 
                    next_y < 0 or next_y >= len(self.terrain_mtx)):
                    continue
                
                # Check if walkable (not a wall)
                if self.terrain_mtx[next_y][next_x] == "X":
                    continue
                
                # Check if blocked by other agent
                if next_pos in blocked:
                    continue
                
                # Check if visited
                if next_pos in visited:
                    continue
                
                # Add to visited
                visited.add(next_pos)
                
                # Check if we reached the goal
                if next_pos == goal:
                    return path + [next_pos]
                
                # Add to queue
                queue.append((next_pos, path + [next_pos]))
        
        # No path found
        return []
    
    def _fallback_translation(self, command: str, current_state: OvercookedState, agent_idx: int = 0) -> ActionPlan:
        """Fallback rule-based translation if LLM fails."""
        command_lower = command.lower()
        
        # Simple rule-based fallback
        if any(word in command_lower for word in ["pick up onion", "grab onion", "get onion"]):
            return ActionPlan(
                macro_action=MacroAction.PICKUP_ONION,
                low_level_actions=[5],  # INTERACT
                description="Picking up onion (fallback)"
            )
        elif any(word in command_lower for word in ["pick up plate", "grab plate", "get plate", "pick up dish"]):
            return ActionPlan(
                macro_action=MacroAction.PICKUP_PLATE,
                low_level_actions=[5],  # INTERACT
                description="Picking up plate (fallback)"
            )
        elif any(word in command_lower for word in ["turn left", "face left", "look left"]):
            return ActionPlan(
                macro_action=None,
                low_level_actions=[4],  # WEST
                description="Turning left (fallback)"
            )
        elif any(word in command_lower for word in ["turn right", "face right", "look right"]):
            return ActionPlan(
                macro_action=None,
                low_level_actions=[3],  # EAST
                description="Turning right (fallback)"
            )
        elif any(word in command_lower for word in ["move up", "go up", "turn up", "face north"]):
            return ActionPlan(
                macro_action=None,
                low_level_actions=[1],  # NORTH
                description="Moving up (fallback)"
            )
        elif any(word in command_lower for word in ["move down", "go down", "turn down", "face south"]):
            return ActionPlan(
                macro_action=None,
                low_level_actions=[2],  # SOUTH
                description="Moving down (fallback)"
            )
        elif any(word in command_lower for word in ["move left", "go left"]):
            return ActionPlan(
                macro_action=None,
                low_level_actions=[4],  # WEST
                description="Moving left (fallback)"
            )
        elif any(word in command_lower for word in ["move right", "go right"]):
            return ActionPlan(
                macro_action=None,
                low_level_actions=[3],  # EAST
                description="Moving right (fallback)"
            )
        elif any(word in command_lower for word in ["stay", "wait", "stop", "halt"]):
            return ActionPlan(
                macro_action=None,
                low_level_actions=[0],  # STAY
                description="Staying in place (fallback)"
            )
        elif any(word in command_lower for word in ["pick up", "grab", "interact"]):
            return ActionPlan(
                macro_action=None,
                low_level_actions=[5],  # INTERACT
                description="Interacting (fallback)"
            )
        else:
            return ActionPlan(
                macro_action=None,
                low_level_actions=[0],  # STAY
                description="No specific action identified, staying in place (fallback)"
            )

def load_commands(filename: str = "language/llm_generated_commands.json") -> List[str]:
    """Load commands from the JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Flatten all commands into a single list
    all_commands = []
    for category, subcategories in data.items():
        for subcategory, commands in subcategories.items():
            if isinstance(commands, list):
                all_commands.extend(commands)
    
    return all_commands

def create_llm_translation_dataset(commands: List[str], mdp: OvercookedGridworld, 
                                 num_examples: int = 10) -> List[Dict[str, Any]]:
    """Create a dataset of LLM-based command translations."""
    
    translator = LLMCommandTranslator(mdp)
    dataset = []
    
    # Create some sample states for testing
    valid_positions = []
    for y in range(len(mdp.terrain_mtx)):
        for x in range(len(mdp.terrain_mtx[0])):
            if mdp.terrain_mtx[y][x] == " ":
                valid_positions.append((x, y))
    
    for i in range(min(num_examples, len(commands))):
        command = commands[i]
        
        # Create a random state for testing
        if valid_positions:
            player_pos = valid_positions[np.random.choice(len(valid_positions))]
            state = OvercookedState.from_player_positions([player_pos, (0, 0)], order_list=None)
            
            # Translate command using LLM
            plan = translator.translate_command(command, state)
            
            dataset.append({
                'command': command,
                'macro_action': plan.macro_action.value if plan.macro_action else None,
                'low_level_actions': plan.low_level_actions,
                'target_position': plan.target_position,
                'description': plan.description,
                'state_text': describe_state(mdp, state, kind="english")
            })
    
    return dataset

def main():
    """Main function to demonstrate LLM-based command translation."""
    
    # Load commands
    print("Loading commands...")
    commands = load_commands()
    print(f"Loaded {len(commands)} commands")
    
    # Create MDP
    print("Creating MDP...")
    mdp = OvercookedGridworld.from_layout_name("random3")
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Will use fallback translation.")
        print("Set OPENAI_API_KEY environment variable to use LLM translation.")
    else:
        print("OpenAI API key found. Using LLM translation.")
    
    # Create translation dataset
    print("Creating LLM translation dataset...")
    dataset = create_llm_translation_dataset(commands, mdp, num_examples=5)
    
    # Save dataset
    output_file = "language/llm_command_translations.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Saved {len(dataset)} LLM command translations to {output_file}")
    
    # Show some examples
    print("\nExample LLM translations:")
    for i, example in enumerate(dataset):
        print(f"\nExample {i+1}:")
        print(f"  Command: {example['command']}")
        print(f"  Macro Action: {example['macro_action']}")
        print(f"  Low-level Actions: {example['low_level_actions']}")
        print(f"  Description: {example['description']}")

if __name__ == "__main__":
    main()
