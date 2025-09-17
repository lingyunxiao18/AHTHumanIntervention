#!/usr/bin/env python3
"""
LLM-based Command Translator for Medium-Level Actions: Uses sophisticated prompts to translate human commands into medium-level actions.
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

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from shared.envs.envs.overcooked.overcooked_ai_py.planning.planners import MediumLevelActionManager
from shared.utils.state_to_text import describe_state

class MediumLevelActionType(Enum):
    """Medium-level action types that can be performed."""
    PICKUP_ONION = "pickup_onion"
    PICKUP_DISH = "pickup_dish"
    PICKUP_SOUP = "pickup_soup"
    PUT_ONION_IN_POT = "put_onion_in_pot"
    DELIVER_SOUP = "deliver_soup"
    PLACE_ON_COUNTER = "place_on_counter"
    WAIT = "wait"

@dataclass
class MediumLevelActionPlan:
    """Represents a medium-level action plan."""
    action_type: MediumLevelActionType
    action_index: int  # Index in the available medium-level actions
    target_position: Optional[Tuple[int, int]] = None
    description: str = ""

class LLMMediumLevelCommandTranslator:
    """Translates human commands into medium-level actions using LLM."""
    
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
        
        # Initialize medium-level action manager
        self.ml_params = {
            "wait_allowed": True,
            "counter_drop": mdp.get_counter_locations(),
            "counter_pickup": mdp.get_counter_locations(),
            "same_motion_goals": True,
            "start_orientations": False,
            "counter_goals": mdp.get_counter_locations()
        }
        self.ml_action_manager = MediumLevelActionManager(mdp, self.ml_params)
        
        # Define target locations
        self.pot_locations = mdp.get_pot_locations()
        self.onion_dispenser_locations = mdp.get_onion_dispenser_locations()
        self.dish_dispenser_locations = mdp.get_dish_dispenser_locations()
        self.serving_locations = mdp.get_serving_locations()
        
        # Create the sophisticated prompt
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> str:
        """Create a sophisticated prompt for medium-level command translation."""
        
        prompt = f"""
You are an expert command translator for the Overcooked game environment. Your task is to translate human commands into medium-level actions.

GAME ENVIRONMENT:
- Layout: random3
- Available locations:
  * Pots: {self.pot_locations}
  * Onion dispensers: {self.onion_dispenser_locations}
  * Dish dispensers: {self.dish_dispenser_locations}
  * Serving counters: {self.serving_locations}

MEDIUM-LEVEL ACTIONS:
- pickup_onion: Pick up an onion from dispenser or counter
- pickup_dish: Pick up a dish from dispenser or counter
- pickup_soup: Pick up soup from counter
- put_onion_in_pot: Put onion in pot to start cooking
- deliver_soup: Deliver soup to serving counter
- place_on_counter: Place object on empty counter
- wait: Stay in current position

COMMAND TRANSLATION RULES:
1. For pickup commands (e.g., "pick up onion", "grab dish"):
   - Use pickup_onion or pickup_dish based on object type
   - Target the appropriate dispenser or counter location

2. For cooking commands (e.g., "cook soup", "put onion in pot"):
   - Use put_onion_in_pot
   - Target an empty or partially full pot

3. For delivery commands (e.g., "deliver soup", "serve soup"):
   - Use deliver_soup
   - Target a serving counter

4. For placement commands (e.g., "put down", "place on counter"):
   - Use place_on_counter
   - Target an empty counter

5. For waiting commands (e.g., "wait", "stay"):
   - Use wait
   - No target position needed

RESPONSE FORMAT:
Return a JSON object with the following structure:
{{
    "action_type": "action_type_name",
    "description": "brief description of what this action does"
}}

Example responses:
- "pick up onion" → {{"action_type": "pickup_onion", "description": "Pick up onion from dispenser"}}
- "cook soup" → {{"action_type": "put_onion_in_pot", "description": "Put onion in pot to cook"}}
- "deliver soup" → {{"action_type": "deliver_soup", "description": "Deliver soup to serving counter"}}
- "wait" → {{"action_type": "wait", "description": "Stay in current position"}}
"""
        return prompt
    
    def translate_command(self, state: OvercookedState, player_idx: int, human_command: str) -> MediumLevelActionPlan:
        """
        Translate a human command into a medium-level action.
        
        Args:
            state: Current game state
            player_idx: Player index (0 or 1)
            human_command: Human command string
            
        Returns:
            MediumLevelActionPlan with action type and index
        """
        if not human_command.strip():
            # Return wait action for empty commands
            return self._get_wait_action(state, player_idx)
        
        if self.client is None:
            # Fallback translation without LLM
            return self._fallback_translation(state, player_idx, human_command)
        
        try:
            # Get available medium-level actions for the player
            player = state.players[player_idx]
            available_ml_actions = self.ml_action_manager.get_medium_level_actions(state, player)
            
            # Create state description
            state_text = describe_state(self.mdp, state, kind="english")
            
            # Create the full prompt
            full_prompt = f"{self.prompt_template}\n\nCURRENT STATE:\n{state_text}\n\nPLAYER {player_idx} STATUS:\n"
            if player.has_object():
                full_prompt += f"- Holding: {player.get_object().name}\n"
            else:
                full_prompt += "- Holding: nothing\n"
            
            full_prompt += f"\nHUMAN COMMAND: {human_command}\n\nAVAILABLE MEDIUM-LEVEL ACTIONS:\n"
            
            # Add available actions to prompt
            for i, ml_action in enumerate(available_ml_actions):
                pos, orient = ml_action
                full_prompt += f"- Action {i}: Go to position {pos} facing {orient}\n"
            
            full_prompt += "\nRESPONSE:"
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert command translator for Overcooked. Respond only with valid JSON."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_data = json.loads(response_text)
            action_type_str = response_data.get("action_type", "wait")
            description = response_data.get("description", "")
            
            # Map action type string to enum
            action_type_map = {
                "pickup_onion": MediumLevelActionType.PICKUP_ONION,
                "pickup_dish": MediumLevelActionType.PICKUP_DISH,
                "pickup_soup": MediumLevelActionType.PICKUP_SOUP,
                "put_onion_in_pot": MediumLevelActionType.PUT_ONION_IN_POT,
                "deliver_soup": MediumLevelActionType.DELIVER_SOUP,
                "place_on_counter": MediumLevelActionType.PLACE_ON_COUNTER,
                "wait": MediumLevelActionType.WAIT
            }
            
            action_type = action_type_map.get(action_type_str, MediumLevelActionType.WAIT)
            
            # Find the best matching action index
            action_index = self._find_best_action_index(state, player_idx, action_type, available_ml_actions)
            
            return MediumLevelActionPlan(
                action_type=action_type,
                action_index=action_index,
                description=description
            )
            
        except Exception as e:
            print(f"Error in LLM translation: {e}")
            return self._fallback_translation(state, player_idx, human_command)
    
    def _find_best_action_index(self, state: OvercookedState, player_idx: int, 
                               action_type: MediumLevelActionType, 
                               available_actions: List[Tuple]) -> int:
        """Find the best action index for the given action type."""
        
        player = state.players[player_idx]
        
        if action_type == MediumLevelActionType.WAIT:
            # Find wait action (current position)
            wait_goal = (player.position, player.orientation)
            try:
                return available_actions.index(wait_goal)
            except ValueError:
                return 0  # Default to first action
        
        elif action_type == MediumLevelActionType.PICKUP_ONION:
            # Find action closest to onion dispenser
            target_locations = self.onion_dispenser_locations
            return self._find_closest_action(available_actions, target_locations)
        
        elif action_type == MediumLevelActionType.PICKUP_DISH:
            # Find action closest to dish dispenser
            target_locations = self.dish_dispenser_locations
            return self._find_closest_action(available_actions, target_locations)
        
        elif action_type == MediumLevelActionType.PICKUP_SOUP:
            # Find action closest to counter with soup
            counter_objects = self.mdp.get_counter_objects_dict(state, self.ml_params["counter_pickup"])
            target_locations = counter_objects.get("soup", [])
            return self._find_closest_action(available_actions, target_locations)
        
        elif action_type == MediumLevelActionType.PUT_ONION_IN_POT:
            # Find action closest to empty or partially full pot
            pot_states = self.mdp.get_pot_states(state)
            target_locations = pot_states["empty"] + pot_states["onion"]["partially_full"]
            return self._find_closest_action(available_actions, target_locations)
        
        elif action_type == MediumLevelActionType.DELIVER_SOUP:
            # Find action closest to serving counter
            return self._find_closest_action(available_actions, self.serving_locations)
        
        elif action_type == MediumLevelActionType.PLACE_ON_COUNTER:
            # Find action closest to empty counter
            empty_counters = self.mdp.get_empty_counter_locations(state)
            return self._find_closest_action(available_actions, empty_counters)
        
        else:
            return 0  # Default to first action
    
    def _find_closest_action(self, available_actions: List[Tuple], 
                            target_locations: List[Tuple]) -> int:
        """Find the action closest to any target location."""
        if not target_locations:
            return 0
        
        min_dist = float('inf')
        best_index = 0
        
        for i, action in enumerate(available_actions):
            action_pos, _ = action
            
            for target_pos in target_locations:
                dist = abs(action_pos[0] - target_pos[0]) + abs(action_pos[1] - target_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    best_index = i
        
        return best_index
    
    def _get_wait_action(self, state: OvercookedState, player_idx: int) -> MediumLevelActionPlan:
        """Get wait action for the player."""
        player = state.players[player_idx]
        available_actions = self.ml_action_manager.get_medium_level_actions(state, player)
        
        wait_goal = (player.position, player.orientation)
        try:
            action_index = available_actions.index(wait_goal)
        except ValueError:
            action_index = 0
        
        return MediumLevelActionPlan(
            action_type=MediumLevelActionType.WAIT,
            action_index=action_index,
            description="Stay in current position"
        )
    
    def _fallback_translation(self, state: OvercookedState, player_idx: int, 
                            human_command: str) -> MediumLevelActionPlan:
        """Fallback translation without LLM."""
        command_lower = human_command.lower()
        
        if "onion" in command_lower and ("pick" in command_lower or "grab" in command_lower):
            action_type = MediumLevelActionType.PICKUP_ONION
        elif "dish" in command_lower and ("pick" in command_lower or "grab" in command_lower):
            action_type = MediumLevelActionType.PICKUP_DISH
        elif "soup" in command_lower and ("pick" in command_lower or "grab" in command_lower):
            action_type = MediumLevelActionType.PICKUP_SOUP
        elif "cook" in command_lower or "pot" in command_lower:
            action_type = MediumLevelActionType.PUT_ONION_IN_POT
        elif "deliver" in command_lower or "serve" in command_lower:
            action_type = MediumLevelActionType.DELIVER_SOUP
        elif "wait" in command_lower or "stay" in command_lower:
            action_type = MediumLevelActionType.WAIT
        else:
            action_type = MediumLevelActionType.WAIT
        
        # Find best action index
        player = state.players[player_idx]
        available_actions = self.ml_action_manager.get_medium_level_actions(state, player)
        action_index = self._find_best_action_index(state, player_idx, action_type, available_actions)
        
        return MediumLevelActionPlan(
            action_type=action_type,
            action_index=action_index,
            description=f"Fallback translation for: {human_command}"
        )

def main():
    """Test the medium-level command translator."""
    
    # Create MDP
    mdp = OvercookedGridworld.from_layout_name("random3")
    
    # Create translator
    translator = LLMMediumLevelCommandTranslator(mdp)
    
    # Create test state
    from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
    test_state = OvercookedState.from_player_positions(
        ((1, 1), (3, 3)), 
        order_list=["any"]
    )
    
    # Test commands
    test_commands = [
        "Pick up an onion",
        "Cook some soup",
        "Deliver the soup",
        "Wait here",
        "Grab a dish"
    ]
    
    print("Testing Medium-Level Command Translator:")
    print("=" * 50)
    
    for command in test_commands:
        print(f"\nCommand: '{command}'")
        
        # Test for both players
        for player_idx in [0, 1]:
            plan = translator.translate_command(test_state, player_idx, command)
            print(f"  Player {player_idx}: {plan.action_type.value} (index {plan.action_index}) - {plan.description}")
    
    print("\nTranslation complete!")

if __name__ == "__main__":
    main()
