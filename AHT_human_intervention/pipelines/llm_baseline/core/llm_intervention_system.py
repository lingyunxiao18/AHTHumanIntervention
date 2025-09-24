#!/usr/bin/env python3
"""
LLM-based Intervention System: Handles human intervention commands for agent performance correction.
This system processes intervention commands and provides appropriate responses for agent behavior modification.
"""

import json
import os
from typing import Dict, Any, Optional, List
from openai import OpenAI
from dataclasses import dataclass
from .structured_state_formatter import create_structured_state_description

@dataclass
class InterventionResult:
    """Result of processing an intervention command."""
    action_override: Optional[int] = None  # 0=STAY, 1=EAST, 2=SOUTH, 3=NORTH, 4=WEST, 5=INTERACT
    macro_override: Optional[str] = None  # New macro like "GET_ONION", "SERVE_SOUP", "TAKE_SOUP", "GET_DISH", "PUT_ONION_IN_POT", "WAIT_FOR_COOKING", "DROP_OFF"
    duration: Optional[int] = None  # How many steps to persist the override (default: 1 for actions, until completion for macros)
    reasoning: str = ""

PROAGENT_ALLOWED_ML_ACTIONS = {
    # canonical ProAgent medium-level actions
    "pickup(onion)",
    "pickup(dish)",
    "put_onion_in_pot",
    "fill_dish_with_soup",
    "deliver_soup",
    # wait(k) handled dynamically
}


class LLMInterventionSystem:
    """Handles human intervention commands using LLM for agent performance correction."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the LLM intervention system.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: OpenAI model to use
        """
        self.model = model
        
        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
            print("🤖 OpenAI client initialized with provided API key")
        elif os.getenv('OPENAI_API_KEY'):
            self.client = OpenAI()
            print("🤖 OpenAI client initialized with environment API key")
        else:
            self.client = None
            print("❌ OPENAI_API_KEY not set and no api_key provided. LLM interventions disabled.")
        
        # Prompt templates (loaded from files; required)
        self._system_prompt: Optional[str] = None
        self._user_prompt_template: Optional[str] = None
        self._load_prompt_templates()
        
        # Interaction history tracking (kept for potential future use)
        self.interaction_history: List[Dict[str, Any]] = []
        self.max_history_length = 20

    # =============== ProAgent-specific intervention (no templates required) ===============
    def process_intervention_proagent(self, command: str) -> Optional[InterventionResult]:
        """
        Map free-text human command into a ProAgent medium-level action string.
        Returns an InterventionResult with macro_override set to one of:
          - "pickup(onion)"
          - "pickup(dish)"
          - "put_onion_in_pot"
          - "fill_dish_with_soup"
          - "deliver_soup"
          - "wait(k)"  (k parsed; defaults to 1 if missing)

        The caller should forward macro_override directly to
        ProMediumLevelAgent.apply_human_intervention(macro_override).
        """
        try:
            normalized = (command or "").strip().lower()
            if not normalized:
                return None

            def parse_wait_val(text: str) -> int:
                # extract a positive integer following 'wait'
                import re
                m = re.search(r"wait\s*\(?\s*(\d+)\s*\)?", text)
                if m:
                    try:
                        return max(1, int(m.group(1)))
                    except Exception:
                        return 1
                return 1

            # Wait
            if "wait" in normalized:
                k = parse_wait_val(normalized)
                return InterventionResult(macro_override=f"wait({k})", duration=k, reasoning="Mapped to ProAgent 'wait(k)'.")

            # Deliver/serve
            if any(w in normalized for w in ["deliver", "serve", "drop off", "hand in", "submit"]):
                return InterventionResult(macro_override="deliver_soup", reasoning="Mapped to ProAgent 'deliver_soup'.")

            # Fill soup with dish
            if any(w in normalized for w in ["fill", "scoop", "take soup", "get soup"]):
                return InterventionResult(macro_override="fill_dish_with_soup", reasoning="Mapped to ProAgent 'fill_dish_with_soup'.")

            # Put onion in pot
            if any(w in normalized for w in ["put onion", "place onion", "onion in pot", "add onion"]):
                return InterventionResult(macro_override="put_onion_in_pot", reasoning="Mapped to ProAgent 'put_onion_in_pot'.")

            # Pickup dish
            if any(w in normalized for w in ["pickup dish", "pick up dish", "get dish", "grab dish", "take dish", "plate"]):
                return InterventionResult(macro_override="pickup(dish)", reasoning="Mapped to ProAgent 'pickup(dish)'.")

            # Pickup onion
            if any(w in normalized for w in ["pickup onion", "pick up onion", "get onion", "grab onion", "take onion"]):
                return InterventionResult(macro_override="pickup(onion)", reasoning="Mapped to ProAgent 'pickup(onion)'.")

            # Fallback simple keywords
            if "dish" in normalized:
                return InterventionResult(macro_override="pickup(dish)", reasoning="Fallback mapping by 'dish' keyword.")
            if "onion" in normalized:
                return InterventionResult(macro_override="pickup(onion)", reasoning="Fallback mapping by 'onion' keyword.")
            if "pot" in normalized:
                return InterventionResult(macro_override="put_onion_in_pot", reasoning="Fallback mapping by 'pot' keyword.")
            if any(w in normalized for w in ["serve", "deliver", "window"]):
                return InterventionResult(macro_override="deliver_soup", reasoning="Fallback mapping by 'serve/deliver' keyword.")

            # Last resort: wait(1)
            return InterventionResult(macro_override="wait(1)", duration=1, reasoning="Unrecognized; default to 'wait(1)'.")
        except Exception as e:
            print(f"❌ ProAgent intervention mapping failed: {e}")
            return None
    
    def _load_prompt_templates(self) -> None:
        """Load system and user prompt templates from sibling .txt files. Raises if missing."""
        try:
            base_dir = os.path.dirname(__file__)
            system_path = os.path.join(base_dir, 'system_prompt.txt')
            user_path = os.path.join(base_dir, 'user_prompt.txt')
            if os.path.exists(system_path):
                with open(system_path, 'r', encoding='utf-8') as f:
                    self._system_prompt = f.read().strip()
                print("🧩 Loaded system_prompt.txt")
            if os.path.exists(user_path):
                with open(user_path, 'r', encoding='utf-8') as f:
                    self._user_prompt_template = f.read()
                print("🧩 Loaded user_prompt.txt")
            if not self._system_prompt or not self._user_prompt_template:
                print("❌ Missing system_prompt.txt or user_prompt.txt; prompts are required.")
        except Exception as e:
            print(f"⚠️ Failed to load prompt templates: {e}")

    def add_to_history(self, step: int, ego_action: int, mate_action: int, 
                      ego_pos: tuple, mate_pos: tuple, ego_item: str, mate_item: str,
                      intervention: Optional[str] = None, intervention_result: Optional[Dict] = None):
        """Add interaction to history."""
        history_entry = {
            'step': step,
            'ego_action': ego_action,
            'mate_action': mate_action,
            'ego_pos': ego_pos,
            'mate_pos': mate_pos,
            'ego_item': ego_item,
            'mate_item': mate_item,
            'intervention': intervention,
            'intervention_result': intervention_result
        }
        
        self.interaction_history.append(history_entry)
        
        # Keep only recent history
        if len(self.interaction_history) > self.max_history_length:
            self.interaction_history = self.interaction_history[-self.max_history_length:]
    
    def _action_to_name(self, action: int) -> str:
        """Convert action number to name."""
        action_names = {
            0: "STAY",
            1: "EAST", 
            2: "SOUTH",
            3: "NORTH",
            4: "WEST",
            5: "INTERACT"
        }
        return action_names.get(action, f"UNK({action})")
    
    def _format_history_10_steps(self) -> str:
        """Format the last 10 steps of interaction history."""
        if not self.interaction_history:
            return "No history available."
        
        history_lines = []
        recent_history = self.interaction_history[-10:]  # Last 10 steps
        
        for entry in recent_history:
            step = entry['step']
            ego_action = self._action_to_name(entry['ego_action'])
            mate_action = self._action_to_name(entry['mate_action'])
            ego_pos = entry['ego_pos']
            mate_pos = entry['mate_pos']
            ego_item = entry['ego_item']
            mate_item = entry['mate_item']
            intervention = entry.get('intervention', '')
            
            line = f"Step {step}: Ego({ego_action}, pos{ego_pos}, {ego_item}) | Mate({mate_action}, pos{mate_pos}, {mate_item})"
            if intervention:
                line += f" | Intervention: '{intervention}'"
            history_lines.append(line)
        
        return "\n".join(history_lines)

    def _create_layout_grid_from_terrain(self, terrain: List[List[str]]) -> str:
        """Create ASCII grid representation from terrain data."""
        if not terrain:
            return "Layout not available"
        
        grid_lines = []
        for y, row in enumerate(terrain):
            line = ""
            for x, cell in enumerate(row):
                line += cell
            grid_lines.append(line)
        
        return "\n".join(grid_lines)

    def _create_structured_state_from_agent_state(self, agent_state: Dict[str, Any]) -> str:
        """Create structured state description from agent_state using the layout-agnostic formatter."""
        try:
            # Map agent_state to format expected by structured_state_formatter
            state_info = {
                'scene_size': agent_state.get('scene_size', (8, 5)),
                'ego_pos': agent_state.get('agent_pos'),
                'ego_facing': agent_state.get('agent_orient'),
                'ego_item': agent_state.get('ego_item'),
                'mate_pos': agent_state.get('mate_pos'),
                'mate_facing': agent_state.get('mate_orient'),
                'mate_item': agent_state.get('mate_item'),
                'layout_name': agent_state.get('layout', agent_state.get('layout_name', 'unknown')),
                'onion_locations': agent_state.get('onion_locations'),
                'dish_locations': agent_state.get('dish_locations'),
                'serve_locations': agent_state.get('serve_locations'),
                'pot_states': agent_state.get('pot_states'),
            }
            
            structured_desc = create_structured_state_description(
                state_info,
                command="",  # No command for state-only description
                grounding={},
                macro_label=""
            )
            
            return structured_desc['state_text']
        except Exception as e:
            return f"Structured state creation failed: {e}"

    def _render_user_prompt_from_template(self, command: str, agent_state: Dict[str, Any]) -> str:
        """Render the user prompt using the loaded template and agent state. Requires template to be loaded."""
        if not self._user_prompt_template:
            raise RuntimeError("user_prompt.txt not loaded; it is required.")

        # Add template-specific fields to agent_state
        agent_state_with_template_fields = agent_state.copy()
        agent_state_with_template_fields.update({
            'layout_name': agent_state.get('layout', agent_state.get('layout_name', 'unknown')),
            'layout_grid': self._create_layout_grid_from_terrain(agent_state.get('terrain', [])),
            'p0_pos': agent_state.get('agent_pos'),
            'p0_orient': agent_state.get('agent_orient', 'unknown'),
            'p0_holding': agent_state.get('ego_item', agent_state.get('p0_holding', 'none')),
            'p1_pos': agent_state.get('mate_pos'),
            'p1_orient': agent_state.get('mate_orient', 'unknown'),
            'p1_holding': agent_state.get('mate_item', agent_state.get('p1_holding', 'none')),
            'onion_disp_positions': agent_state.get('onion_disp_positions', 'unknown'),
            'dish_disp_position': agent_state.get('dish_disp_position', 'unknown'),
            'pot_summaries': agent_state.get('pot_summaries', 'unknown'),
            'serve_position': agent_state.get('serve_position', 'unknown'),
            'blocked_tiles_or_none': agent_state.get('blocked_tiles', agent_state.get('blocked_tiles_or_none', 'none')),
            'items_list_or_none': agent_state.get('items_list', agent_state.get('items_list_or_none', 'none')),
            'ego_macro_or_none': agent_state.get('current_macro', agent_state.get('ego_macro_or_none', 'none')),
            'brief_history_or_none': agent_state.get('brief_history', agent_state.get('brief_history_or_none', 'none')),
            'history_10_steps': self._format_history_10_steps(),
            'structured_state': self._create_structured_state_from_agent_state(agent_state),
            'human_free_text': command,
        })

        try:
            return self._user_prompt_template.format(**agent_state_with_template_fields)
        except KeyError as e:
            missing = str(e)
            raise RuntimeError(f"Missing required placeholder in agent_state for user_prompt: {missing}")

    def process_intervention(self, command: str, agent_state: Dict[str, Any]) -> Optional[InterventionResult]:
        """
        Process a human intervention command.
        
        Args:
            command: Human intervention command
            agent_state: Current agent state information
            
        Returns:
            InterventionResult with response, or None if processing failed
        """
        if not self.client:
            print("❌ OpenAI client not available - intervention failed")
            return None
        if not self._system_prompt or not self._user_prompt_template:
            print("❌ Prompt templates not loaded - intervention failed")
            return None
        
        try:
            # Build prompts (strictly from templates)
            system_content = self._system_prompt
            user_content = self._render_user_prompt_from_template(command, agent_state)

            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                max_tokens=400
            )

            print(system_content)
            print(user_content)
            
            # Parse response
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "```json" in result_text:
                json_start = result_text.find("```json") + 7
                json_end = result_text.find("```", json_start)
                result_text = result_text[json_start:json_end].strip()
            elif "{" in result_text and "}" in result_text:
                json_start = result_text.find("{")
                json_end = result_text.rfind("}") + 1
                result_text = result_text[json_start:json_end]
            
            result_data = json.loads(result_text)
            
            print(f"🤖 LLM Reasoning: {result_data.get('reasoning', 'none')}")
            
            # Convert to InterventionResult
            return self._parse_intervention_result(result_data)
            
        except Exception as e:
            print(f"❌ LLM intervention failed: {e}")
            return None
    
    def _parse_intervention_result(self, result_data: Dict[str, Any]) -> InterventionResult:
        """Parse LLM response into InterventionResult object."""
        return InterventionResult(
            action_override=result_data.get("action_override"),
            macro_override=result_data.get("macro_override"),
            duration=result_data.get("duration"),
            reasoning=result_data.get("reasoning", "")
        )

def main():
    """Test the intervention system."""
    
    # Create intervention system
    intervention_system = LLMInterventionSystem()
    
    # Test commands
    test_commands = [
        "Move west",
        "Stay",
        "Move east", 
        "Pick up the onion",
        "Interact",
        "Serve the soup",
        "Get a dish"
    ]
    
    # Mock agent state for user_prompt.txt (random3 layout)
    agent_state = {
        'layout': 'random3',
        'layout_name': 'random3',
        'terrain': [
            ['X', 'X', 'X', 'P', 'P', 'X', 'X', 'X'],
            ['X', ' ', ' ', '2', ' ', ' ', ' ', 'X'],
            ['D', ' ', 'X', 'X', 'X', 'X', ' ', 'S'],
            ['X', ' ', ' ', '1', ' ', ' ', ' ', 'X'],
            ['X', 'X', 'X', 'O', 'O', 'X', 'X', 'X']
        ],
        't': 123,
        'agent_pos': (3, 3),  # Player0 start position
        'agent_orient': 'EAST',
        'ego_item': 'none',
        'mate_pos': (3, 1),  # Player1 start position
        'mate_orient': 'WEST',
        'mate_item': 'onion',
        'onion_disp_positions': [(3,4), (4,4)],
        'dish_disp_position': (0,2),
        'pot_summaries': 'left_pot: onions=3, status=ready; right_pot: onions=1, status=cooking(τ=6)',
        'serve_position': (7,2),
        'blocked_tiles': None,
        'items_list': None,
        'current_macro': 'SERVE_SOUP',
        'brief_history': 'none',
    }
    
    print("Testing LLM Intervention System (template-based):")
    print("=" * 50)
    
    for command in test_commands:
        print(f"\nCommand: '{command}'")
        
        result = intervention_system.process_intervention(command, agent_state)
        
        if result:
            print(f"  Action Override: {result.action_override}")
            print(f"  Macro Override: {result.macro_override}")
            print(f"  Reasoning: {result.reasoning}")
        else:
            print("  ❌ Processing failed")
    
    print("\nIntervention testing complete!")

if __name__ == "__main__":
    main()
