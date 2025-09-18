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

@dataclass
class InterventionResult:
    """Result of processing an intervention command."""
    action_override: Optional[int] = None  # 0=STAY, 1=EAST, 2=SOUTH, 3=NORTH, 4=WEST, 5=INTERACT
    macro_override: Optional[str] = None  # New macro like "GET_ONION", "SERVE_SOUP", "TAKE_SOUP", "GET_DISH", "PUT_ONION_IN_POT", "WAIT_FOR_COOKING", "DROP_OFF"
    duration: Optional[int] = None  # How many steps to persist the override (default: 1 for actions, until completion for macros)
    reasoning: str = ""

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
            # Use hardcoded API key as fallback for development
            self.client = OpenAI(api_key="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")
            print("🤖 OpenAI client initialized with hardcoded API key (development mode)")
        
        # Interaction history tracking
        self.interaction_history: List[Dict[str, Any]] = []
        self.max_history_length = 20  # Keep last 20 interactions
    
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
    
    def generate_layout_map(self, terrain: List[List[str]], ego_pos: tuple, mate_pos: tuple, 
                          objects: Dict[tuple, str]) -> str:
        """Generate ASCII layout map for the prompt."""
        if not terrain:
            return "Layout map not available"
        
        height, width = len(terrain), len(terrain[0])
        map_lines = []
        
        # Create header
        map_lines.append("LAYOUT MAP:")
        map_lines.append("Legend: E=ego, M=mate, O=onion, D=dish, P=pot, S=serve, X=wall, .=floor")
        map_lines.append("")
        
        # Create map
        for y in range(height):
            line = ""
            for x in range(width):
                pos = (x, y)
                
                # Check for agents first
                if pos == ego_pos:
                    line += "E"
                elif pos == mate_pos:
                    line += "M"
                # Check for objects
                elif pos in objects:
                    obj = objects[pos]
                    if obj == "onion":
                        line += "O"
                    elif obj == "dish":
                        line += "D"
                    elif obj == "pot":
                        line += "P"
                    elif obj == "serving_counter":
                        line += "S"
                    else:
                        line += "?"
                # Check terrain
                elif terrain[y][x] == "X":
                    line += "X"
                else:
                    line += "."
            
            map_lines.append(f"{y:2d}|{line}")
        
        # Add coordinate labels
        coord_line = "   "
        for x in range(width):
            coord_line += str(x % 10)
        map_lines.append(coord_line)
        
        return "\n".join(map_lines)
    
    def format_interaction_history(self) -> str:
        """Format interaction history for the prompt."""
        if not self.interaction_history:
            return "No previous interactions."
        
        history_lines = ["INTERACTION HISTORY:"]
        history_lines.append("Step | Ego Action | Mate Action | Ego Pos | Mate Pos | Ego Item | Mate Item | Intervention")
        history_lines.append("-" * 100)
        
        for entry in self.interaction_history[-10:]:  # Show last 10 interactions
            ego_action_name = self._action_to_name(entry['ego_action'])
            mate_action_name = self._action_to_name(entry['mate_action'])
            intervention = entry.get('intervention', '')
            
            line = f"{entry['step']:4d} | {ego_action_name:10s} | {mate_action_name:10s} | {str(entry['ego_pos']):8s} | {str(entry['mate_pos']):8s} | {entry['ego_item']:8s} | {entry['mate_item']:8s} | {intervention}"
            history_lines.append(line)
        
        return "\n".join(history_lines)
    
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
    
    def _analyze_recent_behavior(self, horizon: int = 8) -> str:
        """Produce a compact analysis of recent ego behavior to help interpret vague feedback.

        The analysis summarizes repeated actions and oscillations so the LLM can suggest
        de-stuck actions without any client-side heuristics deciding the action.
        """
        if not self.interaction_history:
            return "No recent behavior to analyze."

        recent = self.interaction_history[-horizon:]
        ego_actions = [entry.get('ego_action') for entry in recent]
        # Count trailing repetition
        trailing_action = ego_actions[-1]
        trailing_count = 1
        for idx in range(len(ego_actions) - 2, -1, -1):
            if ego_actions[idx] == trailing_action:
                trailing_count += 1
            else:
                break

        # Detect simple two-action oscillation like EAST/WEST or NORTH/SOUTH
        oscillation = False
        if len(ego_actions) >= 4:
            a, b = ego_actions[-1], ego_actions[-2]
            if a is not None and b is not None and a != b:
                last_four = ego_actions[-4:]
                oscillation = last_four == [a, b, a, b] or last_four == [b, a, b, a]

        action_names = [self._action_to_name(a) for a in ego_actions]
        trailing_name = self._action_to_name(trailing_action)

        lines = [
            "RECENT BEHAVIOR ANALYSIS:",
            f"- Last {len(ego_actions)} ego actions: {action_names}",
            f"- Trailing repetition: {trailing_name} x{trailing_count}",
            f"- Oscillation pattern detected: {oscillation}",
            "- Guidance: If the human says 'stuck' or similar, propose an action or macro that breaks repetition/oscillation given the layout and targets."
        ]
        return "\n".join(lines)
    
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
        
        try:
            # Lightweight normalization for hazard-related cues
            normalized_command = command
            lower_cmd = command.lower()
            hazard_hint = ""
            fire_pos = agent_state.get('fire_pos')
            fire_active = agent_state.get('fire_active', False)
            if fire_active:
                if any(k in lower_cmd for k in ["avoid fire", "fire at", "watch the fire", "there is a fire", "be safe", "stay safe", "danger", "hazard"]):
                    hazard_hint = f"Human indicates a hazard. Never step onto FIRE at {fire_pos}. Prefer safe detours or STAY if necessary."
                # If vague safety cue, append clarification to the command for the LLM
                if any(k in lower_cmd for k in ["be safe", "careful", "watch out", "avoid"]) and "fire" not in lower_cmd:
                    normalized_command = command + f" (avoid stepping onto hazard at {fire_pos})"

            # Create enhanced prompt with history and layout map
            prompt = self._create_enhanced_intervention_prompt(
                state_text=agent_state.get('state_text', ''),
                command=normalized_command,
                current_macro=agent_state.get('current_macro', 'UNKNOWN'),
                agent_pos=agent_state.get('agent_pos', (0, 0)),
                target_pos=agent_state.get('target_pos', None),
                layout=agent_state.get('layout', None),
                terrain=agent_state.get('terrain', []),
                objects=agent_state.get('objects', {}),
                mate_pos=agent_state.get('mate_pos', (0, 0)),
                fire_pos=fire_pos,
                fire_active=fire_active,
                hazard_hint=hazard_hint,
            )
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at interpreting both direct and vague human interventions for a grid-world cooking game. If the human feedback is vague (e.g., 'you're stuck', 'stop oscillating', 'try something else'), infer a single direct, executable override that helps the agent get unstuck based on the provided history, layout map, and current goal. Never output text outside the required JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
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
    
    def _create_enhanced_intervention_prompt(self, state_text: str, command: str, current_macro: str, 
                                           agent_pos: tuple, target_pos: tuple = None, layout: str = None,
                                           terrain: List[List[str]] = None, objects: Dict[tuple, str] = None,
                                           mate_pos: tuple = None, fire_pos: tuple = None, fire_active: bool = False,
                                           hazard_hint: str = "") -> str:
        """Create enhanced prompt with history and layout map."""
        
        # Generate layout map
        layout_map = ""
        if terrain:
            layout_map = self.generate_layout_map(terrain, agent_pos, mate_pos or (0, 0), objects or {})
        
        # Get interaction history
        history = self.format_interaction_history()
        behavior_analysis = self._analyze_recent_behavior()
        
        # Hazard context
        hazard_lines = []
        if fire_active and fire_pos is not None:
            hazard_lines.append(f"HAZARD: FIRE is active at tile {fire_pos}. Stepping onto it ends the episode immediately.")
        if hazard_hint:
            hazard_lines.append(f"HUMAN HAZARD HINT: {hazard_hint}")
        hazard_text = "\n".join(hazard_lines)

        # Create simplified prompt
        return f"""
You are helping an Overcooked agent. The human may give either direct commands or vague diagnostic feedback (e.g., "you're stuck", "stop oscillating", "try something else"). Convert any input into a single direct override.

{layout_map}

{history}

{behavior_analysis}

{hazard_text}

CURRENT SITUATION:
- Agent position: {agent_pos}
- Target position: {target_pos if target_pos else "unknown"}
- Current macro: {current_macro}
- State: {state_text}
- Layout: {layout if layout else "unknown"}
- Human command: "{command}"

Respond with a JSON object:

{{
    "action_override": null or action_number (0=STAY, 1=EAST, 2=SOUTH, 3=NORTH, 4=WEST, 5=INTERACT),
    "macro_override": null or new_macro_name (e.g., "GET_ONION", "SERVE_SOUP", "TAKE_SOUP", "GET_DISH", "PUT_ONION_IN_POT", "WAIT_FOR_COOKING", "DROP_OFF"),
    "duration": null or number_of_steps (how long to persist the override),
    "reasoning": "brief explanation of your response"
}}

Guidelines:
- For explicit movement: Use action_override (e.g., "move west" → 4)
- For explicit tasks: Use macro_override (e.g., "serve soup" → "SERVE_SOUP")
- Only one override per response (either action_override OR macro_override)
- Set duration:
  * Actions: duration=1 unless the human specified multiple steps
  * Macros: duration=null (persist until task completion)
- If the human feedback is vague (diagnostic), infer a de-stuck action or macro that breaks repetition/oscillation and progresses toward the goal. Prefer:
  1) INTERACT (5) if the agent is adjacent to a relevant object it seems to be facing
  2) A movement different from the last repeated action
  3) A small sidestep around obstacles rather than continuing the same direction
  4) Switching to a sensible macro if the current macro is making no progress
- Avoid repeating the same low-level action more than twice when handling vague commands
- If a HAZARD is present: NEVER choose an action that would move the agent onto the hazard tile. If the human says "avoid fire" or "be safe", prefer a sidestep or STAY (0) if no safe immediate move exists. When the agent is adjacent and moving toward the hazard, select a perpendicular move away from the hazard tile.

Examples:
- "Stay" → action_override=0, duration=1
- "Move west" → action_override=4, duration=1 (single step movement)
- "Move west one step" → action_override=4, duration=1 (single step movement)
- "Move west for 3 steps" → action_override=4, duration=3 (multi-step movement)
- "Pick up the onion" → macro_override="GET_ONION", duration=null
- "Drop off what you're holding" → macro_override="DROP_OFF", duration=null
- "You're stuck" → choose an action_override that is different from the recent repeated action (or a relevant INTERACT), duration=1
- "Avoid fire" → choose an action_override that keeps the agent from stepping onto the hazardous tile, e.g., sidestep away; duration=1
- "There is a fire on tile (3,2)" → do not move into (3,2)
- "Be safe" → infer a safe immediate action that avoids hazards and breaks repetition
"""
    
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
        "Move west",  # Will persist for 2 steps (face west + move west)
        "Stay",
        "Move east", 
        "Pick up the onion",
        "Interact",
        "Serve the soup",
        "Get a dish"
    ]
    
    # Mock agent state with enhanced context
    agent_state = {
        'state_text': 'Agent at (5,1) trying to reach serving counter at (7,2)',
        'current_macro': 'SERVE_SOUP',
        'agent_pos': (5, 1),
        'target_pos': (7, 2),
        'mate_pos': (3, 2),
        'layout': 'random3',
        'terrain': [
            ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
            ['X', '.', '.', '.', '.', '.', '.', 'X'],
            ['X', '.', 'P', 'P', '.', '.', 'S', 'X'],
            ['X', '.', '.', '.', '.', '.', '.', 'X'],
            ['X', 'O', 'O', '.', '.', '.', '.', 'X'],
            ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']
        ],
        'objects': {
            (3, 0): 'pot',
            (4, 0): 'pot',
            (7, 2): 'serving_counter',
            (1, 4): 'onion',
            (2, 4): 'onion'
        }
    }
    
    print("Testing Enhanced LLM Intervention System:")
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
    
    print("\nEnhanced intervention testing complete!")

if __name__ == "__main__":
    main()
