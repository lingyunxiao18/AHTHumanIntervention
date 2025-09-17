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
    macro_override: Optional[str] = None  # New macro like "SIMPLE", "SERVE_SOUP", "TAKE_SOUP", etc.
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
            self.client = OpenAI(api_key="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")
            print("🤖 OpenAI client initialized with API key")
        elif os.getenv('OPENAI_API_KEY'):
            self.client = OpenAI()
        else:
            self.client = None
            print("❌ No OpenAI API key found. Intervention will fail.")
        
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
            # Create enhanced prompt with history and layout map
            prompt = self._create_enhanced_intervention_prompt(
                state_text=agent_state.get('state_text', ''),
                command=command,
                current_macro=agent_state.get('current_macro', 'UNKNOWN'),
                agent_pos=agent_state.get('agent_pos', (0, 0)),
                target_pos=agent_state.get('target_pos', None),
                layout=agent_state.get('layout', None),
                terrain=agent_state.get('terrain', []),
                objects=agent_state.get('objects', {}),
                mate_pos=agent_state.get('mate_pos', (0, 0))
            )
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing human intervention commands for game agents. Provide direct, actionable responses without categorizing the command type."},
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
                                           mate_pos: tuple = None) -> str:
        """Create enhanced prompt with history and layout map."""
        
        # Generate layout map
        layout_map = ""
        if terrain:
            layout_map = self.generate_layout_map(terrain, agent_pos, mate_pos or (0, 0), objects or {})
        
        # Get interaction history
        history = self.format_interaction_history()
        
        # Create simplified prompt
        return f"""
You are helping an Overcooked agent. The human is giving direct commands that should override the agent's current behavior.

{layout_map}

{history}

CURRENT SITUATION:
- Agent position: {agent_pos}
- Target position: {target_pos if target_pos else "unknown"}
- Current macro: {current_macro}
- State: {state_text}
- Layout: {layout if layout else "unknown"}
- Human command: "{command}"

The human is giving DIRECT, EXECUTABLE commands. Respond with a JSON object:

{{
    "action_override": null or action_number (0=STAY, 1=EAST, 2=SOUTH, 3=NORTH, 4=WEST, 5=INTERACT),
    "macro_override": null or new_macro_name (e.g., "SIMPLE", "SERVE_SOUP", "TAKE_SOUP", "GET_DISH", "GO_TO_POT"),
    "duration": null or number_of_steps (how long to persist the override),
    "reasoning": "brief explanation of your response"
}}

Guidelines:
- For movement commands: Use action_override (e.g., "move west" → action_override=4, "stay" → action_override=0)
- For macro/task commands: Use macro_override (e.g., "pick up onion" → macro_override="SIMPLE", "serve soup" → macro_override="SERVE_SOUP")
- Only provide one override per response (either action_override OR macro_override, not both)
- Use duration to specify how many steps the override should persist:
  * For simple actions: duration=1 (STAY, INTERACT)
  * For single movement: duration=1 (e.g., "move west" → duration=1, "move west one step" → duration=1)
  * For multi-step movements: duration=N (e.g., "move west for 3 steps" → duration=3)
  * For macros: duration=null (persist until task completion)
  * Overcooked handles facing automatically, so single movements only need 1 step
- Consider the interaction history to understand what the agent has been doing

Examples:
- "Stay" → action_override=0, duration=1
- "Move west" → action_override=4, duration=1 (single step movement)
- "Move west one step" → action_override=4, duration=1 (single step movement)
- "Move west for 3 steps" → action_override=4, duration=3 (multi-step movement)
- "Move east" → action_override=1, duration=1 (single step movement)
- "Move north" → action_override=3, duration=1 (single step movement)
- "Move south" → action_override=2, duration=1 (single step movement)
- "Interact" → action_override=5, duration=1
- "Pick up the onion" → macro_override="SIMPLE", duration=null
- "Serve the soup" → macro_override="SERVE_SOUP", duration=null
- "Get a dish" → macro_override="GET_DISH", duration=null
- "Take soup from pot" → macro_override="TAKE_SOUP", duration=null
- "Go to pot" → macro_override="GO_TO_POT", duration=null
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
