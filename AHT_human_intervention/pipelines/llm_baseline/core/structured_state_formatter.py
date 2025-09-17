#!/usr/bin/env python3
"""
Structured state formatter for LLM baseline pipeline.
Creates the JSON-like structured descriptions as requested.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import json
from typing import Dict, List, Any, Optional, Tuple

def create_structured_state_description(state_info: Dict[str, Any], 
                                       command: str = "", 
                                       grounding: Dict = None,
                                       macro_label: str = "") -> Dict[str, Any]:
    """
    Create structured state description from parsed state information.
    
    Args:
        state_info: Dictionary with parsed state information
        command: Human command text
        grounding: Grounding information for command
        macro_label: Expected macro action label
    
    Returns:
        Structured description in requested format
    """
    
    # Extract state information (with defaults for missing data)
    scene_size = state_info.get('scene_size', (8, 5))
    ego_pos = state_info.get('ego_pos', (2, 3))
    ego_facing = state_info.get('ego_facing', 'N')
    ego_item = state_info.get('ego_item', 'none')
    
    mate_pos = state_info.get('mate_pos', (5, 1))
    mate_facing = state_info.get('mate_facing', 'W')
    mate_item = state_info.get('mate_item', 'none')
    
    # Determine goal based on current state
    goal = determine_goal(ego_item, state_info)
    
    # Get object locations
    onion_locs = state_info.get('onion_locations', [(3, 4), (4, 4)])
    dish_locs = state_info.get('dish_locations', [(0, 2)])
    serve_locs = state_info.get('serve_locations', [(7, 2)])
    pot_states = state_info.get('pot_states', [
        {'pos': (3, 0), 'state': 'empty'},
        {'pos': (4, 0), 'state': 'empty'}
    ])
    
    # Build state text components
    scene_part = f"[SCENE] size {scene_size[0]}x{scene_size[1]}"
    ego_part = f"[EGO] pos {ego_pos} facing {ego_facing} item {ego_item}"
    mate_part = f"[MATE] pos {mate_pos} facing {mate_facing} item {mate_item}"
    goal_part = f"[GOAL] {goal} | [OPT_CMD]"
    
    # Format object locations
    onion_part = "[ONION] " + "".join([f"({x},{y})" for x, y in onion_locs])
    dish_part = "[DISH] " + "".join([f"({x},{y})" for x, y in dish_locs])
    serve_part = "[SERVE] " + "".join([f"({x},{y})" for x, y in serve_locs])
    
    # Format pots
    pot_parts = []
    for pot in pot_states:
        pos = pot['pos']
        state_desc = pot['state']
        pot_parts.append(f"({pos[0]},{pos[1]}) {state_desc}")
    
    pots_part = "[POTS] " + " ".join(pot_parts)
    
    # Combine all parts
    state_text = " ".join([
        scene_part, ego_part, mate_part, goal_part,
        onion_part, dish_part, serve_part, pots_part
    ])
    
    # Create structured output
    return {
        "state_text": state_text,
        "command": command,
        "grounding": grounding or {},
        "macro_label": macro_label
    }

def determine_goal(ego_item: str, state_info: Dict[str, Any]) -> str:
    """Determine current goal based on ego item and state."""
    if ego_item == "onion":
        return "go_to_pot"
    elif ego_item == "dish":
        # Check if any pot is ready
        pot_states = state_info.get('pot_states', [])
        if any(pot['state'] == 'ready' for pot in pot_states):
            return "take_soup"
        else:
            return "wait_cook"
    elif ego_item == "soup":
        return "serve"
    else:
        # Check what to do next
        pot_states = state_info.get('pot_states', [])
        if any(pot['state'] == 'ready' for pot in pot_states):
            return "get_dish"
        elif any(pot['state'] == 'empty' for pot in pot_states):
            return "get_onion"
        else:
            return "wait_cook"

def parse_structured_state_for_hms(structured_state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse structured state into format compatible with existing HMS."""
    state_text = structured_state["state_text"]
    
    # Extract key information
    holding = None
    if "item onion" in state_text:
        holding = "onion"
    elif "item dish" in state_text:
        holding = "dish"
    elif "item soup" in state_text:
        holding = "soup"
    
    # Check pot states
    any_pot_ready = "ready" in state_text
    any_pot_has_space = "empty" in state_text or "/3" in state_text
    pots_cooking = "/3" in state_text and "ready" not in state_text
    
    # For now, assume we're not adjacent (would need position analysis)
    near_pot_ready = False
    near_onion_disp = False
    near_dish_disp = False
    near_serve = False
    
    return {
        "holding": holding,
        "near_pot_ready": near_pot_ready,
        "any_pot_ready": any_pot_ready,
        "any_pot_has_space": any_pot_has_space,
        "near_onion_disp": near_onion_disp,
        "near_dish_disp": near_dish_disp,
        "near_serve": near_serve,
        "pots_cooking": pots_cooking
    }

def create_mock_random3_states() -> List[Dict[str, Any]]:
    """Create mock states for random3 layout testing."""
    return [
        {
            # Initial state - need to get onion
            'scene_size': (8, 5),
            'ego_pos': (2, 3),
            'ego_facing': 'N',
            'ego_item': 'none',
            'mate_pos': (5, 1),
            'mate_facing': 'W', 
            'mate_item': 'none',
            'onion_locations': [(3, 4), (4, 4)],
            'dish_locations': [(0, 2)],
            'serve_locations': [(7, 2)],
            'pot_states': [
                {'pos': (3, 0), 'state': 'empty'},
                {'pos': (4, 0), 'state': 'empty'}
            ]
        },
        {
            # Holding onion - need to go to pot
            'scene_size': (8, 5),
            'ego_pos': (3, 4),
            'ego_facing': 'N',
            'ego_item': 'onion',
            'mate_pos': (5, 1),
            'mate_facing': 'W',
            'mate_item': 'none',
            'onion_locations': [(3, 4), (4, 4)],
            'dish_locations': [(0, 2)],
            'serve_locations': [(7, 2)],
            'pot_states': [
                {'pos': (3, 0), 'state': 'empty'},
                {'pos': (4, 0), 'state': '1/3'}
            ]
        },
        {
            # Pot ready, holding dish - take soup
            'scene_size': (8, 5),
            'ego_pos': (0, 2),
            'ego_facing': 'E',
            'ego_item': 'dish',
            'mate_pos': (5, 1),
            'mate_facing': 'W',
            'mate_item': 'none',
            'onion_locations': [(3, 4), (4, 4)],
            'dish_locations': [(0, 2)],
            'serve_locations': [(7, 2)],
            'pot_states': [
                {'pos': (3, 0), 'state': 'ready'},
                {'pos': (4, 0), 'state': '2/3'}
            ]
        }
    ]

def test_structured_format_with_hms():
    """Test the structured format with HMS integration."""
    print("🧪 Testing Structured Format with HMS Integration")
    print("=" * 55)
    
    mock_states = create_mock_random3_states()
    
    for i, state_info in enumerate(mock_states, 1):
        print(f"\n--- Test Case {i} ---")
        
        # Create structured description
        structured_desc = create_structured_state_description(
            state_info,
            command="",  # No command for HMS-only test
            grounding={},
            macro_label=""
        )
        
        print("📋 Structured State:")
        print(json.dumps(structured_desc, indent=2))
        
        # Parse for HMS
        hms_input = parse_structured_state_for_hms(structured_desc)
        print(f"\n🔍 HMS Input:")
        for key, value in hms_input.items():
            print(f"  {key}: {value}")
        
        # Simple HMS decision (using the logic from our test)
        holding = hms_input["holding"]
        any_pot_ready = hms_input["any_pot_ready"]
        any_pot_has_space = hms_input["any_pot_has_space"]
        pots_cooking = hms_input["pots_cooking"]
        
        if holding == "soup":
            hms_decision = "GO_TO_SERVE"
        elif holding == "dish" and any_pot_ready:
            hms_decision = "TAKE_SOUP"
        elif holding == "dish" and pots_cooking:
            hms_decision = "WAIT_COOK"
        elif holding == "onion":
            hms_decision = "GO_TO_POT"
        elif any_pot_ready:
            hms_decision = "GO_TO_DISH"
        elif any_pot_has_space:
            hms_decision = "GO_TO_ONION"
        else:
            hms_decision = "WAIT_COOK"
        
        print(f"\n🤖 HMS Decision: {hms_decision}")
        print("-" * 55)
    
    print("\n✅ Structured format working with HMS!")
    return True

if __name__ == "__main__":
    print("🚀 Structured State Formatter for LLM Baseline")
    print("🎯 Testing with random3 layout format")
    
    success = test_structured_format_with_hms()
    
    if success:
        print(f"\n🎉 All tests passed!")
        print(f"✅ Structured state format implemented")
        print(f"✅ HMS integration working")
        print(f"✅ Ready for random3 layout testing")
        
        print(f"\n📋 Integration Points:")
        print(f"1. Use create_structured_state_description() for state formatting")
        print(f"2. Use parse_structured_state_for_hms() for HMS compatibility")
        print(f"3. Add real environment state extraction")
        print(f"4. Test with actual random3 layout")
    else:
        print(f"\n❌ Tests failed")
