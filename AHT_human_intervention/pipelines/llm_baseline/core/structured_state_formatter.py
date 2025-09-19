#!/usr/bin/env python3
"""
Structured state formatter for LLM baseline pipeline.
Creates the JSON-like structured descriptions as requested.
Layout-agnostic version that parses actual layout files.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import json
from typing import Dict, List, Any, Optional, Tuple

def parse_layout_for_objects(layout_name: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse layout file to extract object locations.
    
    Args:
        layout_name: Name of the layout (e.g., 'random3', 'simple')
        
    Returns:
        Dictionary with object type -> list of positions
    """
    try:
        from shared.envs.envs.overcooked.overcooked_ai_py.data.layouts import read_layout_dict
        
        layout_dict = read_layout_dict(layout_name)
        grid = layout_dict["grid"]
        
        # Parse grid into 2D array
        grid_lines = [line.strip() for line in grid.split("\n") if line.strip()]
        height = len(grid_lines)
        width = len(grid_lines[0]) if grid_lines else 0
        
        # Find object locations
        onion_locs = []
        dish_locs = []
        serve_locs = []
        pot_locs = []
        
        for y, line in enumerate(grid_lines):
            for x, char in enumerate(line):
                if char == 'O':  # Onion dispenser
                    onion_locs.append((x, y))
                elif char == 'D':  # Dish dispenser
                    dish_locs.append((x, y))
                elif char == 'S':  # Serving station
                    serve_locs.append((x, y))
                elif char == 'P':  # Pot
                    pot_locs.append((x, y))
        
        return {
            'onion_locations': onion_locs,
            'dish_locations': dish_locs,
            'serve_locations': serve_locs,
            'pot_locations': pot_locs,
            'scene_size': (width, height)
        }
        
    except Exception as e:
        print(f"Warning: Could not parse layout '{layout_name}': {e}")
        # Return empty locations as fallback
        return {
            'onion_locations': [],
            'dish_locations': [],
            'serve_locations': [],
            'pot_locations': [],
            'scene_size': (8, 5)
        }

def create_structured_state_description(state_info: Dict[str, Any], 
                                       command: str = "", 
                                       grounding: Dict = None,
                                       macro_label: str = "") -> Dict[str, Any]:
    """
    Create structured state description from parsed state information.
    Layout-agnostic version that dynamically parses layout files.
    
    Args:
        state_info: Dictionary with parsed state information
        command: Human command text
        grounding: Grounding information for command
        macro_label: Expected macro action label
    
    Returns:
        Structured description in requested format
    """
    
    # Extract state information
    scene_size = state_info.get('scene_size', (8, 5))
    ego_pos = state_info.get('ego_pos', 'unknown')
    ego_facing = state_info.get('ego_facing', 'N')
    ego_item = state_info.get('ego_item', 'none')
    
    mate_pos = state_info.get('mate_pos', 'unknown')
    mate_facing = state_info.get('mate_facing', 'W')
    mate_item = state_info.get('mate_item', 'none')
    
    # Get layout name and parse object locations
    layout_name = state_info.get('layout_name', state_info.get('layout', 'unknown'))
    layout_objects = parse_layout_for_objects(layout_name)
    
    # Use parsed locations or fallback to provided ones
    onion_locs = state_info.get('onion_locations', layout_objects['onion_locations'])
    dish_locs = state_info.get('dish_locations', layout_objects['dish_locations'])
    serve_locs = state_info.get('serve_locations', layout_objects['serve_locations'])
    pot_locs = layout_objects['pot_locations']
    
    # Ensure all locations are lists (handle None values)
    onion_locs = onion_locs if onion_locs is not None else []
    dish_locs = dish_locs if dish_locs is not None else []
    serve_locs = serve_locs if serve_locs is not None else []
    pot_locs = pot_locs if pot_locs is not None else []
    
    # Get pot states from state_info or create default empty states
    pot_states = state_info.get('pot_states', [])
    if pot_states is None:
        pot_states = []
    if not pot_states and pot_locs:
        # Create default empty pot states for all pots in layout
        pot_states = [{'pos': pos, 'state': 'empty'} for pos in pot_locs]
    
    # Determine goal based on current state
    goal = determine_goal(ego_item, state_info)
    
    # Build state text components
    scene_part = f"[SCENE] size {scene_size[0]}x{scene_size[1]}"
    ego_part = f"[EGO] pos {ego_pos} facing {ego_facing} item {ego_item}"
    mate_part = f"[MATE] pos {mate_pos} facing {mate_facing} item {mate_item}"
    goal_part = f"[GOAL] {goal} | [OPT_CMD]"
    
    # Format object locations
    onion_part = "[ONION] " + " ".join([f"({x},{y})" for x, y in onion_locs])
    dish_part = "[DISH] " + " ".join([f"({x},{y})" for x, y in dish_locs])
    serve_part = "[SERVE] " + " ".join([f"({x},{y})" for x, y in serve_locs])
    
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