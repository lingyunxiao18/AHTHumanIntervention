#!/usr/bin/env python3
"""
Structured state description for Overcooked environments.
Creates JSON-like structured descriptions as requested.
"""

import sys
import os
sys.path.append('.')

import json
from typing import Dict, List, Any, Optional, Tuple
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from shared.utils.state_to_text import _ori_str, _pots, _onion_disp, _dish_disp, _serving, _loose_items, _soups_not_in_pots, _pots_with_state

def create_structured_state_description(mdp: OvercookedGridworld, state: OvercookedState, 
                                       command: str = "", grounding: Dict = None,
                                       macro_label: str = "") -> Dict[str, Any]:
    """
    Create structured state description in the requested format.
    
    Format:
    {
      "state_text": "[SCENE] size 8x5 [EGO] pos (2,3) facing N item onion ...",
      "command": "Bring the onion to the left pot.",
      "grounding": { "target_pot": { "rule": "leftmost", "coord": [3,0] } },
      "macro_label": "GO_TO_POT"
    }
    """
    
    # Get basic layout info
    terrain_mtx = getattr(mdp, 'terrain_mtx', [])
    if terrain_mtx:
        height, width = len(terrain_mtx), len(terrain_mtx[0])
    else:
        height, width = 5, 8  # Default for random3
    
    # Get player info
    ego = state.players[0] if len(state.players) > 0 else None
    mate = state.players[1] if len(state.players) > 1 else None
    
    ego_pos = ego.position if ego else (0, 0)
    ego_facing = _ori_str(ego.orientation) if ego else "N"
    ego_item = ego.get_object().name if ego and ego.has_object() else "none"
    
    mate_pos = mate.position if mate else (0, 0)
    mate_facing = _ori_str(mate.orientation) if mate else "N"
    mate_item = mate.get_object().name if mate and mate.has_object() else "none"
    
    # Get object locations
    onion_locations = _onion_disp(mdp)
    dish_locations = _dish_disp(mdp)
    serve_locations = _serving(mdp)
    pot_locations = _pots(mdp)
    
    # Get pot states
    pots_with_state_info = _pots_with_state(mdp, state)
    
    # Build state text components
    scene_part = f"[SCENE] size {width}x{height}"
    ego_part = f"[EGO] pos {ego_pos} facing {ego_facing} item {ego_item}"
    mate_part = f"[MATE] pos {mate_pos} facing {mate_facing} item {mate_item}"
    
    # Determine goal based on current state (simple heuristic)
    if ego_item == "onion":
        goal = "go_to_pot"
    elif ego_item == "dish":
        goal = "take_soup"
    elif ego_item == "soup":
        goal = "serve"
    else:
        goal = "get_onion"
    
    goal_part = f"[GOAL] {goal} | [OPT_CMD]"
    
    # Format object locations
    onion_part = "[ONION] " + "".join([f"({x},{y})" for x, y in onion_locations]) if onion_locations else "[ONION] none"
    dish_part = "[DISH] " + "".join([f"({x},{y})" for x, y in dish_locations]) if dish_locations else "[DISH] none"
    serve_part = "[SERVE] " + "".join([f"({x},{y})" for x, y in serve_locations]) if serve_locations else "[SERVE] none"
    
    # Format pots with states
    pot_parts = []
    for pot_info in pots_with_state_info:
        pos = pot_info["pos"]
        onions = pot_info["onions"]
        cook_time = pot_info["cook"]
        
        if cook_time == -1:  # Empty pot
            state_desc = "empty"
        elif cook_time == 0 and onions >= 3:  # Ready soup
            state_desc = "ready"
        elif onions > 0:  # Cooking
            state_desc = f"{onions}/3"
        else:
            state_desc = "empty"
        
        pot_parts.append(f"({pos[0]},{pos[1]}) {state_desc}")
    
    pots_part = "[POTS] " + " ".join(pot_parts) if pot_parts else "[POTS] none"
    
    # Combine all parts
    state_text = " ".join([
        scene_part,
        ego_part, 
        mate_part,
        goal_part,
        onion_part,
        dish_part,
        serve_part,
        pots_part
    ])
    
    # Create structured output
    structured_description = {
        "state_text": state_text,
        "command": command,
        "grounding": grounding or {},
        "macro_label": macro_label
    }
    
    return structured_description

def test_structured_description():
    """Test the structured description with random3 layout."""
    print("🧪 Testing Structured State Description...")
    
    try:
        # Create MDP for random3 layout
        mdp = OvercookedGridworld.from_layout_name("random3")
        print(f"✅ Created MDP for random3 layout")
        
        # Create environment and get initial state
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        env = OvercookedEnv(mdp, horizon=100)
        obs = env.reset()
        state = env.state
        
        print(f"✅ Created environment and got initial state")
        
        # Create structured description
        structured_desc = create_structured_state_description(
            mdp, state,
            command="Bring the onion to the left pot.",
            grounding={"target_pot": {"rule": "leftmost", "coord": [3, 0]}},
            macro_label="GO_TO_POT"
        )
        
        print(f"✅ Created structured description")
        print("\n📋 Structured State Description:")
        print(json.dumps(structured_desc, indent=2))
        
        return structured_desc
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_hand_coded_policy():
    """Test hand-coded policy with structured description."""
    print("\n🤖 Testing Hand-Coded Policy...")
    
    try:
        from pipelines.llm_baseline.core.llm_translator_baseline import (
            create_hms_only_agent, parse_minimal_state, handcoded_next_macro
        )
        
        # Create MDP and environment
        mdp = OvercookedGridworld.from_layout_name("random3")
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        env = OvercookedEnv(mdp, horizon=50)
        obs = env.reset()
        
        print(f"✅ Environment setup complete")
        
        # Create HMS-only agent
        hms_agent = create_hms_only_agent(mdp, player_id=0)
        print(f"✅ HMS agent created")
        
        # Run a few steps to test
        for step in range(10):
            # Get structured description
            structured_desc = create_structured_state_description(mdp, env.state)
            
            print(f"\n--- Step {step + 1} ---")
            print(f"State: {structured_desc['state_text']}")
            
            # Test HMS decision using the structured state text
            ps = parse_minimal_state(structured_desc['state_text'])
            hms_decision = handcoded_next_macro(ps)
            print(f"HMS Decision: {hms_decision}")
            
            # Get action from agent
            action_idx = hms_agent.act_low_level(env.state)
            print(f"Action: {action_idx}")
            
            # Step environment
            from pipelines.llm_baseline.demo.play_llm_translator_baseline import convert_action
            ego_action = convert_action(action_idx)
            teammate_action = convert_action(0)  # Teammate stays
            
            obs, reward, done, info = env.step((ego_action, teammate_action))
            print(f"Reward: {reward}, Done: {done}")
            
            if done:
                print("Episode finished!")
                break
        
        print("✅ Hand-coded policy test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing hand-coded policy: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Structured State Description Test")
    print("=" * 50)
    
    # Test structured description format
    structured_desc = test_structured_description()
    
    if structured_desc:
        # Test hand-coded policy
        success = test_hand_coded_policy()
        
        if success:
            print("\n🎉 All tests passed!")
            print("✅ Structured state description working")
            print("✅ Hand-coded policy working with new format")
        else:
            print("\n⚠️ Hand-coded policy test failed")
    else:
        print("\n❌ Structured description test failed")
