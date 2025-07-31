"""
Experimental scenarios for human intervention in Overcooked.
Matrix: 3x3 grid of intervention types vs. information types
Layout: "random3" (counter circuit) - an 5x8 layout with counters around the perimeter
"""

EXPERIMENT_SCENARIOS = {
    # Row 1: Direct Command
    "direct_pickup": {
        "name": "Direct Command - Pickup",
        "command": "Go pick up the onion.",
        "intervention_type": "Direct Command",
        "information_type": "Agent Performance Correction",
        "trigger_step": 30,
        "expected_behavior": "Agent should move toward and pick up the nearest onion from counters",
        "setup": "Onions spawn on perimeter counters in counter circuit layout",
        "confederate_behavior": "normal",
        "layout_specific": "Onions typically spawn at (4,3) and (4,4) in counter circuit"
    },
    
    "direct_serve": {
        "name": "Direct Command - Serve",
        "command": "Go serve that soup.",
        "intervention_type": "Direct Command", 
        "information_type": "Environmental State Update",
        "trigger_step": 60,
        "expected_behavior": "Agent should pick up cooked soup from pot and deliver to serving area",
        "setup": "Soup ready in one of the two pots (typically at (0,3) and (0,4))",
        "confederate_behavior": "normal",
        "layout_specific": "Pots at (0,3) and (0,4), serving area at (2,7) in counter circuit"
    },
    
    "direct_cover": {
        "name": "Direct Command - Cover Role",
        "command": "Cover your teammate's role.",
        "intervention_type": "Direct Command",
        "information_type": "Teammate Model Update", 
        "trigger_step": 40,
        "expected_behavior": "Agent should take over teammate's current task (cooking or serving)",
        "setup": "Confederate switches to StayAgent at step 35, agent should notice and adapt",
        "confederate_behavior": "switch_to_stay",
        "layout_specific": "In counter circuit, roles are typically split between cooking at (0,3)/(0,4) and serving at (2,7)"
    },
    
    # Row 2: Factual Information
    "factual_pot_ready": {
        "name": "Factual Information - Pot Ready",
        "command": "The pot is ready!",
        "intervention_type": "Factual Information",
        "information_type": "Agent Performance Correction",
        "trigger_step": 50,
        "expected_behavior": "Agent should check pots at (0,3) and (0,4) and pick up cooked soup",
        "setup": "Ensure a pot has finished cooking (20 steps cooking time)",
        "confederate_behavior": "normal",
        "layout_specific": "Pots at (0,3) and (0,4) in counter circuit layout"
    },
    
    "factual_fire": {
        "name": "Factual Information - Fire",
        "command": "There is a fire on tile (1,5).",
        "intervention_type": "Factual Information",
        "information_type": "Environmental State Update",
        "trigger_step": 25,
        "expected_behavior": "Agent should avoid the specified tile and find alternative path",
        "setup": "Mark tile (1,5) as blocked - this blocks path between pots and serving area",
        "confederate_behavior": "normal",
        "layout_specific": "Tile (1,5) is between pots at (0,3)/(0,4) and serving area at (2,7) in counter circuit"
    },
    
    "factual_teammate_limitation": {
        "name": "Factual Information - Teammate Limitation",
        "command": "Your teammate can't pick up onions.",
        "intervention_type": "Factual Information",
        "information_type": "Teammate Model Update",
        "trigger_step": 35,
        "expected_behavior": "Agent should prioritize onion pickup tasks from perimeter counters",
        "setup": "Confederate agent restricted from onion pickup, onions available on counters",
        "confederate_behavior": "no_onion_pickup",
        "layout_specific": "Onions spawn at (4,3) and (4,4) in counter circuit"
    },
    
    # Row 3: Strategic Guidance
    "strategic_deliver_first": {
        "name": "Strategic Guidance - Deliver First",
        "command": "Focus on delivering the soup first.",
        "intervention_type": "Strategic Guidance",
        "information_type": "Agent Performance Correction",
        "trigger_step": 45,
        "expected_behavior": "Agent should prioritize soup delivery over cooking new soup",
        "setup": "Multiple tasks available - cooked soup in pot and ingredients on counters",
        "confederate_behavior": "normal",
        "layout_specific": "Serving area at (2,7), pots at (0,3) and (0,4) in counter circuit"
    },
    
    "strategic_alternate_pots": {
        "name": "Strategic Guidance - Alternate Pots",
        "command": "Alternate between pots to keep both cooking.",
        "intervention_type": "Strategic Guidance",
        "information_type": "Environmental State Update",
        "trigger_step": 55,
        "expected_behavior": "Agent should distribute ingredients between pots at (0,3) and (0,4)",
        "setup": "Multiple pots available, ingredients on perimeter counters",
        "confederate_behavior": "normal",
        "layout_specific": "Two pots at (0,3) and (0,4), ingredients at (4,3) and (4,4)"
    },
    
    "strategic_trade_roles": {
        "name": "Strategic Guidance - Trade Roles",
        "command": "Trade roles: you cook, they serve.",
        "intervention_type": "Strategic Guidance",
        "information_type": "Teammate Model Update",
        "trigger_step": 40,
        "expected_behavior": "Agent should switch from serving to cooking tasks at pots",
        "setup": "Agent currently serving, confederate cooking - switch responsibilities",
        "confederate_behavior": "switch_to_cooking",
        "layout_specific": "Cooking at pots (0,3) and (0,4), serving at (2,7) in counter circuit"
    }
}

# Helper function to get scenarios by type
def get_scenarios_by_type(intervention_type=None, information_type=None):
    """Filter scenarios by intervention type and/or information type."""
    filtered = {}
    for key, scenario in EXPERIMENT_SCENARIOS.items():
        if intervention_type and scenario["intervention_type"] != intervention_type:
            continue
        if information_type and scenario["information_type"] != information_type:
            continue
        filtered[key] = scenario
    return filtered

# Helper function to get scenario by name
def get_scenario(scenario_name):
    """Get a specific scenario by its key name."""
    return EXPERIMENT_SCENARIOS.get(scenario_name)

# Helper function to list all scenario names
def list_scenario_names():
    """Return list of all scenario names."""
    return list(EXPERIMENT_SCENARIOS.keys())

# Layout information for counter circuit (random3)
COUNTER_CIRCUIT_LAYOUT = {
    "name": "counter circuit (random3)",
    "size": "5x8",
    "pots": [(0, 3), (0, 4)],
    "serving_area": (2, 7),  # S position
    "agent_spawns": [(3, 3), (3, 1)],
    "dish_spawn": (2, 0),    # D position
    "ingredient_spawns": [(4, 3), (4, 4)],  # O positions
    "description": "5x8 layout with two pots at top, serving area at S, dishes at D"
} 