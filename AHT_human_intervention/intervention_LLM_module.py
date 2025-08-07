import os
import json
from openai import OpenAI
from typing import Optional, Dict, Any

# Initialize OpenAI client only if API key is available
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    print("Warning: OPENAI_API_KEY not set. LLM interventions will not work.")

# Ensure your OpenAI API key is set in the environment variable 'OPENAI_API_KEY'

def process_command(user_command: str, layout_info: Optional[Dict[str, Any]] = None) -> dict:
    """
    Uses OpenAI's GPT-4.1 nano to process a human intervention command.
    Based on the input command, outputs a JSON object with suggested agent behavior changes.
    
    Args:
        user_command: The human's intervention command
        layout_info: Optional dictionary containing layout information such as:
            - layout_name: Name of the current kitchen layout
            - pot_locations: Positions of pots in the kitchen
            - dish_locations: Positions of dish dispensers
            - onion_locations: Positions of onion dispensers
            - serving_locations: Positions where soup can be served
            - counter_locations: Available counter space
    
    Returns:
        A dictionary with agent behavior suggestions. Example outputs:
        {"ego_agent_new_heuristic": "place_onion_in_pot"}
        {"ego_agent_new_heuristic": "deliver_soup"}
        {"ego_agent_new_heuristic": "place_onion_and_deliver_soup"}
        {"ego_agent_new_heuristic": null}  # if no change needed
    """
    
    # Build layout context string
    layout_context = ""
    if layout_info:
        layout_context = f"\nCurrent kitchen layout: {layout_info.get('layout_name', 'unknown')}"
        if 'pot_locations' in layout_info:
            layout_context += f"\nPot locations: {layout_info['pot_locations']}"
        if 'serving_locations' in layout_info:
            layout_context += f"\nServing locations: {layout_info['serving_locations']}"
        if 'onion_locations' in layout_info:
            layout_context += f"\nOnion dispenser locations: {layout_info['onion_locations']}"
        if 'dish_locations' in layout_info:
            layout_context += f"\nDish dispenser locations: {layout_info['dish_locations']}"
    
    prompt = (
        "You are the ego agent in an Overcooked simulation working in a kitchen environment. "
        "You are collaborating with another agent (the confederate) to efficiently complete cooking tasks. "
        f"{layout_context}\n\n"
        "Your current behavior can follow various heuristic strategies such as:\n"
        "- 'place_onion_in_pot': Focus on picking up onions and placing them in pots\n"
        "- 'deliver_soup': Focus on picking up ready soup and delivering it to serving locations\n"
        "- 'place_onion_and_deliver_soup': Alternate between placing onions and delivering soup\n"
        "- 'put_onion_everywhere': Place onions on any available counter space\n"
        "- 'put_dish_everywhere': Place dishes on any available counter space\n"
        "- 'clockwise': Move in a clockwise pattern around the kitchen\n"
        "- 'counterclockwise': Move in a counterclockwise pattern around the kitchen\n"
        "- 'stay_in_region': Focus on a specific region of the kitchen\n"
        "- 'follow_confederate': Coordinate closely with the other agent's movements\n"
        "- 'avoid_confederate': Maintain distance from the other agent to avoid collisions\n\n"
        "A human controller provides high-level commands to help you adjust your strategy. "
        "Carefully analyze the human command below and determine:\n"
        "1. What the human wants you to do differently\n"
        "2. Which heuristic strategy would best achieve this goal\n"
        "3. How this change would improve coordination with your teammate\n\n"
        "Output a JSON object with a single key 'ego_agent_new_heuristic' containing:\n"
        "- The name of the new heuristic strategy (as a string) if a change is needed\n"
        "- null if no change is necessary\n"
        "- You may also suggest custom behavior descriptions if the predefined heuristics don't match\n\n"
        "Output valid JSON only, with no additional commentary.\n\n"
        f"Human command: \"{user_command}\""
    )

    try:
        if client is None:
            print("OpenAI client not initialized. Returning null heuristic.")
            return {"ego_agent_new_heuristic": None}
        
        # Handle empty commands
        if not user_command.strip():
            return {"ego_agent_new_heuristic": None}
            
        response = client.chat.completions.create(
            model="gpt-4.1-nano",  # Updated to use GPT-4.1 nano model.
            messages=[
                {"role": "system", "content": "You are a strategic assistant that analyzes human commands and suggests appropriate behavior heuristics for an agent in an Overcooked kitchen simulation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        result = json.loads(content)
    except Exception as e:
        print("Error during LLM processing:", e)
        result = {"ego_agent_new_heuristic": None}
    return result

# Example usage with layout information
def process_command_with_context(user_command: str, env_state: Optional[Any] = None) -> dict:
    """
    Wrapper function that extracts layout information from environment state if available.
    
    Args:
        user_command: The human's intervention command
        env_state: Optional environment state object containing layout information
    
    Returns:
        A dictionary with agent behavior suggestions
    """
    layout_info = None
    
    if env_state and hasattr(env_state, 'mdp'):
        mdp = env_state.mdp
        layout_info = {
            'layout_name': getattr(mdp, 'layout_name', 'unknown'),
        }
        
        # Extract pot locations if available
        if hasattr(mdp, 'get_pot_locations'):
            layout_info['pot_locations'] = mdp.get_pot_locations()
        
        # Extract serving locations if available  
        if hasattr(mdp, 'get_serving_locations'):
            layout_info['serving_locations'] = mdp.get_serving_locations()
            
        # Extract dispenser locations if available
        if hasattr(mdp, 'get_onion_dispenser_locations'):
            layout_info['onion_locations'] = mdp.get_onion_dispenser_locations()
        if hasattr(mdp, 'get_dish_dispenser_locations'):
            layout_info['dish_locations'] = mdp.get_dish_dispenser_locations()
    
    return process_command(user_command, layout_info)