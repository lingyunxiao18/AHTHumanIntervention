#!/usr/bin/env python
import os
import sys
import random
import pygame
import time
import numpy as np

# --- Direct import for OvercookedEnv ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

# --- Import Overcooked components ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction

# --- Import visualization for rendering ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# --- Import Agent classes ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.agents.agent import AgentPair
from heuristic_agent import RotateAgent, OnionToPotAgent, PlateAgent


# --- Import the intervention module (LLM integration) ---
from AHT_human_intervention.intervention_LLM_module import process_command

def switch_confederate_agent(env, ego_agent, agent_type, mdp):
    """
    Creates a new confederate agent of the specified type and returns a new agent pair.
    
    Args:
        env: The environment instance
        ego_agent: The ego agent instance
        agent_type: String indicating which agent type to create
        mdp: The MDP instance
    
    Returns:
        A new AgentPair instance with the ego agent and new confederate
    """
    if agent_type == "clockwise":
        confederate = RotateAgent(direction=True)
    elif agent_type == "counterclockwise":
        confederate = RotateAgent(direction=False)
    elif agent_type == "onion_to_pot":
        confederate = OnionToPotAgent(direction=True)
    elif agent_type == "plate_agent":
        confederate = PlateAgent(direction=True)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    confederate.set_agent_index(1)
    confederate.set_mdp(mdp)
    confederate.reset()  # Reset the agent's internal state
    return AgentPair(ego_agent, confederate, allow_duplicate_agents=True)

# --- Global configuration parameters ---
LAYOUT_NAME = "random3"
HORIZON = 100000       # Maximum steps per episode
NUM_GAMES = 1        # Number of games to simulate
DISPLAY = True       # Enable display

def convert_action(action):
    """
    Convert various action formats to valid action constants.
    Valid motion actions can be:
      - Direction objects (NORTH, SOUTH, EAST, WEST)
      - Action objects (STAY, INTERACT)
      - Tuples: (0, -1), (0, 1), (1, 0), (-1, 0)
      - Integers: 0-5
      - Strings: "interact"
    Returns the appropriate Action or Direction constant.
    """
    # If it's already a Direction or Action constant, return as is
    if isinstance(action, (Direction, Action)):
        return action
    
    # Handle integer actions from script agents
    if isinstance(action, int):
        # Handle negative integers (convert to Direction)
        if action == -1:
            return Direction.WEST
        
        mapping = {
            0: Action.STAY,
            1: Direction.EAST,
            2: Direction.SOUTH,
            3: Direction.NORTH,
            4: Direction.WEST,
            5: Action.INTERACT
        }
        if action in mapping:
            return mapping[action]
        raise ValueError(f"Invalid integer action: {action}")
    
    # Handle tuple actions from movement agents
    if isinstance(action, tuple) and len(action) == 2:
        mapping = {
            (0, 0): Action.STAY,
            (0, 1): Direction.SOUTH,
            (0, -1): Direction.NORTH,
            (1, 0): Direction.EAST,
            (-1, 0): Direction.WEST,
        }
        if action in mapping:
            return mapping[action]
        raise ValueError(f"Invalid tuple action: {action}")
    
    # Handle string actions (like "interact")
    if isinstance(action, str):
        if action == "interact":
            return Action.INTERACT
        raise ValueError(f"Invalid string action: {action}")
        
    raise ValueError(f"Unknown action type: {type(action)}, value: {action}")

def wrap_text(text, font, max_width):
    """
    Splits the text into multiple lines so that each line's width does not exceed max_width.
    Returns a list of strings (lines).
    """
    words = text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    return lines

def main():
    # Create the MDP instance from layout.
    mdp = OvercookedGridworld.from_layout_name(LAYOUT_NAME, start_order_list=None, cook_time=5)

    # Create the environment using the legacy OvercookedEnv.
    env = OvercookedEnv(mdp, horizon=HORIZON)
    env.reset()  
    state = env.state  # Access initial OvercookedState.

    # --- Agent Initialization ---
    # Initialize ego agent (agent 0) as RotateAgent (clockwise)
    ego_agent = RotateAgent(direction=True)
    ego_agent.set_agent_index(0)
    ego_agent.set_mdp(mdp)
    
    # Initialize confederate (agent 1) as RotateAgent (clockwise)
    confederate = RotateAgent(direction=True)
    confederate.set_agent_index(1)
    confederate.set_mdp(mdp)

    # Compose the agent pair.
    agent_pair = AgentPair(ego_agent, confederate, allow_duplicate_agents=True)
    agent_pair.set_mdp(mdp)

    # --- Setup Pygame Display ---
    GAME_WIDTH, GAME_HEIGHT = 800, 600
    TEXTBOX_HEIGHT = 100
    window_size = (GAME_WIDTH, GAME_HEIGHT + TEXTBOX_HEIGHT)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Overcooked Simulation: Intervention Module Active")
    clock = pygame.time.Clock()
    fps = 5  # Frames per second

    # Create a StateVisualizer to render the OvercookedState.
    state_visualizer = StateVisualizer(grid=mdp.terrain_mtx)

    # Setup font for text input.
    font = pygame.font.Font(None, 32)
    input_text = ""
    show_textbox = False

    # Step counter to simulate sudden behavior changes.
    step_counter = 0

    # --- Main Simulation Loop ---
    while True:
        # Process Pygame events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if show_textbox:
                    if event.key == pygame.K_RETURN:
                        user_cmd = input_text.strip()
                        # Use the real LLM-based intervention module.
                        intervention = process_command(user_cmd)
                        new_heuristic = intervention.get("ego_agent_new_heuristic", None)
                        if new_heuristic:
                            if new_heuristic == "counterclockwise":
                                print("Intervention: Ego agent switching to counterclockwise.")
                                ego_agent.direction = False
                            elif new_heuristic == "clockwise":
                                print("Intervention: Ego agent switching to clockwise.")
                                ego_agent.direction = True
                            elif new_heuristic == "onion_to_pot":
                                print("Intervention: Ego agent switching to onion_to_pot.")
                                ego_agent = OnionToPotAgent(direction=True)
                                ego_agent.set_agent_index(0)
                                ego_agent.set_mdp(mdp)
                                agent_pair = AgentPair(ego_agent, confederate, allow_duplicate_agents=True)
                                agent_pair.set_mdp(mdp)
                            elif new_heuristic == "plate_agent":
                                print("Intervention: Ego agent switching to plate_agent.")
                                ego_agent = PlateAgent(direction=True)
                                ego_agent.set_agent_index(0)
                                ego_agent.set_mdp(mdp)
                                agent_pair = AgentPair(ego_agent, confederate, allow_duplicate_agents=True)
                                agent_pair.set_mdp(mdp)
                        input_text = ""
                        show_textbox = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode
                else:
                    if event.key == pygame.K_p:
                        show_textbox = True

        # Render the environment.
        env_surface = state_visualizer.render_state(env.state, grid=None)
        game_surface = pygame.transform.scale(env_surface, (GAME_WIDTH, GAME_HEIGHT))
        screen.blit(game_surface, (0, 0))

        # Render the textbox area.
        textbox_rect = pygame.Rect(0, GAME_HEIGHT, GAME_WIDTH, TEXTBOX_HEIGHT)
        pygame.draw.rect(screen, (200, 200, 200), textbox_rect)
        # Combine prompt and text input.
        if show_textbox:
            text_to_render = "Enter command: " + input_text
        else:
            text_to_render = "Press 'p' for command"
        # Wrap text into multiple lines if too long.
        lines = wrap_text(text_to_render, font, GAME_WIDTH - 20)
        y_offset = GAME_HEIGHT + 10  # A little padding from the top of the textbox.
        for line in lines:
            line_surface = font.render(line, True, (0, 0, 0))
            screen.blit(line_surface, (10, y_offset))
            y_offset += line_surface.get_height() + 2  # 2 pixels of spacing between lines.

        pygame.display.flip()
        clock.tick(fps)

        # At some step, switch confederate from clockwise to onion_to_pot
        if step_counter == 20:
            print("Simulation: Confederate switching from clockwise to onion_to_pot...")
            agent_pair = switch_confederate_agent(env, ego_agent, "onion_to_pot", mdp)
            agent_pair.set_mdp(mdp)
        # Advance simulation if not in command input mode.
        if not show_textbox:
            # Each agent returns (action, info); extract the action and convert if needed.
            try:
                raw_joint_actions = agent_pair.joint_action(env.state)
                # Handle the case where agents return tuples (action, info) or just actions
                joint_action = []
                for a_info in raw_joint_actions:
                    if isinstance(a_info, tuple):
                        action = a_info[0]  # Extract action from tuple
                    else:
                        action = a_info  # Action is already the value
                    joint_action.append(convert_action(action))
                joint_action = tuple(joint_action)
                next_state, reward, done, info = env.step(joint_action)
                state = next_state
                step_counter += 1
                if done:
                    print(f"Episode ended at step {step_counter}. Resetting environment.")
                    print(f"Environment timestep: {env.t}, Horizon: {env.horizon}")
                    env.reset()
                    state = env.state
                    step_counter = 0
                    # Reset agents when environment resets
                    ego_agent.reset()
                    confederate.reset()
                    ego_agent.set_agent_index(0)
                    confederate.set_agent_index(1)
                    ego_agent.set_mdp(mdp)
                    confederate.set_mdp(mdp)
            except Exception as e:
                print(f"Error during simulation step: {e}")
                print(f"Step counter: {step_counter}")
                break

if __name__ == "__main__":
    main()
