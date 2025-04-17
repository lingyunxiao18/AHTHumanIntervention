#!/usr/bin/env python
import os
import sys
import random
import pygame
import time
import numpy as np
from stable_baselines3 import PPO

# --- Direct import for OvercookedEnv ---
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

# --- Import Overcooked components ---
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction

# --- Import visualization for rendering ---
from zsceval.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# --- Import Agent classes ---
from zsceval.envs.overcooked.overcooked_ai_py.agents.agent import AgentPair
from heuristic_agent import RotateAgent

# --- Import the intervention module (LLM integration) ---
from zsceval.intervention_LLM_module import process_command

# --- Global configuration parameters ---
LAYOUT_NAME = "random3"
HORIZON = 1000       # Maximum steps per episode
NUM_GAMES = 1        # Number of games to simulate
DISPLAY = True       # Enable display

def convert_action(action):
    """
    Convert an action expressed as a tuple (e.g. (0, 1)) to a valid action constant.
    Valid motion actions are:
      Direction.NORTH: (0, -1)
      Direction.SOUTH: (0, 1)
      Direction.EAST:  (1, 0)
      Direction.WEST:  (-1, 0)
    Action.STAY is (0,0) and Action.INTERACT is "interact".
    If the action is already valid (for example "interact"), it is returned unchanged.
    """
    if isinstance(action, tuple) and len(action) == 2:
        mapping = {
            (0, 0): Action.STAY,
            (0, 1): Direction.SOUTH,
            (0, -1): Direction.NORTH,
            (1, 0): Direction.EAST,
            (-1, 0): Direction.WEST,
        }
        return mapping.get(action, action)
    return action

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
    mdp = OvercookedGridworld.from_layout_name(LAYOUT_NAME, start_order_list=["any"], cook_time=5)

    # --- Agent Initialization ---
    # Ego agent (agent 0) starts with clockwise heuristic.
    # ego_agent = RotateAgent(direction=True)
    # ego_agent.set_agent_index(0)
    # ego_agent.set_mdp(mdp)
    ego_agent = PPO.load("best_response_clockwise.zip", device="cpu") 
    
    # Confederate agent (agent 1) starts with clockwise heuristic.
    confederate = RotateAgent(direction=True)
    confederate.set_agent_index(1)
    confederate.set_mdp(mdp)

    # # Compose the agent pair.
    # agent_pair = AgentPair(ego_agent, confederate, allow_duplicate_agents=True)
    # agent_pair.set_mdp(mdp)

    # Create the environment using the legacy OvercookedEnv.
    env = OvercookedEnv(mdp, horizon=HORIZON)
    env.reset()  
    state = env.state  # Access initial OvercookedState.

    # --- Setup Pygame Display ---
    GAME_WIDTH, GAME_HEIGHT = 800, 600
    TEXTBOX_HEIGHT = 100
    window_size = (GAME_WIDTH, GAME_HEIGHT + TEXTBOX_HEIGHT)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Overcooked Simulation: PPO Best-Response vs Heuristic")
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
            text_to_render = "Please enter your intervention command here: " + input_text
        else:
            text_to_render = "Press 'p' to pause the game and enter command."
        # Wrap text into multiple lines if too long.
        lines = wrap_text(text_to_render, font, GAME_WIDTH - 20)
        y_offset = GAME_HEIGHT + 10  # A little padding from the top of the textbox.
        for line in lines:
            line_surface = font.render(line, True, (0, 0, 0))
            screen.blit(line_surface, (10, y_offset))
            y_offset += line_surface.get_height() + 2  # 2 pixels of spacing between lines.

        pygame.display.flip()
        clock.tick(fps)

        # Every 10 steps, simulate that the confederate suddenly changes behavior.
        if step_counter == 10:
            print("Simulation: Confederate switching to counterclockwise.")
            confederate.direction = False

        # Advance simulation if not in command input mode.
        if not show_textbox:
            obs = np.zeros((84, 84, 3), dtype=np.uint8)
            a0_idx, _ = ego_agent.predict(obs, deterministic=True)
            a0 = Action.ALL_ACTIONS[int(a0_idx)]
            a1, _ = confederate.action(env.state)
            next_state, reward, done, info = env.step((a0, a1))
            state = next_state
            step_counter += 1

if __name__ == "__main__":
    main()
