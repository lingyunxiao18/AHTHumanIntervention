#!/usr/bin/env python
import os
import sys
import random
import pygame
import time
import numpy as np

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

def main():
    # Create the MDP instance from layout.
    mdp = OvercookedGridworld.from_layout_name(LAYOUT_NAME, start_order_list=["any"], cook_time=5)

    # --- Agent Initialization ---
    # Ego agent (agent 0) starts with clockwise heuristic.
    ego_agent = RotateAgent(direction=True)
    ego_agent.set_agent_index(0)
    ego_agent.set_mdp(mdp)
    
    # Confederate agent (agent 1) starts with clockwise heuristic.
    confederate = RotateAgent(direction=True)
    confederate.set_agent_index(1)
    confederate.set_mdp(mdp)

    # Compose the agent pair.
    agent_pair = AgentPair(ego_agent, confederate, allow_duplicate_agents=True)
    agent_pair.set_mdp(mdp)

    # Create the environment using the legacy OvercookedEnv.
    env = OvercookedEnv(mdp, horizon=HORIZON)
    env.reset()  
    state = env.state  # Access initial OvercookedState.

    # --- Setup Pygame Display ---
    GAME_WIDTH, GAME_HEIGHT = 800, 600
    TEXTBOX_HEIGHT = 100
    window_size = (GAME_WIDTH, GAME_HEIGHT + TEXTBOX_HEIGHT)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Overcooked Simulation: Intervention Module Active")
    clock = pygame.time.Clock()
    fps = 1  # Frames per second

    # Create a StateVisualizer to render the OvercookedState.
    state_visualizer = StateVisualizer(grid=mdp.terrain_mtx)

    # Setup font for text input.
    font = pygame.font.Font(None, 32)
    input_text = ""
    show_textbox = False

    # Step counter to simulate sudden changes.
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

        # Render the text input box.
        textbox_rect = pygame.Rect(0, GAME_HEIGHT, GAME_WIDTH, TEXTBOX_HEIGHT)
        pygame.draw.rect(screen, (200, 200, 200), textbox_rect)
        prompt = "Enter command: " if show_textbox else "Press 'p' for command"
        text_surface = font.render(prompt + input_text, True, (0, 0, 0))
        screen.blit(text_surface, (10, GAME_HEIGHT + (TEXTBOX_HEIGHT - text_surface.get_height()) // 2))
        pygame.display.flip()
        clock.tick(fps)

        # Every 10 steps, simulate that the confederate suddenly changes behavior.
        if step_counter == 5:
            print("Simulation: Confederate switching to counterclockwise.")
            confederate.direction = False

        # Only advance simulation when not in command input mode.
        if not show_textbox:
            # Each agent returns (action, info); extract the action and convert if needed.
            raw_joint_actions = agent_pair.joint_action(env.state)
            joint_action = tuple(convert_action(a_info[0]) for a_info in raw_joint_actions)
            next_state, reward, done, info = env.step(joint_action)
            state = next_state
            step_counter += 1

            if done:
                print("Episode ended. Resetting environment.")
                env.reset()
                state = env.state
                step_counter = 0

if __name__ == "__main__":
    main()
