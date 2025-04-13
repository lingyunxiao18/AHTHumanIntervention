#!/usr/bin/env python
import os
import sys
import random
import pygame
import time
import numpy as np

# --- Use the old (direct) import for OvercookedEnv ---
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

# --- Import Overcooked components ---
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction

# --- Import visualization for rendering ---
from zsceval.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# --- Import Agent classes ---
from zsceval.envs.overcooked.overcooked_ai_py.agents.agent import AgentPair
from heuristic_agent import RotateAgent

# --- Global configuration parameters ---
LAYOUT_NAME = "random3"
HORIZON = 100       # Maximum steps per episode
NUM_GAMES = 1        # Number of games to simulate
DISPLAY = True       # Enable display

def convert_action(action):
    """
    Helper function to ensure an action is valid.
    If the action is given as a tuple (e.g. (0, 1)), we convert it to a valid action constant.
    Valid motion actions are defined as:
      Direction.NORTH: (0,-1)
      Direction.SOUTH: (0,1)
      Direction.EAST:  (1,0)
      Direction.WEST:  (-1,0)
    Action.STAY is (0, 0) and Action.INTERACT is "interact".
    If an action is already valid (for example, the string "interact"), it is returned unchanged.
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

    # Instantiate two heuristic agents.
    # For player 0, use clockwise rotation.
    agent_clockwise = RotateAgent(direction=True)
    agent_clockwise.set_agent_index(0)
    agent_clockwise.set_mdp(mdp)

    # For player 1, use counterclockwise rotation.
    agent_counterclockwise = RotateAgent(direction=True)
    agent_counterclockwise.set_agent_index(1)
    agent_counterclockwise.set_mdp(mdp)

    # Compose the agent pair.
    agent_pair = AgentPair(agent_clockwise, agent_counterclockwise, allow_duplicate_agents=True)
    agent_pair.set_mdp(mdp)

    # Create the environment using the legacy OvercookedEnv.
    env = OvercookedEnv(mdp, horizon=HORIZON)
    
    # Reset the environment.
    env.reset()
    state = env.state  # Access the initial OvercookedState.

    # --- Setup Pygame Display ---
    GAME_WIDTH, GAME_HEIGHT = 800, 600
    TEXTBOX_HEIGHT = 100
    window_size = (GAME_WIDTH, GAME_HEIGHT + TEXTBOX_HEIGHT)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Overcooked Simulation: Heuristic Agents")
    clock = pygame.time.Clock()
    fps = 1  # Frames per second

    # Create a StateVisualizer to render the OvercookedState.
    state_visualizer = StateVisualizer(grid=mdp.terrain_mtx)

    # Setup font for text input (optional).
    font = pygame.font.Font(None, 32)
    input_text = ""
    show_textbox = False

    # --- Main Simulation Loop ---
    for i in range(300):
        # Process Pygame events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if show_textbox:
                    if event.key == pygame.K_RETURN:
                        user_cmd = input_text.strip()
                        # (Optional: process command input here.)
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
        prompt = "Enter command: " if show_textbox else "Press 'p' for command"
        text_surface = font.render(prompt + input_text, True, (0, 0, 0))
        screen.blit(text_surface, (10, GAME_HEIGHT + (TEXTBOX_HEIGHT - text_surface.get_height()) // 2))
        pygame.display.flip()
        clock.tick(fps)

        # Execute an environment step if not waiting for command input.
        if not show_textbox:
            # Get the joint action from the agent pair.
            # Each agent returns a tuple (action, info); extract the action.
            raw_joint_actions = agent_pair.joint_action(env.state)
            joint_action = tuple(convert_action(action_info[0]) for action_info in raw_joint_actions)
            next_state, reward, done, info = env.step(joint_action)
            state = next_state
            print(state)
            if done:
                print("Episode ended. Resetting environment.")
                env.reset()
                state = env.state
            

    pygame.quit()

if __name__ == "__main__":
    main()
