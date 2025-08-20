#!/usr/bin/env python
# play_language_policies_intervention.py

import os
import sys
import pygame
import time
import numpy as np
import torch
import openai
import json
from AHT_human_intervention.intervention_LLM_module import process_command

# --- Direct import for OvercookedEnv ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
# --- Import Overcooked components ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
# --- Import visualization for rendering ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
# --- Import AgentPair class ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.agents.agent import AgentPair
# --- Import a simple heuristic agent for confederate switch demo ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.agents.agent import StayAgent
# --- Import interesting heuristic agents ---
from heuristic_agent import RotateAgent, OnionToPotAgent, PlateAgent

# --- Our language-conditioned policy module ---
from AHT_human_intervention.language.shared_lang_agent import SharedLangAgent

# Configure OpenAI API key (ensure OPENAI_API_KEY is set in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------------------------------------------------------------------------
# Helpers: wrap text and convert actions
# ----------------------------------------------------------------------------
def wrap_text(text, font, max_width):
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


def convert_action(a):
    """
    Convert various action formats to valid action constants.
    Valid motion actions can be:
      - Direction objects (NORTH, SOUTH, EAST, WEST)
      - Action objects (STAY, INTERACT)
      - Tuples: (0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)
      - Integers: 0-5
      - Strings: "interact"
    Returns the appropriate Action or Direction constant.
    """
    try:
        # If it's already a Direction or Action constant, return as is
        if isinstance(a, (Direction, Action)):
            return a
        
        # Handle integer actions from script agents
        if isinstance(a, int):
            # Handle negative integers (convert to Direction)
            if a == -1:
                return Direction.WEST
            
            mapping = {
                0: Action.STAY,
                1: Direction.EAST,
                2: Direction.SOUTH,
                3: Direction.NORTH,
                4: Direction.WEST,
                5: Action.INTERACT
            }
            if a in mapping:
                return mapping[a]
            raise ValueError(f"Invalid integer action: {a}")
        
        # Handle tuple actions from movement agents
        if isinstance(a, tuple) and len(a) == 2:
            mapping = {
                (0, 0): Action.STAY,
                (0, 1): Direction.SOUTH,
                (0, -1): Direction.NORTH,
                (1, 0): Direction.EAST,
                (-1, 0): Direction.WEST,
            }
            if a in mapping:
                return mapping[a]
            raise ValueError(f"Invalid tuple action: {a}")
        
        # Handle string actions (like "interact")
        if isinstance(a, str):
            if a.lower() == "interact":
                return Action.INTERACT
            raise ValueError(f"Invalid string action: {a}")
            
        raise ValueError(f"Unknown action type: {type(a)}, value: {a}")
    except Exception as e:
        print(f"[ERROR] Error converting action {a} (type: {type(a)}): {e}")
        # Return STAY action as fallback
        return Action.STAY

# ----------------------------------------------------------------------------
def main():
    # Configuration
    LAYOUT_NAME = "random3"
    HORIZON = 400
    FPS = 5
    TRADITIONAL_EGO = False  # Set True to run with a traditional (non-language) ego agent for comparison

    # 1) Load MDP and Env
    mdp = OvercookedGridworld.from_layout_name(LAYOUT_NAME)
    env = OvercookedEnv(mdp, horizon=HORIZON)
    env.reset()

    # 2) Visualizer
    visualizer = StateVisualizer(grid=mdp.terrain_mtx)

    # 3) Instantiate agents
    if TRADITIONAL_EGO:
        # Use a simple StayAgent as a traditional baseline
        ego_agent = StayAgent()
        ego_agent.set_agent_index(0)
        ego_agent.set_mdp(mdp)
        print("[INFO] Running with traditional (non-language) ego agent.")
    else:
        ego_agent = SharedLangAgent(mdp, agent_idx=0)
    
    # Start with an interesting confederate agent
    conf_agent = OnionToPotAgent(direction=True)  # Clockwise rotation
    conf_agent.set_agent_index(1)
    conf_agent.set_mdp(mdp)
    
    pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
    pair.set_mdp(mdp)

    # 5) Pygame setup
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    TB_H = 180  # Increased height for step information
    screen = pygame.display.set_mode((WIDTH, HEIGHT + TB_H))
    pygame.display.set_caption("Overcooked: Language-Conditioned Intervention")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)

    input_text = ""
    show_tb = False
    step = 0
    conf_switched = False
    last_command = ""
    last_heuristic = None

    # 6) Main loop
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif e.type == pygame.KEYDOWN:
                if show_tb:
                    if e.key == pygame.K_RETURN:
                        cmd = input_text.strip()
                        ego_agent.set_command(cmd)
                        last_command = cmd
                        print(f"[LOG] Human intervention: '{cmd}' at step {step}")
                        input_text = ""
                        show_tb = False
                    elif e.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += e.unicode
                else:
                    if e.key == pygame.K_p:
                        show_tb = True

        # 7) Render
        surf = visualizer.render_state(env.state, grid=None)
        gs = pygame.transform.scale(surf, (WIDTH, HEIGHT))
        screen.blit(gs, (0,0))
        pygame.draw.rect(screen, (200,200,200), (0,HEIGHT, WIDTH, TB_H))
        txt = "Enter cmd: " + input_text if show_tb else "Press 'p' to command"
        for i, line in enumerate(wrap_text(txt, font, WIDTH-20)):
            screen.blit(font.render(line, True, (0,0,0)), (10, HEIGHT+10 + i*30))
        # Display last command and ego heuristic
        if last_command:
            screen.blit(font.render(f'Last command: {last_command}', True, (0,0,0)), (10, HEIGHT+70))
        if hasattr(ego_agent, 'heuristic') and getattr(ego_agent, 'heuristic', None):
            screen.blit(font.render(f'Ego heuristic: {ego_agent.heuristic}', True, (0,0,0)), (10, HEIGHT+100))
        
        # Display current confederate agent type
        conf_agent_type = type(conf_agent).__name__
        screen.blit(font.render(f'Confederate: {conf_agent_type}', True, (0,0,0)), (10, HEIGHT+130))
        
        # Display current step
        screen.blit(font.render(f'Step: {step}', True, (0,0,0)), (10, HEIGHT+160))
        pygame.display.flip()
        clock.tick(FPS)

        # 8) Simulation step
        if not show_tb:
            # Confederate switching at different steps to create dynamic behavior
            if step == 20 and not conf_switched:
                # Switch to PlateAgent at step 20
                conf_agent = PlateAgent(direction=True)
                conf_agent.set_agent_index(1)
                conf_agent.set_mdp(mdp)
                pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
                pair.set_mdp(mdp)
                conf_switched = True
                print(f"[LOG] Confederate agent switched to PlateAgent at step {step}")
            elif step == 40 and conf_switched:
                # Switch to RotateAgent at step 40
                conf_agent = RotateAgent(direction=True)
                conf_agent.set_agent_index(1)
                conf_agent.set_mdp(mdp)
                pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
                pair.set_mdp(mdp)
                conf_switched = False  # Reset flag to allow future switches
                print(f"[LOG] Confederate agent switched to RotateAgent at step {step}")
            elif step == 60 and not conf_switched:
                # Switch back to OnionToPotAgent at step 60
                conf_agent = OnionToPotAgent(direction=False)  # Counter-clockwise this time
                conf_agent.set_agent_index(1)
                conf_agent.set_mdp(mdp)
                pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
                pair.set_mdp(mdp)
                conf_switched = True
                print(f"[LOG] Confederate agent switched to OnionToPotAgent (counter-clockwise) at step {step}")
            # Step environment
            raw = pair.joint_action(env.state)
            # Extract actions from (action, info) tuples or just actions
            ja = []
            for a in raw:
                if isinstance(a, tuple) and len(a) == 2:
                    # Agent returned (action, info) tuple
                    ja.append(convert_action(a[0]))
                else:
                    # Agent returned just action
                    ja.append(convert_action(a))
            ja = tuple(ja)
            nxt, r, done, _ = env.step(ja)
            env.state = nxt
            step += 1
            # Log heuristic change
            if hasattr(ego_agent, 'heuristic'):
                if ego_agent.heuristic != last_heuristic:
                    print(f"[LOG] Ego heuristic changed to: {ego_agent.heuristic} at step {step}")
                    last_heuristic = ego_agent.heuristic
            if step == 20:
                print("Sim: step 20 reached—fallback logic active for unknown commands.")
            if done:
                print("Episode done, resetting.")
                env.reset()
                step = 0
                conf_switched = False
                # Reset confederate agent to initial state
                conf_agent = OnionToPotAgent(direction=True)
                conf_agent.set_agent_index(1)
                conf_agent.set_mdp(mdp)
                pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
                pair.set_mdp(mdp)

if __name__ == "__main__":
    main()
