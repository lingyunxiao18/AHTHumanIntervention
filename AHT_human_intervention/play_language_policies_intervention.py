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
from language.language_conditioned_policy import HuggingFaceLangConditionedPolicy
from transformers import DistilBertTokenizer
import torch

class TrainedPolicyAgent:
    """Agent that uses our trained language-conditioned policy."""
    
    def __init__(self, mdp, agent_idx=0, model_path=None):
        self.mdp = mdp
        self.agent_idx = agent_idx
        self.command = None
        self.heuristic = "No command set"
        
        # Set default model path if none provided
        if model_path is None:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "language", "test_policy_best.pth")
        
        # Load the trained policy
        print(f"[INFO] Loading trained policy from {model_path}")
        print(f"[DEBUG] Model path exists: {os.path.exists(model_path)}")
        print(f"[DEBUG] Model path is file: {os.path.isfile(model_path)}")
        print(f"[DEBUG] Model path is absolute: {os.path.isabs(model_path)}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize policy architecture
        state_dim = 20
        num_actions = 6
        self.policy = HuggingFaceLangConditionedPolicy(
            state_dim=state_dim,
            num_actions=num_actions,
            text_dim=768,
            hidden_dim=256,
            freeze_bert=False
        ).to(self.device)
        
        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            print(f"[INFO] Policy loaded successfully! Best validation accuracy: {checkpoint.get('val_acc', 'Unknown')}")
        except Exception as e:
            print(f"[WARNING] Could not load trained policy: {e}")
            print("[WARNING] Using untrained policy (random actions)")
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Set to evaluation mode
        self.policy.eval()
    
    def set_agent_index(self, agent_idx):
        self.agent_idx = agent_idx
    
    def set_mdp(self, mdp):
        self.mdp = mdp
    
    def set_command(self, command):
        """Set the human intervention command."""
        self.command = command
        self.heuristic = f"Following command: {command}"
        print(f"[INFO] Command set: {command}")
    
    def get_action(self, state):
        """Get action from the trained policy."""
        if not self.command:
            # No command set, return STAY
            return 0
        
        try:
            # Extract state features (simplified - you may need to adjust this)
            state_features = self._extract_state_features(state)
            state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Tokenize command
            command_tokens = self.tokenizer(
                self.command,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            command_input_ids = command_tokens['input_ids'].to(self.device)
            command_attention_mask = command_tokens['attention_mask'].to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                logits = self.policy(state_tensor, command_input_ids, command_attention_mask)
                action = torch.argmax(logits, dim=1).item()
            
            return action
            
        except Exception as e:
            print(f"[ERROR] Policy inference failed: {e}")
            return 0  # STAY as fallback
    
    def action(self, state):
        """Method that AgentPair calls to get actions."""
        return self.get_action(state)
    
    def _extract_state_features(self, state):
        """Extract state features similar to training data."""
        # This is a simplified version - you may need to match the exact format from training
        features = []
        
        # Player positions and held objects
        for player in state.players:
            features.extend([float(player.position[0]), float(player.position[1])])
            features.append(1.0 if player.held_object else 0.0)
        
        # Object counts
        object_counts = {'onion': 0, 'dish': 0, 'soup': 0}
        for obj in state.objects.values():
            if obj.name in object_counts:
                object_counts[obj.name] += 1
        
        for obj_type in ['onion', 'dish', 'soup']:
            features.append(float(object_counts[obj_type]))
        
        # Layout features
        features.append(float(len(self.mdp.get_pot_locations())))
        features.append(float(len(self.mdp.get_counter_locations())))
        features.append(float(len(self.mdp.get_onion_dispenser_locations())))
        features.append(float(len(self.mdp.get_dish_dispenser_locations())))
        features.append(float(len(self.mdp.get_serving_locations())))
        
        # Timestep (normalized)
        features.append(float(state.timestep) / 400.0)
        
        # Pad to 20 features
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]

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
        # Load the policy-based agent
        ego_agent = TrainedPolicyAgent(mdp, agent_idx=0)
    
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
