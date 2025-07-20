#!/usr/bin/env python
import os
import pickle
import yaml
import torch
import numpy as np
import random
import threading
import time
import sys
import pygame
from typing import Dict, Tuple

# Import Overcooked evaluation components
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.agents.agent import AgentPairPettingZoo, KeyboardAgent
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from AHT_human_intervention.runner.shared.base_runner import make_trainer_policy_cls
from AHT_human_intervention.algorithms.population.utils import EvalPolicy
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from AHT_human_intervention.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnvPettingZoo

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Global configuration variables (update these as needed)
LAYOUT_NAME = "random3"
POLICY_NAME0 = None   # initial ego policy name
POLICY_NAME1 = None   # second pre-trained policy name
POPULATION_YAML_PATH = "/home/lingyun/LARG/AHTHumanIntervention/AHT_human_intervention/human_exp/configs/benchmark_configs/random3_benchmark.yml"
POLICY_POOL_PATH = "/home/lingyun/LARG/AHTHumanIntervention/AHT_human_intervention/policy_pool"
HORIZON = 1000   # Maximum steps per episode
NUM_GAMES = 1   # Number of games to play
DISPLAY = True  # Enable display

# Set environment variable for policy pool path
os.environ["POLICY_POOL"] = POLICY_POOL_PATH

from AHT_human_intervention.algorithms.population.policy_pool import add_path_prefix
from AHT_human_intervention.envs.overcooked_new.src.overcooked_ai_py.agents.agent import Agent

class LoadedPolicyAgent(Agent):
    """Agent that wraps a pre-trained policy to act in the Overcooked environment."""
    
    def __init__(self, policy_name: str, policy_args, policy, featurize_type: str = "ppo", player_id=-1, device: str = "cpu"):
        self.policy_name = policy_name
        self.policy_args = policy_args
        self.featurize_type = featurize_type
        self.device = torch.device(device)
        
        self.policy = EvalPolicy(policy_args, policy)
        self.policy.reset(1, 1)
        self.policy.register_control_agent(0, 0)
        
        self.deterministic = True
        self.epsilon = 0.3
        self.mdp = None
        
        if player_id not in [0, 1]:
            raise ValueError(f"Player ID must be 0 or 1, not {player_id}.")
        self.player_id = player_id
        
    def featurize(self, overcookedState):
        foo = self.mdp.lossless_state_encoding(overcookedState)[self.player_id] * 255
        return [foo, foo]

    def set_mdp(self, mdp: OvercookedGridworld):
        self.mdp = mdp

    def _process_state(self, state: Dict, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        def object_from_dict(object_dict: Dict):
            return ObjectState(**object_dict)
        def player_from_dict(player_dict: Dict):
            held_obj = player_dict.get("held_object")
            if held_obj is not None:
                player_dict["held_object"] = object_from_dict(held_obj)
            return PlayerState(**player_dict)
        def state_from_dict(state_dict: Dict):
            state_dict["players"] = [player_from_dict(p) for p in state_dict["players"]]
            object_list = [object_from_dict(o) for o in state_dict["objects"].values()]
            state_dict["objects"] = {ob.position: ob for ob in object_list}
            return OvercookedState(**state_dict)
        if self.featurize_type == "ppo":
            state_obj = state.deepcopy()
            return (
                self.mdp.lossless_state_encoding(state_obj)[pos] * 255,
                self._get_available_actions(state_obj)[pos]
            )
        else:
            raise NotImplementedError(f"Featurize type {self.featurize_type} not supported.")
        
    def _get_available_actions(self, state: OvercookedState) -> np.ndarray:
        num_agents = len(state.players)
        available_actions = np.ones((num_agents, len(Action.ALL_ACTIONS)), dtype=np.uint8)
        interact_index = Action.ACTION_TO_INDEX["interact"]
        for agent_idx in range(num_agents):
            player = state.players[agent_idx]
            pos = player.position
            o = player.orientation
            for move_i, move in enumerate(Direction.ALL_DIRECTIONS):
                new_pos = Action.move_in_direction(pos, move)
                if new_pos not in self.mdp.get_valid_player_positions() and o == move:
                    available_actions[agent_idx, move_i] = 0
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.mdp.get_terrain_type_at_pos(i_pos)
            if (
                terrain_type == " "
                or (
                    terrain_type == "X"
                    and (
                        (not player.has_object() and not state.has_object(i_pos))
                        or (player.has_object() and state.has_object(i_pos))
                    )
                )
                or (terrain_type in ["O", "T", "D"] and player.has_object())
                or (
                    terrain_type == "P"
                    and (not player.has_object() or player.get_object().name not in ["dish", "onion", "tomato"])
                )
                or (terrain_type == "S" and (not player.has_object() or player.get_object().name not in ["soup"]))
            ):
                available_actions[agent_idx, interact_index] = 0
        return available_actions

    def action(self, state):
        self.policy.prep_rollout()
        epsilon = random.random()
        if not self.deterministic or epsilon < self.epsilon:
            rtn = self.policy.step(
                np.array([state]),
                [(0, 0)],
                deterministic=False,
                available_actions=None,
            )[0]
            if isinstance(rtn, np.ndarray):
                return rtn[0]
        else:
            rtn = self.policy.step(
                np.array([state]),
                [(0, 0)],
                deterministic=True,
                available_actions=None,
            )[0]
            if isinstance(rtn, np.ndarray):
                return rtn[0]

    def _action(self, state: Dict) -> int:
        if self.mdp is None:
            raise ValueError("MDP not set. Call set_mdp() before using the agent.")
        processed_state, available_actions = self._process_state(state, pos=self.agent_pos)
        state_tensor = np.array(processed_state, dtype=np.float32)
        self.policy.prep_rollout()
        epsilon = random.random()
        if not self.deterministic or epsilon < self.epsilon:
            return self.policy.step(
                np.array([state_tensor]),
                [(0, 0)],
                deterministic=False,
                available_actions=np.array([available_actions]),
            )[0]
        else:
            return self.policy.step(
                np.array([state_tensor]),
                [(0, 0)],
                deterministic=True,
                available_actions=np.array([available_actions]),
            )[0]

    def set_position(self, pos: int):
        self.agent_pos = pos

    def switch_policy(self, new_policy_tuple: Tuple[str, object, object, str]):
        self.policy_name = new_policy_tuple[0]
        self.policy_args = new_policy_tuple[1]
        self.policy = EvalPolicy(new_policy_tuple[1], new_policy_tuple[2])
        self.policy.reset(1, 1)
        self.policy.register_control_agent(0, 0)
        self.featurize_type = new_policy_tuple[3]
        print(f"Switched policy to {self.policy_name}")

def main():
    global LAYOUT_NAME, POLICY_NAME0, POLICY_NAME1, POPULATION_YAML_PATH, POLICY_POOL_PATH
    if not all([LAYOUT_NAME, POLICY_NAME0, POLICY_NAME1, POPULATION_YAML_PATH]):
        raise ValueError("Global variables LAYOUT_NAME, POLICY_NAME0, POLICY_NAME1, and POPULATION_YAML_PATH must be set.")

    population_config = yaml.load(open(POPULATION_YAML_PATH, "r", encoding="utf-8"), yaml.Loader)
    policy_data = {}

    print(f"Available policies: {population_config.keys()}")
    for policy_name in population_config.keys():
        if policy_name not in population_config:
            raise ValueError(f"Policy {policy_name} not found in {POPULATION_YAML_PATH}")
        try:
            policy_config_path = os.path.join(
                POLICY_POOL_PATH,
                population_config[policy_name]["policy_config_path"],
            )
            policy_config = list(pickle.load(open(policy_config_path, "rb")))
            policy_args = policy_config[0]
            _, policy_cls = make_trainer_policy_cls(policy_args.algorithm_name)
            policy = policy_cls(*policy_config, device=torch.device("cpu"))
            if population_config[policy_name].get("model_path", None):
                model_path = add_path_prefix(POLICY_POOL_PATH, population_config[policy_name]["model_path"])
                policy.load_checkpoint(model_path)
            featurize_type = population_config[policy_name]["featurize_type"]
            policy_data[policy_name] = (policy_name, policy_args, policy, featurize_type)
        except Exception as e:
            print(f"Error loading policy {policy_name}: {e}")
            raise e

    agent0 = LoadedPolicyAgent(*policy_data[POLICY_NAME0], player_id=0)
    agent1 = LoadedPolicyAgent(*policy_data[POLICY_NAME1], player_id=1)
    
    mdp = OvercookedGridworld.from_layout_name(LAYOUT_NAME, start_order_list=["any"], cook_time=5)
    agent0.set_mdp(mdp)
    agent1.set_mdp(mdp)
    agent0.set_position(0)
    agent1.set_position(1)
    
    agent_pair = AgentPairPettingZoo(agent0, agent1, allow_duplicate_agents=True)

    large_mdp = OvercookedGridworld.from_layout_name("random3", start_order_list=["any"], cook_time=5)
    base_env = OvercookedEnv(large_mdp, horizon=HORIZON)
    env = OvercookedEnvPettingZoo(base_env, agent_pair)

    # Set window size to include game area and textbox area
    GAME_WIDTH, GAME_HEIGHT = 800, 600
    TEXTBOX_HEIGHT = 100
    window_size = (GAME_WIDTH, GAME_HEIGHT + TEXTBOX_HEIGHT)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Overcooked Simulation with Command Input")
    clock = pygame.time.Clock()
    fps = 12  # Adjust frame rate as needed

    # Reset the environment and get the initial observation
    obs, _ = env.reset()

    # Font for rendering text input
    font = pygame.font.Font(None, 32)
    input_text = ""  # This holds the current text input
    show_textbox = False  # Flag to show/hide the textbox

    # Main simulation loop
    for i in range(300):
        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if show_textbox:
                    # When textbox is active, capture text input
                    if event.key == pygame.K_RETURN:
                        # Process command when Enter is pressed
                        user_cmd = input_text.strip()
                        if user_cmd.startswith("switch from"):
                            parts = user_cmd.split()
                            if len(parts) >= 5:
                                source_policy = parts[2]
                                target_policy = parts[4]
                                if agent0.policy_name == source_policy:
                                    if target_policy in policy_data:
                                        agent0.switch_policy(policy_data[target_policy])
                                    else:
                                        print(f"Target policy '{target_policy}' not found in loaded policies.")
                                else:
                                    print(f"Agent0 is not using policy '{source_policy}'; current policy: {agent0.policy_name}")
                        # Clear text and hide textbox after processing
                        input_text = ""
                        show_textbox = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode
                else:
                    # When textbox is not active, check for command trigger
                    if event.key == pygame.K_p:
                        # Activate textbox for command input
                        show_textbox = True

        # Render game area (assume env.render returns a surface)
        # We let the render function cover the top GAME_HEIGHT pixels
        env_surface = env.state_visualizer.render_state(env.base_env.state, grid=None)
        # If necessary, scale or adjust env_surface to fit GAME_WIDTH x GAME_HEIGHT
        game_surface = pygame.transform.scale(env_surface, (GAME_WIDTH, GAME_HEIGHT))
        screen.blit(game_surface, (0, 0))

        # Render textbox area at the bottom
        textbox_rect = pygame.Rect(0, GAME_HEIGHT, GAME_WIDTH, TEXTBOX_HEIGHT)
        pygame.draw.rect(screen, (200, 200, 200), textbox_rect)  # light gray background

        # Display prompt and current input if textbox is active
        prompt = "Enter command (e.g., 'switch from fcp1 to cole1'): " if show_textbox else "Press 'p' to intervene"
        text_surface = font.render(prompt + input_text, True, (0, 0, 0))
        screen.blit(text_surface, (10, GAME_HEIGHT + (TEXTBOX_HEIGHT - text_surface.get_height()) // 2))

        pygame.display.flip()
        clock.tick(fps)

        # Take a step in the environment if not waiting for input (or you can modify as needed)
        if not show_textbox:
            a = agent_pair.joint_action(obs)
            obs, reward, terminated, truncated, info = env.step(a)
            if any(truncated.values()):
                obs, _ = env.reset()
                print("Environment reset.")

    # Final steps after the loop (if needed)
    pygame.quit()

if __name__ == "__main__":
    # Set the global variables before running (example values)
    LAYOUT_NAME = "random3"
    # Replace with actual policy names from your YAML
    POLICY_NAME0 = "fcp1"  
    POLICY_NAME1 = "fcp1"  
    POPULATION_YAML_PATH = "/home/lingyun/LARG/AHTHumanIntervention/AHT_human_intervention/human_exp/configs/benchmark_configs/random3_benchmark.yml"
    main()
