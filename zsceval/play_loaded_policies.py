import os
import pickle
import yaml
import torch
import numpy as np
import random
from typing import Dict, Callable, Tuple

from zsceval.envs.overcooked.overcooked_ai_py.agents.agent import AgentPair, AgentPairPettingZoo, KeyboardAgent
from zsceval.envs.overcooked.overcooked_ai_py.agents.benchmarking import AgentEvaluator
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
    ObjectState,
)
from zsceval.runner.shared.base_runner import make_trainer_policy_cls
from zsceval.algorithms.population.utils import EvalPolicy
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction

from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnvPettingZoo

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Global configuration variables (to be set before running main)
LAYOUT_NAME = "random1"
POLICY_NAME0 = None  # Set to the first policy name, e.g., "policy1"
POLICY_NAME1 = None  # Set to the second policy name, e.g., "policy2"
POPULATION_YAML_PATH = "/home/zhihan/Documents/Code/ZSC-Eval/zsceval/human_exp/configs/benchmark_configs/random1_benchmark.yml"
POLICY_POOL_PATH = "/home/zhihan/Documents/Code/ZSC-Eval/zsceval/policy_pool"
HORIZON = 100  # Maximum steps per episode
NUM_GAMES = 1  # Number of games to play
DISPLAY = True  # Enable display

import os
# set policy pool path
os.environ["POLICY_POOL"] = POLICY_POOL_PATH

from zsceval.algorithms.population.policy_pool import add_path_prefix
from zsceval.envs.overcooked_new.src.overcooked_ai_py.agents.agent import Agent

class LoadedPolicyAgent(Agent):
    """A class to load a pre-trained policy and act as an agent in the Overcooked environment."""
    
    def __init__(self, policy_name: str, policy_args, policy, featurize_type: str = "ppo", player_id=-1, device: str = "cpu"):
        self.policy_name = policy_name
        self.policy_args = policy_args
        self.featurize_type = featurize_type
        self.device = torch.device(device)
        
        
        # Wrap the policy in EvalPolicy as per the reference system
        self.policy = EvalPolicy(policy_args, policy)
        self.policy.reset(1, 1)  # Reset with dummy values (batch_size=1, num_agents=1)
        self.policy.register_control_agent(0, 0)
        
        # self.policy.policy.eval()  # Set to evaluation mode
        # self.policy.policy.to(self.device)
        self.deterministic = True
        self.epsilon = 0.3
        
        self.mdp = None  # Will be set later
        
        assert player_id in [0, 1], f"Player ID must be 0 or 1 (correspods to 'player_0', 'player_1'), not {player_id}."
        self.player_id = player_id
        
    def featurize(self, overcookedState):
        # foo = self.mdp.lossless_state_encoding(overcookedState)[self.player_id] / 255 # they were all int, why multiply by 255?
        foo = self.mdp.lossless_state_encoding(overcookedState)[self.player_id] * 255
        return [foo, foo]

    def set_mdp(self, mdp: OvercookedGridworld):
        """Set the MDP for state processing."""
        self.mdp = mdp

    def _process_state(self, state: Dict, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """Process the raw state dictionary into a format suitable for the policy."""
        def object_from_dict(object_dict: Dict):
            return ObjectState(**object_dict)

        def player_from_dict(player_dict: Dict):
            held_obj = player_dict.get("held_object")
            if held_obj is not None:
                player_dict["held_object"] = object_from_dict(held_obj)
            return PlayerState(**player_dict)

        def state_from_dict(state_dict: Dict):
            state_dict["players"] = [player_from_dict(p) for p in state_dict["players"]]
            object_list = [object_from_dict(o) for _, o in state_dict["objects"].items()]
            state_dict["objects"] = {ob.position: ob for ob in object_list}
            return OvercookedState(**state_dict)

        if self.featurize_type == "ppo":
            state_obj = state.deepcopy()
            # state_obj = state_from_dict(state.copy())
            return (
                self.mdp.lossless_state_encoding(state_obj)[pos] * 255,
                # self.mdp.get_actions(state_obj)[pos]
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
            # if rtn is not scalar, return the first element
            if isinstance(rtn, np.ndarray):
                return rtn[0]
            else:
                assert "Good, rtn is scalar"
        else:
            rtn = self.policy.step(
                np.array([state]),
                [(0, 0)],
                deterministic=True,
                available_actions=None,
            )[0]
            if isinstance(rtn, np.ndarray):
                return rtn[0]
            else:
                assert "Good, rtn is scalar"
        
        
        
        
    def _action(self, state: Dict) -> int:
        """Get an action from the loaded policy given the state."""
        if self.mdp is None:
            raise ValueError("MDP not set. Call set_mdp() before using the agent.")
        
        # Process state for this agent's position
        # ZHNOTE Here state is a OvercookedState object
        processed_state, available_actions = self._process_state(state, pos=self.agent_pos)
        
        # Convert to tensor
        # ZHNOTE: Require nparray of shape like 5,5,20, available actions is flat vector 
        state_tensor = np.array(processed_state, dtype=np.float32)
        # state_tensor = torch.tensor(processed_state, dtype=torch.float32).to(self.device)
        
        # Get action from the EvalPolicy
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
        """Set the agent's position (0 or 1) in the pair."""
        self.agent_pos = pos

def main():
    """Run two pre-loaded policies in an Overcooked game with display using global variables."""
    global LAYOUT_NAME, POLICY_NAME0, POLICY_NAME1, POPULATION_YAML_PATH, POLICY_POOL_PATH
    if not all([LAYOUT_NAME, POLICY_NAME0, POLICY_NAME1, POPULATION_YAML_PATH]):
        raise ValueError("Global variables LAYOUT_NAME, POLICY_NAME0, POLICY_NAME1, and POPULATION_YAML_PATH must be set.")

    # Load the YAML configuration
    population_config = yaml.load(open(POPULATION_YAML_PATH, "r", encoding="utf-8"), yaml.Loader)
    
    # Temporary structure to hold policy data (not instantiating a full PolicyPool class)
    policy_data = {}
    
    # Print Available policies
    print(f"Available policies: {population_config.keys()}")
    
    # Populate policy data for the two specified policies
    for policy_name in [POLICY_NAME0, POLICY_NAME1]:
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

    # Create two LoadedPolicyAgent instances
    agent0 = LoadedPolicyAgent(*policy_data[POLICY_NAME0], player_id=0)
    agent1 = LoadedPolicyAgent(*policy_data[POLICY_NAME1], player_id=1)
    

    # Initialize the environment and evaluator
    mdp = OvercookedGridworld.from_layout_name(LAYOUT_NAME, start_order_list=["any"], cook_time=5)
    agent0.set_mdp(mdp)
    agent1.set_mdp(mdp)
    
    
    
    # Assign positions
    agent0.set_position(0)
    agent1.set_position(1)
    
    agent1 = KeyboardAgent()
    agent1.featurize = agent0.featurize

    # Pair the agents
    agent_pair = AgentPairPettingZoo(agent0, agent1, allow_duplicate_agents=True)

    # Configure and run the evaluator
    env_config = {"layout_name": LAYOUT_NAME}
    eval_config = {"horizon": HORIZON}
    
    large_mdp = OvercookedGridworld.from_layout_name("random1", start_order_list=["any"], cook_time=5)
    base_env = OvercookedEnv(large_mdp, horizon=100)
    env = OvercookedEnvPettingZoo(base_env, agent_pair)
    
    observation_space = env.observation_space("agent_0")
    action_space = env.action_space("agent_0")
    print(f"Observation space: {observation_space}")
    print(f"observation_spaces: {env.observation_spaces}")
    print(f"Action space: {action_space}")
    print(f"action_spaces: {env.action_spaces}")
    # obs = env.observation_spaces.sample()
    obs, _ = env.reset()
    print(f"Observation: {obs['agent_0'].shape}")
    # print(f"Reward: {reward}")
    
    a = agent_pair.joint_action(obs) # assuming both agent have same action
    print(f"Joint action: {a}")
    
    fps = 12
    for i in range(300):
        # sleep(1/fps)
        import time
        time.sleep(1/fps)
        a = agent_pair.joint_action(obs)
        obs, reward, terminated, truncated, info = env.step(a)
        
        # if any value in the truncated list is True, reset the environment
        if any(truncated.values()):
            obs, _ = env.reset()
            print(f"Environment reset.")

        env.render("human")
        print(f"Observation: {obs['agent_0'].shape}")
    

    
    obs, reward, terminated, truncated, info = env.step(a)
    

    print(f"Observation: {obs['agent_0'].shape}")
    print(f"Reward: {reward}")  
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")

    env.render("human")
    # show surface in pygame window
    # surface.show()
    
    
    
    # trajectory, time_taken, _, _ = env.run_agents(agent_pair, include_final_state=True, display=DISPLAY)
    
    
    
    # # Evaluate the agent pair with display
    # agent_evaluator =AgentEvaluator({"layout_name": "random1"}, {"horizon": 100})
    # trajectories = agent_evaluator.evaluate_agent_pair(agent_pair, num_games=NUM_GAMES)

    # Print results
    print(f"Completed {NUM_GAMES} games.")
    # print(f"trajectory: {trajectory}")
    # print(f"Total rewards: {trajectories['ep_rewards']}")
    # print(f"Average reward: {np.mean(trajectories['ep_rewards'])}")

if __name__ == "__main__":
    # Set the global variables before running (example values)
    LAYOUT_NAME = "random1"
    # POLICY_NAME0 = "policy1"  # Replace with an actual policy name from your YAML
    # vailable policies: dict_keys([
        # 'fcp1', 'fcp2', 'fcp3', 
        # 'mep1', 'mep2', 'mep3', 
        # 'traj1', 'traj2', 'traj3', 
        # 'hsp1', 'hsp2', 'hsp3', 
        # 'sp1', 'sp2', 'sp3', 
        # 'e3t1', 'e3t2', 'e3t3', 
        # 'cole1', 'cole2', 'cole3'])
    POLICY_NAME0 = "cole1"  # Replace with an actual policy name from your YAML
    POLICY_NAME1 = "cole1"  # Replace with an actual policy name from your YAML
    POPULATION_YAML_PATH = "/home/zhihan/Documents/Code/ZSC-Eval/zsceval/human_exp/configs/benchmark_configs/random1_benchmark.yml"
    
    main()