#!/usr/bin/env python
"""
This script trains two best-response policies against fixed confederate agents in the Overcooked environment.
One training run uses a RotateAgent with a clockwise heuristic and the other uses a counterclockwise heuristic.
The RL learner (a dummy agent whose action is overridden by the PPO policy) is trained using PPO from stable-baselines3.
Trained models are saved as "best_response_clockwise.zip" and "best_response_counterclockwise.zip".

Requirements:
    - stable-baselines3
    - gym
    - torch
    - numpy
    - Overcooked code from zsceval (OvercookedGridworld, OvercookedEnv, etc.)
    - heuristic_agent.py (defines RotateAgent) :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
"""

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
# Optionally, for environment checking:
# from stable_baselines3.common.env_checker import check_env

# Import Overcooked MDP and Environment components
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv  
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action
from zsceval.envs.overcooked.overcooked_ai_py.agents.agent import AgentPair

# Import the heuristic RotateAgent from heuristic_agent.py 
from heuristic_agent import RotateAgent

# Define a dummy RL agent that will be controlled by the RL policy.
class DummyRLAgent:
    def __init__(self):
        self.agent_index = None
        self.mdp = None

    def set_agent_index(self, idx):
        self.agent_index = idx

    def set_mdp(self, mdp):
        self.mdp = mdp

    def reset(self):
        pass

    # This action method is not used during training as it is overridden by the PPO policy.
    def action(self, state):
        return (0, {})

# Custom Gym environment wrapper that pairs the dummy RL agent (best-response) with a fixed RotateAgent.
class OvercookedRLWrapper(gym.Env):
    def __init__(self, layout_name, horizon, confederate_direction, best_response_agent):
        """
        Args:
            layout_name (str): Layout of the grid (e.g. "random3").
            horizon (int): Maximum steps per episode.
            confederate_direction (bool): True for clockwise RotateAgent; False for counterclockwise.
            best_response_agent: Instance of the RL agent (e.g. DummyRLAgent).
        """
        # Build the Overcooked gridworld.
        self.mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=["any"], cook_time=5)
        # Create the Overcooked environment.
        self.env = OvercookedEnv(self.mdp, horizon=horizon)  # :contentReference[oaicite:6]{index=6}&#8203;:contentReference[oaicite:7]{index=7}
        self.env.reset()

        # Initialize the confederate agent using the RotateAgent heuristic.
        confederate = RotateAgent(direction=confederate_direction)
        confederate.set_agent_index(1)
        confederate.set_mdp(self.mdp)

        # Initialize the best-response (learning) agent.
        best_response_agent.set_agent_index(0)
        best_response_agent.set_mdp(self.mdp)

        # Create an agent pair consisting of the learning agent and the confederate.
        self.agent_pair = AgentPair(best_response_agent, confederate, allow_duplicate_agents=True)
        self.agent_pair.set_mdp(self.mdp)

        # In this example we use a dummy observation space.
        # Replace with a proper state encoding if needed.
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))
        self.current_action = None

    def step(self, action):
        """
        Injects the provided action into the best-response agent and steps the environment.
        Args:
            action (int): Discrete action index chosen by the RL policy.
        Returns:
            next_obs, reward, done, info: Standard Gym step tuple.
        """
        # Save the RL action.
        self.current_action = action

        # Override the best-response agent's action to return the chosen action.
        original_action_fn = self.agent_pair.agents[0].action
        self.agent_pair.agents[0].action = lambda state: (self.current_action, {})

        # Compute joint actions and step the environment.
        joint_action = self.agent_pair.joint_action(self.env.state)
        next_state, reward, done, info = self.env.step(joint_action)
        next_obs = np.zeros((84, 84, 3), dtype=np.uint8)  # Dummy observation; replace as needed

        # Restore the original action function.
        self.agent_pair.agents[0].action = original_action_fn
        return next_obs, reward, done, info

    def reset(self):
        self.env.reset()
        # Return a dummy initial observation.
        return np.zeros((84, 84, 3), dtype=np.uint8)

    def render(self, mode='human'):
        pass

# Training function for best-response RL policy.
def train_best_response(confederate_direction, model_save_path, total_timesteps=100000):
    """
    Trains a best-response policy using PPO against a fixed confederate.
    Args:
        confederate_direction (bool): True for clockwise; False for counterclockwise.
        model_save_path (str): File path to save the trained model.
        total_timesteps (int): Number of training timesteps.
    """
    dummy_agent = DummyRLAgent()
    env = OvercookedRLWrapper(layout_name="random3", horizon=1000,
                              confederate_direction=confederate_direction,
                              best_response_agent=dummy_agent)
    # Uncomment the following line to check the environment for errors.
    # check_env(env)

    # Create a PPO model (adjust policy and hyperparameters as needed).
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_save_path)
    print("Saved best-response model to", model_save_path)

if __name__ == '__main__':
    # Train best-response policy against the confederate with clockwise rotation.
    train_best_response(confederate_direction=True, model_save_path="best_response_clockwise.zip", total_timesteps=100000)
    
    # Train best-response policy against the confederate with counterclockwise rotation.
    train_best_response(confederate_direction=False, model_save_path="best_response_counterclockwise.zip", total_timesteps=100000)
