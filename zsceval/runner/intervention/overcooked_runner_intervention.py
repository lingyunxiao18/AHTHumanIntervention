import time
import os
import threading
import queue
from collections import defaultdict

import numpy as np
import torch
import wandb
from icecream import ic
from loguru import logger

from zsceval.runner.intervention.base_runner import Runner
from zsceval.utils.log_util import eta

# Import openai for LLM integration.
import openai

# Set the API key from environment variable.
if not openai.api_key:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OPENAI_API_KEY environment variable is not set. LLM calls will fail.")


def _t2n(x):
    return x.detach().cpu().numpy()


class OvercookedRunner(Runner):
    def __init__(self, config):
        super().__init__(config)
        self.paused = False
        # Holds the override action for the EGO agent (player 0)
        self.human_override_action = None
        # A queue to store real-time human input commands.
        self.human_input_queue = queue.Queue()
        # Start background thread to read user input continuously.
        self._start_input_thread()

    def _start_input_thread(self):
        def read_input(q):
            while True:
                # Blocking call; input() runs in a separate thread.
                cmd = input()
                q.put(cmd.strip().lower())
        thread = threading.Thread(target=read_input, args=(self.human_input_queue,), daemon=True)
        thread.start()
        logger.info("Started background input thread for real-time human intervention.")

    def check_for_pause_trigger(self):
        """
        Checks for pause trigger by prompting the user.
        """
        user_input = input("Press Enter to continue or type 'p' to pause: ").strip().lower()
        return user_input == "p"

    def translate_command_to_actions(self, command):
        """
        Uses GPT-4 via the OpenAI API to translate a text command into an action
        for the EGO agent (player 0) in Overcooked.
        Allowed actions:
            0: up, 1: down, 2: left, 3: right, 4: interact, -1: noop.
        Returns only the corresponding integer.
        """
        if not openai.api_key:
            logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return -1  # Stay if API key is missing

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": (
                        "You are an assistant that translates text commands into a simple action number for controlling "
                        "the EGO agent (player 0) in Overcooked. Allowed actions are: up (0), down (1), left (2), "
                        "right (3), interact (4), and noop (-1). Return only the corresponding integer."
                    )},
                    {"role": "user", "content": command}
                ],
                temperature=0.0,
            )
            action_text = response.choices[0].message['content'].strip()
            action = int(action_text)
            return action
        except Exception as e:
            logger.error(f"Error translating command '{command}': {e}")
            return -1

    def apply_human_action(self, agent_id, action):
        """
        For now, the human overrides action directly for the specified agent.
        """
        if agent_id == 0:
            self.human_override_action = action
            logger.info(f"Human override for agent {agent_id} set to action {action}")

    def process_human_commands(self):
        """
        Check the input queue and process any human commands in real time.
        If a command is received:
          - If the command is "resume", clear the override.
          - Otherwise, translate the command via GPT-4 and set the override.
        """
        while not self.human_input_queue.empty():
            command = self.human_input_queue.get()
            if command == "resume":
                logger.info("Received 'resume' command. Clearing human override.")
                self.human_override_action = None
            else:
                action = self.translate_command_to_actions(command)
                logger.info(f"Real-time command '{command}' translated to action {action} for agent 0.")
                self.apply_human_action(agent_id=0, action=action)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            # Optionally, you can still check for a pause trigger at episode boundaries.
            if self.check_for_pause_trigger():
                self.paused = True
                self.human_intervention_loop()

            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Process any human commands in real time.
                self.process_human_commands()

                # Sample actions from agents.
                (values,
                 actions,
                 action_log_probs,
                 rnn_states,
                 rnn_states_critic) = self.collect(step)

                # If a human override is active for the EGO agent, apply it.
                if self.human_override_action is not None:
                    for i in range(len(actions)):
                        actions[i][0] = self.human_override_action
                    # Clear the override once applied.
                    self.human_override_action = None

                # Obtain next observations, rewards, etc.
                (obs,
                 share_obs,
                 rewards,
                 dones,
                 infos,
                 available_actions) = self.envs.step(actions)
                obs = np.stack(obs)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)
                data = (obs,
                        share_obs,
                        rewards,
                        dones,
                        infos,
                        available_actions,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic)

                # Insert data into the buffer.
                self.insert(data)

            # Compute returns and update network.
            self.compute()
            train_infos = self.train(total_num_steps)

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # Save the model based on episode count.
            if episode < 50:
                if episode % 2 == 0:
                    self.save(total_num_steps)
            elif episode < 100:
                if episode % 5 == 0:
                    self.save(total_num_steps)
            else:
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save(total_num_steps)

            # Logging.
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                eta_t = eta(start, end, self.num_env_steps, total_num_steps)
                logger.info(
                    "Layout {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, ETA {}.".format(
                        self.all_args.layout_name,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                        eta_t,
                    )
                )

                for a in range(self.num_agents):
                    train_infos[a]["average_episode_rewards"] = np.mean(self.buffer[a].rewards) * self.episode_length
                    logger.info("agent {} average episode rewards is {}".format(
                        a, train_infos[a]["average_episode_rewards"])
                    )

                env_infos = defaultdict(list)
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)
                if self.env_name == "Overcooked":
                    if self.all_args.overcooked_version == "old":
                        from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
                        shaped_info_keys = SHAPED_INFOS
                    else:
                        from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
                        shaped_info_keys = SHAPED_INFOS

                    for info in infos:
                        for a in range(self.num_agents):
                            env_infos[f"ep_sparse_r_by_agent{a}"].append(
                                info["episode"]["ep_sparse_r_by_agent"][a])
                            env_infos[f"ep_shaped_r_by_agent{a}"].append(
                                info["episode"]["ep_shaped_r_by_agent"][a])
                            if "ep_hidden_r_by_agent" in info["episode"]:
                                env_infos[f"ep_hidden_r_by_agent{a}"].append(
                                    info["episode"]["ep_hidden_r_by_agent"][a])
                            for i, k in enumerate(shaped_info_keys):
                                env_infos[f"ep_{k}_by_agent{a}"].append(
                                    info["episode"]["ep_category_r_by_agent"][a][i])
                        env_infos["ep_sparse_r"].append(info["episode"]["ep_sparse_r"])
                        env_infos["ep_shaped_r"].append(info["episode"]["ep_shaped_r"])

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                logger.info(f'average sparse rewards is {np.mean(env_infos["ep_sparse_r"]):.3f}')

            # Evaluation step.
            if (episode % self.eval_interval == 0 and self.use_eval) or episode == episodes - 1:
                self.eval(total_num_steps)

        # Final cleanup.
        if self.envs is not None:
            self.envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()

    def human_intervention_loop(self):
        """
        (Legacy intervention loop)
        This loop can still be triggered explicitly (e.g., via the pause trigger).
        """
        print("Game paused. You are now controlling the EGO agent (player 0).")
        print("Enter text commands to change its behavior (e.g., 'up', 'interact', 'left', etc.).")
        print("Type 'resume' to continue the game.")
        while True:
            command = input("Enter command for player 0: ").strip().lower()
            if command == "resume":
                print("Resuming game...")
                break
            else:
                action = self.translate_command_to_actions(command)
                print(f"Translated command '{command}' to action {action} for player 0.")
                self.apply_human_action(agent_id=0, action=action)
        self.paused = False

    def warmup(self):
        # Reset the environment.
        obs, share_obs, available_actions = self.envs.reset()
        obs = np.stack(obs)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()
