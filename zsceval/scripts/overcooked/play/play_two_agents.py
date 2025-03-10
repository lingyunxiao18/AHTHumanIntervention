#!/usr/bin/env python
"""
Play script for testing real-time human intervention when two independent Overcooked agents are playing.
This script loads two agent checkpoints, sets up the environment, and runs the game loop.
Press "p" (or type "p" in real time) to pause and intervene with text commands.
"""
import os
import sys
import socket
from pathlib import Path

import torch
import wandb
import setproctitle
from loguru import logger

from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args
from zsceval.envs.env_wrappers import ShareSubprocDummyBatchVecEnv
from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from zsceval.utils.train_util import setup_seed

def make_play_env(all_args, run_dir):
    # Create playing environments.
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                # Use the appropriate version based on configuration.
                if all_args.overcooked_version == "old":
                    env = Overcooked(all_args, run_dir, rank=rank, evaluation=False)
                else:
                    env = Overcooked_new(all_args, run_dir, rank=rank)
            else:
                raise NotImplementedError(f"Environment {all_args.env_name} not supported.")
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    return ShareSubprocDummyBatchVecEnv(
        [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)],
        all_args.dummy_batch_size,
    )

def parse_args(args, parser):
    # Add only the essential arguments for playing.
    parser = get_overcooked_args(parser)
    parser.add_argument("--agent0_model", type=str, required=True,
                        help="Checkpoint (.pt file) for agent 0 (ego agent).")
    parser.add_argument("--agent1_model", type=str, required=True,
                        help="Checkpoint (.pt file) for agent 1.")
    parser.add_argument("--play_result_path", type=str, required=False,
                        help="File path to optionally save play results.", default="play_results.json")
    all_args = parser.parse_args(args)
    from zsceval.overcooked_config import OLD_LAYOUTS
    all_args.old_dynamics = all_args.layout_name in OLD_LAYOUTS
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # For independent agents, share_policy should be False.
    if all_args.share_policy:
        logger.error("share_policy must be False for independent agents.")
        sys.exit(1)

    # Setup device.
    device = torch.device("cuda:0") if (all_args.cuda and torch.cuda.is_available()) else torch.device("cpu")
    torch.set_num_threads(all_args.n_training_threads)
    if all_args.cuda and torch.cuda.is_available() and all_args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Setup run directory.
    run_dir = Path(os.path.expanduser("~")) / "ZSC" / "results" / all_args.env_name / all_args.layout_name / all_args.algorithm_name / all_args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    all_args.run_dir = run_dir

    # Initialize wandb if enabled.
    if all_args.use_wandb:
        run = wandb.init(config=all_args, project=all_args.env_name, entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=f"{all_args.algorithm_name}_{all_args.experiment_name}_seed{all_args.seed}",
                         group=all_args.layout_name, dir=str(run_dir),
                         job_type="play", reinit=True)

    setproctitle.setproctitle(f"{all_args.algorithm_name}-{all_args.env_name}_{all_args.layout_name}-{all_args.experiment_name}@{all_args.user_name}")
    setup_seed(all_args.seed)

    # Create playing environments.
    play_envs = make_play_env(all_args, run_dir)

    config = {
        "all_args": all_args,
        "envs": play_envs,
        "eval_envs": play_envs,  
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # Use the intervention-enabled runner.
    from zsceval.runner.separated.overcooked_runner_intervention import OvercookedRunner as Runner
    runner = Runner(config)

    # Load checkpoints into independent trainers.
    state_dict0 = torch.load(all_args.agent0_model, map_location=device)
    state_dict1 = torch.load(all_args.agent1_model, map_location=device)
    runner.trainer[0].policy.actor.load_state_dict(state_dict0)
    runner.trainer[1].policy.actor.load_state_dict(state_dict1)

    logger.info("Starting play loop. You can intervene in real time during play.")
    runner.run()

    # Optionally, save play results.
    if all_args.play_result_path:
        with open(all_args.play_result_path, "w", encoding="utf-8") as f:
            import json
            json.dump({"result": "play session complete"}, f, indent=4)
        logger.info(f"Play results saved to {all_args.play_result_path}")

    play_envs.close()
    if all_args.use_wandb:
        run.finish()


if __name__ == "__main__":
    import sys
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")
    main(sys.argv[1:])
