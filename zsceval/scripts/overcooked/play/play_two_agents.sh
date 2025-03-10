#!/bin/bash
# play_two_agents.sh
#
# Shell wrapper for playing two independent agents with real-time human intervention.
# Usage: bash play_two_agents.sh <layout> <agent0_relative_path> <agent1_relative_path>

env="Overcooked"
layout=$1

# Set Overcooked version based on layout.
if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

num_agents=2
algo="population"  # Kept for configuration consistency.

# Base directory for checkpoints.
POLICY_POOL_PATH="$(dirname $(dirname $(realpath $0)))/policy_pool"
export POLICY_POOL=${POLICY_POOL_PATH}

# Construct full checkpoint paths.
agent0_model="${POLICY_POOL}/${layout}/$2"
agent1_model="${POLICY_POOL}/${layout}/$3"

echo "Using agent0 model: ${agent0_model}"
echo "Using agent1 model: ${agent1_model}"

python play_two_agents.py \
    --env_name ${env} \
    --layout_name ${layout} \
    --algorithm_name ${algo} \
    --experiment_name "play_two_agents_${layout}" \
    --user_name "your_username" \
    --seed 1 \
    --num_agents ${num_agents} \
    --n_eval_rollout_threads 20 \
    --n_training_threads 4 \
    --episode_length 400 \
    --dummy_batch_size 2 \
    --agent0_model ${agent0_model} \
    --agent1_model ${agent1_model} \
    --play_result_path "play/results/${layout}/${algo}/play_two_agents.json" \
    --overcooked_version ${version} \
    --use_wandb
