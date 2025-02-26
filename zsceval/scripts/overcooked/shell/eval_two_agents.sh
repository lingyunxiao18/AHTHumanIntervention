#!/bin/bash
# Usage: ./eval_two_agents.sh <layout> <agent0_policy_name> <agent1_policy_name>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <layout> <agent0_policy_name> <agent1_policy_name>"
    exit 1
fi

layout=$1
agent0_policy_name=$2
agent1_policy_name=$3

# Determine which version of Overcooked to use based on the layout
if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || \
      "${layout}" == "random1" || "${layout}" == "random3" || \
      "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

env="Overcooked"
num_agents=2
algo="population"  # using the same evaluation framework

# Create a temporary population YAML file for our two policies.
yaml_dir="eval/eval_policy_pool/${layout}/two_agents"
mkdir -p ${yaml_dir}
population_yaml="${yaml_dir}/two_agents.yml"

cat > ${population_yaml} <<EOF
# Minimal population file for two agents
agent0: ${agent0_policy_name}
agent1: ${agent1_policy_name}
EOF

# Define directories for results and run
run_dir="$HOME/ZSC/results/${env}/${layout}/${algo}/eval_two_agents"
mkdir -p ${run_dir}
eval_result_path="eval/results/${layout}/${algo}/eval_two_agents.json"

# Set evaluation parameters
episode_length=400
n_eval_rollout_threads=20
eval_episodes=40
dummy_batch_size=2

# Execute the evaluation using the Python evaluation script
python eval.py \
  --env_name ${env} \
  --algorithm_name ${algo} \
  --experiment_name "eval_two_agents" \
  --layout_name ${layout} \
  --num_agents ${num_agents} \
  --seed 1 \
  --episode_length ${episode_length} \
  --n_eval_rollout_threads ${n_eval_rollout_threads} \
  --eval_episodes ${eval_episodes} \
  --eval_stochastic \
  --dummy_batch_size ${dummy_batch_size} \
  --use_proper_time_limits \
  --use_wandb \
  --population_yaml_path ${population_yaml} \
  --overcooked_version ${version} \
  --eval_result_path ${eval_result_path} \
  --agent0_policy_name ${agent0_policy_name} \
  --agent1_policy_name ${agent1_policy_name}
