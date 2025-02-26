#!/bin/bash
# Usage: ./eval_two_agents.sh <layout> <agent0_policy_name> <agent1_policy_name>

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <layout> <agent0_policy_name> <agent1_policy_name>"
    exit 1
fi

# Set basic parameters.
env="Overcooked"
layout=$1
agent0_policy_name=$2
agent1_policy_name=$3
num_agents=2
algo="population"

# Determine the version of Overcooked based on layout.
if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

# Set POLICY_POOL relative to this script's location.
export POLICY_POOL="$(cd "$(dirname "$0")/../.." && pwd)/policy_pool"
echo "POLICY_POOL is set to: $POLICY_POOL"

# Define directories for the output YAML and results.
yml_dir="$(cd "$(dirname "$0")/../../eval/eval_policy_pool/${layout}/results" && pwd)"
mkdir -p "${yml_dir}"

# Create a minimal population YAML file for the two agents.
# Adjust the inner paths (configs/ and models/) as appropriate for your repository.
population_yaml="${yml_dir}/two_agents.yml"
cat > "${population_yaml}" <<EOF
agent0:
  policy_config_path: "configs/${agent0_policy_name}_config.pkl"
  model_path: "models/${agent0_policy_name}_model.pt"
  featurize_type: "ppo"
  train: false
agent1:
  policy_config_path: "configs/${agent1_policy_name}_config.pkl"
  model_path: "models/${agent1_policy_name}_model.pt"
  featurize_type: "ppo"
  train: false
EOF

echo "Generated population YAML at: ${population_yaml}"

# Set evaluation parameters.
episode_length=400
n_eval_rollout_threads=20
eval_episodes=40
dummy_batch_size=2

# Set run directory and evaluation result path.
run_dir="$(cd "$(dirname "$0")/../../ZSC/results/${env}/${layout}/${algo}" && pwd)/eval_two_agents"
mkdir -p "${run_dir}"
eval_result_path="eval/results/${layout}/${algo}/eval_two_agents.json"

# Launch the evaluation. Here we use eval.py (adjust the path if needed).
python "$(cd "$(dirname "$0")/../../eval" && pwd)/eval.py" \
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
  --population_yaml_path "${population_yaml}" \
  --overcooked_version ${version} \
  --eval_result_path "${eval_result_path}" \
  --agent0_policy_name ${agent0_policy_name} \
  --agent1_policy_name ${agent1_policy_name}
