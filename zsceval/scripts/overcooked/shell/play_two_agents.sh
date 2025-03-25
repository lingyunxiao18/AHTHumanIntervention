#!/bin/bash
# eval_two_agents_dynamic.sh

env="Overcooked"
layout=$1

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

num_agents=2
algo="population"

# Agent 0: set algorithm and experiment options
if [[ $2 == "fcp" ]]; then
    algorithm_0="fcp"
    exps_0=("fcp-S2-s24" "fcp-S2-s36")
elif [[ $2 == "mep" ]]; then
    algorithm_0="mep"
    exps_0=("mep-S2-s24" "mep-S2-s36")
elif [[ $2 == "traj" ]]; then
    algorithm_0="traj"
    exps_0=("traj-S2-s24" "traj-S2-s36")
elif [[ $2 == "hsp" ]]; then
    algorithm_0="hsp"
    exps_0=("hsp-S2-s12" "hsp-S2-s24" "hsp-S2-s36")
elif [[ $2 == "cole" ]]; then
    algorithm_0="cole"
    exps_0=("cole-S2-s50" "cole-S2-s75")
else
    echo "Usage: bash $0 {layout} {algo0} {algo1}"
    exit 0
fi

# Agent 1: set algorithm and experiment options
if [[ $3 == "fcp" ]]; then
    algorithm_1="fcp"
    exps_1=("fcp-S2-s24" "fcp-S2-s36")
elif [[ $3 == "mep" ]]; then
    algorithm_1="mep"
    exps_1=("mep-S2-s24" "mep-S2-s36")
elif [[ $3 == "traj" ]]; then
    algorithm_1="traj"
    exps_1=("traj-S2-s24" "traj-S2-s36")
elif [[ $3 == "hsp" ]]; then
    algorithm_1="hsp"
    exps_1=("hsp-S2-s12" "hsp-S2-s24" "hsp-S2-s36")
elif [[ $3 == "cole" ]]; then
    algorithm_1="cole"
    exps_1=("cole-S2-s50" "cole-S2-s75")
else
    echo "Usage: bash $0 {layout} {algo0} {algo1}"
    exit 0
fi

# For two-agent evaluation we need one checkpoint per agent.
population_size=2

# (If necessary, you can adjust these numbers for other layouts.)
declare -A LAYOUTS_KS
LAYOUTS_KS["random0"]=10
LAYOUTS_KS["random0_medium"]=10
LAYOUTS_KS["random1"]=10
LAYOUTS_KS["random3"]=10
LAYOUTS_KS["small_corridor"]=10
LAYOUTS_KS["unident_s"]=10
LAYOUTS_KS["random0_m"]=15
LAYOUTS_KS["random1_m"]=15
LAYOUTS_KS["random3_m"]=15

# Set the policy pool path (the root directory for your policies)
path=../../policy_pool
export POLICY_POOL=${path}

# By default use the RNN configuration file (adjust later with sed if you want mlp)
config_file="policy_config/rnn_policy_config.pkl"

ulimit -n 65536

yml_dir=eval/eval_policy_pool/${layout}/results
mkdir -p ${yml_dir}

# Loop over experiment options for both agents and for seeds 1..5.
for exp0 in "${exps_0[@]}"; do
    for exp1 in "${exps_1[@]}"; do
        for seed in $(seq 1 5); do
            # In the evaluation YAML we use fixed keys "agent0" and "agent1"
            eval_exp="eval-${exp0}-${seed}-vs-${exp1}-${seed}"
            yml=${yml_dir}/${eval_exp}.yml

            # Construct the model paths for each agent.
            # (Assuming the checkpoint path is of the form: ${layout}/${algorithm}/s2/${experiment_name}/${seed}.pt)
            model_path0="${layout}/${algorithm_0}/s2/${exp0}/${seed}.pt"
            model_path1="${layout}/${algorithm_1}/s2/${exp1}/${seed}.pt"

            # Create the YAML file.
            cat > ${yml} <<EOF
agent0:
  policy_config_path: ${layout}/${config_file}
  model_path: ${model_path0}
  featurize_type: ppo
  train: false
agent1:
  policy_config_path: ${layout}/${config_file}
  model_path: ${model_path1}
  featurize_type: ppo
  train: false
EOF
            python eval/eval.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${eval_exp}" --layout_name "${layout}" \
            --num_agents ${num_agents} --seed ${seed} --episode_length 400 --n_eval_rollout_threads 20 --eval_episodes 40 --eval_stochastic --dummy_batch_size 2 \
            --use_proper_time_limits --use_wandb --population_yaml_path "${yml}" --population_size ${population_size} \
            --overcooked_version ${version} --eval_result_path "eval/results/${layout}/${algo}/${eval_exp}.json" \
            --agent0_policy_name "agent0" --agent1_policy_name "agent1"
        done
    done
done
