### PARTNR environment integration (experimental)

This branch vendors the PARTNR planner environment as a git submodule for experimentation and future integration.

- Submodule path: `AHT_human_intervention/external/partnr-planner`
- Upstream repository: `https://github.com/facebookresearch/partnr-planner`

#### Getting started

1) Update and initialize submodules (after cloning this branch):

```bash
git submodule update --init --recursive
```

2) Follow the upstream installation guide for PARTNR. See their INSTALLATION and README docs:

- INSTALLATION: `https://github.com/facebookresearch/partnr-planner/blob/main/INSTALLATION.md`
- README: `https://github.com/facebookresearch/partnr-planner`

3) Example upstream commands (see upstream docs for details). After installing dependencies and datasets per upstream instructions, you can run demo planners such as:

```bash
python -m habitat_llm.examples.planner_demo --config-name baselines/single_agent_zero_shot_react_summary.yaml \
    habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
    evaluation.agents.agent_0.planner.plan_config.llm.inference_mode=hf \
    evaluation.agents.agent_0.planner.plan_config.llm.generation_params.engine=meta-llama/Meta-Llama-3-8B-Instruct
```

For multi-agent decentralized or centralized variants, see upstream README examples.

#### Notes

- This branch currently provides the submodule and documentation pointers; deeper adapters into this codebase will be added incrementally.
- If you modify the submodule, remember to push to the upstream fork or update the submodule reference here accordingly.


