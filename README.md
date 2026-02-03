# HINT-Agent Demo - macOS Setup Guide

This guide will help you set up and run the `hint_agent_overcooked_demo.py` on macOS. The demo features an interactive Overcooked game where you can play alongside an AI agent (HINT-Agent) and provide real-time interventions to guide its behavior.

## Prerequisites

- macOS
- Conda (miniforge3 recommended)
- OpenAI API key

## Step 1: Create Conda Environment

1. **Navigate to the project root:**
   ```bash
   cd /path/to/AHTHumanIntervention
   ```

2. **Activate conda (if using miniforge3):**
   ```bash
   source ~/miniforge3/etc/profile.d/conda.sh
   ```

3. **Create the conda environment from the macOS-specific environment file:**
   ```bash
   conda env create -f environment_mac.yml
   ```

4. **Activate the environment:**
   ```bash
   conda activate AHT_human_intervention_import
   ```

   **Note:** If you're using miniforge3, you may need to run:
   ```bash
   source ~/miniforge3/etc/profile.d/conda.sh && conda activate AHT_human_intervention_import
   ```

The `requirements.txt` file includes only the packages needed to run the demo (numpy, pygame, openai, jsonschema, gym, scipy, tqdm), making the installation lightweight and fast.

## Step 2: Set Up OpenAI API Key

The demo requires an OpenAI API key to use GPT models.

### Environment Variable
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Step 3: Run the Demo

Navigate to the demos directory and run the script:

```bash
cd AHT_human_intervention/demos
python hint_agent_overcooked_demo.py
```

### Command-Line Arguments

Common flags:
- `--layout` (default `counter_circuit`) or `--layout_name` (direct layout name)
- `--teammate` (default `onion_specialist`)
- `--horizon` (default `200`)
- random start is on by default; use `--fixed_start` to disable
- `--no_cot`, `--no_memory`, `--ego_variant`

Examples:
```bash
python hint_agent_overcooked_demo.py --layout cramped_room --teammate onion_specialist
python hint_agent_overcooked_demo.py --no_cot --no_memory --layout counter_circuit
```

## Step 4: Game Controls

Once the game window opens, you can use the following controls:

- **P** - Start an intervention (type a command to guide the agent)
- **ESC** - Quit the game

### Providing Interventions

Press **P** to enter intervention mode, then type a command and press **Enter**. Example interventions:

- `"Focus on cooking onions first"`
- `"Help your teammate by delivering soup"`
- `"You seem to be stuck for a while"`
- `"Your teammate is idle now"`

---

# CrowdNav Demo - Simulation Guide

This guide explains how to run the CrowdNav-AHT demo simulation, which features a robot navigating through a crowd of humans with AI intervention capabilities.

## Prerequisites

The CrowdNav demo shares the same prerequisites as the HINT-Agent demo:
- macOS (recommended) or Linux
- Conda (miniforge3 recommended)
- OpenAI API key (same setup as Step 2 above)

## Running the CrowdNav Demo

Navigate to the demos directory and run the script:

```bash
cd AHT_human_intervention/demos
python crowdnav_demo.py
```

### Command-Line Arguments

Common flags:
- `--horizon` (default `20`)
- `--seed` (default `6`)
- `--human_num` (default `4`)
- `--circle_radius` (default `6.0`)
- `--human_v_pref` (default `1.0`)
- `--robot_v_pref` (default `1.0`)
- random start is on by default; use `--fixed_start` to disable
- `--no_cot`, `--no_memory`
- `--teammate_style` (`aggressive|conservative|switching`, default `conservative`)
- `--switch_prob` (only used when style is `switching`, default `0.5`)
- `--ego_variant`

Example:
```bash
python crowdnav_demo.py --horizon 50 --seed 42 --human_num 6 --teammate_style switching --switch_prob 0.7
```

## CrowdNav Demo Controls

Once the simulation window opens, you can use the following controls:

- **P** - Pause/resume simulation and enter intervention mode
- **ESC** - Quit the simulation

### Providing Interventions

1. Press **P** to pause the simulation
2. Type your intervention command (e.g., `"Your teammate is almost there!"`)
3. Press **Enter** to apply the intervention and resume the simulation
4. Press **ESC** while typing to cancel the intervention

Example interventions:

- `"Avoid the crowded area to the left"`
- `"Take a more direct path to the goal"`
- `"Your teammate is at another meeting location"`

## License
This code is released under the Apache License 2.0.