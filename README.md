# ProAgent Demo - macOS Setup Guide

This guide will help you set up and run the `proagent_demo.py` on macOS. The demo features an interactive Overcooked game where you can play alongside an AI agent (ProAgent) and provide real-time interventions to guide its behavior.

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

The demo requires an OpenAI API key to use GPT models. You have two options:

### Option A: Environment Variable (Recommended)
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Option B: Create Key File
Create a file named `openai_key.txt` in the project root directory:
```bash
echo "your-api-key-here" > openai_key.txt
```

**Important:** Make sure the file contains only the API key with no blank lines or extra whitespace.

## Step 3: Run the Demo

Navigate to the demos directory and run the script:

```bash
cd AHT_human_intervention/demos
python proagent_demo.py
```

### Command-Line Arguments

You can customize the demo with the following arguments:

#### Layout Selection

**Option 1: Use predefined layout keys**
```bash
python proagent_demo.py --layout <layout_key>
```

Available layout keys:
- `counter_circuit` (default) - Maps to "random3"
- `forced_coordination` - Maps to "random0"
- `cramped_room` - Maps to "simple"
- `coordination_ring` - Maps to "random1"
- `asymmetric_advantages` - Maps to "unident_s"

**Option 2: Use direct Overcooked layout name**
```bash
python proagent_demo.py --layout_name <direct_layout_name>
```

This overrides the `--layout` argument. Examples: `random3`, `random0`, `simple`, `random1`, `unident_s`. You may find all these layouts under ```shared/envs/envs/overcooked/overcooked_ai_py/layouts```

#### Teammate Selection

Choose your AI teammate using the `--teammate` argument:

```bash
python proagent_demo.py --teammate <teammate_type>
```

Available teammates:
- `onion_specialist` (default) - Specializes in collecting and preparing onions
- `dish_specialist` - Specializes in handling dishes
- `greedy` - Uses a greedy heuristic strategy
- `random` - Takes random actions
- `stay` - Stays in place (useful for testing)
- `hand_coded` - Uses a simple hand-coded strategy

#### Horizon (Episode Length)

Set the maximum number of steps in the episode:

```bash
python proagent_demo.py --horizon <number>
```

Default: `400` steps

#### Random Start Positions

Enable random starting positions for players:

```bash
python proagent_demo.py --random_start
```

By default, players start at fixed positions defined by the layout.

#### Ablation Study Options

Control the Chain-of-Thought (CoT) reasoning and memory modules for ablation studies:

**Disable Chain-of-Thought reasoning:**
```bash
python proagent_demo.py --no_cot
```

**Disable Memory module:**
```bash
python proagent_demo.py --no_memory
```

**Disable both (Baseline mode):**
```bash
python proagent_demo.py --no_cot --no_memory
```

### Example Commands

**Basic run with defaults:**
```bash
python proagent_demo.py
```

**Custom layout and teammate:**
```bash
python proagent_demo.py --layout cramped_room --teammate dish_specialist
```

**Full customization:**
```bash
python proagent_demo.py --layout coordination_ring --teammate greedy --horizon 500 --random_start
```

**Using direct layout name:**
```bash
python proagent_demo.py --layout_name random3 --teammate onion_specialist --horizon 300
```

**Ablation study examples:**
```bash
# Baseline mode (no CoT, no memory)
python proagent_demo.py --no_cot --no_memory --layout counter_circuit

# CoT only (reasoning without memory)
python proagent_demo.py --no_memory --layout counter_circuit

# Memory only (no explicit reasoning display)
python proagent_demo.py --no_cot --layout counter_circuit
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

---

# CrowdNav Demo - Simulation Guide

This guide explains how to run the CrowdNav-AHT demo simulation, which features a robot navigating through a crowd of humans with AI intervention capabilities.

## Prerequisites

The CrowdNav demo shares the same prerequisites as the ProAgent demo:
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

You can customize the demo with the following arguments:

#### Horizon (Episode Length)

Set the maximum number of steps in the episode:

```bash
python crowdnav_demo.py --horizon <number>
```

Default: `200` steps

#### Random Seed

Set the random seed for environment initialization:

```bash
python crowdnav_demo.py --seed <number>
```

Default: `6`

#### Background Agent Configuration

Control the simulation environment parameters:

**Number of humans (total, includes teammate + background agents):**
```bash
python crowdnav_demo.py --human_num <number>
```
Default: `4`

**Human number variation range:**
```bash
python crowdnav_demo.py --human_num_range <number>
```
Actual number will be (human_num - range) to (human_num + range). Default: `2`

**Circle radius (meters) - starting area:**
```bash
python crowdnav_demo.py --circle_radius <float>
```
Default: `6.0` meters

**Human agent radius (meters):**
```bash
python crowdnav_demo.py --human_radius <float>
```
Default: `0.3` meters

**Human maximum velocity (m/s):**
```bash
python crowdnav_demo.py --human_v_pref <float>
```
Default: `1.0` m/s

**Robot radius (meters):**
```bash
python crowdnav_demo.py --robot_radius <float>
```
Default: `0.3` meters

**Robot maximum velocity (m/s):**
```bash
python crowdnav_demo.py --robot_v_pref <float>
```
Default: `1.0` m/s

### Example Commands

**Basic run with defaults:**
```bash
python crowdnav_demo.py
```

**Custom horizon and seed:**
```bash
python crowdnav_demo.py --horizon 300 --seed 42
```

**Customized environment:**
```bash
python crowdnav_demo.py --human_num 6 --circle_radius 8.0 --human_v_pref 1.5
```

**Full customization:**
```bash
python crowdnav_demo.py --horizon 500 --seed 10 --human_num 5 --human_num_range 1 --circle_radius 7.0 --robot_v_pref 1.2
```

## CrowdNav Demo Controls

Once the simulation window opens, you can use the following controls:

- **P** - Pause/resume simulation and enter intervention mode
- **ESC** - Quit the simulation

### Providing Interventions

1. Press **P** to pause the simulation
2. Type your intervention command (e.g., `"Avoid the crowded area to the left"`)
3. Press **Enter** to apply the intervention and resume the simulation
4. Press **ESC** while typing to cancel the intervention

Example interventions:

- `"Avoid the crowded area to the left"`
- `"Take a more direct path to the goal"`
- `"Slow down and wait for the humans to pass"`
- `"You're getting too close to the humans"`

The demo displays real-time information including:
- Current step number
- Chain-of-thought reasoning
- Intervention reasons
- Action plans
- Current action being taken

The visualization shows:
- Robot position and trajectory (in red)
- Human agent positions and trajectories (in blue)
- Goal location
- Obstacles and environment boundaries