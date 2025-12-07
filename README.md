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

## Step 4: Game Controls

Once the game window opens, you can use the following controls:

- **P** - Start an intervention (type a command to guide the agent)
- **ESC** - Quit the game

### Providing Interventions

Press **P** to enter intervention mode, then type a command and press **Enter**. Example interventions:

- `"Focus on cooking onions first"`
- `"Help your teammate by delivering soup"`
- `"You seem to be stuck for a while"`