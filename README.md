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

---

# Collaborative Gym (Co-Gym) — AHT × HINT paper arc

HINT-Agent is the **ego** under ad hoc teamwork: the **peer teammate** type is unknown a priori, and an **outside supervisor** may inject sparse free-text guidance (CoT + Memory). Teammates take **task actions only** (no critique chat). CoGym's native `simulated_user` remains available as a **legacy** baseline.

## Paper roles

| Role | Who | Behavior |
|---|---|---|
| Ego | `HINTAgentCoGym` | Plans with CoT+Memory; does not know teammate type |
| Teammate | `PeerTeammateAgent` persona | `idle` / `complementary_searcher` / `greedy_editor` / `follower` — actions only |
| Outside supervisor | Redis CLI or oracle | Complementary prefs/constraints; applied on next `get_action` (no pause) |

## Prerequisites

- macOS or Linux, Conda Python 3.11, Redis (`conda install -c conda-forge redis-server` or Docker), OpenAI API key
- Vendored clone: `third_party/collaborative-gym/`
- TravelPlanner DB unzipped under `third_party/collaborative-gym/datasets/TravelPlanner/database/` (see that folder's README)
- `pip install -r third_party/collaborative-gym/requirements.txt` in env `cogym`
- Secrets in `third_party/collaborative-gym/secrets.example.toml` or `secrets.toml` (**do not commit real keys**)

## Demo (default: peer teammate + human CLI supervisor)

```bash
conda activate cogym
# Redis: redis-server --daemonize yes   # or docker redis-stack
python -m AHT_human_intervention.demos.hint_agent_cogym_demo \
  --task travel_planning --idx 0 --model gpt-5-mini \
  --teammate-persona complementary_searcher \
  --supervisor-mode human_cli
```

In a second terminal:

```bash
python AHT_human_intervention/hintagent/src/hintagent/supervisor_cli.py \
  --env-uuid env_<printed_uuid> --redis-url redis://localhost:6379/0
```

### Supervisor modes

```bash
--supervisor-mode off          # no-intervention baseline
--supervisor-mode oracle       # scripted hints from TravelPlanner hidden prefs (--max-interventions K)
--supervisor-mode human_cli    # you type interventions (default)
```

### Teammate personas

```bash
--teammate-persona idle
--teammate-persona complementary_searcher   # default
--teammate-persona greedy_editor
--teammate-persona follower
--teammate-persona legacy_simulated_user    # CoGym critic-user (legacy)
```

### Cross-episode memory

```bash
python -m AHT_human_intervention.demos.hint_agent_cogym_demo \
  --supervisor-mode oracle --memory-path /tmp/hint_patterns.json ...
```

Patterns load at ego start and merge-save on `end()`.

### Ablation / sweep harness

```bash
python -m AHT_human_intervention.demos.run_cogym_aht_sweep \
  --personas idle,complementary_searcher,greedy_editor,follower \
  --idxs 0 \
  --conditions no_intervention,oracle_no_memory,oracle_with_memory \
  --sweep-tag pilot
```

Writes `AHT_human_intervention/run_logs/cogym/sweeps/<tag>/summary.csv` and `summary.json`
(performance_rating, interventions, event histogram, hint-before-editor, etc.).

## Key files

| File | Role |
|---|---|
| `hintagent/src/hintagent/hint_agent_cogym.py` | Ego + supervisor inbox + `--memory-path` |
| `hintagent/src/hintagent/teammates/peer_personas.py` | Action-only personas |
| `hintagent/src/hintagent/teammates/peer_teammate_launcher.py` | Peer AgentNode launcher |
| `hintagent/src/hintagent/oracle_supervisor.py` | Scripted outside supervisor |
| `hintagent/src/hintagent/supervisor_cli.py` | Human CLI |
| `configs/teams/hint_agent_cogym_supervisor_team_config.toml` | Default: ego + peer agent |
| `configs/teams/hint_agent_cogym_simulated_user_team_config.toml` | Legacy simulated_user |
| `demos/hint_agent_cogym_demo.py` | Session orchestrator |
| `demos/run_cogym_aht_sweep.py` | Persona × condition sweep |

## Architecture map

| HINT module | Co-Gym mapping |
|---|---|
| Planner (CoT + Memory) | One full action string |
| Verifier | Regex `fullmatch` + retry → `WAIT_TEAMMATE_CONTINUE()` |
| Events | `lack_of_progress`, `repeated_action`, `stalled_teammate`, `supervisor_intervention` |
| Human channel | Redis outside-supervisor inbox (not teammate chat) |
| Memory | `intervention_patterns` (+ optional cross-episode `--memory-path`) |

## License
This code is released under the Apache License 2.0.