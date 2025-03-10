# Real-Time Human Intervention Play Session

This repository follows the framework of ZSC-Eval (https://github.com/sjtu-marl/ZSC-Eval) and provides a framework for running two Overcooked agents in a live play session with real-time human intervention. In this setup, the ego agent (player 0, e.g., Cole) can be directly controlled by a human via text commands during gameplay. The commands are translated into in-game actions (for now) using GPT‑4 via the OpenAI API.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment Configuration](#environment-configuration)
- [Running a Play Session](#running-a-play-session)
- [Real-Time Intervention](#real-time-intervention)

## Prerequisites

- **Python Packages:**  
  Ensure the following Python packages are installed:
  - numpy
  - torch
  - wandb
  - loguru
  - openai
  - setproctitle
  - icecream

## Installation

1. **Create Environment:**  
    ```bash
    conda env create -f environment.yml
    ```

2. **Set OpenAI API Key:**  
   The play session uses GPT‑4 to translate human text commands into in-game actions. Set your OpenAI API key in your shell environment:
    ```bash
    export OPENAI_API_KEY="your_actual_openai_api_key" 
    ```

3. **Download Pre-trained Models:**
   Download the pre-trained models for all the baselines to 
   ```
   cd AHTHumanIntervention
   git clone https://huggingface.co/Leoxxxxh/ZSC-Eval-policy_pool policy_pool
   ```

## Environment Configuration

- **Overcooked Version:**  
  The play session supports different layouts. The provided shell script automatically determines whether to use the "old" or "new" Overcooked version based on the layout you choose.

- **Runner:**  
  The file `overcooked_runner_intervention.py` implements a real-time human intervention loop. It starts a background thread to capture user input continuously and checks for any commands at every game step.

- **Play Script:**  
  The `play_two_agents.py` script loads two agents and starts a continuous game loop by calling the runner's `run()` method. This loop processes human commands in real time so that you can intervene while the agents are playing.

## Running a Play Session

   The shell script `play_two_agents.sh` wraps the play session. To launch a session, run:
   ```bash
   bash play_two_agents.sh <layout> <agent0_relative_path> <agent1_relative_path>
   ```

   Example: 
   ```bash
   bash play_two_agents.sh small_corridor cole_checkpoint.pt fcp_checkpoint.pt
   ``` 

## Real-Time Intervention

- **Background Input Thread:**  
  The intervention-enabled runner starts a background thread that continuously reads user input from the terminal and places commands into a queue.

- **Command Processing:**  
  At every step in the game loop, the current runner checks the input queue:
  - If a command is present, it is translated via GPT‑4 into an action.
  - The translated action is applied as an override to the ego agent’s (player 0) action for that step.
  - If you type `"resume"`, any override is cleared so that the agent resumes its normal behavior.

- **Special Command Handling:**  
  If you wish to treat a specific command (e.g., `"p"`) as a full pause rather than sending it for translation, modify the command processing logic in the runner accordingly.

## Visualization (TODO)