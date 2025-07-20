#!/usr/bin/env python
# play_language_policies_intervention.py

import os
import sys
import pygame
import time
import numpy as np
import torch
import openai
import json

# --- Direct import for OvercookedEnv ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
# --- Import Overcooked components ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
# --- Import visualization for rendering ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
# --- Import AgentPair class ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.agents.agent import AgentPair

# --- Our language-conditioned policy module ---
from AHT_human_intervention.language.language_conditioned_policy import (
    build_env_prompt,
    LangConditionedPolicy,
    tokenize,
    VOCAB,
    MAX_LEN,
)

# Configure OpenAI API key (ensure OPENAI_API_KEY is set in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------------------------------------------------------------------------
# Helpers: wrap text and convert actions
# ----------------------------------------------------------------------------
def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    return lines


def convert_action(a):
    if isinstance(a, tuple) and len(a) == 2:
        mapping = {
            (0, 0): Action.STAY,
            (0, 1): Direction.SOUTH,
            (0, -1): Direction.NORTH,
            (1, 0): Direction.EAST,
            (-1, 0): Direction.WEST,
        }
        return mapping.get(a, a)
    return a

# ----------------------------------------------------------------------------
def main():
    # Configuration
    LAYOUT_NAME = "random3"
    HORIZON = 400
    FPS = 5

    # 1) Load MDP and Env
    mdp = OvercookedGridworld.from_layout_name(LAYOUT_NAME)
    env = OvercookedEnv(mdp, horizon=HORIZON)
    env.reset()

    # 2) Visualizer
    visualizer = StateVisualizer(grid=mdp.terrain_mtx)

    # 3) Instantiate language-conditioned policy
    start_state = env.state
    state_dim = mdp.lossless_state_encoding(start_state)[0].flatten().size
    num_actions = len(mdp.action_idx_to_name)
    policy = LangConditionedPolicy(
        state_dim=state_dim,
        vocab_size=len(VOCAB),
        text_dim=128,
        hidden_dim=256,
        nhead=4,
        num_layers=2,
        max_len=MAX_LEN,
        num_actions=num_actions,
    )

    # load the checkpoint if pretrained policy is available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ckpt = torch.load("checkpoints/lang_pretrained.pt", map_location=device)
    # policy.load_state_dict(ckpt)
    # policy.to(device)
    # policy.eval()
    # print(f"✅ Loaded pretrained policy from checkpoints/lang_pretrained.pt on {device}")

    # 4) Wrap into agents with fallback logic
    class LangAgent:
        def __init__(self, policy, mdp, idx):
            self.policy = policy
            self.mdp = mdp
            self.idx = idx
            self.current_cmd = ""

        def act(self, state):
            cmd = self.current_cmd.strip().lower()
            # Fallback to LLM if any token is out-of-vocab
            tokens = cmd.split()
            if any(tok not in VOCAB for tok in tokens):
                prompt = build_env_prompt(state) + "\nHuman: " + self.current_cmd + "\nAI:"
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.0
                )
                try:
                    action_dict = json.loads(resp.choices[0].message.content)
                    act_name = action_dict.get("action", "stay").upper()
                    return getattr(Action, act_name, Action.STAY)
                except Exception as e:
                    print("LLM fallback failed, using local policy.", e)
            # Otherwise use local transformer policy
            arr = self.mdp.lossless_state_encoding(state)[self.idx].flatten()
            s = torch.FloatTensor(arr).unsqueeze(0)
            prompt = build_env_prompt(state) + "\nHuman: " + self.current_cmd
            t = tokenize(prompt, VOCAB, MAX_LEN).unsqueeze(0)
            logits = self.policy(s, t)
            probs = torch.softmax(logits, dim=-1)
            act_idx = int(torch.argmax(probs, dim=-1)[0])
            return self.mdp.action_idx_to_action[act_idx]

    ego_agent = LangAgent(policy, mdp, idx=0)
    conf_agent = LangAgent(policy, mdp, idx=1)
    pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
    pair.set_mdp(mdp)

    # 5) Pygame setup
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    TB_H = 100
    screen = pygame.display.set_mode((WIDTH, HEIGHT + TB_H))
    pygame.display.set_caption("Overcooked: Language-Conditioned Intervention")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)

    input_text = ""
    show_tb = False
    step = 0

    # 6) Main loop
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif e.type == pygame.KEYDOWN:
                if show_tb:
                    if e.key == pygame.K_RETURN:
                        cmd = input_text.strip()
                        ego_agent.current_cmd = cmd
                        conf_agent.current_cmd = cmd
                        input_text = ""
                        show_tb = False
                    elif e.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += e.unicode
                else:
                    if e.key == pygame.K_p:
                        show_tb = True

        # 7) Render
        surf = visualizer.render_state(env.state, grid=None)
        gs = pygame.transform.scale(surf, (WIDTH, HEIGHT))
        screen.blit(gs, (0,0))
        pygame.draw.rect(screen, (200,200,200), (0,HEIGHT, WIDTH, TB_H))
        txt = "Enter cmd: " + input_text if show_tb else "Press 'p' to command"
        for i, line in enumerate(wrap_text(txt, font, WIDTH-20)):
            screen.blit(font.render(line, True, (0,0,0)), (10, HEIGHT+10 + i*30))
        pygame.display.flip()
        clock.tick(FPS)

        # 8) Simulation step
        if not show_tb:
            raw = pair.joint_action(env.state)
            ja = tuple(convert_action(a[0]) for a in raw)
            nxt, r, done, _ = env.step(ja)
            env.state = nxt
            step += 1
            if step == 20:
                print("Sim: step 20 reached—fallback logic active for unknown commands.")
            if done:
                print("Episode done, resetting.")
                env.reset(); step = 0

if __name__ == "__main__":
    main()
