import os
import sys
# Ensure project root is on sys.path so `envs/...` imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import json
import random
from transformers import AutoTokenizer, AutoModel
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from language.train_macro_policy import MacroPolicy, MACROS, MID
from macro_data_generator import MacroDataGenerator


def pick_macro(model: MacroPolicy, text: str, legal_mask):
    with torch.no_grad():
        logits = model([text])
        mask = torch.tensor([legal_mask], dtype=torch.float32, device=logits.device)
        pred = (logits.masked_fill(mask==0, float("-inf")).argmax(dim=-1).item())
    return MACROS[pred]


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = MacroPolicy().to(device)
    ckpt_path = "trained_policies/macro_policy/macro_policy.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Train first with: python language/train_macro_policy.py")
        return
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    mdp = OvercookedGridworld.from_layout_name("random3")
    env = OvercookedEnv(mdp, horizon=200)
    gen = MacroDataGenerator("random3")

    # sample 10 decision states and print the model's choice
    for seed in range(10):
        s = gen._generate_single_sample(seed)
        pred = pick_macro(model, s.text, s.legal_macro_mask)
        print(f"[seed {seed}] gold={s.macro_id:12s}  pred={pred}")


if __name__ == "__main__":
    main()
