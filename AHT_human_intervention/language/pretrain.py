#!/usr/bin/env python

import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from AHT_human_intervention.intervention_LLM_module import process_command

from language_conditioned_policy import (
    build_env_prompt,
    LangConditionedPolicy,
    tokenize,
    VOCAB,
    MAX_LEN,
)

class InterventionDataset(Dataset):
    def __init__(self, mdp, examples_path):
        self.mdp = mdp
        self.examples = []
        # build map from action name to index
        self.act2idx = self.mdp.action_name_to_idx  
        with open(examples_path, 'r') as f:
            for line in f:
                ex = json.loads(line)
                # state is a flat list: convert to tensor
                st = torch.FloatTensor(ex['state'])
                cmd = ex['command']
                act = ex['action']
                # map action name to index, fallback to STAY if missing
                idx = self.act2idx.get(act, self.act2idx['stay'])
                self.examples.append((st, cmd, idx))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        state_vec, cmd, act_idx = self.examples[i]
        token_ids = tokenize(cmd, VOCAB, MAX_LEN)
        return state_vec, token_ids, act_idx

def collate_fn(batch):
    # batch: list of (state_vec, token_ids, act_idx)
    states, tokens, acts = zip(*batch)
    s = torch.stack(states, dim=0)
    t = torch.stack(tokens, dim=0)
    a = torch.LongTensor(acts)
    return s, t, a

def train(args):
    # 1) Load MDP to get dims and action mapping
    mdp = OvercookedGridworld.from_layout_name(args.layout)
    # 2) Create dataset & split
    ds = InterventionDataset(mdp, args.examples)
    n_train = int(len(ds)* (1-args.val_split))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds)-n_train])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3) Instantiate model
    example_state = torch.FloatTensor(mdp.lossless_state_encoding(mdp.get_standard_start_state())[0].flatten())
    state_dim = example_state.numel()
    num_actions = len(mdp.action_idx_to_name)
    model = LangConditionedPolicy(
        state_dim=state_dim,
        vocab_size=len(VOCAB),
        text_dim=args.text_dim,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.nlayers,
        max_len=MAX_LEN,
        num_actions=num_actions,
    )
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 4) Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss, total_acc, cnt = 0.0, 0.0, 0
        for s_b, t_b, a_b in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            s_b, t_b, a_b = s_b.to(device), t_b.to(device), a_b.to(device)
            logits = model(s_b, t_b)
            loss = F.cross_entropy(logits, a_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * s_b.size(0)
            preds = logits.argmax(dim=-1)
            total_acc += (preds == a_b).sum().item()
            cnt += s_b.size(0)
        train_loss = total_loss / cnt
        train_acc  = total_acc / cnt

        # Validation
        model.eval()
        v_loss, v_acc, v_cnt = 0.0, 0.0, 0
        with torch.no_grad():
            for s_b, t_b, a_b in val_loader:
                s_b, t_b, a_b = s_b.to(device), t_b.to(device), a_b.to(device)
                logits = model(s_b, t_b)
                loss = F.cross_entropy(logits, a_b)
                v_loss += loss.item() * s_b.size(0)
                preds = logits.argmax(dim=-1)
                v_acc += (preds == a_b).sum().item()
                v_cnt += s_b.size(0)
        val_loss = v_loss / v_cnt
        val_acc  = v_acc / v_cnt

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

    # 5) Save model
    ckpt_path = Path(args.save_path)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layout", type=str, default="random3")
    p.add_argument("--examples", type=str, required=True,
                   help="Path to examples.jsonl")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--text_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save_path", type=str, default="lang_policy_pretrained.pt")
    args = p.parse_args()

    train(args)
