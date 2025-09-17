#!/usr/bin/env python3
import argparse, json, os, random, time
from collections import defaultdict

import torch
import torch.nn.functional as F

# Local env & helpers (these filenames exist in your project)
import sys
sys.path.append('..')
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_state_converter import state_to_text   # provided text describer you used for training
from language_conditioned_policy import TextBasedLangConditionedPolicy, get_hf_tokenizer
from coordinated_agent import CoordinatedAgent

ACTIONS = ["STAY","LEFT","RIGHT","DOWN","UP","INTERACT"]  # must match training order

def make_overcooked_env(layout_name="random3"):
    """Factory function to create Overcooked environment."""
    mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=["any"])
    env = OvercookedEnv(mdp, horizon=200)  # Use 200 steps horizon
    return env

def make_teammate(teammate_type, mdp=None):
    """Factory function to create teammate agent."""
    if teammate_type == "coordinated":
        if mdp is None:
            # Create a default MDP if none provided
            mdp = OvercookedGridworld.from_layout_name("random3")
        return CoordinatedAgent(1, mdp)  # Player1 (teammate)
    else:
        raise ValueError(f"Unknown teammate type: {teammate_type}")

def pick_command(timing, t, episode_len, rng):
    """Return a command string (or None) based on the timing scheme."""
    if timing == "none": return None
    if timing == "early" and t == 5: return "Please fetch onions; I'll plate."
    if timing == "mid" and t == episode_len//2: return "Switch to serving; I'll cook."
    if timing == "late" and t == max(1, episode_len-20): return "Deliver soup now."
    if timing == "random" and rng.random() < 0.03:  # ~3% chance each step
        return rng.choice([
            "Focus on onions.",
            "I'll handle dishes; you cook.",
            "Go to the right pot.",
            "Pass me a plate at the window.",
        ])
    return None

def load_text_policy(dir_path, ckpt_path, device):
    tok = get_hf_tokenizer()
    policy = TextBasedLangConditionedPolicy.from_dir(dir_path, ckpt_path=ckpt_path)
    policy.to(device).eval()
    return policy, tok

def act_text_policy(policy, tok, state_text, command_text, device, max_len=256):
    if command_text:
        txt = state_text.strip() + " [COMMAND] " + command_text.strip()
    else:
        txt = state_text.strip()
    
    # Use the policy's forward method with the correct interface
    logits = policy.forward(None, [txt], [command_text] if command_text else [""])
    probs = F.softmax(logits, dim=-1).squeeze(0)
    action_idx = int(torch.argmax(probs).item())
    conf = float(probs[action_idx].item())
    
    # For now, use a simple heuristic: prefer STAY and INTERACT over movement
    # This is a temporary fix until we implement proper action validation
    if action_idx in [1, 2, 3, 4]:  # Movement actions
        # 50% chance to use STAY instead of movement
        import random
        if random.random() < 0.5:
            action_idx = 0  # STAY
            conf = 0.5
    
    return action_idx, conf

def run_episode(env, ego_policy, tok, teammate, timing, device, seed=0, max_len=256, max_steps=200):
    rng = random.Random(seed)
    env.reset()
    obs = env.state  # Get the state after reset
    t = 0
    done = False

    deliveries = 0
    steps_to_first = None
    collisions = 0
    idle = 0
    
    # Track intermediate progress
    progress = {
        "onion_pickups": 0,
        "dish_pickups": 0,
        "soup_cooking_started": 0,
        "soup_pickups": 0,
        "soup_deliveries": 0,
        "ego_holding_onion": 0,
        "ego_holding_dish": 0,
        "ego_holding_soup": 0,
        "teammate_holding_onion": 0,
        "teammate_holding_dish": 0,
        "teammate_holding_soup": 0,
        "pots_with_onions": 0,
        "cooking_soups": 0,
        "ready_soups": 0
    }

    current_cmd = None

    while not done and t < max_steps:
        state_text = state_to_text(obs)  # matches what you used for training
        # possibly update command
        new_cmd = pick_command(timing, t, env.horizon, rng)
        if new_cmd:
            current_cmd = new_cmd

        # ego action via text policy
        ego_act, conf = act_text_policy(ego_policy, tok, state_text, current_cmd, device, max_len=max_len)
        # teammate action (heuristic or loaded policy)
        tm_act = teammate.action(obs)

        # Convert teammate action to index format
        if isinstance(tm_act, tuple):
            if tm_act == (0, 0):
                tm_act_idx = 0  # STAY
            elif tm_act == (0, -1):
                tm_act_idx = 1  # UP
            elif tm_act == (0, 1):
                tm_act_idx = 2  # DOWN
            elif tm_act == (-1, 0):
                tm_act_idx = 3  # LEFT
            elif tm_act == (1, 0):
                tm_act_idx = 4  # RIGHT
        elif tm_act == "interact":
            tm_act_idx = 5  # INTERACT
        else:
            tm_act_idx = 0  # STAY

        # Convert ego action to tuple format
        if ego_act == 0:
            ego_act_tuple = (0, 0)  # STAY
        elif ego_act == 1:
            ego_act_tuple = (0, -1)  # UP
        elif ego_act == 2:
            ego_act_tuple = (0, 1)  # DOWN
        elif ego_act == 3:
            ego_act_tuple = (-1, 0)  # LEFT
        elif ego_act == 4:
            ego_act_tuple = (1, 0)  # RIGHT
        elif ego_act == 5:
            ego_act_tuple = "interact"  # INTERACT
        else:
            ego_act_tuple = (0, 0)  # STAY

        joint_action = [ego_act_tuple, tm_act]
        obs, reward, done, info = env.step(joint_action)

        # Track intermediate progress from info
        if info.get("delivered", False):
            deliveries += 1
            progress["soup_deliveries"] += 1
            if steps_to_first is None:
                steps_to_first = t + 1
                
        # Track shaped rewards which indicate progress
        shaped_info = info.get("shaped_info_by_agent", [{}])
        if len(shaped_info) > 0:
            ego_shaped = shaped_info[0]
            progress["onion_pickups"] += ego_shaped.get("pickup_onion_from_O", 0)
            progress["dish_pickups"] += ego_shaped.get("pickup_dish_from_D", 0)
            progress["soup_pickups"] += ego_shaped.get("SOUP_PICKUP", 0)
            progress["soup_cooking_started"] += ego_shaped.get("PLACEMENT_IN_POT", 0)
            
        # Track what agents are holding
        if obs.players and len(obs.players) >= 2:
            ego_holding = obs.players[0].held_object
            tm_holding = obs.players[1].held_object
            
            if ego_holding and "onion" in str(ego_holding):
                progress["ego_holding_onion"] += 1
            elif ego_holding and "dish" in str(ego_holding):
                progress["ego_holding_dish"] += 1
            elif ego_holding and "soup" in str(ego_holding):
                progress["ego_holding_soup"] += 1
                
            if tm_holding and "onion" in str(tm_holding):
                progress["teammate_holding_onion"] += 1
            elif tm_holding and "dish" in str(tm_holding):
                progress["teammate_holding_dish"] += 1
            elif tm_holding and "soup" in str(tm_holding):
                progress["teammate_holding_soup"] += 1
        
        # Track objects in the environment
        for obj_pos, obj in obs.objects.items():
            if "soup" in str(obj):
                if "onion" in str(obj) and "3" in str(obj):  # Soup with 3 onions
                    if "20" in str(obj):  # Fully cooked
                        progress["ready_soups"] += 1
                    else:  # Still cooking
                        progress["cooking_soups"] += 1
                else:  # Soup with onions but not 3 yet
                    progress["pots_with_onions"] += 1
                    
        collisions += int(info.get("collision", 0))
        idle += int(info.get("idle_ego", 0))
        t += 1

    if steps_to_first is None: steps_to_first = max_steps
    return {
        "deliveries": deliveries,
        "steps_to_first": steps_to_first,
        "collisions": collisions,
        "idle_steps": idle,
        "progress": progress
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_dir", type=str, default="../trained_policies/text_policy")
    ap.add_argument("--policy_ckpt", type=str, default="../trained_policies/text_policy/text_policy.pt")
    ap.add_argument("--layouts", type=str, default="random3")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--teammate", type=str, default="coordinated", help="coordinated")
    ap.add_argument("--command_timing", type=str, default="none", help="none,early,mid,late,random")
    ap.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--switch_threshold", type=float, default=0.75, help="(optional) if using intervention switcher")
    ap.add_argument("--save", type=str, default="rollout_results.json")
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    device = torch.device(args.device)
    layouts = [x.strip() for x in args.layouts.split(",")]
    rng = random.Random(0)

    # env factory
    results = []
    for layout in layouts:
        env = make_overcooked_env(layout_name=layout)
        # teammate factory
        teammate = make_teammate(args.teammate, env.mdp)

        # text policy
        policy, tok = load_text_policy(args.policy_dir, args.policy_ckpt, device)

        for ep in range(args.episodes):
            seed = rng.randint(0, 10_000_000)
            ep_res = run_episode(env, policy, tok, teammate, args.command_timing, device, seed, args.max_len, max_steps=200)
            ep_res["layout"] = layout
            ep_res["teammate"] = args.teammate
            ep_res["timing"] = args.command_timing
            results.append(ep_res)
            if (ep+1) % max(1, args.episodes//5) == 0:
                print(f"[{layout}] {ep+1}/{args.episodes} episodes done")

    # aggregate
    by_key = defaultdict(list)
    for r in results:
        key = (r["layout"], r["teammate"], r["timing"])
        by_key[key].append(r)

    summary = []
    for (layout, tm, timing), arr in by_key.items():
        n = len(arr)
        avg = lambda k: sum(x[k] for x in arr)/n
        avg_progress = lambda k: sum(x["progress"][k] for x in arr)/n
        summary.append({
            "layout": layout,
            "teammate": tm,
            "timing": timing,
            "episodes": n,
            "avg_deliveries": round(avg("deliveries"), 3),
            "avg_steps_to_first": round(avg("steps_to_first"), 3),
            "avg_collisions": round(avg("collisions"), 3),
            "avg_idle_steps": round(avg("idle_steps"), 3),
            "avg_progress": {
                "onion_pickups": round(avg_progress("onion_pickups"), 3),
                "dish_pickups": round(avg_progress("dish_pickups"), 3),
                "soup_cooking_started": round(avg_progress("soup_cooking_started"), 3),
                "soup_pickups": round(avg_progress("soup_pickups"), 3),
                "soup_deliveries": round(avg_progress("soup_deliveries"), 3),
                "ego_holding_onion": round(avg_progress("ego_holding_onion"), 3),
                "ego_holding_dish": round(avg_progress("ego_holding_dish"), 3),
                "ego_holding_soup": round(avg_progress("ego_holding_soup"), 3),
                "teammate_holding_onion": round(avg_progress("teammate_holding_onion"), 3),
                "teammate_holding_dish": round(avg_progress("teammate_holding_dish"), 3),
                "teammate_holding_soup": round(avg_progress("teammate_holding_soup"), 3),
                "pots_with_onions": round(avg_progress("pots_with_onions"), 3),
                "cooking_soups": round(avg_progress("cooking_soups"), 3),
                "ready_soups": round(avg_progress("ready_soups"), 3)
            }
        })
    out = {
        "args": vars(args),
        "summary": summary,
        "episodes": results[:50]  # keep first 50 epi details; adjust if you want all
    }
    with open(args.save, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] wrote {args.save}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
