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
import argparse
from collections import deque
from AHT_human_intervention.intervention_LLM_module import process_command

# --- Direct import for OvercookedEnv ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
# --- Import Overcooked components ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
# --- Import visualization for rendering ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
# --- Import AgentPair class ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.agents.agent import AgentPair
# --- Import a simple heuristic agent for confederate switch demo ---
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.agents.agent import StayAgent
# --- Import interesting heuristic agents ---
from heuristic_agent import RotateAgent, OnionToPotAgent, PlateAgent

# --- Our language-conditioned policy module ---
from language.language_conditioned_policy import HuggingFaceLangConditionedPolicy
from transformers import DistilBertTokenizer
import torch

class TrainedPolicyAgent:
    """Agent that uses our trained language-conditioned policy."""
    
    def __init__(self, mdp, agent_idx=0, model_path=None):
        self.mdp = mdp
        self.agent_idx = agent_idx
        self.command = None
        self.heuristic = "No command set"
        
        # Set default model path if none provided
        if model_path is None:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "language", "test_policy_best.pth")
        
        # Load the trained policy
        print(f"[INFO] Loading trained policy from {model_path}")
        print(f"[DEBUG] Model path exists: {os.path.exists(model_path)}")
        print(f"[DEBUG] Model path is file: {os.path.isfile(model_path)}")
        print(f"[DEBUG] Model path is absolute: {os.path.isabs(model_path)}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize policy architecture
        state_dim = 20
        num_actions = 6
        self.policy = HuggingFaceLangConditionedPolicy(
            state_dim=state_dim,
            num_actions=num_actions,
            text_dim=768,
            hidden_dim=256,
            freeze_bert=False
        ).to(self.device)
        
        # Load trained weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            print(f"[INFO] Policy loaded successfully! Best validation accuracy: {checkpoint.get('val_acc', 'Unknown')}")
        except Exception as e:
            print(f"[WARNING] Could not load trained policy: {e}")
            print("[WARNING] Using untrained policy (random actions)")
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Set to evaluation mode
        self.policy.eval()
    
    def set_agent_index(self, agent_idx):
        self.agent_idx = agent_idx
    
    def set_mdp(self, mdp):
        self.mdp = mdp
    
    def set_command(self, command):
        """Set the human intervention command."""
        self.command = command
        self.heuristic = f"Following command: {command}"
        print(f"[INFO] Command set: {command}")
    
    def get_action(self, state):
        """Get action from the trained policy."""
        if not self.command:
            # No command set, return STAY
            return 0
        
        try:
            # Extract state features (simplified - you may need to adjust this)
            state_features = self._extract_state_features(state)
            state_tensor = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Tokenize command
            command_tokens = self.tokenizer(
                self.command,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            command_input_ids = command_tokens['input_ids'].to(self.device)
            command_attention_mask = command_tokens['attention_mask'].to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                logits = self.policy(state_tensor, command_input_ids, command_attention_mask)
                action = torch.argmax(logits, dim=1).item()
            
            return action
            
        except Exception as e:
            print(f"[ERROR] Policy inference failed: {e}")
            return 0  # STAY as fallback
    
    def action(self, state):
        """Method that AgentPair calls to get actions."""
        return self.get_action(state)
    
    def _extract_state_features(self, state):
        """Extract state features similar to training data."""
        # This is a simplified version - you may need to match the exact format from training
        features = []
        
        # Player positions and held objects
        for player in state.players:
            features.extend([float(player.position[0]), float(player.position[1])])
            features.append(1.0 if player.held_object else 0.0)
        
        # Object counts
        object_counts = {'onion': 0, 'dish': 0, 'soup': 0}
        for obj in state.objects.values():
            if obj.name in object_counts:
                object_counts[obj.name] += 1
        
        for obj_type in ['onion', 'dish', 'soup']:
            features.append(float(object_counts[obj_type]))
        
        # Layout features
        features.append(float(len(self.mdp.get_pot_locations())))
        features.append(float(len(self.mdp.get_counter_locations())))
        features.append(float(len(self.mdp.get_onion_dispenser_locations())))
        features.append(float(len(self.mdp.get_dish_dispenser_locations())))
        features.append(float(len(self.mdp.get_serving_locations())))
        
        # Timestep (normalized)
        features.append(float(state.timestep) / 400.0)
        
        # Pad to 20 features
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]

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
    """
    Convert various action formats to valid action constants.
    Valid motion actions can be:
      - Direction objects (NORTH, SOUTH, EAST, WEST)
      - Action objects (STAY, INTERACT)
      - Tuples: (0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)
      - Integers: 0-5
      - Strings: "interact"
    Returns the appropriate Action or Direction constant.
    """
    try:
        # If it's already a Direction or Action constant, return as is
        if isinstance(a, (Direction, Action)):
            return a
        
        # Handle integer actions from script agents
        if isinstance(a, int):
            # Handle negative integers (convert to Direction)
            if a == -1:
                return Direction.WEST
            
            mapping = {
                0: Action.STAY,
                1: Direction.EAST,
                2: Direction.SOUTH,
                3: Direction.NORTH,
                4: Direction.WEST,
                5: Action.INTERACT
            }
            if a in mapping:
                return mapping[a]
            raise ValueError(f"Invalid integer action: {a}")
        
        # Handle tuple actions from movement agents
        if isinstance(a, tuple) and len(a) == 2:
            mapping = {
                (0, 0): Action.STAY,
                (0, 1): Direction.SOUTH,
                (0, -1): Direction.NORTH,
                (1, 0): Direction.EAST,
                (-1, 0): Direction.WEST,
            }
            if a in mapping:
                return mapping[a]
            raise ValueError(f"Invalid tuple action: {a}")
        
        # Handle string actions (like "interact")
        if isinstance(a, str):
            if a.lower() == "interact":
                return Action.INTERACT
            raise ValueError(f"Invalid string action: {a}")
            
        raise ValueError(f"Unknown action type: {type(a)}, value: {a}")
    except Exception as e:
        print(f"[ERROR] Error converting action {a} (type: {type(a)}): {e}")
        # Return STAY action as fallback
        return Action.STAY

# ----------------------------------------------------------------------------
class MacroPolicyAgent:
    """
    Wraps a macro-action .pt policy and emits primitive actions by expanding each macro.
    Assumes MACROS = ["GO_TO_ONION","TAKE_ONION","GO_TO_POT","PUT_IN_POT",
                      "GO_TO_DISH","TAKE_DISH","GO_TO_SERVE","SERVE","WAIT_COOK","TAKE_SOUP"].
    """

    MACROS = ["GO_TO_ONION","TAKE_ONION","GO_TO_POT","PUT_IN_POT",
              "GO_TO_DISH","TAKE_DISH","GO_TO_SERVE","SERVE","WAIT_COOK","TAKE_SOUP"]

    def __init__(self, mdp, agent_idx=0, model_path=None, device=None):
        self.mdp = mdp
        self.agent_idx = agent_idx
        self.command = None
        self.heuristic = "MacroPolicy (no command)"
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self._pending_prims = deque()

        # Tokenizer for command text (reuse DistilBERT like your language agent)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "trained_policies", "macro_policy", "macro_policy.pt")

        self.model = None
        try:
            # Load the macro policy model
            from language.train_macro_policy import MacroPolicy
            self.model = MacroPolicy().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"[INFO] Loaded macro model: {model_path}")
        except Exception as e:
            print(f"[WARNING] Could not load macro policy from {model_path}: {e}")
            print("[WARNING] Falling back to random macros.")

    def set_agent_index(self, agent_idx): 
        self.agent_idx = agent_idx
        
    def set_mdp(self, mdp): 
        self.mdp = mdp

    def set_command(self, command: str):
        self.command = (command or "").strip()
        self.heuristic = f"MacroPolicy: {self.command}" if self.command else "MacroPolicy (no command)"
        print(f"[INFO] Macro command set: {self.command}")

    def _extract_state_features(self, state):
        """Reuse the same 20-d feature recipe used by TrainedPolicyAgent so your .pt can work out-of-the-box."""
        features = []
        for player in state.players:
            features.extend([float(player.position[0]), float(player.position[1])])
            features.append(1.0 if player.held_object else 0.0)

        # Object counts
        object_counts = {'onion': 0, 'dish': 0, 'soup': 0}
        for obj in state.objects.values():
            if obj.name in object_counts:
                object_counts[obj.name] += 1
        
        for obj_type in ['onion', 'dish', 'soup']:
            features.append(float(object_counts[obj_type]))

        # Layout info
        features.append(float(len(self.mdp.get_pot_locations())))
        features.append(float(len(self.mdp.get_onion_dispenser_locations())))
        features.append(float(len(self.mdp.get_dish_dispenser_locations())))
        features.append(float(len(self.mdp.get_serving_locations())))

        # Time step
        features.append(float(state.timestep) / 400.0)

        while len(features) < 20:
            features.append(0.0)
        return features[:20]

    def _format_state_for_model(self, state):
        """Format OvercookedState into text matching training data format."""
        player = state.players[self.agent_idx]
        mate = state.players[1 - self.agent_idx]
        
        # Basic scene info
        H, W = len(self.mdp.terrain_mtx), len(self.mdp.terrain_mtx[0])
        text = f"[SCENE] size {W}x{H} "
        
        # Player positions and items
        text += f"[EGO] pos {player.position} facing {self._direction_to_str(player.orientation)} "
        text += f"item {self._get_item_name(player.held_object)} "
        text += f"[MATE] pos {mate.position} facing {self._direction_to_str(mate.orientation)} "
        text += f"item {self._get_item_name(mate.held_object)} "
        
        # Goal and command
        text += f"[GOAL] macro_action | [COMMAND] {self.command or 'NO_CMD'} "
        
        # NEAREST - distances to key objects (match training format)
        pos = player.position
        onion_dispensers = self.mdp.get_onion_dispenser_locations()
        pot_locations = self.mdp.get_pot_locations()
        dish_dispensers = self.mdp.get_dish_dispenser_locations()
        serve_locations = self.mdp.get_serving_locations()
        
        nearest_parts = []
        
        if onion_dispensers:
            closest_onion = min(onion_dispensers, key=lambda p: abs(p[0]-pos[0]) + abs(p[1]-pos[1]))
            dr, dc = closest_onion[1] - pos[1], closest_onion[0] - pos[0]
            if dr == 0 and dc == 0:
                nearest_parts.append("onion Δr=0 Δc=0 (HERE)")
            else:
                dir_str = ""
                if dr < 0: dir_str += f"N{abs(dr)} "
                elif dr > 0: dir_str += f"S{dr} "
                if dc < 0: dir_str += f"W{abs(dc)}"
                elif dc > 0: dir_str += f"E{dc}"
                nearest_parts.append(f"onion Δr={dr} Δc={dc} ({dir_str.strip()})")
        
        if pot_locations:
            closest_pot = min(pot_locations, key=lambda p: abs(p[0]-pos[0]) + abs(p[1]-pos[1]))
            dr, dc = closest_pot[1] - pos[1], closest_pot[0] - pos[0]
            if dr == 0 and dc == 0:
                nearest_parts.append("pot Δr=0 Δc=0 (HERE)")
            else:
                dir_str = ""
                if dr < 0: dir_str += f"N{abs(dr)} "
                elif dr > 0: dir_str += f"S{dr} "
                if dc < 0: dir_str += f"W{abs(dc)}"
                elif dc > 0: dir_str += f"E{dc}"
                nearest_parts.append(f"pot Δr={dr} Δc={dc} ({dir_str.strip()})")
        
        if dish_dispensers:
            closest_dish = min(dish_dispensers, key=lambda p: abs(p[0]-pos[0]) + abs(p[1]-pos[1]))
            dr, dc = closest_dish[1] - pos[1], closest_dish[0] - pos[0]
            if dr == 0 and dc == 0:
                nearest_parts.append("dish Δr=0 Δc=0 (HERE)")
            else:
                dir_str = ""
                if dr < 0: dir_str += f"N{abs(dr)} "
                elif dr > 0: dir_str += f"S{dr} "
                if dc < 0: dir_str += f"W{abs(dc)}"
                elif dc > 0: dir_str += f"E{dc}"
                nearest_parts.append(f"dish Δr={dr} Δc={dc} ({dir_str.strip()})")
        
        if serve_locations:
            closest_serve = min(serve_locations, key=lambda p: abs(p[0]-pos[0]) + abs(p[1]-pos[1]))
            dr, dc = closest_serve[1] - pos[1], closest_serve[0] - pos[0]
            if dr == 0 and dc == 0:
                nearest_parts.append("serve Δr=0 Δc=0 (HERE)")
            else:
                dir_str = ""
                if dr < 0: dir_str += f"N{abs(dr)} "
                elif dr > 0: dir_str += f"S{dr} "
                if dc < 0: dir_str += f"W{abs(dc)}"
                elif dc > 0: dir_str += f"E{dc}"
                nearest_parts.append(f"serve Δr={dr} Δc={dc} ({dir_str.strip()})")
        
        text += f"[NEAREST] {'; '.join(nearest_parts)} "
        
        # LOCAL - simplified movement info (match training format)
        text += f"[LOCAL] ahead_blocked 0 junction 0 legal_moves N1 S1 E1 W1 "
        
        # LEGAL - legal actions (match training format)
        text += f"[LEGAL] MOVE_N MOVE_S MOVE_E MOVE_W INTERACT STAY "
        
        # Adjacency info
        adj_onion = 1 if any(self._is_adjacent(pos, d) for d in onion_dispensers) else 0
        adj_pot = 1 if any(self._is_adjacent(pos, p) for p in pot_locations) else 0
        adj_dish = 1 if any(self._is_adjacent(pos, d) for d in dish_dispensers) else 0
        adj_serve = 1 if any(self._is_adjacent(pos, s) for s in serve_locations) else 0
        holding = self._get_item_name(player.held_object)
        
        text += f"adj_onion={adj_onion} adj_pot={adj_pot} adj_dish={adj_dish} adj_serve={adj_serve} holding={holding} "
        
        # POTS - pot status (simplified for now)
        pot_states = []
        for pot_pos in pot_locations:
            pot_states.append(f"{pot_pos} empty")
        
        text += f"[POTS] {' | '.join(pot_states)}"
        
        return text
    
    def _direction_to_str(self, direction):
        """Convert direction tuple to string."""
        if direction == (0, -1): return "N"
        elif direction == (0, 1): return "S"
        elif direction == (1, 0): return "E"
        elif direction == (-1, 0): return "W"
        else: return "?"
    
    def _get_item_name(self, obj):
        """Get item name from object."""
        if obj is None: return "empty"
        return getattr(obj, 'name', 'unknown')
    
    def _is_adjacent(self, pos1, pos2):
        """Check if two positions are adjacent."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def _get_direction_to_face(self, from_pos, to_pos):
        """Get the direction to face from one position to another."""
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        if dx == 1 and dy == 0: return Direction.EAST
        if dx == -1 and dy == 0: return Direction.WEST
        if dx == 0 and dy == 1: return Direction.SOUTH
        if dx == 0 and dy == -1: return Direction.NORTH
        return Action.STAY

    def _predict_macro_idx(self, state):
        if self.model is None:
            return np.random.randint(0, len(self.MACROS))
        try:
            # Format state as text for the macro model
            text = self._format_state_for_model(state)
            
            # Tokenize and run model
            with torch.no_grad():
                logits_macro, _, _, _, _ = self.model.forward([text])
                return int(torch.argmax(logits_macro, dim=-1).item())
        except Exception as e:
            print(f"[ERROR] Macro policy forward failed: {e}")
            return np.random.randint(0, len(self.MACROS))

    @staticmethod
    def _neighbors(p):
        x, y = p
        return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

    def _valid_positions(self):
        try:
            return set(self.mdp.get_valid_player_positions())
        except Exception:
            # Fallback: any non-wall cell in terrain_mtx (X = wall)
            valids = set()
            T = self.mdp.terrain_mtx
            for y in range(len(T)):
                for x in range(len(T[0])):
                    if T[y][x] != 'X':
                        valids.add((x, y))
            return valids

    def _bfs_path(self, start, goal):
        valids = self._valid_positions()
        if start not in valids:
            return []
        q = deque([start]); parent = {start: None}
        while q:
            cur = q.popleft()
            if cur == goal: break
            for nb in self._neighbors(cur):
                if nb in valids and nb not in parent:
                    parent[nb] = cur
                    q.append(nb)
        if goal not in parent:
            return []
        # Reconstruct path
        path = []
        cur = goal
        while parent[cur] is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    @staticmethod
    def _step_to_direction(prev, nxt):
        dx, dy = (nxt[0]-prev[0], nxt[1]-prev[1])
        if dx == 1 and dy == 0: return Direction.EAST
        if dx == -1 and dy == 0: return Direction.WEST
        if dx == 0 and dy == 1: return Direction.SOUTH
        if dx == 0 and dy == -1: return Direction.NORTH
        return Action.STAY

    def _closest_adjacent_walkable(self, me, targets):
        """Pick a walkable cell adjacent to any target; choose the closest by Manhattan to 'me'."""
        valids = self._valid_positions()
        cands = []
        for t in targets:
            for nb in self._neighbors(t):
                if nb in valids:
                    cands.append(nb)
        if not cands:
            return None
        return min(cands, key=lambda c: abs(c[0]-me[0]) + abs(c[1]-me[1]))

    def _expand_macro(self, macro, state):
        pl = state.players[self.agent_idx]
        me = pl.position
        prims = []

        if macro == "GO_TO_ONION":
            targets = self.mdp.get_onion_dispenser_locations()
            goal = self._closest_adjacent_walkable(me, targets)
            if goal:
                path = [me] + self._bfs_path(me, goal)
                prims += [self._step_to_direction(path[i], path[i+1]) for i in range(len(path)-1)]
        elif macro == "TAKE_ONION":
            # Find the adjacent onion dispenser and face it
            targets = self.mdp.get_onion_dispenser_locations()
            for target in targets:
                if self._is_adjacent(me, target):
                    # Face the dispenser first, then interact
                    face_dir = self._get_direction_to_face(me, target)
                    if face_dir != Action.STAY:
                        prims += [face_dir]
                    prims += [Action.INTERACT]
                    break
        elif macro == "GO_TO_POT":
            targets = self.mdp.get_pot_locations()
            goal = self._closest_adjacent_walkable(me, targets)
            if goal:
                path = [me] + self._bfs_path(me, goal)
                prims += [self._step_to_direction(path[i], path[i+1]) for i in range(len(path)-1)]
        elif macro == "PUT_IN_POT":
            # Find the adjacent pot and face it
            targets = self.mdp.get_pot_locations()
            for target in targets:
                if self._is_adjacent(me, target):
                    # Face the pot first, then interact
                    face_dir = self._get_direction_to_face(me, target)
                    if face_dir != Action.STAY:
                        prims += [face_dir]
                    prims += [Action.INTERACT]
                    break
        elif macro == "GO_TO_DISH":
            targets = self.mdp.get_dish_dispenser_locations()
            goal = self._closest_adjacent_walkable(me, targets)
            if goal:
                path = [me] + self._bfs_path(me, goal)
                prims += [self._step_to_direction(path[i], path[i+1]) for i in range(len(path)-1)]
        elif macro == "TAKE_DISH":
            # Find the adjacent dish dispenser and face it
            targets = self.mdp.get_dish_dispenser_locations()
            for target in targets:
                if self._is_adjacent(me, target):
                    # Face the dispenser first, then interact
                    face_dir = self._get_direction_to_face(me, target)
                    if face_dir != Action.STAY:
                        prims += [face_dir]
                    prims += [Action.INTERACT]
                    break
        elif macro == "GO_TO_SERVE":
            targets = self.mdp.get_serving_locations()
            goal = self._closest_adjacent_walkable(me, targets)
            if goal:
                path = [me] + self._bfs_path(me, goal)
                prims += [self._step_to_direction(path[i], path[i+1]) for i in range(len(path)-1)]
        elif macro == "SERVE":
            # Find the adjacent serve location and face it
            targets = self.mdp.get_serving_locations()
            for target in targets:
                if self._is_adjacent(me, target):
                    # Face the serve location first, then interact
                    face_dir = self._get_direction_to_face(me, target)
                    if face_dir != Action.STAY:
                        prims += [face_dir]
                    prims += [Action.INTERACT]
                    break
        elif macro == "WAIT_COOK":
            prims += [Action.STAY] * 5
        elif macro == "TAKE_SOUP":
            # Find the adjacent pot with soup and face it
            targets = self.mdp.get_pot_locations()
            for target in targets:
                if self._is_adjacent(me, target):
                    # Face the pot first, then interact
                    face_dir = self._get_direction_to_face(me, target)
                    if face_dir != Action.STAY:
                        prims += [face_dir]
                    prims += [Action.INTERACT]
                    break

        return prims if prims else [Action.STAY]

    def action(self, state):
        if not self._pending_prims:
            macro_idx = self._predict_macro_idx(state)
            macro = self.MACROS[macro_idx]
            self._pending_prims.extend(self._expand_macro(macro, state))
            # For UI/logging
            self.heuristic = f"Macro: {macro}" + (f" | cmd: {self.command}" if self.command else "")
        return self._pending_prims.popleft()

def main():
    # ---- CLI args ----
    parser = argparse.ArgumentParser(description="Play Overcooked with language or macro policies + rule-based teammate")
    parser.add_argument("--layout", default="random3")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--fps", type=int, default=5)

    parser.add_argument("--ego", choices=["lang","macro","traditional"], default="macro")
    parser.add_argument("--macro-pt", type=str, default=os.getenv("MACRO_PT"),
                        help="Path to macro policy .pt (from train_macro_policy)")
    parser.add_argument("--no-switch", action="store_true",
                        help="Disable confederate switching; keep a fixed rule-based teammate")
    parser.add_argument("--teammate", choices=["onion","plate","rotate"], default="onion",
                        help="Initial rule-based teammate when --no-switch is given")

    args = parser.parse_args()

    LAYOUT_NAME = args.layout
    HORIZON = args.horizon
    FPS = args.fps

    # 1) Load MDP and Env
    mdp = OvercookedGridworld.from_layout_name(LAYOUT_NAME)
    env = OvercookedEnv(mdp, horizon=HORIZON)
    env.reset()

    # 2) Visualizer
    visualizer = StateVisualizer(grid=mdp.terrain_mtx)

    # 3) Instantiate EGO
    if args.ego == "traditional":
        ego_agent = StayAgent()
        ego_agent.set_agent_index(0)
        ego_agent.set_mdp(mdp)
        print("[INFO] Using StayAgent as traditional baseline.")
    elif args.ego == "macro":
        ego_agent = MacroPolicyAgent(mdp, agent_idx=0, model_path=args.macro_pt)
        ego_agent.set_agent_index(0)
        ego_agent.set_mdp(mdp)
        print(f"[INFO] Using MacroPolicyAgent (pt={args.macro_pt})")
    else:  # "lang"
        ego_agent = TrainedPolicyAgent(mdp, agent_idx=0)
        print("[INFO] Using language-conditioned policy (primitive actions).")

    # 4) Confederate (rule-based) teammate
    def make_teammate(kind: str, clockwise=True):
        if kind == "onion":   return OnionToPotAgent(direction=clockwise)
        if kind == "plate":   return PlateAgent(direction=clockwise)
        if kind == "rotate":  return RotateAgent(direction=clockwise)
        return OnionToPotAgent(direction=clockwise)

    conf_agent = make_teammate(args.teammate, clockwise=True)
    conf_agent.set_agent_index(1)
    conf_agent.set_mdp(mdp)

    pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
    pair.set_mdp(mdp)

    # 5) Pygame setup
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    TB_H = 180  # Increased height for step information
    screen = pygame.display.set_mode((WIDTH, HEIGHT + TB_H))
    pygame.display.set_caption("Overcooked: Language-Conditioned Intervention")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 32)

    input_text = ""
    show_tb = False
    step = 0
    conf_switched = False
    last_command = ""
    last_heuristic = None

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
                        ego_agent.set_command(cmd)
                        last_command = cmd
                        print(f"[LOG] Human intervention: '{cmd}' at step {step}")
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
        # Display last command and ego heuristic
        if last_command:
            screen.blit(font.render(f'Last command: {last_command}', True, (0,0,0)), (10, HEIGHT+70))
        if hasattr(ego_agent, 'heuristic') and getattr(ego_agent, 'heuristic', None):
            screen.blit(font.render(f'Ego heuristic: {ego_agent.heuristic}', True, (0,0,0)), (10, HEIGHT+100))
        
        # Display current confederate agent type
        conf_agent_type = type(conf_agent).__name__
        screen.blit(font.render(f'Confederate: {conf_agent_type}', True, (0,0,0)), (10, HEIGHT+130))
        
        # Display current step
        screen.blit(font.render(f'Step: {step}', True, (0,0,0)), (10, HEIGHT+160))
        pygame.display.flip()
        clock.tick(FPS)

        # 8) Simulation step
        if not show_tb:
            if not args.no_switch:
                # Confederate switching at different steps to create dynamic behavior
                if step == 20 and not conf_switched:
                    # Switch to PlateAgent at step 20
                    conf_agent = PlateAgent(direction=True)
                    conf_agent.set_agent_index(1)
                    conf_agent.set_mdp(mdp)
                    pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
                    pair.set_mdp(mdp)
                    conf_switched = True
                    print(f"[LOG] Confederate agent switched to PlateAgent at step {step}")
                elif step == 40 and conf_switched:
                    # Switch to RotateAgent at step 40
                    conf_agent = RotateAgent(direction=True)
                    conf_agent.set_agent_index(1)
                    conf_agent.set_mdp(mdp)
                    pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
                    pair.set_mdp(mdp)
                    conf_switched = False  # Reset flag to allow future switches
                    print(f"[LOG] Confederate agent switched to RotateAgent at step {step}")
                elif step == 60 and not conf_switched:
                    # Switch back to OnionToPotAgent at step 60
                    conf_agent = OnionToPotAgent(direction=False)  # Counter-clockwise this time
                    conf_agent.set_agent_index(1)
                    conf_agent.set_mdp(mdp)
                    pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
                    pair.set_mdp(mdp)
                    conf_switched = True
                    print(f"[LOG] Confederate agent switched to OnionToPotAgent (counter-clockwise) at step {step}")
            # Step environment
            raw = pair.joint_action(env.state)
            # Extract actions from (action, info) tuples or just actions
            ja = []
            for a in raw:
                if isinstance(a, tuple) and len(a) == 2:
                    # Agent returned (action, info) tuple
                    ja.append(convert_action(a[0]))
                else:
                    # Agent returned just action
                    ja.append(convert_action(a))
            ja = tuple(ja)
            nxt, r, done, _ = env.step(ja)
            env.state = nxt
            step += 1
            # Log heuristic change
            if hasattr(ego_agent, 'heuristic'):
                if ego_agent.heuristic != last_heuristic:
                    print(f"[LOG] Ego heuristic changed to: {ego_agent.heuristic} at step {step}")
                    last_heuristic = ego_agent.heuristic
            if step == 20:
                print("Sim: step 20 reached—fallback logic active for unknown commands.")
            if done:
                print("Episode done, resetting.")
                env.reset()
                step = 0
                conf_switched = False
                # Reset confederate agent to initial state
                conf_agent = OnionToPotAgent(direction=True)
                conf_agent.set_agent_index(1)
                conf_agent.set_mdp(mdp)
                pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
                pair.set_mdp(mdp)

if __name__ == "__main__":
    main()
