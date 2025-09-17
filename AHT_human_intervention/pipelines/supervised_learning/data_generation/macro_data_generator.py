"""
Macro Action Training Data Generator

Generates high-quality training data for macro action policies following the blueprint:
- Label only at macro decision points
- Include arguments + legality
- Verify feasibility and success under oracle executor
"""

import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))


from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from shared.envs.envs.overcooked.script_agent.utils import bfs

from pipelines.rl_finetuning.macro_actions.macro_actions import MacroAction, MacroActionExecutor, MacroActionPolicy
from pipelines.rl_finetuning.macro_actions.macro_action_agents import CoordinatedCookingAgent, SoupServerAgent


@dataclass
class MacroTrainingSample:
    """One training sample for macro action learning."""
    # Required fields
    traj_id: str
    t: int
    layout_id: str
    teammate_mode: str
    text: str
    command_text: str
    macro_id: str
    macro_args: Dict[str, Any]
    legal_macro_mask: List[int]
    oracle_success: bool
    oracle_steps: int
    decision_flags: Dict[str, bool]
    seed: int
    
    # Optional fields
    illegal_reasons: Optional[Dict[str, str]] = None
    primitive_trace: Optional[List[str]] = None
    teacher_top2: Optional[List[str]] = None


class MacroOrder(Enum):
    """Order of macros in legal_macro_mask."""
    GO_TO_ONION = 0
    TAKE_ONION = 1
    GO_TO_POT = 2
    PUT_IN_POT = 3
    GO_TO_DISH = 4
    TAKE_DISH = 5
    GO_TO_SERVE = 6
    SERVE = 7
    WAIT_COOK = 8
    TAKE_SOUP = 9  # appended


class MacroDataGenerator:
    """Generates macro action training data."""
    
    def __init__(self, layout_name: str = "random3"):
        self.layout_name = layout_name
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.env = OvercookedEnv(self.mdp, horizon=200)
        self.executor = MacroActionExecutor(self.mdp)
        self.macro_order = [m.value for m in MacroOrder]
        
        # Get layout-specific info
        self.onion_dispensers = self.mdp.get_onion_dispenser_locations()
        self.dish_dispensers = self.mdp.get_dish_dispenser_locations()
        self.serve_locations = self.mdp.get_serving_locations()
        self.pot_locations = self.mdp.get_pot_locations()
        
    def generate_training_data(self, num_samples: int = 1000, 
                             seeds: Optional[List[int]] = None) -> List[MacroTrainingSample]:
        """Generate training data with quality gates."""
        samples = []
        
        if seeds is None:
            seeds = list(range(num_samples))
        
        for i, seed in enumerate(seeds):
            if i >= num_samples:
                break
                
            try:
                sample = self._generate_single_sample(seed)
                if self._passes_quality_gates(sample):
                    samples.append(sample)
                    
                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1} samples...")
                    
            except Exception as e:
                print(f"Failed to generate sample {i} with seed {seed}: {e}")
                continue
        
        print(f"Generated {len(samples)} valid samples out of {num_samples} attempts")
        return samples
    
    def _spawn_bias_for_act_macros(self, state: OvercookedState, rng: random.Random) -> bool:
        """Occasionally force spawn to create ACT macros (TAKE_*, PUT_IN_POT, SERVE, TAKE_SOUP)."""
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
        choice = rng.random()
        # TAKE_ONION
        if choice < 0.2 and self.onion_dispensers:
            d = rng.choice(self.onion_dispensers)
            for nx,ny in [(d[0]+1,d[1]),(d[0]-1,d[1]),(d[0],d[1]+1),(d[0],d[1]-1)]:
                if self._is_valid_position((nx,ny)):
                    state.players[0].position=(nx,ny); state.players[0].held_object=None; return True
        # PUT_IN_POT
        elif choice < 0.4 and self.pot_locations:
            p = rng.choice(self.pot_locations)
            # ensure pot can accept onion
            state.objects["soup_for_put"] = ObjectState(name="soup", position=p, state=("onion", rng.choice([0,1,2]), 0))
            for nx,ny in [(p[0]+1,p[1]),(p[0]-1,p[1]),(p[0],p[1]+1),(p[0],p[1]-1)]:
                if self._is_valid_position((nx,ny)):
                    state.players[0].position=(nx,ny)
                    state.players[0].held_object = ObjectState(name="onion", position=(nx,ny), state=None)
                    return True
        # TAKE_DISH (requires some pot ready somewhere)
        elif choice < 0.6 and self.dish_dispensers and any(self._pot_has_ready_soup(state, p) for p in self.pot_locations):
            d = rng.choice(self.dish_dispensers)
            for nx,ny in [(d[0]+1,d[1]),(d[0]-1,d[1]),(d[0],d[1]+1),(d[0],d[1]-1)]:
                if self._is_valid_position((nx,ny)):
                    state.players[0].position=(nx,ny); state.players[0].held_object=None; return True
        # TAKE_SOUP
        elif choice < 0.8 and self.pot_locations:
            p = rng.choice(self.pot_locations)
            state.objects["soup_ready_for_take"] = ObjectState(name="soup", position=p, state=("onion", 3, 20))
            for nx,ny in [(p[0]+1,p[1]),(p[0]-1,p[1]),(p[0],p[1]+1),(p[0],p[1]-1)]:
                if self._is_valid_position((nx,ny)):
                    state.players[0].position=(nx,ny)
                    state.players[0].held_object = ObjectState(name="dish", position=(nx,ny), state=None)
                    return True
        # SERVE
        else:
            if self.serve_locations:
                s_loc = rng.choice(self.serve_locations)
                for nx,ny in [(s_loc[0]+1,s_loc[1]),(s_loc[0]-1,s_loc[1]),(s_loc[0],s_loc[1]+1),(s_loc[0],s_loc[1]-1)]:
                    if self._is_valid_position((nx,ny)):
                        state.players[0].position=(nx,ny)
                        # Held soup requires a 3-tuple state (soup_type, num_items, cook_time)
                        state.players[0].held_object = ObjectState(name="soup", position=(nx,ny), state=("onion", 3, 20))
                        return True
        return False

    def _spawn_bias_for_nav_macros(self, state: OvercookedState, rng: random.Random) -> bool:
        """Create NAV states: not-adjacent, path exists, preconditions set."""
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState

        # Ensure decision cache exists
        if not hasattr(self, "_decision_cache"):
            self._spawn_at_decision_state(state, rng)  # builds cache

        def place_far_from(target: Tuple[int,int]) -> bool:
            # pick a walkable tile with Manhattan dist >= 3 and path exists
            cand = list(self._decision_cache["walkable"]) if hasattr(self, "_decision_cache") else []
            rng.shuffle(cand)
            for (x,y) in cand:
                if abs(x-target[0]) + abs(y-target[1]) >= 3:
                    distmap = self._bfs_from((x,y))
                    if self._path_exists_map(distmap, target):
                        state.players[0].position = (x,y)
                        state.players[0].orientation = rng.choice([(0,-1),(1,0),(0,1),(-1,0)])
                        return True
            return False

        r = rng.random()
        # GO_TO_ONION (empty hands)
        if r < 0.25 and self.onion_dispensers:
            d = rng.choice(self.onion_dispensers)
            state.players[0].held_object = None
            return place_far_from(d)

        # GO_TO_POT (holding onion → pot needs onions)
        if r < 0.50 and self.pot_locations:
            p = rng.choice(self.pot_locations)
            # ensure pot can accept onions
            state.objects["soup_for_nav_put"] = ObjectState(name="soup", position=p, state=("onion", rng.choice([0,1,2]), 0))
            state.players[0].held_object = ObjectState(name="onion", position=state.players[0].position, state=None)
            return place_far_from(p)

        # GO_TO_DISH (empty hands + some ready pot)
        if r < 0.75 and self.dish_dispensers:
            if not any(self._pot_has_ready_soup(state, pp) for pp in self.pot_locations) and self.pot_locations:
                rp = rng.choice(self.pot_locations)
                state.objects["soup_ready_nav"] = ObjectState(name="soup", position=rp, state=("onion",3,20))
            d = rng.choice(self.dish_dispensers)
            state.players[0].held_object = None
            return place_far_from(d)

        # GO_TO_SERVE (holding soup)
        if self.serve_locations:
            s_loc = rng.choice(self.serve_locations)
            state.players[0].held_object = ObjectState(name="soup", position=state.players[0].position, state=("onion",3,20))
            return place_far_from(s_loc)

        return False

    def _generate_single_sample(self, seed: int) -> MacroTrainingSample:
        """Generate one training sample."""
        rng = random.Random(seed)
        np.random.seed(seed)
        
        # Initialize environment
        self.env.reset()
        state = self.env.state
        
        # Create diverse scenarios by manipulating the state
        self._create_diverse_scenario(state, seed)
        
        # Mix ACT and NAV biased spawns
        ok = False
        if rng.random() < 0.5:
            ok = self._spawn_bias_for_act_macros(state, rng)
            if not ok:
                ok = self._spawn_bias_for_nav_macros(state, rng)
        else:
            ok = self._spawn_bias_for_nav_macros(state, rng)
            if not ok:
                ok = self._spawn_bias_for_act_macros(state, rng)
        
        if not ok:
            # Fallback: directly spawn at a decision point
            self._spawn_at_decision_state(state, rng)
        
        # Generate the sample
        return self._create_sample_from_state(state, t=0, seed=seed)
    
    def _create_diverse_scenario(self, state: OvercookedState, seed: int):
        """Create diverse scenarios by manipulating the state."""
        # Use seed to create different scenarios
        scenario_type = seed % 5
        
        if scenario_type == 0:
            # Scenario 0: Empty state (default)
            pass
        elif scenario_type == 1:
            # Scenario 1: Pots with onions
            self._add_onions_to_pots(state, 1)
        elif scenario_type == 2:
            # Scenario 2: Pots with 2 onions
            self._add_onions_to_pots(state, 2)
        elif scenario_type == 3:
            # Scenario 3: Cooking pots
            self._add_cooking_pots(state)
        elif scenario_type == 4:
            # Scenario 4: Ready soup
            self._add_ready_soup(state)
        
        # Randomize player positions
        self._randomize_player_positions(state, seed)
    
    def _add_onions_to_pots(self, state: OvercookedState, num_onions: int):
        """Add onions to pots."""
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
        
        for i, pot_pos in enumerate(self.pot_locations):
            # Create soup object with onions
            soup_obj = ObjectState(
                name="soup",
                position=pot_pos,
                state=("onion", num_onions, 0)  # (type, num_items, cook_time)
            )
            state.objects[f"soup_{i}"] = soup_obj
    
    def _add_cooking_pots(self, state: OvercookedState):
        """Add cooking pots."""
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
        
        for i, pot_pos in enumerate(self.pot_locations):
            # Create soup object with 3 onions, cooking
            soup_obj = ObjectState(
                name="soup",
                position=pot_pos,
                state=("onion", 3, 10)  # (type, num_items, cook_time)
            )
            state.objects[f"soup_{i}"] = soup_obj
    
    def _add_ready_soup(self, state: OvercookedState):
        """Add ready soup to pots."""
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
        
        for i, pot_pos in enumerate(self.pot_locations):
            # Create soup object with 3 onions, ready
            soup_obj = ObjectState(
                name="soup",
                position=pot_pos,
                state=("onion", 3, 20)  # (type, num_items, cook_time)
            )
            state.objects[f"soup_{i}"] = soup_obj
    
    def _randomize_player_positions(self, state: OvercookedState, seed: int):
        """Randomize player positions."""
        # Get valid positions
        valid_positions = []
        for y in range(len(self.mdp.terrain_mtx)):
            for x in range(len(self.mdp.terrain_mtx[0])):
                if self.mdp.terrain_mtx[y][x] != 'X':
                    valid_positions.append((x, y))
        
        # Use seed to select positions
        random.seed(seed)
        pos1 = random.choice(valid_positions)
        pos2 = random.choice([p for p in valid_positions if p != pos1])
        
        # Update player positions
        state.players[0].position = pos1
        state.players[1].position = pos2
    
    def _is_macro_decision_point(self, state: OvercookedState) -> bool:
        """Check if current state is a macro decision point."""
        player = state.players[0]
        
        # Check decision flags
        flags = self._compute_decision_flags(state)
        
        # Decision point if:
        # 1. Near an interactable (high priority)
        if flags["near_interactable"]:
            return True
        
        # 2. At a junction with multiple valid moves
        if flags["junction"]:
            return True
        
        # 3. Pot ready somewhere and we're not holding anything
        if flags["pot_ready_somewhere"] and not player.held_object:
            return True
        
        # 4. Holding something and near a relevant target
        if player.held_object:
            if player.held_object.name == "onion":
                # Near a pot that can accept onions
                for pot_pos in self.pot_locations:
                    if self._is_adjacent(player.position, pot_pos) and self._pot_can_accept_onions(state, pot_pos):
                        return True
            elif player.held_object.name == "dish":
                # Near a pot with ready soup
                for pot_pos in self.pot_locations:
                    if self._is_adjacent(player.position, pot_pos) and self._pot_has_ready_soup(state, pot_pos):
                        return True
            elif player.held_object.name == "soup":
                # Near serve location
                for serve_pos in self.serve_locations:
                    if self._is_adjacent(player.position, serve_pos):
                        return True
        
        # 5. Just completed a macro (empty hands after holding something)
        # This would need history tracking, simplified for now
        
        return False
    
    def _compute_decision_flags(self, state: OvercookedState) -> Dict[str, bool]:
        """Compute decision flags for the current state."""
        player = state.players[0]
        pos = player.position
        
        # Check if near any interactable
        near_interactable = self._is_adjacent_to_interactable(pos)
        
        # Check if at junction (multiple valid moves)
        junction = self._is_at_junction(pos)
        
        # Check if any pot is ready
        pot_ready_somewhere = self._any_pot_ready(state)
        
        # Check what player is holding
        holding_onion = player.held_object and player.held_object.name == "onion"
        holding_dish = player.held_object and player.held_object.name == "dish"
        
        # Check if changed direction (simplified - would need history)
        changed_dir_prev_step = False  # TODO: track direction history
        
        return {
            "near_interactable": near_interactable,
            "junction": junction,
            "pot_ready_somewhere": pot_ready_somewhere,
            "holding_onion": holding_onion,
            "holding_dish": holding_dish,
            "changed_dir_prev_step": changed_dir_prev_step
        }
    
    def _is_adjacent_to_interactable(self, pos: Tuple[int, int]) -> bool:
        """Check if position is adjacent to any interactable."""
        x, y = pos
        
        # Check all adjacent positions
        adjacent = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for adj_x, adj_y in adjacent:
            # Check if adjacent to onion dispenser
            if (adj_x, adj_y) in self.onion_dispensers:
                return True
            # Check if adjacent to dish dispenser
            if (adj_x, adj_y) in self.dish_dispensers:
                return True
            # Check if adjacent to serve location
            if (adj_x, adj_y) in self.serve_locations:
                return True
            # Check if adjacent to pot
            if (adj_x, adj_y) in self.pot_locations:
                return True
        
        return False
    
    def _is_at_junction(self, pos: Tuple[int, int]) -> bool:
        """Check if position is at a junction (multiple valid moves)."""
        x, y = pos
        valid_moves = 0
        
        # Check all four directions
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if self._is_valid_position((new_x, new_y)):
                valid_moves += 1
        
        return valid_moves >= 3  # Junction if 3+ valid moves
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (not wall)."""
        x, y = pos
        if y < 0 or y >= len(self.mdp.terrain_mtx):
            return False
        if x < 0 or x >= len(self.mdp.terrain_mtx[0]):
            return False
        
        terrain = self.mdp.terrain_mtx[y][x]
        return terrain != 'X'
    
    def _any_pot_ready(self, state: OvercookedState) -> bool:
        """Check if any pot has ready soup."""
        for obj in state.objects.values():
            if obj.name == "soup" and obj.state[2] >= 20:  # cook_time >= 20
                return True
        return False
    
    def _get_masks_and_indices(self, state: OvercookedState) -> Tuple[List[int], List[int], int, List[int], int]:
        """Return (pot_mask, onion_mask, onion_id, serve_mask, serve_id) for current state."""
        # masks are 1 for each existing candidate
        pot_mask = [1]*len(self.pot_locations)
        onion_mask = [1]*len(self.onion_dispensers)
        serve_mask = [1]*len(self.serve_locations)
        # pick nearest indices
        player = state.players[0]
        pos = player.position
        if self.onion_dispensers:
            dists = [abs(pos[0]-x)+abs(pos[1]-y) for (x,y) in self.onion_dispensers]
            onion_id = int(min(range(len(dists)), key=lambda i: dists[i]))
        else:
            onion_id = 0
        if self.serve_locations:
            dists = [abs(pos[0]-x)+abs(pos[1]-y) for (x,y) in self.serve_locations]
            serve_id = int(min(range(len(dists)), key=lambda i: dists[i]))
        else:
            serve_id = 0
        return pot_mask, onion_mask, onion_id, serve_mask, serve_id

    def _create_sample_from_state(self, state: OvercookedState, t: int, seed: int) -> MacroTrainingSample:
        """Create a training sample from the current state."""
        player = state.players[0]
        
        # Generate state text
        text = self._format_state_text(state)
        
        # Infer gold macro
        macro_id, macro_args = self._infer_gold_macro(state)
        
        # Compute legality mask
        legal_mask = self._compute_legal_macro_mask(state)
        
        # Check if gold macro is legal
        if not self._is_macro_legal(macro_id, legal_mask):
            # Try to find an alternative legal macro
            macro_id, macro_args = self._find_alternative_macro(state, legal_mask)
        
        # Oracle execute for verification
        oracle_success, oracle_steps, primitive_trace = self._oracle_execute(state, macro_id, macro_args)
        
        # Compute decision flags
        decision_flags = self._compute_decision_flags(state)
        
        # Generate command text (simplified - would be from human commands)
        command_text = "[NO_CMD]"
        
        # Teammate mode passthrough if set
        teammate_mode = getattr(self, 'teammate_mode', "SCRIPTED_SAFE")
        
        # Attach argument masks/indices into macro_args for training
        pot_mask, onion_mask, onion_id, serve_mask, serve_id = self._get_masks_and_indices(state)
        macro_args = dict(macro_args or {})
        macro_args.setdefault("pot_mask", pot_mask)
        macro_args.setdefault("onion_mask", onion_mask)
        macro_args.setdefault("onion_id", onion_id)
        macro_args.setdefault("serve_mask", serve_mask)
        macro_args.setdefault("serve_id", serve_id)
        
        # Create sample
        return MacroTrainingSample(
            traj_id=f"L3_seed{seed}_ep{t}",
            t=t,
            layout_id=self.layout_name,
            teammate_mode=teammate_mode,
            text=text,
            command_text=command_text,
            macro_id=macro_id,
            macro_args=macro_args,
            legal_macro_mask=legal_mask,
            oracle_success=oracle_success,
            oracle_steps=oracle_steps,
            decision_flags=decision_flags,
            seed=seed,
            primitive_trace=primitive_trace if oracle_success else None
        )
    
    def _format_state_text(self, state: OvercookedState) -> str:
        """Format state into informative text."""
        player = state.players[0]
        mate = state.players[1]
        
        # Basic scene info
        text = f"[SCENE] size {len(self.mdp.terrain_mtx[0])}x{len(self.mdp.terrain_mtx)} "
        
        # Player positions and orientations
        text += f"[EGO] pos {player.position} facing {self._direction_to_str(player.orientation)} "
        text += f"item {self._get_item_name(player.held_object)} "
        
        text += f"[MATE] pos {mate.position} facing {self._direction_to_str(mate.orientation)} "
        text += f"item {self._get_item_name(mate.held_object)} "
        
        # Goal inference
        goal = self._infer_goal(state)
        text += f"[GOAL] {goal} | [NO_CMD] "
        
        # Nearest objects
        text += self._format_nearest_objects(player.position)
        
        # Local legality
        text += self._format_local_legality(player.position, player.orientation)
        
        # Pot summaries
        text += self._format_pot_summaries(state)
        
        return text
    
    def _direction_to_str(self, direction: Tuple[int, int]) -> str:
        """Convert direction tuple to string."""
        if direction == (0, -1):
            return "N"
        elif direction == (0, 1):
            return "S"
        elif direction == (1, 0):
            return "E"
        elif direction == (-1, 0):
            return "W"
        else:
            return "?"
    
    def _get_item_name(self, obj) -> str:
        """Get item name from held object."""
        if obj is None:
            return "empty"
        return obj.name
    
    def _infer_goal(self, state: OvercookedState) -> str:
        """Infer the current goal from state."""
        player = state.players[0]
        
        if player.held_object is None:
            # Check if any pot needs onions
            if self._any_pot_needs_onions(state):
                return "GO_TO_ONION"
            # Check if any pot is ready
            elif self._any_pot_ready(state):
                return "GO_TO_DISH"
            else:
                return "WAIT_COOK"
        elif player.held_object.name == "onion":
            return "GO_TO_POT"
        elif player.held_object.name == "dish":
            return "GO_TO_POT"
        else:
            return "GO_TO_SERVE"
    
    def _any_pot_needs_onions(self, state: OvercookedState) -> bool:
        """Check if any pot needs onions."""
        for obj in state.objects.values():
            if obj.name == "soup" and obj.state[1] < 3:  # num_items < 3
                return True
        return False
    
    def _format_nearest_objects(self, pos: Tuple[int, int]) -> str:
        """Format nearest objects information."""
        text = "[NEAREST] "
        
        # Find nearest onion dispenser
        nearest_onion = self._find_nearest(pos, self.onion_dispensers)
        if nearest_onion:
            delta = self._compute_delta(pos, nearest_onion)
            text += f"onion Δr={delta[0]} Δc={delta[1]} ({self._delta_to_str(delta)}); "
        
        # Find nearest pot
        nearest_pot = self._find_nearest(pos, self.pot_locations)
        if nearest_pot:
            delta = self._compute_delta(pos, nearest_pot)
            text += f"pot Δr={delta[0]} Δc={delta[1]} ({self._delta_to_str(delta)}); "
        
        # Find nearest dish dispenser
        nearest_dish = self._find_nearest(pos, self.dish_dispensers)
        if nearest_dish:
            delta = self._compute_delta(pos, nearest_dish)
            text += f"dish Δr={delta[0]} Δc={delta[1]} ({self._delta_to_str(delta)}); "
        
        # Find nearest serve location
        nearest_serve = self._find_nearest(pos, self.serve_locations)
        if nearest_serve:
            delta = self._compute_delta(pos, nearest_serve)
            text += f"serve Δr={delta[0]} Δc={delta[1]} ({self._delta_to_str(delta)}) "
        
        return text
    
    def _find_nearest(self, pos: Tuple[int, int], locations: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find nearest location from list."""
        if not locations:
            return None
        
        min_dist = float('inf')
        nearest = None
        
        for loc in locations:
            dist = abs(pos[0] - loc[0]) + abs(pos[1] - loc[1])  # Manhattan distance
            if dist < min_dist:
                min_dist = dist
                nearest = loc
        
        return nearest
    
    def _compute_delta(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Compute delta from from_pos to to_pos."""
        fx,fy = from_pos; tx,ty = to_pos
        # Δr = ty - fy (N negative), Δc = tx - fx (E positive)
        return (ty - fy, tx - fx)
    
    def _delta_to_str(self, delta: Tuple[int, int]) -> str:
        """Convert delta to string representation."""
        dr, dc = delta
        card = []
        if dr < 0: card.append(f"N{abs(dr)}")
        elif dr > 0: card.append(f"S{dr}")
        if dc > 0: card.append(f"E{dc}")
        elif dc < 0: card.append(f"W{abs(dc)}")
        return " ".join(card) or "HERE"
    
    def _format_local_legality(self, pos: Tuple[int, int], orient: Tuple[int, int]) -> str:
        """Format local legality information."""
        x,y = pos
        dx,dy = orient
        ax,ay = x+dx, y+dy
        H = len(self.mdp.terrain_mtx); W = len(self.mdp.terrain_mtx[0])
        def inb(xx,yy): return 0 <= xx < W and 0 <= yy < H
        ahead_blocked = (not inb(ax,ay)) or (self.mdp.terrain_mtx[ay][ax] == 'X')

        junction = self._is_at_junction(pos)
        legal_moves = self._get_legal_moves(pos)
        legal_actions = []
        if "N1" in legal_moves: legal_actions.append("MOVE_N")
        if "S1" in legal_moves: legal_actions.append("MOVE_S")
        if "E1" in legal_moves: legal_actions.append("MOVE_E")
        if "W1" in legal_moves: legal_actions.append("MOVE_W")
        legal_actions += ["INTERACT","STAY"]  # you can refine INTERACT with adjacency if you want

        # Local adjacency tags
        adj_onion = 1 if any(self._is_adjacent(pos, d) for d in self.onion_dispensers) else 0
        adj_pot = 1 if any(self._is_adjacent(pos, p) for p in self.pot_locations) else 0
        adj_dish = 1 if any(self._is_adjacent(pos, d) for d in self.dish_dispensers) else 0
        adj_serve = 1 if any(self._is_adjacent(pos, s) for s in self.serve_locations) else 0
        holding = self._get_item_name(self.env.state.players[0].held_object)

        return (f"[LOCAL] ahead_blocked {1 if ahead_blocked else 0} "
                f"junction {1 if junction else 0} "
                f"legal_moves {' '.join(legal_moves)} "
                f"[LEGAL] {' '.join(legal_actions)} "
                f"adj_onion={adj_onion} adj_pot={adj_pot} adj_dish={adj_dish} adj_serve={adj_serve} holding={holding} ")
    
    def _get_legal_moves(self, pos: Tuple[int, int]) -> List[str]:
        """Get legal moves from position."""
        x, y = pos
        legal = []
        
        # Check all four directions
        if self._is_valid_position((x, y-1)):
            legal.append("N1")
        if self._is_valid_position((x, y+1)):
            legal.append("S1")
        if self._is_valid_position((x+1, y)):
            legal.append("E1")
        if self._is_valid_position((x-1, y)):
            legal.append("W1")
        
        return legal
    
    def _format_pot_summaries(self, state: OvercookedState) -> str:
        """Format pot summaries."""
        text = "[POTS] "
        
        pot_info = []
        for i, pot_pos in enumerate(self.pot_locations):
            pot_state = self._get_pot_state(state, pot_pos)
            pot_info.append(f"({pot_pos[0]},{pot_pos[1]}) {pot_state}")
        
        text += " | ".join(pot_info)
        return text
    
    def _get_pot_state(self, state: OvercookedState, pot_pos: Tuple[int, int]) -> str:
        """Get pot state string."""
        # Find soup object at this pot
        for obj in state.objects.values():
            if obj.name == "soup" and obj.position == pot_pos:
                num_items, cook_time = obj.state[1], obj.state[2]
                if cook_time >= 20:
                    return "ready"
                elif cook_time > 0:
                    return f"{num_items}/3,cook={cook_time}"
                else:
                    return f"{num_items}/3"
        
        return "empty"
    
    def _infer_gold_macro(self, state: OvercookedState) -> Tuple[str, Dict[str, Any]]:
        """Infer the gold macro from state."""
        player = state.players[0]
        pos = player.position
        held = self._get_item_name(player.held_object)
        
        # Prefer ACT when adjacent and preconditions satisfied
        if held == "empty":
            # TAKE_ONION if adjacent to onion dispenser
            for d in self.onion_dispensers:
                if self._is_adjacent(pos, d):
                    return "TAKE_ONION", {}
            # TAKE_DISH if any pot ready and adjacent to dish dispenser
            if self._any_pot_ready(state):
                for dd in self.dish_dispensers:
                    if self._is_adjacent(pos, dd):
                        return "TAKE_DISH", {}
        elif held == "onion":
            # PUT_IN_POT if adjacent to pot that can accept onions
            for p in self.pot_locations:
                if self._is_adjacent(pos, p) and self._pot_can_accept_onions(state, p):
                    return "PUT_IN_POT", {}
        elif held == "dish":
            # TAKE_SOUP if adjacent to ready pot
            for p in self.pot_locations:
                if self._is_adjacent(pos, p) and self._pot_has_ready_soup(state, p):
                    return "TAKE_SOUP", {}
        elif held == "soup":
            # SERVE if adjacent to serve location
            for s in self.serve_locations:
                if self._is_adjacent(pos, s):
                    return "SERVE", {}
        
        # Otherwise NAV/WAIT decisions
        if held == "empty":
            if self._any_pot_needs_onions(state):
                # Pick nearest onion dispenser
                dispenser_id = self._find_nearest_onion_dispenser(state)
                return "GO_TO_ONION", {"onion_id": dispenser_id}
            elif self._any_pot_ready(state):
                return "GO_TO_DISH", {"dispenser": 0}
            else:
                return "WAIT_COOK", {}
        elif held == "onion":
            # Holding onion - go to pot
            pot_id = self._find_nearest_available_pot(state)
            return "GO_TO_POT", {"pot_id": pot_id}
        elif held == "dish":
            # Holding dish - go to pot to get soup
            pot_id = self._find_ready_pot(state)
            return "GO_TO_POT", {"pot_id": pot_id}
        else:
            # Holding soup - go to serve
            serve_id = self._find_nearest_serve_location(state)
            return "GO_TO_SERVE", {"serve_id": serve_id}
    
    def _find_nearest_onion_dispenser(self, state: OvercookedState) -> int:
        """Find nearest onion dispenser."""
        player = state.players[0]
        pos = player.position
        if not self.onion_dispensers:
            return 0
        dists = [abs(pos[0]-x)+abs(pos[1]-y) for (x,y) in self.onion_dispensers]
        return int(min(range(len(dists)), key=lambda i: dists[i]))
    
    def _find_nearest_serve_location(self, state: OvercookedState) -> int:
        """Find nearest serve location."""
        player = state.players[0]
        pos = player.position
        if not self.serve_locations:
            return 0
        dists = [abs(pos[0]-x)+abs(pos[1]-y) for (x,y) in self.serve_locations]
        return int(min(range(len(dists)), key=lambda i: dists[i]))

    def _find_nearest_available_pot(self, state: OvercookedState) -> int:
        """Find nearest pot that can accept onions."""
        player = state.players[0]
        pos = player.position
        distmap = self._bfs_from(pos)
        best = None; best_id = 0
        for i,p in enumerate(self.pot_locations):
            if self._pot_can_accept_onions(state, p) and self._path_exists_map(distmap, p):
                d = distmap[p[1]][p[0]]
                if (best is None) or (d < best): best, best_id = d, i
        return best_id
    
    def _pot_can_accept_onions(self, state: OvercookedState, pot_pos: Tuple[int, int]) -> bool:
        """Check if pot can accept onions."""
        for obj in state.objects.values():
            if obj.name == "soup" and obj.position == pot_pos:
                return obj.state[1] < 3  # num_items < 3
        return True  # No soup object means empty pot
    
    def _find_ready_pot(self, state: OvercookedState) -> int:
        """Find a pot with ready soup."""
        for i, pot_pos in enumerate(self.pot_locations):
            for obj in state.objects.values():
                if obj.name == "soup" and obj.position == pot_pos and obj.state[2] >= 20:
                    return i
        return 0  # Default
    
    def _compute_legal_macro_mask(self, state: OvercookedState) -> List[int]:
        """Compute legal macro mask using fast BFS."""
        mask = [0] * len(self.macro_order)
        player = state.players[0]
        pos = player.position
        held_item = self._get_item_name(player.held_object)

        distmap = self._bfs_from(pos)

        # GO_TO_ONION: path to any onion dispenser + NOT already adjacent
        if held_item in ["empty", "dish"]:
            for dispenser_pos in self.onion_dispensers:
                if self._path_exists_map(distmap, dispenser_pos) and not self._is_adjacent(pos, dispenser_pos):
                    mask[MacroOrder.GO_TO_ONION.value] = 1
                    break

        # TAKE_ONION: adjacent to onion dispenser + empty hands
        if held_item == "empty":
            for dispenser_pos in self.onion_dispensers:
                if self._is_adjacent(pos, dispenser_pos):
                    mask[MacroOrder.TAKE_ONION.value] = 1
                    break

        # GO_TO_POT: path to valid pot + NOT already adjacent
        if held_item == "onion":
            # Find pot that needs onions
            for i, pot_pos in enumerate(self.pot_locations):
                if self._pot_can_accept_onions(state, pot_pos) and self._path_exists_map(distmap, pot_pos) and not self._is_adjacent(pos, pot_pos):
                    mask[MacroOrder.GO_TO_POT.value] = 1
                    break
        elif held_item == "dish":
            # Find pot with ready soup
            for i, pot_pos in enumerate(self.pot_locations):
                if self._pot_has_ready_soup(state, pot_pos) and self._path_exists_map(distmap, pot_pos) and not self._is_adjacent(pos, pot_pos):
                    mask[MacroOrder.GO_TO_POT.value] = 1
                    break

        # PUT_IN_POT: adjacent to pot + holding onion + pot not full
        if held_item == "onion":
            for pot_pos in self.pot_locations:
                if self._is_adjacent(pos, pot_pos) and self._pot_can_accept_onions(state, pot_pos):
                    mask[MacroOrder.PUT_IN_POT.value] = 1
                    break

        # GO_TO_DISH: path to dish dispenser + empty hands + pot ready + NOT already adjacent
        if held_item == "empty" and self._any_pot_ready(state):
            for dispenser_pos in self.dish_dispensers:
                if self._path_exists_map(distmap, dispenser_pos) and not self._is_adjacent(pos, dispenser_pos):
                    mask[MacroOrder.GO_TO_DISH.value] = 1
                    break

        # TAKE_DISH: adjacent to dish dispenser + empty hands + pot ready
        if held_item == "empty" and self._any_pot_ready(state):
            for dispenser_pos in self.dish_dispensers:
                if self._is_adjacent(pos, dispenser_pos):
                    mask[MacroOrder.TAKE_DISH.value] = 1
                    break

        # GO_TO_SERVE: path to serve location + holding soup + NOT already adjacent
        if held_item == "soup":
            for serve_pos in self.serve_locations:
                if self._path_exists_map(distmap, serve_pos) and not self._is_adjacent(pos, serve_pos):
                    mask[MacroOrder.GO_TO_SERVE.value] = 1
                    break

        # SERVE: adjacent to serve location + holding soup
        if held_item == "soup":
            for serve_pos in self.serve_locations:
                if self._is_adjacent(pos, serve_pos):
                    mask[MacroOrder.SERVE.value] = 1
                    break

        # TAKE_SOUP: holding dish + adjacent to a pot with ready soup
        if held_item == "dish":
            for pot_pos in self.pot_locations:
                if self._is_adjacent(pos, pot_pos) and self._pot_has_ready_soup(state, pot_pos):
                    mask[MacroOrder.TAKE_SOUP.value] = 1
                    break

        # WAIT_COOK: any pot cooking
        if self._any_pot_cooking(state):
            mask[MacroOrder.WAIT_COOK.value] = 1

        return mask
    
    def _is_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
    
    def _pot_has_ready_soup(self, state: OvercookedState, pot_pos: Tuple[int, int]) -> bool:
        """Check if pot has ready soup."""
        for obj in state.objects.values():
            if obj.name == "soup" and obj.position == pot_pos and obj.state[2] >= 20:
                return True
        return False
    
    def _any_pot_cooking(self, state: OvercookedState) -> bool:
        """Check if any pot is cooking."""
        for obj in state.objects.values():
            if obj.name == "soup" and 0 < obj.state[2] < 20:  # 0 < cook_time < 20
                return True
        return False
    
    def _is_macro_legal(self, macro_id: str, legal_mask: List[int]) -> bool:
        """Check if macro is legal."""
        try:
            macro_idx = MacroOrder[macro_id].value
            return legal_mask[macro_idx] == 1
        except:
            return False
    
    def _find_alternative_macro(self, state: OvercookedState, legal_mask: List[int]) -> Tuple[str, Dict[str, Any]]:
        """Find alternative legal macro."""
        # Find first legal macro
        for i, legal in enumerate(legal_mask):
            if legal == 1:
                macro_name = MacroOrder(i).name
                # Generate basic args
                if macro_name == "GO_TO_ONION":
                    return macro_name, {"dispenser": 0}
                elif macro_name == "GO_TO_POT":
                    return macro_name, {"pot_id": 0}
                elif macro_name == "GO_TO_DISH":
                    return macro_name, {"dispenser": 0}
                elif macro_name == "GO_TO_SERVE":
                    return macro_name, {"serve_id": 0}
                elif macro_name == "TAKE_SOUP":
                    return macro_name, {}
                else:
                    return macro_name, {}
        
        # Fallback
        return "WAIT_COOK", {}
    
    def _oracle_execute(self, state: OvercookedState, macro_id: str, macro_args: Dict[str, Any]) -> Tuple[bool, int, List[str]]:
        """Fast oracle: success = preconditions true + path existence (if GO_TO_*)."""
        player = state.players[0]
        pos = player.position
        distmap = self._bfs_from(pos)

        def dist_to(xy): 
            x,y = xy; d = distmap[y][x]
            return None if d >= 10**9 else d

        if macro_id == "GO_TO_ONION":
            targets = self.onion_dispensers
            d = min((dist_to(t) for t in targets if dist_to(t) is not None), default=None)
            return (d is not None, int(d or 0), [])
        if macro_id == "GO_TO_DISH":
            if not self._any_pot_ready(state): return (False,0,[])
            targets = self.dish_dispensers
            d = min((dist_to(t) for t in targets if dist_to(t) is not None), default=None)
            return (d is not None, int(d or 0), [])
        if macro_id == "GO_TO_POT":
            pid = macro_args.get("pot_id", 0)
            if pid < 0 or pid >= len(self.pot_locations): return (False,0,[])
            target = self.pot_locations[pid]
            if self._get_item_name(player.held_object) == "onion" and not self._pot_can_accept_onions(state, target):
                return (False,0,[])
            d = dist_to(target)
            return (d is not None, int(d or 0), [])
        if macro_id in ("TAKE_ONION","TAKE_DISH","PUT_IN_POT","SERVE"):
            # Just check adjacency + preconditions
            return (True, 1, [])
        if macro_id == "TAKE_SOUP":
            # adjacency + dish + ready pot already in legality; treat as 1-step success
            return (True, 1, [])
        if macro_id == "GO_TO_SERVE":
            targets = self.serve_locations
            d = min((dist_to(t) for t in targets if dist_to(t) is not None), default=None)
            return (d is not None, int(d or 0), [])
        if macro_id == "WAIT_COOK":
            return (self._any_pot_cooking(state), 1, [])

        return (False,0,[])
    
    def _passes_quality_gates(self, sample: MacroTrainingSample) -> bool:
        """Check if sample passes quality gates."""
        # Check if gold macro is legal
        if sample.legal_macro_mask[MacroOrder[sample.macro_id].value] == 0:
            return False
        
        # Check if oracle execution succeeded
        if not sample.oracle_success:
            return False
        
        # Check if macro_args are present for macros that need them
        if sample.macro_id in ["GO_TO_ONION", "GO_TO_DISH"] and "dispenser" not in sample.macro_args:
            return False
        if sample.macro_id in ["GO_TO_POT"] and "pot_id" not in sample.macro_args:
            return False
        if sample.macro_id in ["GO_TO_SERVE"] and "serve_id" not in sample.macro_args:
            return False
        
        # Check text length (simplified)
        if len(sample.text) > 500:
            return False
        
        return True
    
    def save_samples(self, samples: List[MacroTrainingSample], filename: str):
        """Save samples to JSONL file."""
        with open(filename, 'w') as f:
            for sample in samples:
                json.dump(asdict(sample), f)
                f.write('\n')
        
        print(f"Saved {len(samples)} samples to {filename}")

    def _spawn_at_decision_state(self, state: OvercookedState, rng: random.Random):
        """Place ego at a macro decision tile (near interactable or junction) with a plausible holding item."""
        H = len(self.mdp.terrain_mtx)
        W = len(self.mdp.terrain_mtx[0])

        # Build (and cache) decision tiles
        if not hasattr(self, "_decision_cache"):
            walkable = [(x,y) for y in range(H) for x in range(W) if self.mdp.terrain_mtx[y][x] != 'X']
            inter_adj = set()
            for lst in (self.onion_dispensers, self.dish_dispensers, self.serve_locations, self.pot_locations):
                for (x,y) in lst:
                    for nx,ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                        if 0 <= nx < W and 0 <= ny < H and self.mdp.terrain_mtx[ny][nx] != 'X':
                            inter_adj.add((nx,ny))
            # Junctions = tiles with ≥3 legal moves
            def deg(x,y):
                d=0
                for nx,ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                    if 0 <= nx < W and 0 <= ny < H and self.mdp.terrain_mtx[ny][nx] != 'X':
                        d+=1
                return d
            junctions = {xy for xy in walkable if deg(*xy) >= 3}
            self._decision_cache = {
                "near_interactable": list(inter_adj),
                "junctions": list(junctions),
                "walkable": walkable
            }

        mode = rng.random()
        if mode < 0.6 and self._decision_cache["near_interactable"]:
            pos = rng.choice(self._decision_cache["near_interactable"])
        elif mode < 0.85 and self._decision_cache["junctions"]:
            pos = rng.choice(self._decision_cache["junctions"])
        else:
            pos = rng.choice(self._decision_cache["walkable"])

        # Place ego here; mate somewhere else
        state.players[0].position = pos
        state.players[0].orientation = rng.choice([(0,-1),(1,0),(0,1),(-1,0)])
        mate_pos = rng.choice([p for p in self._decision_cache["walkable"] if p != pos])
        state.players[1].position = mate_pos
        state.players[1].orientation = rng.choice([(0,-1),(1,0),(0,1),(-1,0)])

        # Give ego a plausible held item ~30% of time to diversify macros
        held_roll = rng.random()
        state.players[0].held_object = None
        if held_roll < 0.15 and self.onion_dispensers:
            # as if just took onion elsewhere
            from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
            state.players[0].held_object = ObjectState(name="onion", position=pos, state=None)
        elif held_roll < 0.30 and any(self._pot_has_ready_soup(state, p) for p in self.pot_locations):
            from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
            state.players[0].held_object = ObjectState(name="dish", position=pos, state=None)
    
    def _bfs_from(self, start: Tuple[int,int]):
        """Return a 2D dist array from start over static walkables (no players)."""
        H = len(self.mdp.terrain_mtx); W = len(self.mdp.terrain_mtx[0])
        INF = 10**9
        dist = [[INF]*W for _ in range(H)]
        sx,sy = start
        if self.mdp.terrain_mtx[sy][sx] == 'X':
            return dist
        from collections import deque
        q = deque([(sx,sy)])
        dist[sy][sx] = 0
        while q:
            x,y = q.popleft()
            for nx,ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if 0 <= nx < W and 0 <= ny < H and self.mdp.terrain_mtx[ny][nx] != 'X' and dist[ny][nx] == INF:
                    dist[ny][nx] = dist[y][x] + 1
                    q.append((nx,ny))
        return dist

    def _path_exists_map(self, distmap, to_pos: Tuple[int,int]) -> bool:
        x,y = to_pos
        H = len(self.mdp.terrain_mtx); W = len(self.mdp.terrain_mtx[0])
        if not (0 <= x < W and 0 <= y < H): return False
        return distmap[y][x] < 10**9


# ------------------------------
# Pretraining corpus helpers
# ------------------------------

LAYOUTS = [
    "random3"
]

TEAMMATE_MODES = [
    "NO_OP",
    "SCRIPTED_SAFE",
    "SCRIPTED_NOISY",
]

# Tiny synthetic command bank (optional)
CMD_TEMPLATES = {
    "GO_TO_ONION": [
        "head to the onion dispenser",
        "go get an onion",
        "move toward the onion source",
    ],
    "TAKE_ONION": [
        "pick up an onion",
        "grab the onion",
        "take one onion",
    ],
    "GO_TO_POT": [
        "go to the pot",
        "move to the cooking pot",
        "head for the nearest pot",
    ],
    "PUT_IN_POT": [
        "put the onion in the pot",
        "drop the onion into the pot",
    ],
    "GO_TO_DISH": [
        "go get a dish",
        "head to the dish dispenser",
    ],
    "TAKE_DISH": [
        "pick up a dish",
        "grab a plate",
    ],
    "GO_TO_SERVE": [
        "go to the serving window",
        "head to the serving area",
    ],
    "SERVE": [
        "serve the soup",
        "deliver the order",
    ],
    "WAIT_COOK": [
        "wait for the soup to cook",
        "hold until it’s ready",
    ],
}

# Add TAKE_SOUP command templates (optional)
CMD_TEMPLATES.setdefault("TAKE_SOUP", [
    "scoop the soup",
    "take soup from the pot",
    "grab the soup",
])

TARGET_GROUPS = {"NAV": 0.5, "ACT": 0.45, "WAIT": 0.05}

def render_synthetic_command(macro_id: str, macro_args: Dict[str, Any]) -> str:
    bank = CMD_TEMPLATES.get(macro_id, [])
    if not bank:
        return "[NO_CMD]"
    return random.choice(bank)

def _macro_group(mid: str) -> str:
    if mid in ("GO_TO_ONION","GO_TO_POT","GO_TO_DISH","GO_TO_SERVE"):
        return "NAV"
    if mid in ("TAKE_ONION","TAKE_DISH","PUT_IN_POT","SERVE"):
        return "ACT"
    return "WAIT"


def build_generator(layout: str, teammate_mode: str):
    gen = MacroDataGenerator(layout)
    # Store teammate mode for record-keeping (can integrate later if used in env)
    gen.teammate_mode = teammate_mode
    return gen


def _layout_exists(layout_name: str) -> bool:
    import os
    base = os.path.join(
        os.path.dirname(__file__),
        "envs",
        "overcooked",
        "overcooked_ai_py",
        "data",
        "layouts",
        f"{layout_name}.layout",
    )
    return os.path.exists(base)


def _filter_available_layouts(layouts: List[str]) -> List[str]:
    avail = [l for l in layouts if _layout_exists(l)]
    if not avail:
        # Always keep at least random3 if present
        return [l for l in ["random3"] if _layout_exists(l)]
    return avail


def generate_pretrain_corpus(
    total_samples: int = 20000,
    per_shard: int = 5000,
    layouts: List[str] = LAYOUTS,
    teammate_modes: List[str] = TEAMMATE_MODES,
    jobs: int = 1,
    synth_cmd_frac: float = 0.15,
    balance_groups: bool = True,
):
    """Generate macro pretraining shards across layouts/teammates.
    Writes JSONL shards under pretrain_shards/.
    """
    import os, math
    from dataclasses import asdict

    # Filter layouts to those available on disk
    layouts = _filter_available_layouts(layouts)
    if not layouts:
        print("No available layouts found; aborting generation.")
        return
    else:
        print(f"Using layouts: {layouts}")

    shards = math.ceil(total_samples / per_shard)
    seeds = list(range(total_samples))

    # Round-robin pairing for balanced coverage
    pairs = [(l, m) for l in layouts for m in teammate_modes]
    def pair_for(i: int):
        return pairs[i % len(pairs)]

    os.makedirs("pretrain_shards", exist_ok=True)

    for si in range(shards):
        lo, hi = si * per_shard, min((si + 1) * per_shard, total_samples)
        shard_seeds = seeds[lo:hi]

        counts = {"NAV": 0, "ACT": 0, "WAIT": 0}
        def want(sample_macro_id: str) -> bool:
            if not balance_groups:
                return True
            g = _macro_group(sample_macro_id)
            tgt = TARGET_GROUPS[g] * per_shard
            return counts[g] < tgt

        if jobs == 1:
            samples = []
            for k, seed in enumerate(shard_seeds):
                layout, mode = pair_for(lo + k)
                gen = build_generator(layout, mode)
                s = gen._generate_single_sample(seed)
                # Optional synthetic command
                if random.random() < synth_cmd_frac:
                    cmd = render_synthetic_command(s.macro_id, s.macro_args)
                    s.command_text = cmd
                    s.text = s.text.replace("[NO_CMD]", f"[COMMAND] {cmd}")
                if gen._passes_quality_gates(s) and want(s.macro_id):
                    samples.append(s)
                    counts[_macro_group(s.macro_id)] += 1
        else:
            import multiprocessing as mp
            def worker(task):
                seed, idx = task
                layout, mode = pair_for(idx)
                try:
                    gen = build_generator(layout, mode)
                    s = gen._generate_single_sample(seed)
                    if random.random() < synth_cmd_frac:
                        cmd = render_synthetic_command(s.macro_id, s.macro_args)
                        s.command_text = cmd
                        s.text = s.text.replace("[NO_CMD]", f"[COMMAND] {cmd}")
                    return (s, _macro_group(s.macro_id), gen._passes_quality_gates(s))
                except Exception:
                    return (None, None, False)
            tasks = [(seed, lo + i) for i, seed in enumerate(shard_seeds)]
            with mp.Pool(processes=jobs) as pool:
                out = pool.map(worker, tasks, chunksize=64)
            samples = []
            for s, grp, ok in out:
                if s is None or not ok:
                    continue
                if want(s.macro_id):
                    samples.append(s)
                    counts[grp] += 1

        # Save shard
        fname = f"pretrain_shards/macro_pretrain_{si:02d}.jsonl"
        with open(fname, "w") as f:
            for s in samples:
                json.dump(asdict(s), f)
                f.write("\n")
        print(f"[shard {si+1}/{shards}] wrote {len(samples)} → {fname}")


if __name__ == "__main__":
    # If run directly, produce a tiny sanity shard
    generate_pretrain_corpus(total_samples=200, per_shard=200, jobs=1)
