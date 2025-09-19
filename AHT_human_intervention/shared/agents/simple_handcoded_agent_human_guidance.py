#!/usr/bin/env python3
"""
Simple Handcoded Agent - "Dumber" version without teammate modeling

A simplified agent that:
- Does no teammate modeling (teammate never treated specially in planning)
- No collision-aware replanning (if blocked, tries cached step or stays)
- No task allocation or division of labor (only reasons about its own hands)
- Uses A* pathfinding with teammate avoidance
"""

import sys
import random
import numpy as np
from collections import deque
import heapq

# Add project root to path
sys.path.append('.')

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction


class SimpleHandcodedAgentHumanGuidance:
    """Simple Handcoded Agent without sophisticated teammate modeling."""
    
    def __init__(self, mdp, agent_idx=0, agent_name="SimpleHandcodedAgentHumanGuidance"):
        self.mdp = mdp
        self.agent_idx = agent_idx
        self.agent_name = agent_name
        self.current_macro = "GET_ONION"
        self.step_count = 0
        self.heuristic = f"{agent_name}: Initializing"
        
        # Cache important locations
        self.onion_locations = list(self.mdp.get_onion_dispenser_locations())
        self.dish_locations = list(self.mdp.get_dish_dispenser_locations())
        self.serve_locations = list(self.mdp.get_serving_locations())
        self.pot_locations = list(self.mdp.get_pot_locations())
        
        # Cache valid positions for BFS
        self.valid_positions = self._get_valid_positions()
        
        # Current path being followed
        self.current_path = []
        self.path_target = None
        
        # Current target for each macro (to avoid oscillation)
        self.current_onion_target = None
        self.current_pot_target = None
        self.current_dish_target = None
        self.current_serve_target = None
        self.target_persistence = 3  # Lower persistence - give up targets more easily
        self.target_failures = {}  # Track failures for each target
        
        # Simple anti-stuck mechanism (no collision-aware replanning)
        self.last_positions = []
        self.stuck_counter = 0
        self.max_stuck_steps = 10  # Lower threshold to get stuck more easily
        
        # LLM Intervention capabilities (kept for compatibility)
        self.next_action_override = None
        self.intervention_steps_remaining = 0  # Track how many steps to persist intervention
        self.replan_after_override = False
        self.replanning_needed = False
        self.execution_guidance = None
        self.macro_override = None  # Track macro override from interventions
        
        # Alternative pathfinding strategies for interventions
        self.pathfinding_strategy = "shortest"  # "shortest", "counterclockwise", "clockwise", "random"
        self.strategy_until_step = None  # When to revert to normal strategy
        
        print(f"🔧 {agent_name} initialized:")
        print(f"   Onion dispensers: {self.onion_locations}")
        print(f"   Dish dispensers: {self.dish_locations}")
        print(f"   Pots: {self.pot_locations}")
        print(f"   Serving counters: {self.serve_locations}")
        print(f"   Valid positions: {len(self.valid_positions)} walkable cells")
    
    def apply_intervention(self, intervention_result: dict):
        """Apply LLM intervention result."""
        print(f"🎯 Applying intervention: {intervention_result.get('reasoning', 'No reasoning provided')}")
        
        # Handle action override (immediate action)
        action_override = intervention_result.get("action_override")
        if action_override is not None:
            self.next_action_override = action_override
            # Use duration from LLM, default to 1 for all actions
            duration = intervention_result.get("duration")
            if duration is None:
                duration = 1  # Default to single step for all actions
            self.intervention_steps_remaining = duration
            self.replan_after_override = True
            print(f"🎯 Action override: {action_override} (will persist for {duration} steps)")
        
        # Handle macro override
        macro_override = intervention_result.get("macro_override")
        if macro_override:
            self.macro_override = macro_override  # Set the override flag
            self.current_macro = macro_override
            # Clear current targets to force re-evaluation with new macro
            self.current_onion_target = None
            self.current_pot_target = None
            self.current_dish_target = None
            self.current_serve_target = None
            self.replanning_needed = True
            print(f"🎯 Overriding macro to {macro_override}")
            print(f"🔄 Replanning triggered for new macro")
    
    def get_intervention_state(self, state):
        """Get current state information for LLM intervention."""
        ego = state.players[self.agent_idx]
        mate = state.players[1 - self.agent_idx]  # Other agent
        ego_pos = ego.position
        mate_pos = mate.position
        ego_item = self._get_item_name(ego.get_object()) if ego.has_object() else "none"
        mate_item = self._get_item_name(mate.get_object()) if mate.has_object() else "none"
        
        # Get agent orientations
        ego_facing = self._direction_to_str(ego.orientation)
        mate_facing = self._direction_to_str(mate.orientation)
        
        # Get layout dimensions
        layout_name = getattr(self.mdp, 'layout_name', 'unknown')
        terrain = self.mdp.terrain_mtx
        height, width = len(terrain), len(terrain[0])
        
        # Get all object locations
        onion_locs = []
        dish_locs = []
        serve_locs = []
        pot_states = []
        
        for obj_pos, obj in state.objects.items():
            obj_name = self._get_item_name(obj)
            if obj_name == "onion":
                onion_locs.append(obj_pos)
            elif obj_name == "dish":
                dish_locs.append(obj_pos)
            elif obj_name == "serving_counter":
                serve_locs.append(obj_pos)
            elif obj_name == "pot":
                # Get pot state
                if hasattr(obj, 'state'):
                    soup_type, num_items, cook_time = obj.state
                    if num_items > 0:
                        pot_states.append(f"{obj_pos} {num_items}/3")
                    else:
                        pot_states.append(f"{obj_pos} empty")
                else:
                    pot_states.append(f"{obj_pos} empty")
        
        # Determine target position based on current macro
        target_pos = None
        if self.current_macro == "SERVE_SOUP":
            target_pos = self.serve_locations[0] if self.serve_locations else None
        elif self.current_macro == "GET_DISH":
            target_pos = self.dish_locations[0] if self.dish_locations else None
        elif self.current_macro == "TAKE_SOUP":
            # Find ready soup pot
            for pot_pos in self.pot_locations:
                pot_obj = None
                for obj_pos, obj in state.objects.items():
                    if obj_pos == pot_pos and hasattr(obj, 'state'):
                        pot_obj = obj
                        break
                if pot_obj and hasattr(pot_obj, 'state'):
                    soup_type, num_items, cook_time = pot_obj.state
                    if cook_time == 20:  # Ready soup
                        target_pos = pot_pos
                        break
        
        # Create enhanced state text
        state_text = f"""[SCENE] size {width}x{height}
[EGO] pos {ego_pos} facing {ego_facing} item {ego_item}
[MATE] pos {mate_pos} facing {mate_facing} item {mate_item}
[GOAL] {self.current_macro.lower()} | [OPT_CMD]"""
        
        # Add object locations
        if onion_locs:
            onion_str = "".join([str(pos) for pos in onion_locs])
            state_text += f"\n[ONION] {onion_str}"
        
        if dish_locs:
            dish_str = "".join([str(pos) for pos in dish_locs])
            state_text += f"\n[DISH] {dish_str}"
            
        if serve_locs:
            serve_str = "".join([str(pos) for pos in serve_locs])
            state_text += f"\n[SERVE] {serve_str}"
        
        if pot_states:
            pot_str = " ".join(pot_states)
            state_text += f"\n[POTS] {pot_str}"
        
        # Get terrain matrix for layout map
        terrain = self.mdp.terrain_mtx if hasattr(self.mdp, 'terrain_mtx') else []
        
        # Create objects dictionary for layout map
        objects = {}
        for obj_pos, obj in state.objects.items():
            obj_name = self._get_item_name(obj)
            if obj_name in ["onion", "dish", "pot", "serving_counter"]:
                objects[obj_pos] = obj_name
        
        # Convert pot_states to proper format
        formatted_pot_states = []
        for pot_state_str in pot_states:
            parts = pot_state_str.split()
            if len(parts) >= 2:
                pos_str = parts[0]
                state_str = ' '.join(parts[1:])
                # Parse position from string like "(3,0)"
                try:
                    pos = eval(pos_str)  # Convert "(3,0)" to (3,0)
                    formatted_pot_states.append({'pos': pos, 'state': state_str})
                except:
                    formatted_pot_states.append({'pos': (0, 0), 'state': 'unknown'})
        
        return {
            'agent_pos': ego_pos,
            'target_pos': target_pos,
            'current_macro': self.current_macro,
            'ego_item': ego_item,  # Renamed from holding_item for consistency
            'state_text': state_text,
            'layout': layout_name,
            'terrain': terrain,
            'objects': objects,
            'mate_pos': mate_pos,
            # Additional fields for layout-agnostic formatter
            'scene_size': (width, height),
            'agent_orient': ego_facing,
            'mate_orient': mate_facing,
            'mate_item': mate_item,
            'onion_locations': onion_locs,
            'dish_locations': dish_locs,
            'serve_locations': serve_locs,
            'pot_states': formatted_pot_states,
            # Template-specific fields
            't': getattr(state, 'timestep', 0),  # Time step
            'onion_disp_positions': onion_locs,
            'dish_disp_position': dish_locs[0] if dish_locs else 'unknown',
            'pot_summaries': '; '.join([f"{state['pos']}: {state['state']}" for state in formatted_pot_states]) if formatted_pot_states else 'none',
            'serve_position': serve_locs[0] if serve_locs else 'unknown',
            'blocked_tiles_or_none': 'none',  # Could be enhanced to detect blocked tiles
            'items_list_or_none': 'none',  # Could be enhanced to list items on counters
            'ego_macro_or_none': self.current_macro,
            'brief_history_or_none': 'none',  # Could be enhanced to track recent events
        }
    
    def _get_valid_positions(self):
        """Get all valid (walkable) positions."""
        try:
            return set(self.mdp.get_valid_player_positions())
        except:
            # Fallback: scan terrain matrix
            valid_positions = set()
            terrain = self.mdp.terrain_mtx
            for y in range(len(terrain)):
                for x in range(len(terrain[0])):
                    if terrain[y][x] != 'X':  # X = wall
                        valid_positions.add((x, y))
            return valid_positions
    
    def _get_item_name(self, obj):
        """Get the name of an object."""
        if obj is None:
            return None
        return getattr(obj, 'name', str(obj))
    
    def _direction_to_str(self, direction):
        """Convert direction tuple to string."""
        if direction == (0, -1):
            return "N"
        elif direction == (0, 1):
            return "S"
        elif direction == (1, 0):
            return "E"
        elif direction == (-1, 0):
            return "W"
        return str(direction)
    
    def _get_neighbors(self, pos):
        """Get valid neighboring positions."""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if new_pos in self.valid_positions:
                neighbors.append(new_pos)
        return neighbors
    
    def _get_direction_to_face(self, from_pos, to_pos):
        """Get the direction action needed to face from_pos toward to_pos."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if dx == 1 and dy == 0:
            return 1  # EAST
        elif dx == -1 and dy == 0:
            return 4  # WEST
        elif dx == 0 and dy == 1:
            return 2  # SOUTH
        elif dx == 0 and dy == -1:
            return 3  # NORTH
        else:
            return 0  # STAY
    
    def _get_direction_from_positions(self, current_pos, next_pos):
        """Get the movement direction from current position to next position."""
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        if dx == 1 and dy == 0:
            return 1  # EAST
        elif dx == -1 and dy == 0:
            return 4  # WEST
        elif dx == 0 and dy == 1:
            return 2  # SOUTH
        elif dx == 0 and dy == -1:
            return 3  # NORTH
        else:
            return 0  # STAY
    
    def _is_facing_target(self, agent_pos, agent_orientation, target_pos):
        """Check if agent is facing the target."""
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        
        # Expected direction to face target
        if dx > 0:  # Target is to the right
            expected_orientation = (1, 0)  # EAST
        elif dx < 0:  # Target is to the left
            expected_orientation = (-1, 0)  # WEST
        elif dy > 0:  # Target is below
            expected_orientation = (0, 1)  # SOUTH
        elif dy < 0:  # Target is above
            expected_orientation = (0, -1)  # NORTH
        else:
            return True  # Same position
        
        return agent_orientation == expected_orientation
    
    def _is_adjacent(self, pos1, pos2):
        """Check if two positions are adjacent (Manhattan distance = 1)."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1
    
    def _analyze_pot_states(self, state):
        """Analyze the state of all pots for cooking logic."""
        pot_states = {
            'any_pot_ready': False,
            'any_pot_cooking': False,
            'any_pot_has_space': False,
            'pots_info': []
        }
        
        try:
            # Get pot objects from state
            for pot_pos in self.pot_locations:
                # In Overcooked, pots are objects in the state
                pot_obj = None
                for obj_pos, obj in state.objects.items():
                    if obj_pos == pot_pos and hasattr(obj, 'state'):
                        pot_obj = obj
                        break
                
                if pot_obj and hasattr(pot_obj, 'state'):
                    # Pot state is a tuple: (soup_type, num_items, cook_time)
                    soup_type, num_items, cook_time = pot_obj.state
                    
                    if num_items == 3 and cook_time >= 20:  # Ready soup
                        pot_states['any_pot_ready'] = True
                        pot_states['pots_info'].append({'pos': pot_pos, 'state': 'ready', 'onions': num_items, 'cook_time': cook_time})
                    elif num_items == 3 and cook_time < 20:  # Cooking
                        pot_states['any_pot_cooking'] = True
                        pot_states['pots_info'].append({'pos': pot_pos, 'state': 'cooking', 'onions': num_items, 'cook_time': cook_time})
                    elif num_items < 3:  # Has space for more onions
                        pot_states['any_pot_has_space'] = True
                        if num_items > 0:
                            pot_states['pots_info'].append({'pos': pot_pos, 'state': 'partial', 'onions': num_items, 'cook_time': cook_time})
                        else:
                            pot_states['pots_info'].append({'pos': pot_pos, 'state': 'empty', 'onions': num_items, 'cook_time': cook_time})
                else:
                    # No pot object found, assume empty
                    pot_states['any_pot_has_space'] = True
                    pot_states['pots_info'].append({'pos': pot_pos, 'state': 'empty', 'onions': 0, 'cook_time': 0})
            
            # Debug output
            print(f"[POT_ANALYSIS] {self.agent_name}: ready={pot_states['any_pot_ready']}, "
                  f"cooking={pot_states['any_pot_cooking']}, has_space={pot_states['any_pot_has_space']}")
            for pot_info in pot_states['pots_info']:
                print(f"  Pot {pot_info['pos']}: {pot_info['state']} ({pot_info['onions']} onions, cook_time={pot_info['cook_time']})")
            
        except Exception as e:
            print(f"[ERROR] Pot analysis failed: {e}")
            # Fallback: assume pots have space
            pot_states['any_pot_has_space'] = True
        
        return pot_states
    
    def hms_decision(self, state):
        """Make HMS decision based on state with cooking logic."""
        try:
            ego = state.players[self.agent_idx]
            ego_item = self._get_item_name(ego.get_object()) if ego.has_object() else None
            
            # Analyze pot states for smart cooking logic
            pot_states = self._analyze_pot_states(state)
            
            # Debug: Print decision reasoning
            print(f"[HMS] {self.agent_name} decision: holding={ego_item}, ready={pot_states['any_pot_ready']}, cooking={pot_states['any_pot_cooking']}, has_space={pot_states['any_pot_has_space']}")
            print(f"[HMS] {self.agent_name} ego.has_object()={ego.has_object()}, ego.get_object()={ego.get_object() if ego.has_object() else None}")
            
            # Debug soup state if holding soup
            if ego_item == "soup" and ego.has_object():
                soup_obj = ego.get_object()
                print(f"[SOUP_DEBUG] {self.agent_name} holding soup: {soup_obj}")
            
            if ego_item == "soup":
                print(f"[HMS] {self.agent_name} holding soup - selecting SERVE_SOUP macro")
                return "SERVE_SOUP"
            elif ego_item == "dish":
                # SIMPLIFIED: Always try to take soup, even if not ready (more likely to get stuck)
                return "TAKE_SOUP"
            elif ego_item == "onion":
                # Check if pots need more onions
                if pot_states['any_pot_has_space']:
                    return "PUT_ONION_IN_POT"
                else:
                    # All pots are full/cooking, get a dish to prepare for soup
                    return "GET_DISH"
            elif ego_item and ego_item != "soup":
                # Holding something that's not soup - drop it off
                print(f"[HMS] {self.agent_name} holding {ego_item} (not soup) - selecting DROP_OFF macro")
                return "DROP_OFF"
            else:
                # Not holding anything - decide what to do
                if pot_states['any_pot_ready']:
                    return "GET_DISH"  # Get dish to take ready soup
                elif pot_states['any_pot_cooking']:
                    return "GET_DISH"  # Get dish and wait for cooking
                elif pot_states['any_pot_has_space']:
                    return "GET_ONION"  # Get onion to fill pots
                else:
                    return "GET_ONION"  # Default
            
        except Exception as e:
            print(f"[ERROR] HMS decision failed: {e}")
            return "GET_ONION"
    
    def _astar_path(self, start, goal, state=None):
        """Find shortest path using A* with teammate avoidance."""
        if start == goal:
            return []
        
        if start not in self.valid_positions or goal not in self.valid_positions:
            return []
        
        # Get occupied positions (other agents) to avoid
        occupied_positions = set()
        if state is not None:
            for i, player in enumerate(state.players):
                if i != self.agent_idx:  # Don't avoid self
                    occupied_positions.add(player.position)
        
        # A* search with priority queue: (f_cost, g_cost, position, path)
        # f_cost = g_cost + h_cost (heuristic)
        # g_cost = actual distance from start
        # h_cost = Manhattan distance to goal
        heap = [(0, 0, start, [])]
        visited = set()
        g_costs = {start: 0}
        
        while heap:
            f_cost, g_cost, current_pos, path = heapq.heappop(heap)
            
            if current_pos in visited:
                continue
                
            visited.add(current_pos)
            
            if current_pos == goal:
                return path + [current_pos]
            
            for neighbor in self._get_neighbors(current_pos):
                # Avoid other agents
                if neighbor in occupied_positions:
                    continue
                
                if neighbor in visited:
                    continue
                
                # Calculate costs
                new_g_cost = g_cost + 1
                h_cost = self._manhattan_distance(neighbor, goal)
                new_f_cost = new_g_cost + h_cost
                
                # Only add if we found a better path to this neighbor
                if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = new_g_cost
                    heapq.heappush(heap, (new_f_cost, new_g_cost, neighbor, path + [current_pos]))
        
        return []  # No path found
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _find_adjacent_positions(self, target_pos):
        """Find positions adjacent to target."""
        adjacent_positions = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_pos = (target_pos[0] + dx, target_pos[1] + dy)
            if adj_pos in self.valid_positions:
                adjacent_positions.append(adj_pos)
        return adjacent_positions
    
    def _get_best_adjacent_position(self, start_pos, target_pos, state=None):
        """Get the best adjacent position to target (closest to start)."""
        adjacent_positions = self._find_adjacent_positions(target_pos)
        if not adjacent_positions:
            return None
        
        # If we are already on a valid adjacent tile, prefer staying
        if start_pos in adjacent_positions:
            return start_pos
        
        # Find the adjacent position with shortest path from start
        best_pos = None
        shortest_path_length = float('inf')
        
        for adj_pos in adjacent_positions:
            path = self._astar_path(start_pos, adj_pos, state)
            # Treat zero-length path (already adjacent) as valid and shortest
            if path is not None:
                path_len = len(path)
                if path_len < shortest_path_length:
                    shortest_path_length = path_len
                    best_pos = adj_pos
        
        return best_pos
    
    def _get_closest_target(self, current_pos, targets, state=None):
        """Get closest target using BFS path length with alternative strategies."""
        if not targets:
            return None
        
        # Check if we should revert to normal strategy
        if self.strategy_until_step and self.step_count >= self.strategy_until_step:
            self.pathfinding_strategy = "shortest"
            self.strategy_until_step = None
            print(f"✅ Reverted to SHORTEST pathfinding strategy")
        
        # Apply alternative pathfinding strategies
        if self.pathfinding_strategy == "counterclockwise":
            return self._get_target_counterclockwise(current_pos, targets, state)
        elif self.pathfinding_strategy == "clockwise":
            return self._get_target_clockwise(current_pos, targets, state)
        elif self.pathfinding_strategy == "random":
            return self._get_target_random(current_pos, targets, state)
        else:
            # Default: shortest path
            return self._get_target_shortest(current_pos, targets, state)
    
    def _get_target_shortest(self, current_pos, targets, state=None):
        """Get closest target using shortest A* path with teammate avoidance."""
        best_target = None
        min_distance = float('inf')
        
        # Filter out targets that have failed too many times due to teammate blocking
        available_targets = []
        for target in targets:
            failures = self.target_failures.get(target, 0)
            if failures < self.target_persistence:
                available_targets.append(target)
            else:
                print(f"[TARGET_GIVEUP] Giving up on target {target} after {failures} failures")
        
        if not available_targets:
            # Reset failures if all targets failed
            print(f"[TARGET_RESET] All targets failed, resetting failure counts")
            self.target_failures.clear()
            available_targets = targets
        
        for target in available_targets:
            adj_pos = self._get_best_adjacent_position(current_pos, target, state)
            if adj_pos:
                if adj_pos == current_pos:
                    path_len = 0
                    no_path = False
                else:
                    path = self._astar_path(current_pos, adj_pos, state)
                    no_path = (path is None or (isinstance(path, list) and len(path) == 0))
                    path_len = len(path) if path else float('inf')
                if path_len < min_distance:
                    min_distance = path_len
                    best_target = target
                elif no_path and adj_pos != current_pos:
                    # Target is blocked by teammates - track failure (only when not already adjacent)
                    self.target_failures[target] = self.target_failures.get(target, 0) + 1
                    print(f"[TARGET_BLOCKED] Target {target} blocked by teammates, failures: {self.target_failures[target]}")
        
        return best_target
    
    def _get_target_counterclockwise(self, current_pos, targets, state=None):
        """Get target using counterclockwise routing strategy."""
        if not targets:
            return None
        
        # For counterclockwise, prefer targets that are "left" of the direct path
        # This is a simplified heuristic - in practice you'd want more sophisticated routing
        best_target = None
        best_score = float('inf')
        
        for target in targets:
            adj_pos = self._get_best_adjacent_position(current_pos, target, state)
            if adj_pos:
                path = self._astar_path(current_pos, adj_pos, state)
                if path:
                    # Add penalty for direct paths, bonus for longer/indirect paths
                    path_length = len(path)
                    # Simple counterclockwise heuristic: prefer paths that go "around"
                    counterclockwise_bonus = path_length * 0.5  # Bonus for longer paths
                    score = path_length - counterclockwise_bonus
                    
                    if score < best_score:
                        best_score = score
                        best_target = target
        
        print(f"🔄 COUNTERCLOCKWISE: Selected target {best_target} (score: {best_score:.1f})")
        return best_target
    
    def _get_target_clockwise(self, current_pos, targets, state=None):
        """Get target using clockwise routing strategy."""
        if not targets:
            return None
        
        # Similar to counterclockwise but with different preferences
        best_target = None
        best_score = float('inf')
        
        for target in targets:
            adj_pos = self._get_best_adjacent_position(current_pos, target, state)
            if adj_pos:
                path = self._astar_path(current_pos, adj_pos, state)
                if path:
                    path_length = len(path)
                    # Clockwise heuristic: different routing preferences
                    clockwise_bonus = path_length * 0.3
                    score = path_length - clockwise_bonus
                    
                    if score < best_score:
                        best_score = score
                        best_target = target
        
        print(f"🔄 CLOCKWISE: Selected target {best_target} (score: {best_score:.1f})")
        return best_target
    
    def _get_target_random(self, current_pos, targets, state=None):
        """Get target using random selection."""
        if not targets:
            return None
        
        # Randomly select from available targets
        target = random.choice(targets)
        print(f"🎲 RANDOM: Selected target {target}")
        return target
    
    def _follow_astar_path_to_target(self, current_pos, target_pos, state=None):
        """Follow A* path to target with simple anti-stuck mechanism."""
        # Simple anti-stuck mechanism (no collision-aware replanning)
        self.last_positions.append(current_pos)
        if len(self.last_positions) > self.max_stuck_steps:
            self.last_positions.pop(0)
        
        # Check if we're stuck (same position for too long)
        if len(self.last_positions) == self.max_stuck_steps and len(set(self.last_positions)) <= 2:
            self.stuck_counter += 1
            print(f"[STUCK] {self.agent_name} detected stuck at {current_pos}, counter={self.stuck_counter}")
            
            # Simple stuck resolution: try random move or give up path
            if self.stuck_counter >= 10:  # Much higher threshold - less likely to escape
                # Try random move
                neighbors = self._get_neighbors(current_pos)
                if neighbors:
                    random_move = neighbors[self.stuck_counter % len(neighbors)]
                    print(f"[ANTI-STUCK] {self.agent_name} making random move to {random_move}")
                    direction = self._get_direction_from_positions(current_pos, random_move)
                    self.stuck_counter = 0
                    self.last_positions = []
                    return direction
                else:
                    # Give up current path
                    print(f"[GIVE_UP] {self.agent_name} giving up path to {target_pos}")
                    self.current_path = []
                    self.path_target = None
                    self.stuck_counter = 0
                    self.last_positions = []
                    return 0  # Stay
        else:
            # Reset stuck counter if making progress (but less aggressively)
            if self.stuck_counter > 0:
                self.stuck_counter = max(0, self.stuck_counter - 0.5)  # Slower recovery
        
        # Check if we need a new path (target changed or path is empty)
        if self.path_target != target_pos or not self.current_path:
            # Find best adjacent position to target
            best_adj_pos = self._get_best_adjacent_position(current_pos, target_pos, state)
            if not best_adj_pos:
                return 0
            
            # Calculate new BFS path
            new_path = self._astar_path(current_pos, best_adj_pos, state)
            if not new_path:
                return 0
            
            self.current_path = new_path
            self.path_target = target_pos
            print(f"[A*] {self.agent_name} new path to {target_pos}: {len(new_path)} steps")
            print(f"[A*] {self.agent_name} path: {new_path}")
        
        # Follow the current path
        if self.current_path:
            next_pos = self.current_path.pop(0)
            direction = self._get_direction_from_positions(current_pos, next_pos)
            print(f"[A*] {self.agent_name} following path: {current_pos} -> {next_pos} ({['STAY', 'EAST', 'SOUTH', 'NORTH', 'WEST'][direction]})")
            return direction
        
        return 0
    
    def _get_onion_with_astar(self, ego_pos, ego_orientation, state):
        """Get onion using A* pathfinding - NO COORDINATION."""
        if not self.onion_locations:
            return 0
        
        # Use persistent target to avoid oscillation
        if self.current_onion_target is None:
            self.current_onion_target = self._get_closest_target(ego_pos, self.onion_locations, state)
            if not self.current_onion_target:
                self.current_onion_target = self.onion_locations[0]
        
        target_onion = self.current_onion_target
        
        # Check if we're stuck getting to the current onion target
        # Track position history for onion-specific stuck detection
        if not hasattr(self, 'onion_position_history'):
            self.onion_position_history = []
        
        self.onion_position_history.append(ego_pos)
        if len(self.onion_position_history) > 2:
            self.onion_position_history.pop(0)
        
        # Check if stuck for 2 steps without reaching the onion
        is_stuck = (len(self.onion_position_history) == 2 and 
                   len(set(self.onion_position_history)) <= 1 and 
                   not self._is_adjacent(ego_pos, target_onion))
        
        if is_stuck:
            # When stuck getting onions, first try to move EAST to break deadlock
            print(f"[ONION_STUCK] {self.agent_name} stuck at {ego_pos}, trying to move EAST to break deadlock")
            east_pos = (ego_pos[0] + 1, ego_pos[1])
            if east_pos in self.valid_positions:
                direction = self._get_direction_from_positions(ego_pos, east_pos)
                self.onion_position_history = []
                return direction
            
            # If can't move EAST, try switching to alternative dispenser
            if len(self.onion_locations) > 1:
                alternative_targets = [loc for loc in self.onion_locations if loc != target_onion]
                if alternative_targets:
                    self.current_onion_target = alternative_targets[0]
                    print(f"[ONION_STUCK] {self.agent_name} switching to alternative dispenser {self.current_onion_target}")
                    # Reset position history after switching
                    self.onion_position_history = []
                    # Clear current path to force new path calculation to new target
                    self.current_path = []
                    self.path_target = None
                    target_onion = self.current_onion_target
        
        print(f"[SIMPLE] {self.agent_name} targeting onion {target_onion}")
        
        if self._is_adjacent(ego_pos, target_onion):
            if self._is_facing_target(ego_pos, ego_orientation, target_onion):
                print(f"[SIMPLE] {self.agent_name} facing {target_onion} - INTERACTING!")
                # Reset target after successful interaction
                self.current_onion_target = None
                # Reset position history after successful interaction
                self.onion_position_history = []
                return 5  # INTERACT
            else:
                return self._get_direction_to_face(ego_pos, target_onion)
        else:
            return self._follow_astar_path_to_target(ego_pos, target_onion, state)
    
    def _put_onion_in_pot_with_astar(self, ego_pos, ego_orientation, state):
        """Put onion in pot using A* pathfinding - NO COORDINATION."""
        if not self.pot_locations:
            return 0
        
        # Use persistent target to avoid oscillation
        if self.current_pot_target is None:
            self.current_pot_target = self._get_closest_target(ego_pos, self.pot_locations, state)
            if not self.current_pot_target:
                self.current_pot_target = self.pot_locations[0]
        
        target_pot = self.current_pot_target
        
        print(f"[SIMPLE] {self.agent_name} targeting pot {target_pot}")
        
        if self._is_adjacent(ego_pos, target_pot):
            if self._is_facing_target(ego_pos, ego_orientation, target_pot):
                print(f"[SIMPLE] {self.agent_name} facing pot {target_pot} - INTERACTING!")
                # Reset target after successful interaction
                self.current_pot_target = None
                return 5  # INTERACT
            else:
                return self._get_direction_to_face(ego_pos, target_pot)
        else:
            return self._follow_astar_path_to_target(ego_pos, target_pot, state)
    
    def _get_dish_with_astar(self, ego_pos, ego_orientation, state):
        """Get dish using A* pathfinding."""
        if not self.dish_locations:
            return 0
        
        # Use persistent target to avoid oscillation
        if self.current_dish_target is None:
            self.current_dish_target = self._get_closest_target(ego_pos, self.dish_locations, state)
            if not self.current_dish_target:
                self.current_dish_target = self.dish_locations[0]
        target_dish = self.current_dish_target
        print(f"[SIMPLE] {self.agent_name} targeting dish dispenser {target_dish}")
        
        if self._is_adjacent(ego_pos, target_dish):
            if self._is_facing_target(ego_pos, ego_orientation, target_dish):
                print(f"[BFS] {self.agent_name} facing dish dispenser {target_dish} - INTERACTING")
                # Reset target after successful interaction
                self.current_dish_target = None
                return 5  # INTERACT
            else:
                return self._get_direction_to_face(ego_pos, target_dish)
        else:
            return self._follow_astar_path_to_target(ego_pos, target_dish, state)
    
    def _take_soup_with_astar(self, ego_pos, ego_orientation, state):
        """Take soup using A* pathfinding - NO COORDINATION."""
        if not self.pot_locations:
            return 0
        
        # Find pots with ready soup (cook_time == 20)
        ready_pots = []
        for pot_pos in self.pot_locations:
            # Check pot state using the same pattern as OnionSpecialistAgent
            pot_obj = None
            for obj_pos, obj in state.objects.items():
                if obj_pos == pot_pos and hasattr(obj, 'state'):
                    pot_obj = obj
                    break
            
            if pot_obj and hasattr(pot_obj, 'state'):
                soup_type, num_items, cook_time = pot_obj.state
                if cook_time == 20:  # Soup is ready
                    ready_pots.append(pot_pos)
                    print(f"[SOUP] Found ready soup at {pot_pos}: {soup_type}, items={num_items}, cook_time={cook_time}")
        
        if not ready_pots:
            print(f"[SIMPLE] {self.agent_name}: No ready soup found!")
            return 0
        
        # Use persistent target to avoid oscillation
        if self.current_pot_target is None or self.current_pot_target not in ready_pots:
            self.current_pot_target = self._get_closest_target(ego_pos, ready_pots, state)
            if not self.current_pot_target:
                self.current_pot_target = ready_pots[0]
        
        target_pot = self.current_pot_target
        
        print(f"[SIMPLE] {self.agent_name} targeting READY pot {target_pot} for soup")
        
        if self._is_adjacent(ego_pos, target_pot):
            if self._is_facing_target(ego_pos, ego_orientation, target_pot):
                print(f"[SIMPLE] {self.agent_name} facing pot {target_pot} - INTERACTING!")
                # Reset target after successful interaction
                self.current_pot_target = None
                return 5  # INTERACT
            else:
                return self._get_direction_to_face(ego_pos, target_pot)
        else:
            return self._follow_astar_path_to_target(ego_pos, target_pot, state)
    
    def _serve_soup_with_astar(self, ego_pos, ego_orientation, state):
        """Serve soup using A* pathfinding."""
        print(f"[SERVE_SOUP] {self.agent_name} at {ego_pos} trying to serve soup")
        if not self.serve_locations:
            print(f"[SERVE_SOUP] {self.agent_name} ERROR: No serving locations!")
            return 0
        
        target_serve = self.serve_locations[0]
        print(f"[SERVE_SOUP] {self.agent_name} targeting serving counter at {target_serve}")
        
        if self._is_adjacent(ego_pos, target_serve):
            print(f"[SERVE_SOUP] {self.agent_name} adjacent to serving counter at {target_serve}")
            if self._is_facing_target(ego_pos, ego_orientation, target_serve):
                print(f"[SERVE_SOUP] {self.agent_name} facing serving counter - INTERACTING!")
                return 5
            else:
                print(f"[SERVE_SOUP] {self.agent_name} turning to face serving counter")
                return self._get_direction_to_face(ego_pos, target_serve)
        else:
            print(f"[SERVE_SOUP] {self.agent_name} not adjacent to serving counter, pathfinding to {target_serve}")
            return self._follow_astar_path_to_target(ego_pos, target_serve, state)
    
    def _wait_for_cooking(self, ego_pos, ego_orientation, state):
        """Wait for cooking - move to a good waiting position near pots."""
        # Simple waiting: find closest position near any pot
        best_waiting_pos = None
        min_distance = float('inf')
        
        for pot_pos in self.pot_locations:
            # Find adjacent positions to the pot
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                waiting_pos = (pot_pos[0] + dx, pot_pos[1] + dy)
                if waiting_pos in self.valid_positions:
                    path_length = len(self._astar_path(ego_pos, waiting_pos))
                    if path_length > 0 and path_length < min_distance:
                        min_distance = path_length
                        best_waiting_pos = waiting_pos
        
        if best_waiting_pos and best_waiting_pos != ego_pos:
            print(f"[COOKING] {self.agent_name} moving to waiting position {best_waiting_pos}")
            return self._follow_astar_path_to_target(ego_pos, best_waiting_pos, state)
        else:
            # Stay in place if no better position
            return 0
    
    def _get_nearest_wall_tile(self, current_pos):
        """Find the nearest wall tile ('X') to drop off objects."""
        min_distance = float('inf')
        nearest_wall = None
        
        # Get terrain matrix
        terrain = self.mdp.terrain_mtx
        height, width = len(terrain), len(terrain[0])
        
        for y in range(height):
            for x in range(width):
                if terrain[y][x] == 'X':  # Wall tile
                    wall_pos = (x, y)
                    distance = self._manhattan_distance(current_pos, wall_pos)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_wall = wall_pos
        
        return nearest_wall
    
    def _drop_off_with_astar(self, ego_pos, ego_orientation, state):
        """Drop off current object at nearest wall tile."""
        print(f"[DROP_OFF] {self.agent_name} at {ego_pos} trying to drop off object")
        
        # Find nearest wall tile
        nearest_wall = self._get_nearest_wall_tile(ego_pos)
        if not nearest_wall:
            print(f"[DROP_OFF] {self.agent_name} ERROR: No wall tiles found!")
            return 0
        
        print(f"[DROP_OFF] {self.agent_name} targeting wall tile at {nearest_wall}")
        
        # Check if we're adjacent to the wall
        if self._is_adjacent(ego_pos, nearest_wall):
            if self._is_facing_target(ego_pos, ego_orientation, nearest_wall):
                print(f"[DROP_OFF] {self.agent_name} facing wall at {nearest_wall} - INTERACTING to drop off!")
                # Clear macro override after successful drop-off
                self.macro_override = None
                return 5  # INTERACT
            else:
                print(f"[DROP_OFF] {self.agent_name} turning to face wall at {nearest_wall}")
                return self._get_direction_to_face(ego_pos, nearest_wall)
        else:
            print(f"[DROP_OFF] {self.agent_name} not adjacent to wall, pathfinding to {nearest_wall}")
            return self._follow_astar_path_to_target(ego_pos, nearest_wall, state)
    
    def _execute_macro_with_astar(self, macro, state):
        """Execute macro action using A* pathfinding."""
        ego = state.players[self.agent_idx]
        ego_pos = ego.position
        ego_orientation = ego.orientation
        
        try:
            if macro == "GET_ONION":
                return self._get_onion_with_astar(ego_pos, ego_orientation, state)
            elif macro == "PUT_ONION_IN_POT":
                return self._put_onion_in_pot_with_astar(ego_pos, ego_orientation, state)
            elif macro == "GET_DISH":
                return self._get_dish_with_astar(ego_pos, ego_orientation, state)
            elif macro == "TAKE_SOUP":
                return self._take_soup_with_astar(ego_pos, ego_orientation, state)
            elif macro == "SERVE_SOUP":
                return self._serve_soup_with_astar(ego_pos, ego_orientation, state)
            elif macro == "WAIT_FOR_COOKING":
                return self._wait_for_cooking(ego_pos, ego_orientation, state)
            elif macro == "DROP_OFF":
                return self._drop_off_with_astar(ego_pos, ego_orientation, state)
            else:
                return self._get_onion_with_astar(ego_pos, ego_orientation, state)
        except Exception as e:
            print(f"[ERROR] Macro execution failed: {e}")
            return 0
    
    def get_action(self, state):
        """Get action using BFS pathfinding."""
        self.step_count += 1
        
        # Check for action override first (LLM intervention)
        if self.next_action_override is not None and self.intervention_steps_remaining > 0:
            action = self.next_action_override
            self.intervention_steps_remaining -= 1
            
            if self.intervention_steps_remaining == 0:
                self.next_action_override = None  # Clear after duration steps
                print(f"⚡ Using intervention override action: {action} (final step)")
            else:
                print(f"⚡ Using intervention override action: {action} (step {self.intervention_steps_remaining} remaining)")
            
            # If we need to replan after this override action
            if self.replan_after_override and self.intervention_steps_remaining == 0:
                self.replan_after_override = False
                self.replanning_needed = True
                # Clear current targets to force re-evaluation
                self.current_onion_target = None
                self.current_pot_target = None
                self.current_dish_target = None
                self.current_serve_target = None
                print(f"🔄 DIRECT intervention: Will replan next step after override action")
            
            return action
        
        # Check for replanning (LLM intervention) - simple version
        if self.replanning_needed:
            print(f"🔄 Replanning triggered - clearing current path")
            self.replanning_needed = False
            self.current_path = []  # Clear current path to force new BFS
        
        try:
            ego = state.players[self.agent_idx]
            ego_pos = ego.position
            ego_orientation = ego.orientation
            ego_item = self._get_item_name(ego.get_object()) if ego.has_object() else "none"
            
            # Get HMS decision (unless we have a macro override from intervention)
            if hasattr(self, 'macro_override') and self.macro_override:
                macro_decision = self.macro_override
                print(f"🎯 Using macro override: {macro_decision}")
            else:
                macro_decision = self.hms_decision(state)
            self.current_macro = macro_decision
            
            # Execute macro with A* pathfinding
            action = self._execute_macro_with_astar(macro_decision, state)
            
            # Update heuristic display
            facing_str = self._direction_to_str(ego_orientation)
            path_info = f"path:{len(self.current_path)}" if self.current_path else "no_path"
            self.heuristic = f"{self.agent_name}: {macro_decision} @ {ego_pos} facing {facing_str} ({path_info}) (holding: {ego_item})"
            
            return action
            
        except Exception as e:
            print(f"[ERROR] Simple BFS agent {self.agent_idx} failed: {e}")
            self.heuristic = f"{self.agent_name}: Error - staying"
            return 0
    
    def action(self, state):
        """Method called by AgentPair."""
        return self.get_action(state)
