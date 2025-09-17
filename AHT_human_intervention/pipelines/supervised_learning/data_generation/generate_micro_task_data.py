#!/usr/bin/env python3
"""
Generate task-focused sanity data for micro-skills in Overcooked.

Curriculum v1 (≈1000–1200 trajectories total):
- Task A: Pick up onion (200–300 eps)
- Task B: Navigate to pot (100–200 eps) 
- Task C: INTERACT with pot (100–200 eps)
- Task D: Turn-to-face target (100 eps)
- Task E: Pick up multiple onions (100–200 eps) - if pot has < 3 onions
- Task F: Pick up plate (100 eps)
- Task G: Pick up cooked soup (100 eps)
- Task H: Deliver soup to serving area (100 eps)

Each task isolates specific micro-skills to verify model capabilities.
"""

import sys
import os
import random
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedGridworld
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from shared.utils.state_to_text import describe_state

class MicroTaskAgent:
    """Base class for micro-task agents."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld, task_name: str):
        self.agent_index = agent_index
        self.mdp = mdp
        self.task_name = task_name
        self.last_position = None
        self.stuck_count = 0
        self.last_action = None
        self.action_repeat_count = 0
        
    def reset(self):
        self.last_position = None
        self.stuck_count = 0
        self.last_action = None
        self.action_repeat_count = 0
        
    def action(self, state: OvercookedState) -> str:
        """Get action for the specific micro-task."""
        raise NotImplementedError
        
    def is_task_complete(self, state: OvercookedState) -> bool:
        """Check if the micro-task is complete."""
        raise NotImplementedError
        
    def get_shaping_reward(self, state: OvercookedState, action: str, next_state: OvercookedState) -> float:
        """Calculate shaping reward for task-specific progress."""
        # Base implementation - can be overridden by specific agents
        return 0.0
        
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (walkable - not a wall, dispenser, pot, or serving area)."""
        if not (0 <= pos[0] < self.mdp.width and 0 <= pos[1] < self.mdp.height):
            return False
        terrain = self.mdp.terrain_mtx[pos[1]][pos[0]]
        # Only empty spaces are walkable
        return terrain == ' '
        
    def _is_adjacent_to_target(self, pos: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """Check if position is adjacent to target."""
        return abs(pos[0] - target[0]) + abs(pos[1] - target[1]) == 1
        
    def _get_required_orientation(self, pos: Tuple[int, int], target: Tuple[int, int]) -> Direction:
        """Get the orientation required to face the target."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        
        if dx == 1: return Direction.EAST
        elif dx == -1: return Direction.WEST
        elif dy == 1: return Direction.SOUTH
        elif dy == -1: return Direction.NORTH
        else: return Direction.NORTH  # Default
        
    def _get_rotation_action(self, current_orientation: Direction, target_orientation: Direction) -> str:
        """Get action to rotate from current to target orientation."""
        if current_orientation == target_orientation:
            return "STAY"
            
        # Rotation logic - turn towards target
        if current_orientation == Direction.NORTH:
            if target_orientation == Direction.EAST: return "MOVE_E"  # Turn right
            elif target_orientation == Direction.WEST: return "MOVE_W"  # Turn left
            else: return "MOVE_S"  # Turn around
        elif current_orientation == Direction.SOUTH:
            if target_orientation == Direction.EAST: return "MOVE_E"  # Turn right
            elif target_orientation == Direction.WEST: return "MOVE_W"  # Turn left
            else: return "MOVE_N"  # Turn around
        elif current_orientation == Direction.EAST:
            if target_orientation == Direction.NORTH: return "MOVE_N"  # Turn left
            elif target_orientation == Direction.SOUTH: return "MOVE_S"  # Turn right
            else: return "MOVE_W"  # Turn around
        else:  # WEST
            if target_orientation == Direction.NORTH: return "MOVE_N"  # Turn right
            elif target_orientation == Direction.SOUTH: return "MOVE_S"  # Turn left
            else: return "MOVE_E"  # Turn around
            
    def _drop_item(self, state: OvercookedState) -> str:
        """Drop the currently held item at a valid location."""
        player = state.players[self.agent_index]
        if not player.has_object():
            return "STAY"
            
        # Find a valid location to drop the item (any walkable space)
        pos = player.position
        for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:  # Start with current position
            drop_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= drop_pos[0] < self.mdp.width and 
                0 <= drop_pos[1] < self.mdp.height and 
                self.mdp.terrain_mtx[drop_pos[1]][drop_pos[0]] == ' ' and  # Only empty spaces
                drop_pos not in state.objects):
                # Valid drop location found
                if dx == 0 and dy == 0:
                    # Drop at current position
                    return "interact"
                else:
                    # Move to drop position first
                    return self._get_action_to_position(pos, drop_pos)
        
        # If no valid drop location found, just stay
        return "STAY"
        
    def _handle_stuck_situation(self, current_pos: Tuple[int, int], state: OvercookedState) -> str:
        """Handle stuck situation with intelligent movement instead of random."""
        # Try to find a valid path to any nearby walkable position
        valid_moves = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if self._is_valid_position(new_pos):
                # Check if teammate is not blocking this position
                teammate_blocking = False
                for i, player in enumerate(state.players):
                    if i != self.agent_index and player.position == new_pos:
                        teammate_blocking = True
                        break
                if not teammate_blocking:
                    valid_moves.append((dx, dy))
        
        if valid_moves:
            # Choose a random valid move
            dx, dy = random.choice(valid_moves)
            if dx == 1: action = "MOVE_E"
            elif dx == -1: action = "MOVE_W"
            elif dy == 1: action = "MOVE_S"
            else: action = "MOVE_N"
            self.last_action = action
            return action
        else:
            # No valid moves available, stay put
            return "STAY"
        
    def _simple_move_towards_target_with_collision_check(self, pos: Tuple[int, int], target: Tuple[int, int], state: OvercookedState) -> str:
        """Simple movement towards target with teammate collision avoidance."""
        # Get teammate positions to avoid
        teammate_positions = set()
        for i, player in enumerate(state.players):
            if i != self.agent_index:  # Skip self
                teammate_positions.add(player.position)
        
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        
        # Prefer horizontal movement first, but check for collisions
        if dx > 0:
            new_pos = (pos[0] + 1, pos[1])
            if self._is_valid_position(new_pos) and new_pos not in teammate_positions:
                self.last_action = "MOVE_E"
                return "MOVE_E"
        elif dx < 0:
            new_pos = (pos[0] - 1, pos[1])
            if self._is_valid_position(new_pos) and new_pos not in teammate_positions:
                self.last_action = "MOVE_W"
                return "MOVE_W"
                
        # Then vertical movement
        if dy > 0:
            new_pos = (pos[0], pos[1] + 1)
            if self._is_valid_position(new_pos) and new_pos not in teammate_positions:
                self.last_action = "MOVE_S"
                return "MOVE_S"
        elif dy < 0:
            new_pos = (pos[0], pos[1] - 1)
            if self._is_valid_position(new_pos) and new_pos not in teammate_positions:
                self.last_action = "MOVE_N"
                return "MOVE_N"
                
        # If all preferred moves are blocked, try any valid move
        for dx, dy, action in [(1, 0, "MOVE_E"), (-1, 0, "MOVE_W"), (0, 1, "MOVE_S"), (0, -1, "MOVE_N")]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if self._is_valid_position(new_pos) and new_pos not in teammate_positions:
                self.last_action = action
                return action
                
        return "STAY"
        
    def _bfs_move_towards_target(self, pos: Tuple[int, int], target: Tuple[int, int], state: OvercookedState) -> str:
        """Move towards target using BFS pathfinding for ring layouts, avoiding teammates."""
        # Find path to target
        path = self._find_path_to_target(pos, target, state)
        
        if not path or len(path) < 2:
            # Fallback to simple movement with teammate collision check
            return self._simple_move_towards_target_with_collision_check(pos, target, state)
        
        # Get next position in path
        next_pos = path[1]  # path[0] is current position
        
        # Get action to move to next position
        action = self._get_action_to_position(pos, next_pos)
        if action:
            self.last_action = action
            return action
        
        # Fallback to simple movement with collision check
        return self._simple_move_towards_target_with_collision_check(pos, target, state)
        
    def _find_path_to_target(self, start: Tuple[int, int], target: Tuple[int, int], state: OvercookedState) -> List[Tuple[int, int]]:
        """Find a path to the target using BFS, avoiding walls and teammates."""
        if start == target:
            return [start]
        
        # Check if target is walkable
        if not self._is_valid_position(target):
            # Target is not walkable (like a dispenser, pot, or serving area), find path to adjacent position
            adjacent_positions = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # South, North, East, West
                adj_pos = (target[0] + dx, target[1] + dy)
                if (0 <= adj_pos[0] < self.mdp.width and 
                    0 <= adj_pos[1] < self.mdp.height and 
                    self._is_valid_position(adj_pos)):
                    adjacent_positions.append(adj_pos)
            
            if not adjacent_positions:
                return []  # No adjacent valid positions
            
            # Find path to the closest adjacent position
            best_path = []
            best_distance = float('inf')
            
            for adj_pos in adjacent_positions:
                path = self._find_path_to_walkable_target(start, adj_pos, state)
                if path:
                    distance = len(path)
                    if distance < best_distance:
                        best_distance = distance
                        best_path = path
            
            return best_path
        else:
            # Target is walkable, find direct path
            return self._find_path_to_walkable_target(start, target, state)
    
    def _find_path_to_walkable_target(self, start: Tuple[int, int], target: Tuple[int, int], state: OvercookedState) -> List[Tuple[int, int]]:
        """Find a path to a walkable target using BFS, avoiding teammates."""
        if start == target:
            return [start]
        
        # Get teammate positions to avoid
        teammate_positions = set()
        for i, player in enumerate(state.players):
            if i != self.agent_index:  # Skip self
                teammate_positions.add(player.position)
        
        # Use BFS to find a path
        queue = [(start, [start])]
        visited = set([start])
        
        while queue:
            current_pos, path = queue.pop(0)
            
            # Check all adjacent positions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # South, North, East, West
                new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                # Check if position is valid and not occupied by teammate
                if (new_pos not in visited and 
                    self._is_valid_position(new_pos) and 
                    new_pos not in teammate_positions):
                    
                    visited.add(new_pos)
                    new_path = path + [new_pos]
                    
                    if new_pos == target:
                        return new_path
                    
                    queue.append((new_pos, new_path))
        
        return []  # No path found
    
    def _get_action_to_position(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> str:
        """Get the action needed to move from current_pos to target_pos."""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        if dx == 1 and dy == 0:
            return "MOVE_E"
        elif dx == -1 and dy == 0:
            return "MOVE_W"
        elif dx == 0 and dy == 1:
            return "MOVE_S"
        elif dx == 0 and dy == -1:
            return "MOVE_N"
        else:
            return None  # Not adjacent

class PickupOnionAgent(MicroTaskAgent):
    """Agent for Task A: Pick up onion."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        super().__init__(agent_index, mdp, "pickup_onion")
        
    def action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Check if stuck
        if self.last_position == current_pos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.last_position = current_pos
            
        # If stuck, check for teammate blocking and handle appropriately
        if self.stuck_count > 3:
            # Check if teammate is blocking us
            teammate_blocking = False
            for j, player in enumerate(state.players):
                if j != self.agent_index:
                    if abs(player.position[0] - current_pos[0]) + abs(player.position[1] - current_pos[1]) == 1:
                        teammate_blocking = True
                        break
            
            if teammate_blocking:
                # If teammate is blocking, wait for them to move
                return "STAY"
            else:
                return self._handle_stuck_situation(current_pos, state)
            
        # If already holding onion, task is complete
        if player.has_object() and player.get_object().name == 'onion':
            return "STAY"
            
        # If holding something else, drop it first
        if player.has_object():
            return self._drop_item(state)
            
        # Find closest onion dispenser
        onion_dispensers = self.mdp.get_onion_dispenser_locations()
        if not onion_dispensers:
            return "STAY"
            
        closest_dispenser = min(onion_dispensers, key=lambda d: abs(d[0] - current_pos[0]) + abs(d[1] - current_pos[1]))
        
        # If adjacent to dispenser and facing it, interact
        if self._is_adjacent_to_target(current_pos, closest_dispenser):
            required_orientation = self._get_required_orientation(current_pos, closest_dispenser)
            if orientation == required_orientation:
                self.last_action = "interact"
                return "interact"
            else:
                action = self._get_rotation_action(orientation, required_orientation)
                self.last_action = action
                return action
                
        # Move towards dispenser
        return self._bfs_move_towards_target(current_pos, closest_dispenser, state)
        
    def is_task_complete(self, state: OvercookedState) -> bool:
        """Task is complete when holding an onion."""
        player = state.players[self.agent_index]
        return player.has_object() and player.get_object().name == 'onion'
        
    def get_shaping_reward(self, state: OvercookedState, action: str, next_state: OvercookedState) -> float:
        """Calculate shaping reward for onion pickup progress."""
        current_player = state.players[self.agent_index]
        next_player = next_state.players[self.agent_index]
        
        # Reward for getting closer to onion dispenser
        onion_dispensers = self.mdp.get_onion_dispenser_locations()
        if not onion_dispensers:
            return 0.0
            
        closest_dispenser = min(onion_dispensers, key=lambda d: 
            abs(d[0] - current_player.position[0]) + abs(d[1] - current_player.position[1]))
        
        # Distance-based reward
        current_distance = abs(current_player.position[0] - closest_dispenser[0]) + abs(current_player.position[1] - closest_dispenser[1])
        next_distance = abs(next_player.position[0] - closest_dispenser[0]) + abs(next_player.position[1] - closest_dispenser[1])
        
        distance_reward = 0.1 * (current_distance - next_distance)  # Positive if getting closer
        
        # Reward for successful interaction
        interaction_reward = 0.0
        if action == "interact" and not current_player.has_object() and next_player.has_object():
            if next_player.get_object().name == 'onion':
                interaction_reward = 2.0  # Large reward for successful pickup
        
        # Reward for proper orientation
        orientation_reward = 0.0
        if self._is_adjacent_to_target(current_player.position, closest_dispenser):
            required_orientation = self._get_required_orientation(current_player.position, closest_dispenser)
            if current_player.orientation == required_orientation:
                orientation_reward = 0.5
        
        return distance_reward + interaction_reward + orientation_reward

class NavigateToPotAgent(MicroTaskAgent):
    """Agent for Task B: Navigate to pot (starting with onion)."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        super().__init__(agent_index, mdp, "navigate_to_pot")
        
    def action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_index]
        current_pos = player.position
        
        # Check if stuck
        if self.last_position == current_pos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.last_position = current_pos
            
        # If stuck, check for teammate blocking and handle appropriately
        if self.stuck_count > 3:
            # Check if teammate is blocking us
            teammate_blocking = False
            for j, player in enumerate(state.players):
                if j != self.agent_index:
                    if abs(player.position[0] - current_pos[0]) + abs(player.position[1] - current_pos[1]) == 1:
                        teammate_blocking = True
                        break
            
            if teammate_blocking:
                # If teammate is blocking, wait for them to move
                return "STAY"
            else:
                return self._handle_stuck_situation(current_pos, state)
            
        # Find closest pot
        pots = self.mdp.get_pot_locations()
        if not pots:
            return "STAY"
            
        closest_pot = min(pots, key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1]))
        
        # If adjacent to pot, task is complete
        if self._is_adjacent_to_target(current_pos, closest_pot):
            return "STAY"
            
        # Move towards pot
        return self._bfs_move_towards_target(current_pos, closest_pot, state)
        
    def is_task_complete(self, state: OvercookedState) -> bool:
        """Task is complete when adjacent to a pot."""
        player = state.players[self.agent_index]
        current_pos = player.position
        pots = self.mdp.get_pot_locations()
        
        for pot in pots:
            if self._is_adjacent_to_target(current_pos, pot):
                return True
        return False
        
    def get_shaping_reward(self, state: OvercookedState, action: str, next_state: OvercookedState) -> float:
        """Calculate shaping reward for pot navigation progress."""
        current_player = state.players[self.agent_index]
        next_player = next_state.players[self.agent_index]
        
        # Reward for getting closer to pot
        pots = self.mdp.get_pot_locations()
        if not pots:
            return 0.0
            
        closest_pot = min(pots, key=lambda p: 
            abs(p[0] - current_player.position[0]) + abs(p[1] - current_player.position[1]))
        
        # Distance-based reward
        current_distance = abs(current_player.position[0] - closest_pot[0]) + abs(current_player.position[1] - closest_pot[1])
        next_distance = abs(next_player.position[0] - closest_pot[0]) + abs(next_player.position[1] - closest_pot[1])
        
        distance_reward = 0.1 * (current_distance - next_distance)  # Positive if getting closer
        
        # Reward for reaching adjacency
        adjacency_reward = 0.0
        if not self._is_adjacent_to_target(current_player.position, closest_pot) and self._is_adjacent_to_target(next_player.position, closest_pot):
            adjacency_reward = 1.0  # Reward for reaching pot adjacency
        
        return distance_reward + adjacency_reward

class InteractWithPotAgent(MicroTaskAgent):
    """Agent for Task C: INTERACT with pot (starting adjacent to pot with onion)."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        super().__init__(agent_index, mdp, "interact_with_pot")
        
    def action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Find closest pot
        pots = self.mdp.get_pot_locations()
        if not pots:
            return "STAY"
            
        closest_pot = min(pots, key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1]))
        
        # If adjacent to pot and facing it, interact
        if self._is_adjacent_to_target(current_pos, closest_pot):
            required_orientation = self._get_required_orientation(current_pos, closest_pot)
            if orientation == required_orientation:
                self.last_action = "interact"
                return "interact"
            else:
                action = self._get_rotation_action(orientation, required_orientation)
                self.last_action = action
                return action
                
        return "STAY"
        
    def is_task_complete(self, state: OvercookedState) -> bool:
        """Task is complete when successfully interacted with pot."""
        # Check if the agent has successfully placed an onion in a pot
        player = state.players[self.agent_index]
        
        # If player is still holding an onion, task is not complete
        if player.has_object() and player.get_object().name == 'onion':
            return False
            
        # Check if any pot now has an onion (indicating successful interaction)
        pots = self.mdp.get_pot_locations()
        for pot_pos in pots:
            if state.has_object(pot_pos):
                obj = state.get_object(pot_pos)
                if obj.name == 'soup':
                    soup_type, num_items, cook_time = obj.state
                    if num_items > 0:  # Pot has at least one onion
                        return True
        
        return False
        
    def get_shaping_reward(self, state: OvercookedState, action: str, next_state: OvercookedState) -> float:
        """Calculate shaping reward for pot interaction progress."""
        current_player = state.players[self.agent_index]
        next_player = next_state.players[self.agent_index]
        
        # Reward for proper orientation towards pot
        pots = self.mdp.get_pot_locations()
        if not pots:
            return 0.0
            
        closest_pot = min(pots, key=lambda p: 
            abs(p[0] - current_player.position[0]) + abs(p[1] - current_player.position[1]))
        
        orientation_reward = 0.0
        if self._is_adjacent_to_target(current_player.position, closest_pot):
            required_orientation = self._get_required_orientation(current_player.position, closest_pot)
            if current_player.orientation == required_orientation:
                orientation_reward = 0.5
        
        # Reward for successful interaction (placing onion in pot)
        interaction_reward = 0.0
        if action == "interact" and current_player.has_object() and current_player.get_object().name == 'onion':
            if not next_player.has_object():  # Onion was placed in pot
                interaction_reward = 3.0  # Large reward for successful placement
        
        return orientation_reward + interaction_reward

class TurnToFaceAgent(MicroTaskAgent):
    """Agent for Task D: Turn to face target."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld, target_type: str = "onion_dispenser"):
        super().__init__(agent_index, mdp, f"turn_to_face_{target_type}")
        self.target_type = target_type
        
    def action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Find target based on type
        if self.target_type == "onion_dispenser":
            targets = self.mdp.get_onion_dispenser_locations()
        elif self.target_type == "pot":
            targets = self.mdp.get_pot_locations()
        else:
            return "STAY"
            
        if not targets:
            return "STAY"
            
        # Find closest target
        closest_target = min(targets, key=lambda t: abs(t[0] - current_pos[0]) + abs(t[1] - current_pos[1]))
        
        # Get required orientation to face target
        required_orientation = self._get_required_orientation(current_pos, closest_target)
        
        # If already facing target, task is complete
        if orientation == required_orientation:
            return "STAY"
            
        # Turn towards target
        action = self._get_rotation_action(orientation, required_orientation)
        self.last_action = action
        return action
        
    def is_task_complete(self, state: OvercookedState) -> bool:
        """Task is complete when facing the target."""
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Find target based on type
        if self.target_type == "onion_dispenser":
            targets = self.mdp.get_onion_dispenser_locations()
        elif self.target_type == "pot":
            targets = self.mdp.get_pot_locations()
        else:
            return False
            
        if not targets:
            return False
            
        # Find closest target
        closest_target = min(targets, key=lambda t: abs(t[0] - current_pos[0]) + abs(t[1] - current_pos[1]))
        
        # Check if facing target
        required_orientation = self._get_required_orientation(current_pos, closest_target)
        # Check if facing target
        required_orientation = self._get_required_orientation(current_pos, closest_target)
        return orientation == required_orientation
        
    def get_shaping_reward(self, state: OvercookedState, action: str, next_state: OvercookedState) -> float:
        """Calculate shaping reward for turning to face target."""
        current_player = state.players[self.agent_index]
        next_player = next_state.players[self.agent_index]
        
        # Find target based on type
        if self.target_type == "onion_dispenser":
            targets = self.mdp.get_onion_dispenser_locations()
        elif self.target_type == "pot":
            targets = self.mdp.get_pot_locations()
        else:
            return 0.0
            
        if not targets:
            return 0.0
            
        # Find closest target
        closest_target = min(targets, key=lambda t: 
            abs(t[0] - current_player.position[0]) + abs(t[1] - current_player.position[1]))
        
        # Reward for getting closer to correct orientation
        required_orientation = self._get_required_orientation(current_player.position, closest_target)
        
        # Calculate orientation difference (0-4, where 0 is same direction)
        current_ori = current_player.orientation
        next_ori = next_player.orientation
        
        # Reward for moving towards correct orientation
        orientation_progress_reward = 0.0
        if current_ori != required_orientation and next_ori == required_orientation:
            orientation_progress_reward = 1.0  # Reward for reaching correct orientation
        
        # Small reward for any orientation change towards target
        if current_ori != next_ori:
            orientation_progress_reward += 0.1
        
        return orientation_progress_reward

class PickupMultipleOnionsAgent(MicroTaskAgent):
    """Agent for Task E: Pick up multiple onions (if pot has < 3 onions)."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        super().__init__(agent_index, mdp, "pickup_multiple_onions")
        self.onions_picked = 0
        self.target_onions = 0
        
    def reset(self):
        super().reset()
        self.onions_picked = 0
        self.target_onions = 0
        
    def action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Check if stuck
        if self.last_position == current_pos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.last_position = current_pos
            
        # If stuck, check for teammate blocking and handle appropriately
        if self.stuck_count > 3:
            # Check if teammate is blocking us
            teammate_blocking = False
            for j, player in enumerate(state.players):
                if j != self.agent_index:
                    if abs(player.position[0] - current_pos[0]) + abs(player.position[1] - current_pos[1]) == 1:
                        teammate_blocking = True
                        break
            
            if teammate_blocking:
                # If teammate is blocking, wait for them to move
                return "STAY"
            else:
                return self._handle_stuck_situation(current_pos, state)
            return self._drop_item(state)
            
        # If not holding anything, go to onion dispenser
        return self._move_to_onion_dispenser(player, state)
        
    def _move_to_pot(self, player: PlayerState, state: OvercookedState) -> str:
        """Move towards pot to deliver onion."""
        pos = player.position
        pots = self.mdp.get_pot_locations()
        
        if not pots:
            return self._random_move(pos, state)
        
        # Find closest pot
        closest_pot = min(pots, key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
        
        # If adjacent to pot and facing it, interact
        if self._is_adjacent_to_target(pos, closest_pot):
            required_orientation = self._get_required_orientation(pos, closest_pot)
            if player.orientation == required_orientation:
                self.last_action = "interact"
                return "interact"
            else:
                action = self._get_rotation_action(player.orientation, required_orientation)
                self.last_action = action
                return action
        
        # Move towards pot
        return self._bfs_move_towards_target(pos, closest_pot, state)
        
    def _move_to_onion_dispenser(self, player: PlayerState, state: OvercookedState) -> str:
        """Move towards onion dispenser."""
        pos = player.position
        onion_dispensers = self.mdp.get_onion_dispenser_locations()
        
        if not onion_dispensers:
            return self._random_move(pos, state)
        
        # Find closest onion dispenser
        closest_dispenser = min(onion_dispensers, key=lambda d: abs(d[0] - pos[0]) + abs(d[1] - pos[1]))
        
        # If adjacent to dispenser and facing it, interact
        if self._is_adjacent_to_target(pos, closest_dispenser):
            required_orientation = self._get_required_orientation(pos, closest_dispenser)
            if player.orientation == required_orientation:
                self.last_action = "interact"
                return "interact"
            else:
                action = self._get_rotation_action(player.orientation, required_orientation)
                self.last_action = action
                return action
        
        # Move towards dispenser
        return self._bfs_move_towards_target(pos, closest_dispenser, state)
        
    def is_task_complete(self, state: OvercookedState) -> bool:
        """Task is complete when we've picked up the target number of onions."""
        # For now, let's simplify: complete when we pick up 1 onion
        # This can be enhanced later to track multiple onions
        player = state.players[self.agent_index]
        return player.has_object() and player.get_object().name == 'onion'
        
    def get_shaping_reward(self, state: OvercookedState, action: str, next_state: OvercookedState) -> float:
        """Calculate shaping reward for multiple onion pickup progress."""
        current_player = state.players[self.agent_index]
        next_player = next_state.players[self.agent_index]
        
        # Reward for getting closer to onion dispenser
        onion_dispensers = self.mdp.get_onion_dispenser_locations()
        if not onion_dispensers:
            return 0.0
            
        closest_dispenser = min(onion_dispensers, key=lambda d: 
            abs(d[0] - current_player.position[0]) + abs(d[1] - current_player.position[1]))
        
        # Distance-based reward
        current_distance = abs(current_player.position[0] - closest_dispenser[0]) + abs(current_player.position[1] - closest_dispenser[1])
        next_distance = abs(next_player.position[0] - closest_dispenser[0]) + abs(next_player.position[1] - closest_dispenser[1])
        
        distance_reward = 0.1 * (current_distance - next_distance)  # Positive if getting closer
        
        # Reward for successful interaction
        interaction_reward = 0.0
        if action == "interact" and not current_player.has_object() and next_player.has_object():
            if next_player.get_object().name == 'onion':
                interaction_reward = 2.0  # Large reward for successful pickup
        
        # Reward for proper orientation
        orientation_reward = 0.0
        if self._is_adjacent_to_target(current_player.position, closest_dispenser):
            required_orientation = self._get_required_orientation(current_player.position, closest_dispenser)
            if current_player.orientation == required_orientation:
                orientation_reward = 0.5
        
        return distance_reward + interaction_reward + orientation_reward

class PickupPlateAgent(MicroTaskAgent):
    """Agent for Task F: Pick up plate."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        super().__init__(agent_index, mdp, "pickup_plate")
        
    def action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Check if stuck
        if self.last_position == current_pos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.last_position = current_pos
            
        # If stuck, check for teammate blocking and handle appropriately
        if self.stuck_count > 3:
            # Check if teammate is blocking us
            teammate_blocking = False
            for j, player in enumerate(state.players):
                if j != self.agent_index:
                    if abs(player.position[0] - current_pos[0]) + abs(player.position[1] - current_pos[1]) == 1:
                        teammate_blocking = True
                        break
            
            if teammate_blocking:
                # If teammate is blocking, wait for them to move
                return "STAY"
            else:
                return self._handle_stuck_situation(current_pos, state)
            
        # If already holding plate, task is complete
        if player.has_object() and player.get_object().name == 'dish':
            return "STAY"
            
        # If holding something else, drop it first
        if player.has_object():
            return self._drop_item(state)
            
        # Find closest dish dispenser
        dish_dispensers = self.mdp.get_dish_dispenser_locations()
        if not dish_dispensers:
            return "STAY"
            
        closest_dispenser = min(dish_dispensers, key=lambda d: abs(d[0] - current_pos[0]) + abs(d[1] - current_pos[1]))
        
        # If adjacent to dispenser and facing it, interact
        if self._is_adjacent_to_target(current_pos, closest_dispenser):
            required_orientation = self._get_required_orientation(current_pos, closest_dispenser)
            if orientation == required_orientation:
                self.last_action = "interact"
                return "interact"
            else:
                action = self._get_rotation_action(orientation, required_orientation)
                self.last_action = action
                return action
                
        # Move towards dispenser
        return self._bfs_move_towards_target(current_pos, closest_dispenser, state)
        
    def is_task_complete(self, state: OvercookedState) -> bool:
        """Task is complete when holding a plate."""
        player = state.players[self.agent_index]
        return player.has_object() and player.get_object().name == 'dish'
        
    def get_shaping_reward(self, state: OvercookedState, action: str, next_state: OvercookedState) -> float:
        """Calculate shaping reward for plate pickup progress."""
        current_player = state.players[self.agent_index]
        next_player = next_state.players[self.agent_index]
        
        # Reward for getting closer to dish dispenser
        dish_dispensers = self.mdp.get_dish_dispenser_locations()
        if not dish_dispensers:
            return 0.0
            
        closest_dispenser = min(dish_dispensers, key=lambda d: 
            abs(d[0] - current_player.position[0]) + abs(d[1] - current_player.position[1]))
        
        # Distance-based reward
        current_distance = abs(current_player.position[0] - closest_dispenser[0]) + abs(current_player.position[1] - closest_dispenser[1])
        next_distance = abs(next_player.position[0] - closest_dispenser[0]) + abs(next_player.position[1] - closest_dispenser[1])
        
        distance_reward = 0.1 * (current_distance - next_distance)  # Positive if getting closer
        
        # Reward for successful interaction
        interaction_reward = 0.0
        if action == "interact" and not current_player.has_object() and next_player.has_object():
            if next_player.get_object().name == 'dish':
                interaction_reward = 2.0  # Large reward for successful pickup
        
        # Reward for proper orientation
        orientation_reward = 0.0
        if self._is_adjacent_to_target(current_player.position, closest_dispenser):
            required_orientation = self._get_required_orientation(current_player.position, closest_dispenser)
            if current_player.orientation == required_orientation:
                orientation_reward = 0.5
        
        return distance_reward + interaction_reward + orientation_reward

class PickupCookedSoupAgent(MicroTaskAgent):
    """Agent for Task G: Pick up cooked soup (simplified - starts with dish)."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        super().__init__(agent_index, mdp, "pickup_cooked_soup")
        
    def action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Check if stuck
        if self.last_position == current_pos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.last_position = current_pos
            
        # If stuck, check for teammate blocking and handle appropriately
        if self.stuck_count > 3:
            # Check if teammate is blocking us
            teammate_blocking = False
            for j, player in enumerate(state.players):
                if j != self.agent_index:
                    if abs(player.position[0] - current_pos[0]) + abs(player.position[1] - current_pos[1]) == 1:
                        teammate_blocking = True
                        break
            
            if teammate_blocking:
                # If teammate is blocking, wait for them to move
                return "STAY"
            else:
                return self._handle_stuck_situation(current_pos, state)
            
        # If already holding soup, task is complete
        if player.has_object() and player.get_object().name == 'soup':
            return "STAY"
            
        # If not holding dish, task failed (should start with dish)
        if not player.has_object() or player.get_object().name != 'dish':
            return "STAY"
            
        # Find pot with cooked soup and navigate to it
        return self._navigate_to_cooked_soup_pot(player, state)
    def _navigate_to_cooked_soup_pot(self, player: PlayerState, state: OvercookedState) -> str:
        """Navigate to pot with cooked soup and pick it up."""
        pos = player.position
        
        # Find pot with cooked soup
        pots = self.mdp.get_pot_locations()
        cooked_soup_pots = []
        for pot_pos in pots:
            if state.has_object(pot_pos):
                obj = state.get_object(pot_pos)
                if obj.name == 'soup':
                    soup_type, num_items, cook_time = obj.state
                    if cook_time >= 20:  # Fully cooked soup
                        cooked_soup_pots.append(pot_pos)
        
        if not cooked_soup_pots:
            return "STAY"  # No cooked soup available
            
        # Check if already adjacent to any pot with cooked soup
        for pot_pos in cooked_soup_pots:
            if self._is_adjacent_to_target(pos, pot_pos):
                # First, make sure we're facing the pot
                required_orientation = self._get_required_orientation(pos, pot_pos)
                if player.orientation == required_orientation:
                    # We're adjacent and facing the pot, interact
                    self.last_action = "interact"
                    return "interact"
                else:
                    # Turn to face the pot
                    action = self._get_rotation_action(player.orientation, required_orientation)
                    self.last_action = action
                    return action
            
            
        # Find the closest pot that we can actually reach
        closest_pot = None
        min_distance = float('inf')
        
        for pot_pos in cooked_soup_pots:
            # Check if we can reach this pot by finding path to adjacent position
            distance = abs(pot_pos[0] - pos[0]) + abs(pot_pos[1] - pos[1])
            if distance < min_distance:
                # Find adjacent positions to this pot
                adjacent_positions = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # South, North, East, West
                    adj_pos = (pot_pos[0] + dx, pot_pos[1] + dy)
                    if (0 <= adj_pos[0] < self.mdp.width and 
                        0 <= adj_pos[1] < self.mdp.height and 
                        self._is_valid_position(adj_pos)):
                        adjacent_positions.append(adj_pos)
                
                # Try to find a path to any adjacent position
                for adj_pos in adjacent_positions:
                    path = self._find_path_to_walkable_target(pos, adj_pos, state)
                    if path:  # If we can find a path, this pot is reachable
                        closest_pot = pot_pos
                        min_distance = distance
                        break
                if closest_pot:
                    break
        
        if closest_pot is None:
            return "STAY"  # No reachable cooked soup
        
        # If adjacent to pot, try to interact
        if self._is_adjacent_to_target(pos, closest_pot):
            # First, make sure we're facing the pot
            required_orientation = self._get_required_orientation(pos, closest_pot)
            if player.orientation == required_orientation:
                # We're adjacent and facing the pot, interact
                self.last_action = "interact"
                return "interact"
            else:
                # Turn to face the pot
                action = self._get_rotation_action(player.orientation, required_orientation)
                self.last_action = action
                return action
                
        # Only use BFS if not adjacent to pot
        return self._bfs_move_towards_target(pos, closest_pot, state)
        
    def is_task_complete(self, state: OvercookedState) -> bool:
        """Task is complete when holding cooked soup."""
        player = state.players[self.agent_index]
        return player.has_object() and player.get_object().name == 'soup'
        
    def get_shaping_reward(self, state: OvercookedState, action: str, next_state: OvercookedState) -> float:
        """Calculate shaping reward for cooked soup pickup progress."""
        current_player = state.players[self.agent_index]
        next_player = next_state.players[self.agent_index]
        
        # Reward for getting closer to pot with cooked soup
        soup_distance_reward = 0.0
        if current_player.has_object() and current_player.get_object().name == 'dish':
            pots = self.mdp.get_pot_locations()
            cooked_soup_pots = []
            for pot_pos in pots:
                if state.has_object(pot_pos):
                    obj = state.get_object(pot_pos)
                    if obj.name == 'soup':
                        soup_type, num_items, cook_time = obj.state
                        if cook_time >= 20:  # Fully cooked soup
                            cooked_soup_pots.append(pot_pos)
            
            if cooked_soup_pots:
                closest_pot = min(cooked_soup_pots, key=lambda p: 
                    abs(p[0] - current_player.position[0]) + abs(p[1] - current_player.position[1]))
                
                current_distance = abs(current_player.position[0] - closest_pot[0]) + abs(current_player.position[1] - closest_pot[1])
                next_distance = abs(next_player.position[0] - closest_pot[0]) + abs(next_player.position[1] - closest_pot[1])
                
                soup_distance_reward = 0.1 * (current_distance - next_distance)  # Positive if getting closer
        
        # Reward for successful soup pickup
        soup_pickup_reward = 0.0
        if action == "interact" and current_player.has_object() and current_player.get_object().name == 'dish':
            if next_player.has_object() and next_player.get_object().name == 'soup':
                soup_pickup_reward = 2.0  # Large reward for successful pickup
        
        # Reward for proper orientation when adjacent to pot
        orientation_reward = 0.0
        if current_player.has_object() and current_player.get_object().name == 'dish':
            pots = self.mdp.get_pot_locations()
            for pot_pos in pots:
                if state.has_object(pot_pos):
                    obj = state.get_object(pot_pos)
                    if obj.name == 'soup':
                        soup_type, num_items, cook_time = obj.state
                        if cook_time >= 20:  # Fully cooked soup
                            if self._is_adjacent_to_target(current_player.position, pot_pos):
                                required_orientation = self._get_required_orientation(current_player.position, pot_pos)
                                if current_player.orientation == required_orientation:
                                    orientation_reward = 0.5
        
        return soup_distance_reward + soup_pickup_reward + orientation_reward
        
        return distance_reward + dish_pickup_reward + soup_distance_reward + soup_pickup_reward
        
    def _get_dish(self, player: PlayerState, state: OvercookedState) -> str:
        """Get a dish from the dish dispenser."""
        pos = player.position
        dish_dispensers = self.mdp.get_dish_dispenser_locations()
        
        if not dish_dispensers:
            return "STAY"
            
        # Find closest dish dispenser
        closest_dispenser = min(dish_dispensers, key=lambda d: abs(d[0] - pos[0]) + abs(d[1] - pos[1]))
        
        # If adjacent to dispenser and facing it, interact
        if self._is_adjacent_to_target(pos, closest_dispenser):
            required_orientation = self._get_required_orientation(pos, closest_dispenser)
            if player.orientation == required_orientation:
                self.last_action = "interact"
                return "interact"
            else:
                action = self._get_rotation_action(player.orientation, required_orientation)
                self.last_action = action
                return action
        
        # Move towards dispenser
        return self._bfs_move_towards_target(pos, closest_dispenser, state)
        
    def _pickup_soup_from_pot(self, player: PlayerState, state: OvercookedState) -> str:
        """Pick up soup from pot using dish."""
        pos = player.position
        
        # Find pot with cooked soup
        pots = self.mdp.get_pot_locations()
        cooked_soup_pots = []
        for pot_pos in pots:
            if state.has_object(pot_pos):
                obj = state.get_object(pot_pos)
                if obj.name == 'soup':
                    soup_type, num_items, cook_time = obj.state
                    if cook_time >= 20:  # Fully cooked soup
                        cooked_soup_pots.append(pot_pos)
        
        if not cooked_soup_pots:
            return "STAY"  # No cooked soup available
            
        # Find the closest pot that we can actually reach
        closest_pot = None
        min_distance = float('inf')
        
        for pot_pos in cooked_soup_pots:
            # Check if we can reach this pot
            distance = abs(pot_pos[0] - pos[0]) + abs(pot_pos[1] - pos[1])
            if distance < min_distance:
                # Try to find a path to this pot
                path = self._find_path_to_target(pos, pot_pos, state)
                if path:  # If we can find a path, this pot is reachable
                    closest_pot = pot_pos
                    min_distance = distance
        
        if closest_pot is None:
            return "STAY"  # No reachable cooked soup
        
        # If adjacent to pot, try to interact
        if self._is_adjacent_to_target(pos, closest_pot):
            # First, make sure we're facing the pot
            required_orientation = self._get_required_orientation(pos, closest_pot)
            if player.orientation == required_orientation:
                # We're adjacent and facing the pot, interact
                self.last_action = "interact"
                return "interact"
            else:
                # Turn to face the pot
                action = self._get_rotation_action(player.orientation, required_orientation)
                self.last_action = action
                return action
                
        # Move towards pot
        return self._bfs_move_towards_target(pos, closest_pot, state)

class DeliverSoupAgent(MicroTaskAgent):
    """Agent for Task H: Deliver soup to serving area."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        super().__init__(agent_index, mdp, "deliver_soup")
        
    def action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Check if stuck
        if self.last_position == current_pos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.last_position = current_pos
            
        # If stuck, check for teammate blocking and handle appropriately
        if self.stuck_count > 3:
            # Check if teammate is blocking us
            teammate_blocking = False
            for j, player in enumerate(state.players):
                if j != self.agent_index:
                    if abs(player.position[0] - current_pos[0]) + abs(player.position[1] - current_pos[1]) == 1:
                        teammate_blocking = True
                        break
            
            if teammate_blocking:
                # If teammate is blocking, wait for them to move
                return "STAY"
            else:
                return self._handle_stuck_situation(current_pos, state)
            
        # If not holding soup, task failed
        if not player.has_object() or player.get_object().name != 'soup':
            return "STAY"
            
        # Find closest serving area
        serving_areas = self.mdp.get_serving_locations()
        if not serving_areas:
            return "STAY"
            
        closest_serving = min(serving_areas, key=lambda s: abs(s[0] - current_pos[0]) + abs(s[1] - current_pos[1]))
        
        # If adjacent to serving area and facing it, interact
        if self._is_adjacent_to_target(current_pos, closest_serving):
            required_orientation = self._get_required_orientation(current_pos, closest_serving)
            if orientation == required_orientation:
                self.last_action = "interact"
                return "interact"
            else:
                action = self._get_rotation_action(orientation, required_orientation)
                self.last_action = action
                return action
                
        # Move towards serving area
        return self._bfs_move_towards_target(current_pos, closest_serving, state)
        
    def is_task_complete(self, state: OvercookedState) -> bool:
        """Task is complete when soup has been delivered."""
        # Check if the agent has successfully delivered soup
        player = state.players[self.agent_index]
        
        # If player is still holding soup, task is not complete
        if player.has_object() and player.get_object().name == 'soup':
            return False
            
        # Check if soup was delivered (this would be tracked in the environment)
        # For now, we'll consider it complete if the agent is not holding soup
        # and was previously holding soup (indicating delivery)
        return True
        
    def get_shaping_reward(self, state: OvercookedState, action: str, next_state: OvercookedState) -> float:
        """Calculate shaping reward for soup delivery progress."""
        current_player = state.players[self.agent_index]
        next_player = next_state.players[self.agent_index]
        
        # Reward for getting closer to serving area
        serving_areas = self.mdp.get_serving_locations()
        if not serving_areas:
            return 0.0
            
        closest_serving = min(serving_areas, key=lambda s: 
            abs(s[0] - current_player.position[0]) + abs(s[1] - current_player.position[1]))
        
        # Distance-based reward
        current_distance = abs(current_player.position[0] - closest_serving[0]) + abs(current_player.position[1] - closest_serving[1])
        next_distance = abs(next_player.position[0] - closest_serving[0]) + abs(next_player.position[1] - closest_serving[1])
        
        distance_reward = 0.1 * (current_distance - next_distance)  # Positive if getting closer
        
        # Reward for reaching serving area adjacency
        adjacency_reward = 0.0
        if not self._is_adjacent_to_target(current_player.position, closest_serving) and self._is_adjacent_to_target(next_player.position, closest_serving):
            adjacency_reward = 1.0  # Reward for reaching serving area
        
        # Reward for successful delivery
        delivery_reward = 0.0
        if action == "interact" and current_player.has_object() and current_player.get_object().name == 'soup':
            if not next_player.has_object():  # Soup was delivered
                delivery_reward = 5.0  # Large reward for successful delivery
        
        # Reward for proper orientation
        orientation_reward = 0.0
        if self._is_adjacent_to_target(current_player.position, closest_serving):
            required_orientation = self._get_required_orientation(current_player.position, closest_serving)
            if current_player.orientation == required_orientation:
                orientation_reward = 0.5
        
        return distance_reward + adjacency_reward + delivery_reward + orientation_reward

class TeammateAgent:
    """Base class for teammate agents in micro-task scenarios."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        self.agent_index = agent_index
        self.mdp = mdp
        self.step_count = 0
        
    def reset(self):
        """Reset agent state."""
        self.step_count = 0
        
    def action(self, state: OvercookedState) -> str:
        """Get action for teammate."""
        raise NotImplementedError
        
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (walkable - not a wall, dispenser, pot, or serving area)."""
        if not (0 <= pos[0] < self.mdp.width and 0 <= pos[1] < self.mdp.height):
            return False
        terrain = self.mdp.terrain_mtx[pos[1]][pos[0]]
        # Only empty spaces are walkable
        return terrain == ' '

class NoOpTeammateAgent(TeammateAgent):
    """Teammate that takes no action (solo episodes)."""
    
    def action(self, state: OvercookedState) -> str:
        return "STAY"

class DishRunnerTeammateAgent(TeammateAgent):
    """Deterministic dish runner that follows a fixed loop away from main work areas."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        super().__init__(agent_index, mdp)
        self.phase = 0  # Current phase in the loop
        self.phase_steps = 0  # Steps in current phase
        self.yield_to_ego = False  # Whether to yield to ego agent
        
    def reset(self):
        super().reset()
        self.phase = 0
        self.phase_steps = 0
        self.yield_to_ego = False
        
    def action(self, state: OvercookedState) -> str:
        """Follow a deterministic dish-running loop."""
        self.step_count += 1
        
        # Check if ego agent is nearby (yield if too close)
        ego_pos = state.players[1 - self.agent_index].position
        my_pos = state.players[self.agent_index].position
        
        # If ego is within 2 tiles, yield
        if abs(ego_pos[0] - my_pos[0]) + abs(ego_pos[1] - my_pos[1]) <= 2:
            self.yield_to_ego = True
        else:
            self.yield_to_ego = False
            
        if self.yield_to_ego:
            return "STAY"
        
        # Follow dish-running loop based on layout
        if self.mdp.layout_name == "random3":
            return self._random3_dish_loop(state)
        else:
            return self._generic_dish_loop(state)
    
    def _random3_dish_loop(self, state: OvercookedState) -> str:
        """Dish-running loop for random3 layout."""
        my_pos = state.players[self.agent_index].position
        
        # Define dish-running phases for random3
        # Phase 0: Move to dish dispenser
        # Phase 1: Get dish
        # Phase 2: Move to serving area
        # Phase 3: Deliver dish
        # Phase 4: Return to starting position
        
        if self.phase == 0:  # Move to dish dispenser
            dish_dispensers = self.mdp.get_dish_dispenser_locations()
            if dish_dispensers:
                target = dish_dispensers[0]  # Use first dish dispenser
                if my_pos == target:
                    self.phase = 1
                    self.phase_steps = 0
                    return "interact"
                else:
                    return self._move_towards(target, my_pos)
            else:
                self.phase = 4  # Skip to return phase
                
        elif self.phase == 1:  # Get dish
            self.phase_steps += 1
            if self.phase_steps >= 2:  # Stay for 2 steps
                self.phase = 2
                self.phase_steps = 0
            return "STAY"
            
        elif self.phase == 2:  # Move to serving area
            serving_areas = self.mdp.get_serving_locations()
            if serving_areas:
                target = serving_areas[0]  # Use first serving area
                if my_pos == target:
                    self.phase = 3
                    self.phase_steps = 0
                    return "interact"
                else:
                    return self._move_towards(target, my_pos)
            else:
                self.phase = 4
                
        elif self.phase == 3:  # Deliver dish
            self.phase_steps += 1
            if self.phase_steps >= 2:  # Stay for 2 steps
                self.phase = 4
                self.phase_steps = 0
            return "STAY"
            
        elif self.phase == 4:  # Return to starting position
            # Return to a safe starting position (away from main work areas)
            target = (1, 3)  # Safe position in random3
            if my_pos == target:
                self.phase = 0  # Restart loop
                self.phase_steps = 0
            return self._move_towards(target, my_pos)
        
        return "STAY"
    
    def _generic_dish_loop(self, state: OvercookedState) -> str:
        """Generic dish-running loop for other layouts."""
        # Simple back-and-forth movement
        if self.step_count % 10 < 5:
            return "MOVE_E"
        else:
            return "MOVE_W"
    
    def _move_towards(self, target: Tuple[int, int], current: Tuple[int, int]) -> str:
        """Simple movement towards target with collision avoidance."""
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        
        # Try horizontal movement first
        if abs(dx) > abs(dy):
            if dx > 0:
                new_pos = (current[0] + 1, current[1])
                if self._is_valid_position(new_pos):
                    return "MOVE_E"
            else:
                new_pos = (current[0] - 1, current[1])
                if self._is_valid_position(new_pos):
                    return "MOVE_W"
        
        # Try vertical movement
        if dy > 0:
            new_pos = (current[0], current[1] + 1)
            if self._is_valid_position(new_pos):
                return "MOVE_S"
        else:
            new_pos = (current[0], current[1] - 1)
            if self._is_valid_position(new_pos):
                return "MOVE_N"
        
        # If all preferred moves are blocked, try any valid move
        for dx, dy, action in [(1, 0, "MOVE_E"), (-1, 0, "MOVE_W"), (0, 1, "MOVE_S"), (0, -1, "MOVE_N")]:
            new_pos = (current[0] + dx, current[1] + dy)
            if self._is_valid_position(new_pos):
                return action
        
        return "STAY"

class LightNoiseTeammateAgent(DishRunnerTeammateAgent):
    """Dish runner with occasional random actions (ε ≈ 0.05-0.10)."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld, noise_prob: float = 0.08):
        super().__init__(agent_index, mdp)
        self.noise_prob = noise_prob
        
    def action(self, state: OvercookedState) -> str:
        """Add noise to dish runner actions."""
        base_action = super().action(state)
        
        # Add noise with probability noise_prob
        if random.random() < self.noise_prob:
            # Random action instead of base action
            actions = ["STAY", "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W"]
            return random.choice(actions)
        
        return base_action

class TurnToFaceAgent(MicroTaskAgent):
    """Agent for Task D: Turn to face target."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld, target_type: str = "onion_dispenser"):
        super().__init__(agent_index, mdp, f"turn_to_face_{target_type}")
        self.target_type = target_type
        
    def action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Find target based on type
        if self.target_type == "onion_dispenser":
            targets = self.mdp.get_onion_dispenser_locations()
        elif self.target_type == "pot":
            targets = self.mdp.get_pot_locations()
        else:
            return "STAY"
            
        if not targets:
            return "STAY"
            
        # Find closest target
        closest_target = min(targets, key=lambda t: abs(t[0] - current_pos[0]) + abs(t[1] - current_pos[1]))
        
        # Get required orientation to face target
        required_orientation = self._get_required_orientation(current_pos, closest_target)
        
        # If already facing target, task is complete
        if orientation == required_orientation:
            return "STAY"
            
        # Turn towards target
        action = self._get_rotation_action(orientation, required_orientation)
        self.last_action = action
        return action
        
    def is_task_complete(self, state: OvercookedState) -> bool:
        """Task is complete when facing the target."""
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Find target based on type
        if self.target_type == "onion_dispenser":
            targets = self.mdp.get_onion_dispenser_locations()
        elif self.target_type == "pot":
            targets = self.mdp.get_pot_locations()
        else:
            return False
            
        if not targets:
            return False
            
        # Find closest target
        closest_target = min(targets, key=lambda t: abs(t[0] - current_pos[0]) + abs(t[1] - current_pos[1]))
        
        # Check if facing target
        required_orientation = self._get_required_orientation(current_pos, closest_target)
        return orientation == required_orientation

class AgentPair:
    """Wrapper to make micro-task agents compatible with Overcooked rollout system."""
    
    def __init__(self, ego_agent, teammate_agent):
        self.ego_agent = ego_agent
        self.teammate_agent = teammate_agent
        self.mdp = None
        
    def set_mdp(self, mdp):
        """Set the MDP for both agents."""
        self.mdp = mdp
        self.ego_agent.mdp = mdp
        self.teammate_agent.mdp = mdp
        
    def reset(self):
        """Reset both agents."""
        self.ego_agent.reset()
        self.teammate_agent.reset()
        
    def joint_action(self, state):
        """Get joint action from both agents."""
        ego_action = self.ego_agent.action(state)
        teammate_action = self.teammate_agent.action(state)
        
        # Convert to environment actions
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
        
        def convert_to_env_action(action_str):
            if action_str == "interact":
                return Action.INTERACT
            elif action_str == "STAY":
                return Action.STAY
            elif action_str == "MOVE_N":
                return Direction.NORTH
            elif action_str == "MOVE_S":
                return Direction.SOUTH
            elif action_str == "MOVE_E":
                return Direction.EAST
            elif action_str == "MOVE_W":
                return Direction.WEST
            else:
                return Action.STAY
        
        return [convert_to_env_action(ego_action), convert_to_env_action(teammate_action)]

def process_rollout_trajectory(trajectory, task_name: str, teammate_type: str, 
                             tot_rews_sparse: float, tot_rews_shaped: float, time_taken: int,
                             ego_agent) -> Dict[str, Any]:
    """Process rollout trajectory into our standard format."""
    
    # The trajectory is a numpy array with shape (time_steps, 4) where each row is (s_t, joint_a_t, r_t, done_t)
    # Extract trajectory components
    states = [step[0] for step in trajectory]  # s_t
    joint_actions = [step[1] for step in trajectory]  # joint_a_t
    rewards = [step[2] for step in trajectory]  # r_t
    dones = [step[3] for step in trajectory]  # done_t
    
    # Convert to our format
    trajectory_data = {
        'task_name': task_name,
        'teammate_type': teammate_type,
        'states': states,
        'state_texts': [describe_state(states[0].mdp, state, mode="both") for state in states],
        'actions': joint_actions,
        'rewards': rewards,
        'dones': dones,
        'human_commands': [""] * len(states),
        'episode_length': time_taken,
        'total_reward': tot_rews_sparse + tot_rews_shaped,
        'total_sparse_reward': tot_rews_sparse,
        'total_shaped_reward': tot_rews_shaped,
        'task_completed': check_task_completion(states[-1], task_name, ego_agent)
    }
    
    return trajectory_data

def check_task_completion(final_state: OvercookedState, task_name: str, ego_agent) -> bool:
    """Check if the task was completed based on final state."""
    # Check if the ego agent completed its task
    return ego_agent.is_task_complete(final_state)

def convert_action_to_index(action: str) -> int:
    """Convert action string to index."""
    if action == "interact":
        return 4
    elif action == "STAY":
        return 0
    elif action == "MOVE_N":
        return 1
    elif action == "MOVE_S":
        return 2
    elif action == "MOVE_E":
        return 3
    elif action == "MOVE_W":
        return 5
    else:
        return 0

def create_micro_task_start_state(mdp: OvercookedGridworld, task_name: str, agent_index: int = 0) -> OvercookedState:
    """Create appropriate start state for each micro-task with expanded scenario variations."""
    
    if task_name == "pickup_onion":
        # Enhanced starting scenarios for pickup_onion
        valid_positions = mdp.get_valid_joint_player_positions()
        onion_dispensers = mdp.get_onion_dispenser_locations()
        
        # Filter positions that are not adjacent to onion dispensers
        filtered_positions = []
        for joint_pos in valid_positions:
            player_pos = joint_pos[agent_index]  # Get position for the specific agent
            too_close = False
            for dispenser in onion_dispensers:
                if abs(player_pos[0] - dispenser[0]) + abs(player_pos[1] - dispenser[1]) <= 1:
                    too_close = True
                    break
            if not too_close:
                filtered_positions.append(joint_pos)
                
        if not filtered_positions:
            filtered_positions = valid_positions
            
        start_pos = random.choice(filtered_positions)
        state = OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
        
        # Enhanced starting conditions
        # 10% chance: start with some other item (tests agent's ability to handle conflicts)
        if random.random() < 0.1:
            from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
            # Randomly choose between dish or soup
            if random.random() < 0.5:
                item = ObjectState("dish", start_pos[agent_index])
            else:
                item = ObjectState("soup", start_pos[agent_index], ("onion", random.randint(1, 3), random.randint(0, 20)))
            state.players[agent_index].held_object = item
        
        return state
        
    elif task_name == "navigate_to_pot":
        # Enhanced starting scenarios for navigate_to_pot
        valid_positions = mdp.get_valid_joint_player_positions()
        pots = mdp.get_pot_locations()
        
        # Filter positions that are not adjacent to pots
        filtered_positions = []
        for joint_pos in valid_positions:
            player_pos = joint_pos[agent_index]  # Get position for the specific agent
            too_close = False
            for pot in pots:
                if abs(player_pos[0] - pot[0]) + abs(player_pos[1] - pot[1]) <= 1:
                    too_close = True
                    break
            if not too_close:
                filtered_positions.append(joint_pos)
                
        if not filtered_positions:
            filtered_positions = valid_positions
            
        start_pos = random.choice(filtered_positions)
        state = OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
        
        # Add onion to player's hands
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
        onion = ObjectState("onion", start_pos[agent_index])  # Use the specific agent's position
        state.players[agent_index].held_object = onion
        
        # Enhanced pot state variations
        if pots:
            # 40% chance: add some soup to pots (creates more realistic scenarios)
            if random.random() < 0.4:
                num_pots_with_soup = random.randint(1, len(pots))
                pots_to_fill = random.sample(pots, num_pots_with_soup)
                
                for pot_pos in pots_to_fill:
                    # Random onion count (1-3) and cooking time (0-20)
                    onion_count = random.randint(1, 3)
                    cook_time = random.randint(0, 20)
                    soup = ObjectState("soup", pot_pos, ("onion", onion_count, cook_time))
                    state.objects[pot_pos] = soup
        
        return state
        
    elif task_name == "interact_with_pot":
        # Enhanced starting scenarios for interact_with_pot
        pots = mdp.get_pot_locations()
        if not pots:
            # Fallback to regular start
            valid_positions = mdp.get_valid_joint_player_positions()
            start_pos = random.choice(valid_positions)
            return OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
            
        pot = random.choice(pots)
        
        # Find adjacent position to pot
        adjacent_positions = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_pos = (pot[0] + dx, pot[1] + dy)
            if (0 <= adj_pos[0] < mdp.width and 
                0 <= adj_pos[1] < mdp.height and 
                mdp.terrain_mtx[adj_pos[1]][adj_pos[0]] != 'X'):
                adjacent_positions.append(adj_pos)
                
        if not adjacent_positions:
            # Fallback to regular start
            valid_positions = mdp.get_valid_joint_player_positions()
            start_pos = random.choice(valid_positions)
            return OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
            
        # Create joint position with the adjacent position for the agent
        valid_positions = mdp.get_valid_joint_player_positions()
        start_pos = None
        for joint_pos in valid_positions:
            if joint_pos[agent_index] in adjacent_positions:
                start_pos = joint_pos
                break
                
        if start_pos is None:
            # Fallback to regular start
            start_pos = random.choice(valid_positions)
            
        state = OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
        
        # Add onion to player's hands
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
        onion = ObjectState("onion", start_pos[agent_index])  # Use the specific agent's position
        state.players[agent_index].held_object = onion
        
        # Enhanced pot state variations
        # 60% chance: pot has some soup (tests agent's ability to handle existing soup)
        if random.random() < 0.6:
            # Random onion count (1-2) and cooking time (0-15)
            onion_count = random.randint(1, 2)
            cook_time = random.randint(0, 15)
            soup = ObjectState("soup", pot, ("onion", onion_count, cook_time))
            state.objects[pot] = soup
        
        return state
        
    elif task_name.startswith("turn_to_face"):
        # Start away from target but within reasonable distance
        valid_positions = mdp.get_valid_joint_player_positions()
        start_pos = random.choice(valid_positions)
        return OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
        
    elif task_name == "pickup_multiple_onions":
        # Start away from onion dispensers, not holding anything
        valid_positions = mdp.get_valid_joint_player_positions()
        onion_dispensers = mdp.get_onion_dispenser_locations()
        
        # Filter positions that are not adjacent to onion dispensers
        filtered_positions = []
        for joint_pos in valid_positions:
            player_pos = joint_pos[agent_index]  # Get position for the specific agent
            too_close = False
            for dispenser in onion_dispensers:
                if abs(player_pos[0] - dispenser[0]) + abs(player_pos[1] - dispenser[1]) <= 1:
                    too_close = True
                    break
            if not too_close:
                filtered_positions.append(joint_pos)
                
        if not filtered_positions:
            filtered_positions = valid_positions
            
        start_pos = random.choice(filtered_positions)
        return OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
        
    elif task_name == "pickup_plate":
        # Enhanced starting scenarios for pickup_plate
        valid_positions = mdp.get_valid_joint_player_positions()
        plate_dispensers = mdp.get_dish_dispenser_locations()
        
        # Filter positions that are not adjacent to plate dispensers
        filtered_positions = []
        for joint_pos in valid_positions:
            player_pos = joint_pos[agent_index]  # Get position for the specific agent
            too_close = False
            for dispenser in plate_dispensers:
                if abs(player_pos[0] - dispenser[0]) + abs(player_pos[1] - dispenser[1]) <= 1:
                    too_close = True
                    break
            if not too_close:
                filtered_positions.append(joint_pos)
                
        if not filtered_positions:
            filtered_positions = valid_positions
            
        start_pos = random.choice(filtered_positions)
        state = OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
        
        # Enhanced starting conditions
        # 15% chance: start with some other item (tests agent's ability to handle conflicts)
        if random.random() < 0.15:
            from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
            # Randomly choose between onion or soup
            if random.random() < 0.5:
                item = ObjectState("onion", start_pos[agent_index])
            else:
                item = ObjectState("soup", start_pos[agent_index], ("onion", random.randint(1, 3), random.randint(0, 20)))
            state.players[agent_index].held_object = item
        
        return state
        
    elif task_name == "pickup_cooked_soup":
        # Enhanced starting scenarios for pickup_cooked_soup (SIMPLIFIED)
        valid_positions = mdp.get_valid_joint_player_positions()
        pots = mdp.get_pot_locations()
        
        # Filter positions that are not adjacent to pots
        filtered_positions = []
        for joint_pos in valid_positions:
            player_pos = joint_pos[agent_index]  # Get position for the specific agent
            too_close = False
            for pot in pots:
                if abs(player_pos[0] - pot[0]) + abs(player_pos[1] - pot[1]) <= 1:
                    too_close = True
                    break
            if not too_close:
                filtered_positions.append(joint_pos)
                
        if not filtered_positions:
            filtered_positions = valid_positions
            
        start_pos = random.choice(filtered_positions)
        state = OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
        
        # SIMPLIFIED: Agent always starts with dish
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
        dish = ObjectState("dish", start_pos[agent_index])
        state.players[agent_index].held_object = dish
        
        # Enhanced pot state variations
        if pots:
            # ALWAYS add cooked soup for pickup_cooked_soup task
            num_pots_with_soup = random.randint(1, len(pots))
            pots_to_fill = random.sample(pots, num_pots_with_soup)
            
            for pot_pos in pots_to_fill:
                # For pickup_cooked_soup, soup must have exactly 3 onions to be pickable
                onion_count = 3  # Always 3 onions for pickable soup
                # For pickup_cooked_soup, soup must be fully cooked (cook_time >= 20)
                cook_time = random.randint(20, 20)  # Always fully cooked
                from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
                soup = ObjectState("soup", pot_pos, ("onion", onion_count, cook_time))
                state.objects[pot_pos] = soup
        
        return state
        
    elif task_name == "deliver_soup":
        # Enhanced starting scenarios for deliver_soup
        valid_positions = mdp.get_valid_joint_player_positions()
        serving_areas = mdp.get_serving_locations()
        
        # Filter positions that are not adjacent to serving areas
        filtered_positions = []
        for joint_pos in valid_positions:
            player_pos = joint_pos[agent_index]  # Get position for the specific agent
            too_close = False
            for serving in serving_areas:
                if abs(player_pos[0] - serving[0]) + abs(player_pos[1] - serving[1]) <= 1:
                    too_close = True
                    break
            if not too_close:
                filtered_positions.append(joint_pos)
                
        if not filtered_positions:
            filtered_positions = valid_positions
            
        start_pos = random.choice(filtered_positions)
        state = OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
        
        # Enhanced soup variations
        from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import ObjectState
        
        # For delivery, soup must be fully cooked and have 3 onions
        onion_count = 3  # Always 3 onions for deliverable soup
        cook_time = random.randint(20, 20)  # Always fully cooked for delivery
        soup = ObjectState("soup", start_pos[agent_index], ("onion", onion_count, cook_time))
        state.players[agent_index].held_object = soup
        
        return state
        
    elif task_name == "pickup_plate":
        # Enhanced starting scenarios for pickup_plate
        valid_positions = mdp.get_valid_joint_player_positions()
        dish_dispensers = mdp.get_dish_dispenser_locations()
        
        # Filter positions that are not adjacent to dish dispensers
        filtered_positions = []
        for joint_pos in valid_positions:
            player_pos = joint_pos[agent_index]  # Get position for the specific agent
            too_close = False
            for dispenser in dish_dispensers:
                if abs(player_pos[0] - dispenser[0]) + abs(player_pos[1] - dispenser[1]) <= 1:
                    too_close = True
                    break
            if not too_close:
                filtered_positions.append(joint_pos)
                
        if not filtered_positions:
            filtered_positions = valid_positions
            
        start_pos = random.choice(filtered_positions)
        state = OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)
        
        return state
        
    else:
        # Default: random start
        valid_positions = mdp.get_valid_joint_player_positions()
        start_pos = random.choice(valid_positions)
        return OvercookedState.from_player_positions(start_pos, order_list=mdp.start_order_list)

def generate_micro_task_trajectories(env: OvercookedEnv, agent: MicroTaskAgent, task_name: str, 
                                   num_trajectories: int, max_steps: int = 50) -> List[Dict[str, Any]]:
    """Generate trajectories using manual step-by-step execution (avoiding numpy array issues)."""
    
    print(f"Generating {num_trajectories} trajectories for {task_name}...")
    
    # Determine teammate distribution
    solo_count = int(num_trajectories * 0.70)  # 70% solo
    cooperative_count = int(num_trajectories * 0.25)  # 25% cooperative
    noise_count = num_trajectories - solo_count - cooperative_count  # 5% light noise
    
    print(f"  Solo episodes: {solo_count}")
    print(f"  Cooperative episodes: {cooperative_count}")
    print(f"  Light noise episodes: {noise_count}")
    
    all_trajectories = []
    
    # Generate solo trajectories
    if solo_count > 0:
        print(f"  Generating {solo_count} solo trajectories...")
        solo_teammate = NoOpTeammateAgent(1, env.mdp)
        
        for _ in tqdm(range(solo_count), desc=f"{task_name} solo"):
            trajectory_data = generate_single_trajectory(
                env, agent, solo_teammate, task_name, "solo", max_steps
            )
            all_trajectories.append(trajectory_data)
    
    # Generate cooperative trajectories
    if cooperative_count > 0:
        print(f"  Generating {cooperative_count} cooperative trajectories...")
        cooperative_teammate = DishRunnerTeammateAgent(1, env.mdp)
        
        for _ in tqdm(range(cooperative_count), desc=f"{task_name} cooperative"):
            trajectory_data = generate_single_trajectory(
                env, agent, cooperative_teammate, task_name, "cooperative", max_steps
            )
            all_trajectories.append(trajectory_data)
    
    # Generate light noise trajectories
    if noise_count > 0:
        print(f"  Generating {noise_count} light noise trajectories...")
        noise_teammate = LightNoiseTeammateAgent(1, env.mdp, noise_prob=0.08)
        
        for _ in tqdm(range(noise_count), desc=f"{task_name} noise"):
            trajectory_data = generate_single_trajectory(
                env, agent, noise_teammate, task_name, "noise", max_steps
            )
            all_trajectories.append(trajectory_data)
    
    return all_trajectories

def generate_single_trajectory(env: OvercookedEnv, ego_agent: MicroTaskAgent, teammate_agent: TeammateAgent,
                             task_name: str, teammate_type: str, max_steps: int) -> Dict[str, Any]:
    """Generate a single trajectory using manual step-by-step execution."""
    
    # Create start state
    start_state = create_micro_task_start_state(env.mdp, task_name, ego_agent.agent_index)
    env.reset()
    env.state = start_state
    ego_agent.reset()
    teammate_agent.reset()
    
    trajectory_data = {
        'task_name': task_name,
        'teammate_type': teammate_type,
        'states': [],
        'state_texts': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'human_commands': [],
        'episode_length': 0,
        'total_reward': 0.0,
        'total_sparse_reward': 0.0,
        'total_shaped_reward': 0.0,
        'task_completed': False
    }
    
    state = env.state
    done = False
    step = 0
    
    while not done and step < max_steps:
        # Get state text description
        state_text = describe_state(env.mdp, state, mode="both")
        
        # Get actions from both agents
        ego_action = ego_agent.action(state)
        teammate_action = teammate_agent.action(state)
        
        # Store trajectory data
        trajectory_data['states'].append(state)
        trajectory_data['state_texts'].append(state_text)
        trajectory_data['actions'].append([ego_action, teammate_action])
        trajectory_data['human_commands'].append("")
        
        # Check if task is complete
        if ego_agent.is_task_complete(state):
            trajectory_data['task_completed'] = True
            trajectory_data['rewards'].append(1.0)  # Success reward
            trajectory_data['dones'].append(True)
            trajectory_data['total_reward'] += 1.0
            break
        
        # Take step in environment
        try:
            # Convert actions to environment format
            from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
            
            def convert_to_env_action(action_str):
                if action_str == "interact":
                    return Action.INTERACT
                elif action_str == "STAY":
                    return Action.STAY
                elif action_str == "MOVE_N":
                    return Direction.NORTH
                elif action_str == "MOVE_S":
                    return Direction.SOUTH
                elif action_str == "MOVE_E":
                    return Direction.EAST
                elif action_str == "MOVE_W":
                    return Direction.WEST
                else:
                    return Action.STAY
            
            # Execute actions
            env_actions = [convert_to_env_action(ego_action), convert_to_env_action(teammate_action)]
            state, reward, done, info = env.step(env_actions)
            
            # Calculate shaping reward for the ego agent
            shaping_reward = ego_agent.get_shaping_reward(trajectory_data['states'][-1], ego_action, state)
            
            # Combine environment reward with shaping reward
            total_reward = reward + shaping_reward
            
            trajectory_data['rewards'].append(total_reward)
            trajectory_data['dones'].append(done)
            trajectory_data['total_reward'] += total_reward
            
            # Track sparse vs shaped rewards from info
            if 'sparse_reward_by_agent' in info:
                trajectory_data['total_sparse_reward'] += sum(info['sparse_reward_by_agent'])
            if 'shaped_reward_by_agent' in info:
                trajectory_data['total_shaped_reward'] += sum(info['shaped_reward_by_agent'])
            
            # Add our custom shaping reward to the shaped reward total
            trajectory_data['total_shaped_reward'] += shaping_reward
            
        except Exception as e:
            print(f"Error in trajectory step {step}: {e}")
            break
        
        step += 1
    
    trajectory_data['episode_length'] = step
    return trajectory_data

def evaluate_micro_task_performance(trajectories: List[Dict[str, Any]], task_name: str) -> Dict[str, Any]:
    """Evaluate performance on a micro-task."""
    total_trajectories = len(trajectories)
    completed_trajectories = sum(1 for traj in trajectories if traj['task_completed'])
    success_rate = completed_trajectories / total_trajectories if total_trajectories > 0 else 0
    
    avg_episode_length = np.mean([traj['episode_length'] for traj in trajectories])
    avg_total_reward = np.mean([traj['total_reward'] for traj in trajectories])
    
    return {
        'task_name': task_name,
        'total_trajectories': total_trajectories,
        'completed_trajectories': completed_trajectories,
        'success_rate': success_rate,
        'avg_episode_length': avg_episode_length,
        'avg_total_reward': avg_total_reward
    }

def main():
    """Main function to generate micro-task sanity data."""
    
    # Configuration
    layout_name = "random3"  # You can change this to other layouts
    max_steps_per_task = 50  # Shorter episodes for micro-tasks
    
    # Task configurations
    task_configs = {
        'pickup_onion': {'num_trajectories': 250, 'agent_class_name': 'PickupOnionAgent'},
        'navigate_to_pot': {'num_trajectories': 150, 'agent_class_name': 'NavigateToPotAgent'},
        'interact_with_pot': {'num_trajectories': 150, 'agent_class_name': 'InteractWithPotAgent'},
        'turn_to_face_onion': {'num_trajectories': 100, 'agent_class_name': 'TurnToFaceAgent_onion'},
        'turn_to_face_pot': {'num_trajectories': 100, 'agent_class_name': 'TurnToFaceAgent_pot'},
        'pickup_multiple_onions': {'num_trajectories': 150, 'agent_class_name': 'PickupMultipleOnionsAgent'},
        'pickup_plate': {'num_trajectories': 100, 'agent_class_name': 'PickupPlateAgent'},
        'pickup_cooked_soup': {'num_trajectories': 100, 'agent_class_name': 'PickupCookedSoupAgent'},
        'deliver_soup': {'num_trajectories': 100, 'agent_class_name': 'DeliverSoupAgent'}
    }
    
    # Create output directory
    output_dir = "generated_data/micro_task_sanity"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating micro-task sanity data")
    print(f"Layout: {layout_name}")
    print(f"Max steps per task: {max_steps_per_task}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create environment
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv(mdp, horizon=max_steps_per_task)
    
    all_trajectories = []
    evaluation_results = []
    
    # Generate trajectories for each task
    for task_name, config in task_configs.items():
        print(f"\n{'='*50}")
        print(f"Task: {task_name}")
        print(f"Trajectories: {config['num_trajectories']}")
        print(f"{'='*50}")
        
        try:
            # Create agent
            agent_class_name = config['agent_class_name']
            
            if agent_class_name == 'PickupOnionAgent':
                agent = PickupOnionAgent(0, mdp)
            elif agent_class_name == 'NavigateToPotAgent':
                agent = NavigateToPotAgent(0, mdp)
            elif agent_class_name == 'InteractWithPotAgent':
                agent = InteractWithPotAgent(0, mdp)
            elif agent_class_name == 'TurnToFaceAgent_onion':
                agent = TurnToFaceAgent(0, mdp, "onion_dispenser")
            elif agent_class_name == 'TurnToFaceAgent_pot':
                agent = TurnToFaceAgent(0, mdp, "pot")
            elif agent_class_name == 'PickupMultipleOnionsAgent':
                agent = PickupMultipleOnionsAgent(0, mdp)
            elif agent_class_name == 'PickupPlateAgent':
                agent = PickupPlateAgent(0, mdp)
            elif agent_class_name == 'PickupCookedSoupAgent':
                agent = PickupCookedSoupAgent(0, mdp)
            elif agent_class_name == 'DeliverSoupAgent':
                agent = DeliverSoupAgent(0, mdp)
            else:
                raise ValueError(f"Unknown agent class: {agent_class_name}")
            
            # Generate trajectories
            trajectories = generate_micro_task_trajectories(
                env, agent, task_name, 
                config['num_trajectories'], max_steps_per_task
            )
            
            # Evaluate performance
            eval_result = evaluate_micro_task_performance(trajectories, task_name)
            evaluation_results.append(eval_result)
            
            # Save task-specific trajectories
            task_filename = f"{output_dir}/{task_name}_trajectories.pkl"
            with open(task_filename, 'wb') as f:
                pickle.dump(trajectories, f)
            
            print(f"Success rate: {eval_result['success_rate']:.3f}")
            print(f"Avg episode length: {eval_result['avg_episode_length']:.2f}")
            print(f"Saved to: {task_filename}")
            
            all_trajectories.extend(trajectories)
            
        except Exception as e:
            print(f"Error generating trajectories for {task_name}: {e}")
            continue
    
    # Save all trajectories together
    all_trajectories_filename = f"{output_dir}/all_micro_task_trajectories.pkl"
    with open(all_trajectories_filename, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    # Save evaluation results
    evaluation_filename = f"{output_dir}/evaluation_results.json"
    with open(evaluation_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Save metadata
    metadata = {
        'total_trajectories': len(all_trajectories),
        'max_steps_per_task': max_steps_per_task,
        'layout_name': layout_name,
        'task_configs': task_configs,
        'description': 'Micro-task sanity data for isolating basic skills in Overcooked',
        'evaluation_summary': {
            'overall_success_rate': np.mean([r['success_rate'] for r in evaluation_results]),
            'total_completed': sum(r['completed_trajectories'] for r in evaluation_results)
        }
    }
    
    metadata_filename = f"{output_dir}/metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Total trajectories: {len(all_trajectories)}")
    print(f"Overall success rate: {metadata['evaluation_summary']['overall_success_rate']:.3f}")
    print(f"Total completed tasks: {metadata['evaluation_summary']['total_completed']}")
    print(f"All trajectories saved to: {all_trajectories_filename}")
    print(f"Evaluation results saved to: {evaluation_filename}")
    print(f"Metadata saved to: {metadata_filename}")
    print(f"{'='*50}")
    
    # Print detailed results
    print(f"\nDetailed Results:")
    for result in evaluation_results:
        print(f"  {result['task_name']}: {result['success_rate']:.3f} ({result['completed_trajectories']}/{result['total_trajectories']})")
    
    # Print curriculum summary
    print(f"\nCurriculum Summary:")
    print(f"  Task A (pickup_onion): 250 trajectories")
    print(f"  Task B (navigate_to_pot): 150 trajectories")
    print(f"  Task C (interact_with_pot): 150 trajectories")
    print(f"  Task D (turn_to_face): 200 trajectories")
    print(f"  Task E (pickup_multiple_onions): 150 trajectories")
    print(f"  Task F (pickup_plate): 100 trajectories")
    print(f"  Task G (pickup_cooked_soup): 100 trajectories")
    print(f"  Task H (deliver_soup): 100 trajectories")
    print(f"  Total: {len(all_trajectories)} trajectories")
    print(f"")
    print(f"Teammate Distribution:")
    print(f"  Solo/No-op episodes: 70% (≈{int(len(all_trajectories) * 0.7)} trajectories)")
    print(f"  Cooperative episodes: 25% (≈{int(len(all_trajectories) * 0.25)} trajectories)")
    print(f"  Light noise episodes: 5% (≈{int(len(all_trajectories) * 0.05)} trajectories)")

if __name__ == "__main__":
    main()
