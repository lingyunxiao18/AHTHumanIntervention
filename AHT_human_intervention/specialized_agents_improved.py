#!/usr/bin/env python3
"""
Improved Specialized Agents for Diverse Training Data

This module contains various agents with limited abilities to create diverse training scenarios
for studying human intervention effects. Includes improved pathfinding.
"""

import random
import numpy as np
from typing import Tuple, List, Optional
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction

class OnionCollectorAgent:
    """Agent that only picks up onions and delivers them to pots repeatedly."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        self.agent_index = agent_index
        self.mdp = mdp
        self.last_position = None
        self.stuck_count = 0
        
    def reset(self):
        self.last_position = None
        self.stuck_count = 0
        
    def action(self, state: OvercookedState) -> Tuple[int, int]:
        """Get action for onion collection only."""
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Check if stuck
        if self.last_position == current_pos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.last_position = current_pos
        
        # If holding onion, go to pot
        if player.has_object() and player.get_object().name == 'onion':
            return self._move_to_pot(player, state)
        
        # If not holding anything, go to onion dispenser
        return self._move_to_onion_dispenser(player, state)
    
    def _move_to_pot(self, player: PlayerState, state: OvercookedState) -> Tuple[int, int]:
        """Move towards pot to deliver onion."""
        pos = player.position
        pots = self.mdp.get_pot_locations()
        
        if not pots:
            return self._random_move(pos)
        
        # Find closest pot
        closest_pot = min(pots, key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
        
        # If adjacent to pot and facing it, interact
        if self._is_adjacent_to_target(pos, closest_pot):
            required_orientation = self._get_required_orientation(pos, closest_pot)
            if player.orientation == required_orientation:
                return "interact"
            else:
                return self._get_rotation_action(player.orientation, required_orientation)
        
        # Move towards pot
        return self._improved_move_towards_target(pos, closest_pot)
    
    def _move_to_onion_dispenser(self, player: PlayerState, state: OvercookedState) -> Tuple[int, int]:
        """Move towards onion dispenser."""
        pos = player.position
        onion_dispensers = self.mdp.get_onion_dispenser_locations()
        
        if not onion_dispensers:
            return self._random_move(pos)
        
        # Find closest onion dispenser
        closest_dispenser = min(onion_dispensers, key=lambda d: abs(d[0] - pos[0]) + abs(d[1] - pos[1]))
        
        # If adjacent to dispenser and facing it, interact
        if self._is_adjacent_to_target(pos, closest_dispenser):
            required_orientation = self._get_required_orientation(pos, closest_dispenser)
            if player.orientation == required_orientation:
                return "interact"
            else:
                return self._get_rotation_action(player.orientation, required_orientation)
        
        # Move towards dispenser
        return self._improved_move_towards_target(pos, closest_dispenser)
    
    def _improved_move_towards_target(self, pos: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
        """Improved movement towards target with better obstacle avoidance."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        
        # Try to find a path around obstacles
        # First try horizontal movement if it reduces distance
        if dx != 0:
            new_pos = (pos[0] + (1 if dx > 0 else -1), pos[1])
            if self._is_valid_position(new_pos):
                return (1 if dx > 0 else -1, 0)
        
        # Then try vertical movement if it reduces distance
        if dy != 0:
            new_pos = (pos[0], pos[1] + (1 if dy > 0 else -1))
            if self._is_valid_position(new_pos):
                return (0, 1 if dy > 0 else -1)
        
        # If direct path blocked, try to find alternative path
        # Try moving in the direction that's not blocked
        for dx_test, dy_test in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (pos[0] + dx_test, pos[1] + dy_test)
            if self._is_valid_position(new_pos):
                return (dx_test, dy_test)
        
        # If all else fails, stay still
        return (0, 0)
    
    def _is_adjacent_to_target(self, pos: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """Check if position is adjacent to target."""
        dx = abs(target[0] - pos[0])
        dy = abs(target[1] - pos[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)
    
    def _get_required_orientation(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Get the orientation required to face the target."""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        if abs(dx) > abs(dy):
            return (1, 0) if dx > 0 else (-1, 0)
        else:
            return (0, 1) if dy > 0 else (0, -1)
    
    def _get_rotation_action(self, current_orientation: Tuple[int, int], target_orientation: Tuple[int, int]) -> Tuple[int, int]:
        """Get action to rotate to target orientation."""
        if current_orientation == target_orientation:
            return (0, 0)
        
        # Simple rotation logic
        if target_orientation == (1, 0):  # East
            return (1, 0)
        elif target_orientation == (-1, 0):  # West
            return (-1, 0)
        elif target_orientation == (0, 1):  # South
            return (0, 1)
        elif target_orientation == (0, -1):  # North
            return (0, -1)
        
        return (0, 0)
    
    def _random_move(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Random valid movement."""
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(actions)
        
        for action in actions:
            new_pos = (pos[0] + action[0], pos[1] + action[1])
            if self._is_valid_position(new_pos):
                return action
        
        return (0, 0)
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid."""
        x, y = pos
        if x < 0 or x >= self.mdp.width or y < 0 or y >= self.mdp.height:
            return False
        
        terrain = self.mdp.terrain_mtx[y][x]
        return terrain == ' '

class DishWaiterAgent:
    """Agent that waits near dish dispenser until soup is cooked, then collects and delivers."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        self.agent_index = agent_index
        self.mdp = mdp
        self.waiting_position = None
        self.last_position = None
        
    def reset(self):
        self.waiting_position = None
        self.last_position = None
        
    def action(self, state: OvercookedState) -> Tuple[int, int]:
        """Get action for dish waiting and delivery."""
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # If holding soup, deliver to serving area
        if player.has_object() and player.get_object().name == 'soup':
            return self._deliver_soup(player, state)
        
        # If holding dish, look for ready soup
        if player.has_object() and player.get_object().name == 'dish':
            return self._collect_soup(player, state)
        
        # If not holding anything, check if soup is ready
        if self._has_ready_soup(state):
            return self._get_dish(player, state)
        else:
            return self._wait_near_dish_dispenser(player, state)
    
    def _has_ready_soup(self, state: OvercookedState) -> bool:
        """Check if there's any ready soup."""
        for obj in state.objects.values():
            if obj.name == 'soup':
                if hasattr(obj, 'state') and len(obj.state) >= 3:
                    soup_type, num_items, cook_time = obj.state
                    if soup_type == 'onion' and num_items == 3 and cook_time >= 20:
                        return True
                elif hasattr(obj, 'cook_time') and obj.cook_time >= 20:
                    if hasattr(obj, 'num_items') and obj.num_items == 3:
                        return True
        return False
    
    def _get_dish(self, player: PlayerState, state: OvercookedState) -> Tuple[int, int]:
        """Get dish from dispenser."""
        pos = player.position
        dish_dispensers = self.mdp.get_dish_dispenser_locations()
        
        if not dish_dispensers:
            return (0, 0)
        
        closest_dispenser = min(dish_dispensers, key=lambda d: abs(d[0] - pos[0]) + abs(d[1] - pos[1]))
        
        if self._is_adjacent_to_target(pos, closest_dispenser):
            required_orientation = self._get_required_orientation(pos, closest_dispenser)
            if player.orientation == required_orientation:
                return "interact"
            else:
                return self._get_rotation_action(player.orientation, required_orientation)
        
        return self._improved_move_towards_target(pos, closest_dispenser)
    
    def _collect_soup(self, player: PlayerState, state: OvercookedState) -> Tuple[int, int]:
        """Collect soup from pot."""
        pos = player.position
        pots = self.mdp.get_pot_locations()
        
        if not pots:
            return (0, 0)
        
        # Find pot with ready soup
        ready_pot = None
        for pot_pos in pots:
            for obj in state.objects.values():
                if obj.name == 'soup' and obj.position == pot_pos:
                    if hasattr(obj, 'state') and len(obj.state) >= 3:
                        soup_type, num_items, cook_time = obj.state
                        if soup_type == 'onion' and num_items == 3 and cook_time >= 20:
                            ready_pot = pot_pos
                            break
                    elif hasattr(obj, 'cook_time') and obj.cook_time >= 20:
                        if hasattr(obj, 'num_items') and obj.num_items == 3:
                            ready_pot = pot_pos
                            break
        
        if not ready_pot:
            return (0, 0)
        
        if self._is_adjacent_to_target(pos, ready_pot):
            required_orientation = self._get_required_orientation(pos, ready_pot)
            if player.orientation == required_orientation:
                return "interact"
            else:
                return self._get_rotation_action(player.orientation, required_orientation)
        
        return self._improved_move_towards_target(pos, ready_pot)
    
    def _deliver_soup(self, player: PlayerState, state: OvercookedState) -> Tuple[int, int]:
        """Deliver soup to serving area."""
        pos = player.position
        serving_areas = self.mdp.get_serving_locations()
        
        if not serving_areas:
            return (0, 0)
        
        closest_serving = min(serving_areas, key=lambda s: abs(s[0] - pos[0]) + abs(s[1] - pos[1]))
        
        if self._is_adjacent_to_target(pos, closest_serving):
            required_orientation = self._get_required_orientation(pos, closest_serving)
            if player.orientation == required_orientation:
                return "interact"
            else:
                return self._get_rotation_action(player.orientation, required_orientation)
        
        return self._improved_move_towards_target(pos, closest_serving)
    
    def _wait_near_dish_dispenser(self, player: PlayerState, state: OvercookedState) -> Tuple[int, int]:
        """Wait near dish dispenser."""
        pos = player.position
        dish_dispensers = self.mdp.get_dish_dispenser_locations()
        
        if not dish_dispensers:
            return (0, 0)
        
        # Find position adjacent to dish dispenser
        if self.waiting_position is None:
            closest_dispenser = min(dish_dispensers, key=lambda d: abs(d[0] - pos[0]) + abs(d[1] - pos[1]))
            # Find adjacent position
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                adj_pos = (closest_dispenser[0] + dx, closest_dispenser[1] + dy)
                if (0 <= adj_pos[0] < self.mdp.width and 
                    0 <= adj_pos[1] < self.mdp.height and 
                    self.mdp.terrain_mtx[adj_pos[1]][adj_pos[0]] == ' '):
                    self.waiting_position = adj_pos
                    break
        
        if self.waiting_position and pos == self.waiting_position:
            return (0, 0)  # Stay still
        
        if self.waiting_position:
            return self._improved_move_towards_target(pos, self.waiting_position)
        
        return (0, 0)
    
    # Helper methods (same as OnionCollectorAgent)
    def _is_adjacent_to_target(self, pos: Tuple[int, int], target: Tuple[int, int]) -> bool:
        dx = abs(target[0] - pos[0])
        dy = abs(target[1] - pos[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)
    
    def _get_required_orientation(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> Tuple[int, int]:
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        if abs(dx) > abs(dy):
            return (1, 0) if dx > 0 else (-1, 0)
        else:
            return (0, 1) if dy > 0 else (0, -1)
    
    def _get_rotation_action(self, current_orientation: Tuple[int, int], target_orientation: Tuple[int, int]) -> Tuple[int, int]:
        if current_orientation == target_orientation:
            return (0, 0)
        
        if target_orientation == (1, 0):
            return (1, 0)
        elif target_orientation == (-1, 0):
            return (-1, 0)
        elif target_orientation == (0, 1):
            return (0, 1)
        elif target_orientation == (0, -1):
            return (0, -1)
        
        return (0, 0)
    
    def _improved_move_towards_target(self, pos: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
        """Improved movement towards target with better obstacle avoidance."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        
        # Try to find a path around obstacles
        # First try horizontal movement if it reduces distance
        if dx != 0:
            new_pos = (pos[0] + (1 if dx > 0 else -1), pos[1])
            if self._is_valid_position(new_pos):
                return (1 if dx > 0 else -1, 0)
        
        # Then try vertical movement if it reduces distance
        if dy != 0:
            new_pos = (pos[0], pos[1] + (1 if dy > 0 else -1))
            if self._is_valid_position(new_pos):
                return (0, 1 if dy > 0 else -1)
        
        # If direct path blocked, try to find alternative path
        # Try moving in the direction that's not blocked
        for dx_test, dy_test in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (pos[0] + dx_test, pos[1] + dy_test)
            if self._is_valid_position(new_pos):
                return (dx_test, dy_test)
        
        # If all else fails, stay still
        return (0, 0)
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.mdp.width or y < 0 or y >= self.mdp.height:
            return False
        
        terrain = self.mdp.terrain_mtx[y][x]
        return terrain == ' '

# Import the other agents from the original file
from specialized_agents import RandomAgent, PotWatcherAgent, ServingSpecialistAgent
