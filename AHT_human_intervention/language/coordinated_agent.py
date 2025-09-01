#!/usr/bin/env python3
"""
Coordinated Agent

A coordinated agent that can work together and avoid blocking each other.
Improved version with proper orientation handling, soup readiness checks, and diverse behavior.
"""

import random
import numpy as np
from typing import Tuple, List, Optional
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction

class CoordinatedAgent:
    """A coordinated agent that can work together and avoid blocking each other."""
    
    def __init__(self, agent_index: int, mdp: OvercookedGridworld):
        self.agent_index = agent_index
        self.mdp = mdp
        self.last_position = None
        self.stuck_count = 0
        self.path_to_target = []
        self.last_action = None
        self.action_repeat_count = 0
        self.target_position = None
        self.target_orientation = None
        self.just_dropped_onion = False  # Track if we just dropped an onion
        
    def set_agent_index(self, agent_index: int):
        self.agent_index = agent_index
        
    def set_mdp(self, mdp: OvercookedGridworld):
        self.mdp = mdp
        
    def reset(self):
        self.last_position = None
        self.stuck_count = 0
        self.path_to_target = []
        self.last_action = None
        self.action_repeat_count = 0
        self.target_position = None
        self.target_orientation = None
        self.just_dropped_onion = False
        
    def action(self, state: OvercookedState) -> Tuple[int, int]:
        """Get action for the agent based on coordinated logic."""
        player = state.players[self.agent_index]
        current_pos = player.position
        orientation = player.orientation
        
        # Check if we're stuck
        if self.last_position == current_pos:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.last_position = current_pos
        
        # Check if we're repeating the same action
        if self.last_action == "interact":
            self.action_repeat_count += 1
        else:
            self.action_repeat_count = 0
        
        # Rule 1: If we're at target position but wrong orientation, rotate to face target
        if self.target_position and current_pos == self.target_position and self.target_orientation:
            if orientation != self.target_orientation:
                action = self._get_rotation_action(orientation, self.target_orientation)
                if action:
                    self.last_action = action
                    return action
        
        # Rule 2: If we're facing an onion dispenser and not holding anything, INTERACT
        if self._can_pickup_onion(player):
            self.last_action = "interact"
            self.just_dropped_onion = False  # Reset flag when picking up new onion
            return "interact"
        
        # Rule 3: If we're facing a dish dispenser and not holding anything, INTERACT
        if self._can_pickup_dish(player, state):
            self.last_action = "interact"
            self.just_dropped_onion = False  # Reset flag when picking up dish
            return "interact"
        
        # Rule 4: If we're holding an onion and facing a pot, INTERACT
        if self._can_put_onion_in_pot(player):
            self.last_action = "interact"
            self.just_dropped_onion = True  # Mark that we just dropped an onion
            return "interact"
        
        # Rule 5: If we're holding soup and facing serving area, INTERACT
        if self._can_deliver_soup(player):
            self.last_action = "interact"
            self.just_dropped_onion = False  # Reset flag when delivering soup
            return "interact"
        
        # Rule 6: If we're holding a dish and facing a pot with ready soup, INTERACT
        if self._can_pickup_soup(player, state):
            self.last_action = "interact"
            self.just_dropped_onion = False  # Reset flag when picking up soup
            return "interact"
        
        # Rule 7: If stuck in interaction loop, try to move away
        if self.action_repeat_count > 5:
            action = self._move_away_from_current_position(current_pos)
            if action:
                self.last_action = action
                return action
        
        # Rule 8: Use coordinated pathfinding to move towards target
        action = self._coordinated_pathfind_to_target(player, state)
        if action:
            self.last_action = action
            return action
        
        # Rule 9: If stuck, try random valid movement
        if self.stuck_count > 3:
            action = self._random_valid_movement(current_pos)
            self.last_action = action
            return action
        
        # Rule 10: Default to staying still
        self.last_action = (0, 0)
        return (0, 0)  # STAY
    
    def _can_pickup_onion(self, player: PlayerState) -> bool:
        """Check if we can pickup an onion from dispenser."""
        if player.has_object():
            return False  # Already holding something
        
        facing_pos = self._get_facing_position(player)
        # Check bounds first
        if facing_pos[0] < 0 or facing_pos[0] >= self.mdp.width or facing_pos[1] < 0 or facing_pos[1] >= self.mdp.height:
            return False
            
        facing_terrain = self.mdp.terrain_mtx[facing_pos[1]][facing_pos[0]]
        return facing_terrain == 'O'  # Onion dispenser
    
    def _can_pickup_dish(self, player: PlayerState, state: OvercookedState) -> bool:
        """Check if we can pickup a dish from dispenser (if there's 3-onion soup)."""
        if player.has_object():
            return False  # Already holding something
        
        facing_pos = self._get_facing_position(player)
        # Check bounds first
        if facing_pos[0] < 0 or facing_pos[0] >= self.mdp.width or facing_pos[1] < 0 or facing_pos[1] >= self.mdp.height:
            return False
            
        facing_terrain = self.mdp.terrain_mtx[facing_pos[1]][facing_pos[0]]
        if facing_terrain != 'D':  # Not a dish dispenser
            return False
        
        # Check if there's any 3-onion soup in any pot (cooking or ready)
        for obj in state.objects.values():
            if obj.name == 'soup':
                if hasattr(obj, 'state') and len(obj.state) >= 3:
                    soup_type, num_items, cook_time = obj.state
                    if soup_type == 'onion' and num_items == 3:
                        return True  # There's a 3-onion soup, can pick up dish
                # Alternative check for different soup format
                elif hasattr(obj, 'num_items') and obj.num_items == 3:
                    return True  # There's a 3-onion soup, can pick up dish
        
        return False  # No 3-onion soup, don't pick up dish
    
    def _can_put_onion_in_pot(self, player: PlayerState) -> bool:
        """Check if we can put onion in pot."""
        if not player.has_object():
            return False
        
        held_object = player.get_object()
        if held_object.name != 'onion':
            return False
        
        facing_pos = self._get_facing_position(player)
        # Check bounds first
        if facing_pos[0] < 0 or facing_pos[0] >= self.mdp.width or facing_pos[1] < 0 or facing_pos[1] >= self.mdp.height:
            return False
            
        facing_terrain = self.mdp.terrain_mtx[facing_pos[1]][facing_pos[0]]
        return facing_terrain == 'P'  # Pot
    
    def _can_deliver_soup(self, player: PlayerState) -> bool:
        """Check if we can deliver soup."""
        if not player.has_object():
            return False
        
        held_object = player.get_object()
        if held_object.name != 'soup':
            return False
        
        facing_pos = self._get_facing_position(player)
        # Check bounds first
        if facing_pos[0] < 0 or facing_pos[0] >= self.mdp.width or facing_pos[1] < 0 or facing_pos[1] >= self.mdp.height:
            return False
            
        facing_terrain = self.mdp.terrain_mtx[facing_pos[1]][facing_pos[0]]
        return facing_terrain == 'S'  # Serving area
    
    def _can_pickup_soup(self, player: PlayerState, state: OvercookedState) -> bool:
        """Check if we can pickup soup from pot (only if soup is ready)."""
        if not player.has_object():
            return False
        
        held_object = player.get_object()
        if held_object.name != 'dish':
            return False
        
        facing_pos = self._get_facing_position(player)
        # Check bounds first
        if facing_pos[0] < 0 or facing_pos[0] >= self.mdp.width or facing_pos[1] < 0 or facing_pos[1] >= self.mdp.height:
            return False
            
        facing_terrain = self.mdp.terrain_mtx[facing_pos[1]][facing_pos[0]]
        if facing_terrain != 'P':  # Not a pot
            return False
        
        # Check if there's a ready soup in the pot
        for obj in state.objects.values():
            if obj.name == 'soup' and obj.position == facing_pos:
                # Check if soup is ready (3 onions and cook_time >= 20)
                if hasattr(obj, 'state') and len(obj.state) >= 3:
                    soup_type, num_items, cook_time = obj.state
                    if soup_type == 'onion' and num_items == 3 and cook_time >= 20:
                        return True
                # Alternative check for different soup format
                elif hasattr(obj, 'cook_time') and obj.cook_time >= 20:
                    # For alternative format, we still need to check num_items
                    if hasattr(obj, 'num_items') and obj.num_items == 3:
                        return True
        
        return False
    
    def _should_place_onion_and_get_dish(self, player: PlayerState, state: OvercookedState) -> bool:
        """Check if we should place onion on counter and get dish (when soup is ready)."""
        if not player.has_object():
            return False
        
        held_object = player.get_object()
        if held_object.name != 'onion':
            return False
        
        # Check if there's any ready soup in any pot
        for obj in state.objects.values():
            if obj.name == 'soup':
                if hasattr(obj, 'state') and len(obj.state) >= 3:
                    soup_type, num_items, cook_time = obj.state
                    if soup_type == 'onion' and num_items == 3 and cook_time >= 20:
                        return True  # There's a ready soup, should get dish
                elif hasattr(obj, 'cook_time') and obj.cook_time >= 20:
                    if hasattr(obj, 'num_items') and obj.num_items == 3:
                        return True  # There's a ready soup, should get dish
        
        return False
    
    def _can_place_onion_on_counter(self, player: PlayerState, state: OvercookedState) -> bool:
        """Check if we can place onion on a counter (must be adjacent to counter tile)."""
        if not player.has_object():
            return False
        
        held_object = player.get_object()
        if held_object.name != 'onion':
            return False
        
        # Check if we're facing a counter tile (adjacent to it)
        facing_pos = self._get_facing_position(player)
        # Check bounds first
        if facing_pos[0] < 0 or facing_pos[0] >= self.mdp.width or facing_pos[1] < 0 or facing_pos[1] >= self.mdp.height:
            return False
            
        facing_terrain = self.mdp.terrain_mtx[facing_pos[1]][facing_pos[0]]
        
        # Can only place on counter tiles (X), and we must be adjacent to them
        if facing_terrain != 'X':
            return False
        
        # Check if there's already an object at the facing position
        for obj in state.objects.values():
            if obj.position == facing_pos:
                return False  # Position is occupied
        
        return True
    
    def _move_towards_dish_dispenser(self, player: PlayerState, state: OvercookedState) -> Optional[Tuple[int, int]]:
        """Move towards dish dispenser to pick up a dish."""
        pos = player.position
        dish_dispensers = self.mdp.get_dish_dispenser_locations()
        
        if not dish_dispensers:
            return None
        
        # Find closest dish dispenser
        closest_dispenser = min(dish_dispensers, key=lambda d: abs(d[0] - pos[0]) + abs(d[1] - pos[1]))
        
        # Get other agent's position to avoid blocking
        other_agent_pos = None
        for i, other_player in enumerate(state.players):
            if i != self.agent_index:
                other_agent_pos = other_player.position
                break
        
        # Use simple movement towards dish dispenser
        return self._simple_move_towards_target(pos, closest_dispenser, other_agent_pos)
    
    def _should_handle_three_onion_soup(self, player: PlayerState, state: OvercookedState) -> bool:
        """Check if we should handle a pot with 3 onions (after dropping an onion)."""
        # Check if there's any pot with exactly 3 onions
        for obj in state.objects.values():
            if obj.name == 'soup':
                # Check if soup has exactly 3 onions
                if hasattr(obj, 'state') and len(obj.state) >= 3:
                    soup_type, num_items, cook_time = obj.state
                    if soup_type == 'onion' and num_items == 3:
                        return True
                # Alternative check for different soup format
                elif hasattr(obj, 'num_items') and obj.num_items == 3:
                    return True
        
        return False
    
    def _move_towards_three_onion_pot(self, player: PlayerState, state: OvercookedState) -> Optional[Tuple[int, int]]:
        """Move towards the pot that has 3 onions."""
        pos = player.position
        
        # Find the pot with 3 onions
        three_onion_pot = None
        for obj in state.objects.values():
            if obj.name == 'soup':
                # Check if soup has exactly 3 onions
                if hasattr(obj, 'state') and len(obj.state) >= 3:
                    soup_type, num_items, cook_time = obj.state
                    if soup_type == 'onion' and num_items == 3:
                        three_onion_pot = obj.position
                        break
                # Alternative check for different soup format
                elif hasattr(obj, 'num_items') and obj.num_items == 3:
                    three_onion_pot = obj.position
                    break
        
        if not three_onion_pot:
            return None
        
        # Get other agent's position to avoid blocking
        other_agent_pos = None
        for i, other_player in enumerate(state.players):
            if i != self.agent_index:
                other_agent_pos = other_player.position
                break
        
        # Use simple movement towards the pot
        return self._simple_move_towards_target(pos, three_onion_pot, other_agent_pos)
    
    def _is_at_three_onion_pot(self, player: PlayerState, state: OvercookedState) -> bool:
        """Check if we're at a pot that has 3 onions."""
        facing_pos = self._get_facing_position(player)
        # Check bounds first
        if facing_pos[0] < 0 or facing_pos[0] >= self.mdp.width or facing_pos[1] < 0 or facing_pos[1] >= self.mdp.height:
            return False
        
        # Check if we're facing a pot
        facing_terrain = self.mdp.terrain_mtx[facing_pos[1]][facing_pos[0]]
        if facing_terrain != 'P':
            return False
        
        # Check if the pot has exactly 3 onions
        for obj in state.objects.values():
            if obj.name == 'soup' and obj.position == facing_pos:
                # Check if soup has exactly 3 onions
                if hasattr(obj, 'state') and len(obj.state) >= 3:
                    soup_type, num_items, cook_time = obj.state
                    if soup_type == 'onion' and num_items == 3:
                        return True
                # Alternative check for different soup format
                elif hasattr(obj, 'num_items') and obj.num_items == 3:
                    return True
        
        return False
    
    def _move_towards_counter(self, player: PlayerState, state: OvercookedState) -> Optional[Tuple[int, int]]:
        """Move towards a counter to place an onion (move to empty space adjacent to counter)."""
        pos = player.position
        
        # Find available counters and their adjacent empty spaces
        available_positions = []
        for y in range(self.mdp.height):
            for x in range(self.mdp.width):
                terrain = self.mdp.terrain_mtx[y][x]
                # Look for counter tiles (X)
                if terrain == 'X':
                    # Check adjacent positions for empty spaces
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # Adjacent positions
                        adj_x, adj_y = x + dx, y + dy
                        if (0 <= adj_x < self.mdp.width and 
                            0 <= adj_y < self.mdp.height and 
                            self.mdp.terrain_mtx[adj_y][adj_x] == ' '):  # Empty space
                            
                            # Check if position is not occupied by objects
                            position_occupied = False
                            for obj in state.objects.values():
                                if obj.position == (adj_x, adj_y):
                                    position_occupied = True
                                    break
                            
                            if not position_occupied:
                                # Check if no player is at this position
                                player_occupied = False
                                for other_player in state.players:
                                    if other_player.position == (adj_x, adj_y):
                                        player_occupied = True
                                        break
                                
                                if not player_occupied:
                                    available_positions.append((adj_x, adj_y))
        
        if not available_positions:
            return None
        
        # Find closest available position adjacent to a counter
        closest_position = min(available_positions, key=lambda p: abs(p[0] - pos[0]) + abs(p[1] - pos[1]))
        
        # Get other agent's position to avoid blocking
        other_agent_pos = None
        for i, other_player in enumerate(state.players):
            if i != self.agent_index:
                other_agent_pos = other_player.position
                break
        
        # Use simple movement towards the position adjacent to counter
        return self._simple_move_towards_target(pos, closest_position, other_agent_pos)
    
    def _get_facing_position(self, player: PlayerState) -> Tuple[int, int]:
        """Get the position the player is facing."""
        pos = player.position
        orientation = player.orientation
        return (pos[0] + orientation[0], pos[1] + orientation[1])
    
    def _get_rotation_action(self, current_orientation: Tuple[int, int], target_orientation: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Get action to rotate from current to target orientation."""
        if current_orientation == target_orientation:
            return None
        
        # Define rotation mappings
        rotations = {
            (0, -1): {  # North
                (0, 1): (0, 1),   # To South
                (1, 0): (1, 0),   # To East
                (-1, 0): (-1, 0)  # To West
            },
            (0, 1): {   # South
                (0, -1): (0, -1), # To North
                (1, 0): (1, 0),   # To East
                (-1, 0): (-1, 0)  # To West
            },
            (1, 0): {   # East
                (0, -1): (0, -1), # To North
                (0, 1): (0, 1),   # To South
                (-1, 0): (-1, 0)  # To West
            },
            (-1, 0): {  # West
                (0, -1): (0, -1), # To North
                (0, 1): (0, 1),   # To South
                (1, 0): (1, 0)    # To East
            }
        }
        
        if current_orientation in rotations and target_orientation in rotations[current_orientation]:
            return rotations[current_orientation][target_orientation]
        
        return None
    
    def _move_away_from_current_position(self, current_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Move away from current position to avoid getting stuck."""
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # South, North, East, West
        random.shuffle(actions)  # Randomize order
        
        for action in actions:
            new_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
            if self._is_valid_position(new_pos):
                return action
        
        return None
    
    def _coordinated_pathfind_to_target(self, player: PlayerState, state: OvercookedState) -> Optional[Tuple[int, int]]:
        """Use coordinated pathfinding to move towards target."""
        pos = player.position
        holding_object = player.has_object()
        
        # Get other agent's position to avoid blocking
        other_agent_pos = None
        for i, other_player in enumerate(state.players):
            if i != self.agent_index:
                other_agent_pos = other_player.position
                break
        
        # Determine target objects based on what we're holding
        if not holding_object:
            # Check if there's any pot with 3 onions (soup cooking or ready)
            has_three_onion_soup = False
            for obj in state.objects.values():
                if obj.name == 'soup':
                    if hasattr(obj, 'state') and len(obj.state) >= 3:
                        soup_type, num_items, cook_time = obj.state
                        if soup_type == 'onion' and num_items == 3:
                            has_three_onion_soup = True
                            break
                    elif hasattr(obj, 'num_items') and obj.num_items == 3:
                        has_three_onion_soup = True
                        break
            
            if has_three_onion_soup:
                # If there's a pot with 3 onions, prioritize dish dispenser to collect soup
                target_objects = self.mdp.get_dish_dispenser_locations()
            else:
                # If no 3-onion soup, prioritize onion dispensers to help complete soups
                target_objects = self.mdp.get_onion_dispenser_locations()
        elif player.get_object().name == 'onion':
            # Holding onion - go to pot
            target_objects = self.mdp.get_pot_locations()
        elif player.get_object().name == 'dish':
            # Holding dish - go to pot to get soup
            target_objects = self.mdp.get_pot_locations()
        elif player.get_object().name == 'soup':
            # Holding soup - go to serving area
            target_objects = self.mdp.get_serving_locations()
        else:
            # Unknown object - go to pot
            target_objects = self.mdp.get_pot_locations()
        
        if not target_objects:
            return None
        
        # Find closest target
        closest_target = min(target_objects, key=lambda t: abs(t[0] - pos[0]) + abs(t[1] - pos[1]))
        
        # Check if we're already adjacent to the target and facing it
        if self._is_adjacent_to_target(pos, closest_target):
            # We're adjacent, check if we're facing the target
            required_orientation = self._get_required_orientation(pos, closest_target)
            if player.orientation == required_orientation:
                # We're adjacent and facing the target, don't move
                return None
            else:
                # We're adjacent but not facing the target, rotate
                return self._get_rotation_action(player.orientation, required_orientation)
        
        # Set target position and required orientation
        self.target_position = closest_target
        self.target_orientation = self._get_required_orientation(pos, closest_target)
        
        # If we have a path to follow, continue following it
        if self.path_to_target and len(self.path_to_target) > 1:
            next_pos = self.path_to_target[1]
            
            # Check if next position is blocked by other agent
            if other_agent_pos and next_pos == other_agent_pos:
                # Find alternative path
                self.path_to_target = []
            else:
                action = self._get_action_to_position(pos, next_pos)
                if action:
                    # Remove the current position from the path
                    self.path_to_target.pop(0)
                    return action
                else:
                    # Path is blocked, recalculate
                    self.path_to_target = []
        
        # Calculate a new path to the target
        path = self._find_path_to_target(pos, closest_target, other_agent_pos)
        if path and len(path) > 1:
            self.path_to_target = path
            next_pos = path[1]
            action = self._get_action_to_position(pos, next_pos)
            if action:
                self.path_to_target.pop(0)
                return action
        
        # Fallback to simple movement
        return self._simple_move_towards_target(pos, closest_target, other_agent_pos)
    
    def _is_adjacent_to_target(self, pos: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """Check if position is adjacent to target."""
        dx = abs(target[0] - pos[0])
        dy = abs(target[1] - pos[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)
    
    def _get_required_orientation(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Get the orientation required to face the target from current position."""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Determine which direction to face
        if abs(dx) > abs(dy):
            # Horizontal movement is larger
            if dx > 0:
                return (1, 0)  # Face East
            else:
                return (-1, 0)  # Face West
        else:
            # Vertical movement is larger
            if dy > 0:
                return (0, 1)  # Face South
            else:
                return (0, -1)  # Face North
    
    def _find_path_to_target(self, start: Tuple[int, int], target: Tuple[int, int], blocked_pos: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """Find a path to the target using BFS, avoiding blocked positions."""
        if start == target:
            return [start]
        
        # Check if target is reachable (not a wall)
        if not self._is_valid_position(target):
            # Target is not walkable (like a dispenser), find path to adjacent position
            # Find all adjacent positions to the target
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
                path = self._find_path_to_walkable_target(start, adj_pos, blocked_pos)
                if path:
                    distance = len(path)
                    if distance < best_distance:
                        best_distance = distance
                        best_path = path
            
            return best_path
        else:
            # Target is walkable, find direct path
            return self._find_path_to_walkable_target(start, target, blocked_pos)
    
    def _find_path_to_walkable_target(self, start: Tuple[int, int], target: Tuple[int, int], blocked_pos: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """Find a path to a walkable target using BFS."""
        if start == target:
            return [start]
        
        # Use BFS to find a path
        queue = [(start, [start])]
        visited = set([start])
        
        while queue:
            current_pos, path = queue.pop(0)
            
            # Check all adjacent positions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # South, North, East, West
                new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                if new_pos not in visited and self._is_valid_position(new_pos):
                    # Avoid blocked positions
                    if blocked_pos and new_pos == blocked_pos:
                        continue
                        
                    visited.add(new_pos)
                    new_path = path + [new_pos]
                    
                    if new_pos == target:
                        return new_path
                    
                    queue.append((new_pos, new_path))
        
        return []
    
    def _get_action_to_position(self, current: Tuple[int, int], target: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Get the action to move from current to target position."""
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        
        if dx == 1:
            return (1, 0)  # East
        elif dx == -1:
            return (-1, 0)  # West
        elif dy == 1:
            return (0, 1)  # South
        elif dy == -1:
            return (0, -1)  # North
        
        return None
    
    def _simple_move_towards_target(self, pos: Tuple[int, int], target: Tuple[int, int], blocked_pos: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        """Use simple movement towards target as fallback."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        
        # Prioritize the direction that reduces distance the most
        # If vertical distance is larger, try vertical movement first
        if abs(dy) > abs(dx):
            # Try vertical movement first
            if dy > 0:
                new_pos = (pos[0], pos[1] + 1)
                if self._is_valid_position(new_pos) and new_pos != blocked_pos:
                    return (0, 1)  # South
            elif dy < 0:
                new_pos = (pos[0], pos[1] - 1)
                if self._is_valid_position(new_pos) and new_pos != blocked_pos:
                    return (0, -1)  # North
            
            # Then try horizontal movement
            if dx > 0:
                new_pos = (pos[0] + 1, pos[1])
                if self._is_valid_position(new_pos) and new_pos != blocked_pos:
                    return (1, 0)  # East
            elif dx < 0:
                new_pos = (pos[0] - 1, pos[1])
                if self._is_valid_position(new_pos) and new_pos != blocked_pos:
                    return (-1, 0)  # West
        else:
            # If horizontal distance is larger or equal, try horizontal movement first
            if dx > 0:
                new_pos = (pos[0] + 1, pos[1])
                if self._is_valid_position(new_pos) and new_pos != blocked_pos:
                    return (1, 0)  # East
            elif dx < 0:
                new_pos = (pos[0] - 1, pos[1])
                if self._is_valid_position(new_pos) and new_pos != blocked_pos:
                    return (-1, 0)  # West
            
            # Then try vertical movement
            if dy > 0:
                new_pos = (pos[0], pos[1] + 1)
                if self._is_valid_position(new_pos) and new_pos != blocked_pos:
                    return (0, 1)  # South
            elif dy < 0:
                new_pos = (pos[0], pos[1] - 1)
                if self._is_valid_position(new_pos) and new_pos != blocked_pos:
                    return (0, -1)  # North
        
        # If we can't move towards the target, try any valid movement
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # South, North, East, West
        random.shuffle(actions)  # Randomize order
        
        for action in actions:
            new_pos = (pos[0] + action[0], pos[1] + action[1])
            if self._is_valid_position(new_pos) and new_pos != blocked_pos:
                return action
        
        return None
    
    def _random_valid_movement(self, current_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Generate random valid movement action when stuck."""
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # South, North, East, West
        random.shuffle(actions)  # Randomize order
        
        for action in actions:
            new_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
            if self._is_valid_position(new_pos):
                return action
        
        # If no valid movement, stay still
        return (0, 0)
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is valid (within bounds and is an empty space)."""
        x, y = pos
        if x < 0 or x >= self.mdp.width or y < 0 or y >= self.mdp.height:
            return False
        
        terrain = self.mdp.terrain_mtx[y][x]
        # In Overcooked, agents can only move on empty spaces (' ')
        # They cannot walk on dispensers ('O', 'D'), counters ('X'), pots ('P'), or serving areas ('S')
        return terrain == ' '  # Only empty spaces are valid for movement

def test_coordinated_agent():
    """Test the coordinated agent."""
    print("🧪 TESTING COORDINATED AGENT")
    print("=" * 50)
    
    from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    
    layout_name = "random3"
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv(mdp, horizon=400)
    
    print(f"Layout: {layout_name}")
    print(f"Layout size: {mdp.width}x{mdp.height}")
    print(f"Onion dispensers: {mdp.get_onion_dispenser_locations()}")
    print(f"Pots: {mdp.get_pot_locations()}")
    print(f"Serving areas: {mdp.get_serving_locations()}")
    print(f"Dish dispensers: {mdp.get_dish_dispenser_locations()}")
    
    # Show layout
    print(f"\n📋 LAYOUT:")
    for y in range(mdp.height):
        row = ""
        for x in range(mdp.width):
            terrain = mdp.terrain_mtx[y][x]
            row += terrain
        print(f"  {row}")
    
    # Create agents
    agent0 = CoordinatedAgent(0, mdp)
    agent1 = CoordinatedAgent(1, mdp)
    
    # Reset environment
    env.reset()
    agent0.reset()
    agent1.reset()
    
    # Test for 400 steps
    total_reward = 0.0
    reward_history = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    action_names = {0: 'STAY', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'INTERACT'}
    
    for step in range(400):
        state = env.state
        player0 = state.players[0]
        player1 = state.players[1]
        
        # Get actions
        action0 = agent0.action(state)
        action1 = agent1.action(state)
        
        # Convert to action indices
        if isinstance(action0, tuple):
            if action0 == (0, 0):
                action0_idx = 0  # STAY
            elif action0 == (0, -1):
                action0_idx = 1  # UP
            elif action0 == (0, 1):
                action0_idx = 2  # DOWN
            elif action0 == (-1, 0):
                action0_idx = 3  # LEFT
            elif action0 == (1, 0):
                action0_idx = 4  # RIGHT
        elif action0 == "interact":
            action0_idx = 5  # INTERACT
        else:
            action0_idx = 0  # STAY
        
        if isinstance(action1, tuple):
            if action1 == (0, 0):
                action1_idx = 0  # STAY
            elif action1 == (0, -1):
                action1_idx = 1  # UP
            elif action1 == (0, 1):
                action1_idx = 2  # DOWN
            elif action1 == (-1, 0):
                action1_idx = 3  # LEFT
            elif action1 == (1, 0):
                action1_idx = 4  # RIGHT
        elif action1 == "interact":
            action1_idx = 5  # INTERACT
        else:
            action1_idx = 0  # STAY
        
        # Count actions
        action_counts[action0_idx] += 1
        action_counts[action1_idx] += 1
        
        # Step environment
        next_state, reward, done, info = env.step([action0, action1])
        
        total_reward += reward
        reward_history.append(reward)
        
        # Print first 50 steps with details
        if step < 50:
            p0_holding = player0.get_object().name if player0.has_object() else "None"
            p1_holding = player1.get_object().name if player1.has_object() else "None"
            print(f"Step {step:2d}: P0={action0}({action0_idx}) at {player0.position} holding {p0_holding:6s}, P1={action1}({action1_idx}) at {player1.position} holding {p1_holding:6s}, reward={reward:6.3f}")
        
        # Check for success conditions
        if reward > 0:
            print(f"\n🎉 POSITIVE REWARD at step {step}: {reward}")
            print(f"   P0 at {player0.position} holding {p0_holding}")
            print(f"   P1 at {player1.position} holding {p1_holding}")
            print(f"   Actions: P0={action0}, P1={action1}")
        
        if done:
            break
    
    # Analyze results
    print(f"\n📊 Results:")
    print(f"  Total steps: {len(reward_history)}")
    print(f"  Total reward: {total_reward:.3f}")
    print(f"  Average reward per step: {total_reward/len(reward_history):.3f}")
    
    # Action distribution
    print(f"\n🎮 Action Distribution:")
    total_actions = sum(action_counts.values())
    for action_idx, count in sorted(action_counts.items()):
        action_name = action_names.get(action_idx, f"Unknown_{action_idx}")
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        print(f"  {action_name}: {count} times ({percentage:.1f}%)")
    
    # Check for non-zero rewards
    non_zero_rewards = [r for r in reward_history if r != 0]
    if non_zero_rewards:
        print(f"\n✅ Found {len(non_zero_rewards)} non-zero rewards!")
        print(f"  Reward range: {min(non_zero_rewards):.3f} to {max(non_zero_rewards):.3f}")
    else:
        print(f"\n❌ No non-zero rewards found")
    
    # Check for task completion
    if done:
        print(f"✅ Task completed!")
    else:
        print(f"❌ Task not completed within {len(reward_history)} steps")

if __name__ == "__main__":
    test_coordinated_agent()
