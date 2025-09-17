#!/usr/bin/env python3
"""
Macro action system for Overcooked with proper preconditions, termination, and failure handling.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import random

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from shared.envs.envs.overcooked.overcooked_ai_py.agents.agent import Agent
from shared.envs.envs.overcooked.script_agent.utils import bfs

class MacroAction(Enum):
    """Macro actions for Overcooked."""
    GO_TO_ONION = "GO_TO_ONION"
    TAKE_ONION = "TAKE_ONION"
    GO_TO_POT = "GO_TO_POT"
    PUT_IN_POT = "PUT_IN_POT"
    GO_TO_DISH = "GO_TO_DISH"
    TAKE_DISH = "TAKE_DISH"
    TAKE_SOUP = "TAKE_SOUP"  # New: take soup from pot with dish
    GO_TO_SERVE = "GO_TO_SERVE"
    SERVE = "SERVE"
    WAIT_COOK = "WAIT_COOK"
    STAY = "STAY"  # Fallback action

class MacroActionState:
    """Tracks the state of a macro action execution."""
    
    def __init__(self, macro_action: MacroAction, args: Dict[str, Any] = None):
        self.macro_action = macro_action
        self.args = args or {}
        self.step_count = 0
        self.failure_count = 0
        self.last_position = None
        self.blocked_steps = 0
        self.interact_failures = 0
        self.is_completed = False
        self.is_failed = False
        self.failure_reason = None

class MacroActionExecutor:
    """Converts macro actions to primitive actions using pathfinding with proper state tracking."""
    
    def __init__(self, mdp: OvercookedGridworld):
        self.mdp = mdp
        self.current_macro_state = None
        self.max_blocked_steps = 10  # K steps for path blocked
        self.max_interact_failures = 3  # Max interact failures
    
    def execute_macro_action(self, macro_action: MacroAction, state: OvercookedState, 
                           player_idx: int = 0, args: Dict[str, Any] = None) -> Tuple[str, bool]:
        """
        Execute a macro action and return (primitive_action, should_continue_macro).
        
        Returns:
            - primitive_action: The action to take
            - should_continue_macro: Whether to continue with this macro action
        """
        player = state.players[player_idx]
        
        # Initialize macro state if starting new macro
        if (self.current_macro_state is None or 
            self.current_macro_state.macro_action != macro_action or
            self.current_macro_state.is_completed or 
            self.current_macro_state.is_failed):
            self.current_macro_state = MacroActionState(macro_action, args)
        
        # Update step count
        self.current_macro_state.step_count += 1
        
        # Check preconditions
        if not self._check_preconditions(state, player_idx):
            self.current_macro_state.is_failed = True
            self.current_macro_state.failure_reason = "Preconditions not met"
            return "STAY", False
        
        # Execute macro action
        if macro_action == MacroAction.GO_TO_ONION:
            return self._go_to_onion(state, player_idx)
        elif macro_action == MacroAction.TAKE_ONION:
            return self._take_onion(state, player_idx)
        elif macro_action == MacroAction.GO_TO_POT:
            return self._go_to_pot(state, player_idx)
        elif macro_action == MacroAction.PUT_IN_POT:
            return self._put_in_pot(state, player_idx)
        elif macro_action == MacroAction.GO_TO_DISH:
            return self._go_to_dish(state, player_idx)
        elif macro_action == MacroAction.TAKE_DISH:
            return self._take_dish(state, player_idx)
        elif macro_action == MacroAction.TAKE_SOUP:
            return self._take_soup(state, player_idx)
        elif macro_action == MacroAction.GO_TO_SERVE:
            return self._go_to_serve(state, player_idx)
        elif macro_action == MacroAction.SERVE:
            return self._serve(state, player_idx)
        elif macro_action == MacroAction.WAIT_COOK:
            return self._wait_cook(state, player_idx)
        elif macro_action == MacroAction.STAY:
            return "STAY", True
        else:
            return "STAY", True
    
    def _check_preconditions(self, state: OvercookedState, player_idx: int) -> bool:
        """Check if preconditions are met for the current macro action."""
        player = state.players[player_idx]
        macro_action = self.current_macro_state.macro_action
        args = self.current_macro_state.args
        
        if macro_action == MacroAction.GO_TO_ONION:
            # Precondition: path exists to onion dispenser
            onion_dispensers = self.mdp.get_onion_dispenser_locations()
            if not onion_dispensers:
                return False
            
            # Check if any onion dispenser is reachable
            dist, _ = bfs(self.mdp, state, player_idx)
            for onion_pos in onion_dispensers:
                if dist[onion_pos[1]][onion_pos[0]] != -1:
                    return True
            return False
            
        elif macro_action == MacroAction.TAKE_ONION:
            # Precondition: adjacent to onion dispenser, hands empty
            if player.held_object is not None:
                return False
            
            onion_dispensers = self.mdp.get_onion_dispenser_locations()
            for onion_pos in onion_dispensers:
                if self._is_adjacent(player.position, onion_pos):
                    return True
            return False
            
        elif macro_action == MacroAction.GO_TO_POT:
            # Precondition: holding onion or dish
            if player.held_object is None:
                return False
            
            # Check if path exists to any pot
            pots = self.mdp.get_pot_locations()
            if not pots:
                return False
            
            dist, _ = bfs(self.mdp, state, player_idx)
            for pot_pos in pots:
                if dist[pot_pos[1]][pot_pos[0]] != -1:
                    return True
            return False
            
        elif macro_action == MacroAction.PUT_IN_POT:
            # Precondition: adjacent to pot, holding onion
            if player.held_object is None or player.held_object.name != "onion":
                return False
            
            pots = self.mdp.get_pot_locations()
            for pot_pos in pots:
                if self._is_adjacent(player.position, pot_pos):
                    return True
            return False
            
        elif macro_action == MacroAction.GO_TO_DISH:
            # Precondition: hands empty
            return player.held_object is None
            
        elif macro_action == MacroAction.TAKE_DISH:
            # Precondition: adjacent to dish dispenser, hands empty
            if player.held_object is not None:
                return False
            
            dish_dispensers = self.mdp.get_dish_dispenser_locations()
            for dish_pos in dish_dispensers:
                if self._is_adjacent(player.position, dish_pos):
                    return True
            return False
            
        elif macro_action == MacroAction.TAKE_SOUP:
            # Precondition: adjacent to pot with cooked soup, holding dish
            if player.held_object is None or player.held_object.name != "dish":
                return False
            
            # Check if any pot has cooked soup
            pots = self.mdp.get_pot_locations()
            for pot_pos in pots:
                if self._is_adjacent(player.position, pot_pos):
                    # Check if pot has cooked soup (simplified check)
                    if pot_pos in state.objects:
                        pot_obj = state.objects[pot_pos]
                        if pot_obj.name == "pot" and len(pot_obj.state) >= 4:
                            onions_count, is_cooking, cook_time, is_ready = pot_obj.state
                            if is_ready and onions_count >= 3:  # Cooked soup ready
                                return True
            return False
            
        elif macro_action == MacroAction.GO_TO_SERVE:
            # Precondition: holding soup or dish with soup
            if player.held_object is None:
                return False
            # Note: This is simplified - should check for soup specifically
            return True
            
        elif macro_action == MacroAction.SERVE:
            # Precondition: adjacent to serve area, has soup on dish
            if player.held_object is None:
                return False
            # Note: This is simplified - should check for soup specifically
            
            serving_locations = self.mdp.get_serving_locations()
            for serve_pos in serving_locations:
                if self._is_adjacent(player.position, serve_pos):
                    return True
            return False
            
        elif macro_action == MacroAction.WAIT_COOK:
            # Precondition: pot is cooking
            # Note: This would need to check pot states
            return True
            
        return True
    
    def _check_termination(self, state: OvercookedState, player_idx: int) -> bool:
        """Check if the macro action has reached its termination condition."""
        player = state.players[player_idx]
        macro_action = self.current_macro_state.macro_action
        
        if macro_action == MacroAction.GO_TO_ONION:
            # Termination: adjacent to onion dispenser AND facing it
            onion_dispensers = self.mdp.get_onion_dispenser_locations()
            for onion_pos in onion_dispensers:
                if self._is_adjacent(player.position, onion_pos):
                    # Also check if facing the dispenser
                    facing_pos = self._get_facing_position(player.position, player.orientation)
                    if facing_pos == onion_pos:
                        return True
            return False
            
        elif macro_action == MacroAction.TAKE_ONION:
            # Termination: holding onion
            return player.held_object is not None and player.held_object.name == "onion"
            
        elif macro_action == MacroAction.GO_TO_POT:
            # Termination: adjacent to target pot AND facing it
            pots = self.mdp.get_pot_locations()
            for pot_pos in pots:
                if self._is_adjacent(player.position, pot_pos):
                    # Also check if facing the pot
                    facing_pos = self._get_facing_position(player.position, player.orientation)
                    if facing_pos == pot_pos:
                        return True
            return False
            
        elif macro_action == MacroAction.PUT_IN_POT:
            # Termination: onion count increased (simplified - just check if not holding onion)
            return player.held_object is None or player.held_object.name != "onion"
            
        elif macro_action == MacroAction.GO_TO_DISH:
            # Termination: adjacent to dish dispenser AND facing it
            dish_dispensers = self.mdp.get_dish_dispenser_locations()
            for dish_pos in dish_dispensers:
                if self._is_adjacent(player.position, dish_pos):
                    # Also check if facing the dispenser
                    facing_pos = self._get_facing_position(player.position, player.orientation)
                    if facing_pos == dish_pos:
                        return True
            return False
            
        elif macro_action == MacroAction.TAKE_DISH:
            # Termination: holding dish
            return player.held_object is not None and player.held_object.name == "dish"
            
        elif macro_action == MacroAction.TAKE_SOUP:
            # Termination: holding soup (dish with soup)
            return player.held_object is not None and "soup" in str(player.held_object.name)
            
        elif macro_action == MacroAction.GO_TO_SERVE:
            # Termination: adjacent to serve area AND facing it
            serving_locations = self.mdp.get_serving_locations()
            for serve_pos in serving_locations:
                if self._is_adjacent(player.position, serve_pos):
                    # Also check if facing the serving area
                    facing_pos = self._get_facing_position(player.position, player.orientation)
                    if facing_pos == serve_pos:
                        return True
            return False
            
        elif macro_action == MacroAction.SERVE:
            # Termination: soup delivered (simplified - just check if not holding soup)
            return player.held_object is None or "soup" not in str(player.held_object.name)
            
        elif macro_action == MacroAction.WAIT_COOK:
            # Termination: pot ready or timeout
            # Note: This would need to check pot states and implement timeout
            return self.current_macro_state.step_count > 20  # Simple timeout
            
        return False
    
    def _check_failure(self, state: OvercookedState, player_idx: int) -> bool:
        """Check if the macro action has failed."""
        player = state.players[player_idx]
        macro_action = self.current_macro_state.macro_action
        
        # Check for blocked path (for navigation macros)
        if macro_action in [MacroAction.GO_TO_ONION, MacroAction.GO_TO_POT, MacroAction.GO_TO_DISH, MacroAction.GO_TO_SERVE]:
            if player.position == self.current_macro_state.last_position:
                self.current_macro_state.blocked_steps += 1
            else:
                self.current_macro_state.blocked_steps = 0
            
            if self.current_macro_state.blocked_steps >= self.max_blocked_steps:
                self.current_macro_state.is_failed = True
                self.current_macro_state.failure_reason = f"Path blocked for {self.max_blocked_steps} steps"
                return True
        
        # Check for interact failures
        if macro_action in [MacroAction.TAKE_ONION, MacroAction.PUT_IN_POT, MacroAction.TAKE_DISH, MacroAction.TAKE_SOUP, MacroAction.SERVE]:
            if self.current_macro_state.interact_failures >= self.max_interact_failures:
                self.current_macro_state.is_failed = True
                self.current_macro_state.failure_reason = f"Interact failed {self.max_interact_failures} times"
                return True
        
        # Update last position
        self.current_macro_state.last_position = player.position
        return False
    
    def _go_to_onion(self, state: OvercookedState, player_idx: int) -> Tuple[str, bool]:
        """Navigate to the nearest onion dispenser using BFS."""
        player = state.players[player_idx]
        onion_dispensers = self.mdp.get_onion_dispenser_locations()
        
        # Check termination
        if self._check_termination(state, player_idx):
            self.current_macro_state.is_completed = True
            return "STAY", False
        
        # Check failure
        if self._check_failure(state, player_idx):
            return "STAY", False
        
        # Find the nearest accessible onion dispenser using BFS
        best_target = None
        best_distance = float('inf')
        
        for onion_pos in onion_dispensers:
            # Find path to this onion dispenser
            dist, path = bfs(self.mdp, state, player_idx)
            distance = dist[onion_pos[1]][onion_pos[0]]
            
            if distance != -1 and distance < best_distance:
                best_distance = distance
                best_target = onion_pos
        
        if best_target is None:
            return "STAY", True  # Continue trying
        
        # Get the next action towards the target
        action = self._get_next_action_towards_target(player.position, best_target, path)
        return action, True
    
    def _take_onion(self, state: OvercookedState, player_idx: int) -> Tuple[str, bool]:
        """Take an onion from the dispenser."""
        player = state.players[player_idx]
        onion_dispensers = self.mdp.get_onion_dispenser_locations()
        
        # Check termination
        if self._check_termination(state, player_idx):
            self.current_macro_state.is_completed = True
            return "STAY", False
        
        # Check failure
        if self._check_failure(state, player_idx):
            return "STAY", False
        
        # Check if player is adjacent to an onion dispenser
        for onion_pos in onion_dispensers:
            if self._is_adjacent(player.position, onion_pos):
                # Check if player is facing the dispenser
                facing_pos = self._get_facing_position(player.position, player.orientation)
                if facing_pos == onion_pos:
                    action = "INTERACT"
                    # Track interact attempts
                    self.current_macro_state.interact_failures += 1
                    return action, True
                else:
                    # Need to rotate to face the dispenser
                    action = self._get_rotation_action(player.orientation, 
                                                   self._get_orientation_to_position(player.position, onion_pos))
                    return action, True
        
        # If not adjacent, return STAY instead of recursive call
        # The agent should call GO_TO_ONION macro first, then TAKE_ONION
        return "STAY", True
    
    def _go_to_pot(self, state: OvercookedState, player_idx: int) -> Tuple[str, bool]:
        """Navigate to the nearest pot using BFS."""
        player = state.players[player_idx]
        pots = self.mdp.get_pot_locations()
        
        # Check termination
        if self._check_termination(state, player_idx):
            self.current_macro_state.is_completed = True
            return "STAY", False
        
        # Check failure
        if self._check_failure(state, player_idx):
            return "STAY", False
        
        # Find the nearest accessible pot using BFS
        best_target = None
        best_distance = float('inf')
        
        for pot_pos in pots:
            # Find path to this pot
            dist, path = bfs(self.mdp, state, player_idx)
            distance = dist[pot_pos[1]][pot_pos[0]]
            
            if distance != -1 and distance < best_distance:
                best_distance = distance
                best_target = pot_pos
        
        if best_target is None:
            return "STAY", True  # Continue trying
        
        # Get the next action towards the target
        action = self._get_next_action_towards_target(player.position, best_target, path)
        return action, True
    
    def _put_in_pot(self, state: OvercookedState, player_idx: int) -> Tuple[str, bool]:
        """Put an onion in a pot."""
        player = state.players[player_idx]
        pots = self.mdp.get_pot_locations()
        
        # Check termination
        if self._check_termination(state, player_idx):
            self.current_macro_state.is_completed = True
            return "STAY", False
        
        # Check failure
        if self._check_failure(state, player_idx):
            return "STAY", False
        
        # Check if player is adjacent to a pot
        for pot_pos in pots:
            if self._is_adjacent(player.position, pot_pos):
                # Check if player is facing the pot
                facing_pos = self._get_facing_position(player.position, player.orientation)
                if facing_pos == pot_pos:
                    action = "INTERACT"
                    # Track interact attempts
                    self.current_macro_state.interact_failures += 1
                    return action, True
                else:
                    # Need to rotate to face the pot
                    action = self._get_rotation_action(player.orientation, 
                                                   self._get_orientation_to_position(player.position, pot_pos))
                    return action, True
        
        # If not adjacent, return STAY instead of recursive call
        # The agent should call GO_TO_POT macro first, then PUT_IN_POT
        return "STAY", True
    
    def _go_to_dish(self, state: OvercookedState, player_idx: int) -> Tuple[str, bool]:
        """Navigate to the nearest dish dispenser using BFS."""
        player = state.players[player_idx]
        dish_dispensers = self.mdp.get_dish_dispenser_locations()
        
        # Check termination
        if self._check_termination(state, player_idx):
            self.current_macro_state.is_completed = True
            return "STAY", False
        
        # Check failure
        if self._check_failure(state, player_idx):
            return "STAY", False
        
        # Find the nearest accessible dish dispenser using BFS
        best_target = None
        best_distance = float('inf')
        
        for dish_pos in dish_dispensers:
            # Find path to this dish dispenser
            dist, path = bfs(self.mdp, state, player_idx)
            distance = dist[dish_pos[1]][dish_pos[0]]
            
            if distance != -1 and distance < best_distance:
                best_distance = distance
                best_target = dish_pos
        
        if best_target is None:
            return "STAY", True  # Continue trying
        
        # Get the next action towards the target
        action = self._get_next_action_towards_target(player.position, best_target, path)
        return action, True
    
    def _take_dish(self, state: OvercookedState, player_idx: int) -> Tuple[str, bool]:
        """Take a dish from the dispenser."""
        player = state.players[player_idx]
        dish_dispensers = self.mdp.get_dish_dispenser_locations()
        
        # Check termination
        if self._check_termination(state, player_idx):
            self.current_macro_state.is_completed = True
            return "STAY", False
        
        # Check failure
        if self._check_failure(state, player_idx):
            return "STAY", False
        
        # Check if player is adjacent to a dish dispenser
        for dish_pos in dish_dispensers:
            if self._is_adjacent(player.position, dish_pos):
                # Check if player is facing the dispenser
                facing_pos = self._get_facing_position(player.position, player.orientation)
                if facing_pos == dish_pos:
                    action = "INTERACT"
                    # Track interact attempts
                    self.current_macro_state.interact_failures += 1
                    return action, True
                else:
                    # Need to rotate to face the dispenser
                    action = self._get_rotation_action(player.orientation, 
                                                   self._get_orientation_to_position(player.position, dish_pos))
                    return action, True
        
        # If not adjacent, return STAY instead of recursive call
        # The agent should call GO_TO_DISH macro first, then TAKE_DISH
        return "STAY", True
    
    def _take_soup(self, state: OvercookedState, player_idx: int) -> Tuple[str, bool]:
        """Take soup from a pot with cooked soup."""
        player = state.players[player_idx]
        pots = self.mdp.get_pot_locations()
        
        # Check termination
        if self._check_termination(state, player_idx):
            self.current_macro_state.is_completed = True
            return "STAY", False
        
        # Check failure
        if self._check_failure(state, player_idx):
            return "STAY", False
        
        # Check if player is adjacent to a pot with cooked soup
        for pot_pos in pots:
            if self._is_adjacent(player.position, pot_pos):
                # Check if pot has cooked soup
                if pot_pos in state.objects:
                    pot_obj = state.objects[pot_pos]
                    if pot_obj.name == "pot" and len(pot_obj.state) >= 4:
                        onions_count, is_cooking, cook_time, is_ready = pot_obj.state
                        if is_ready and onions_count >= 3:  # Cooked soup ready
                            # Check if player is facing the pot
                            facing_pos = self._get_facing_position(player.position, player.orientation)
                            if facing_pos == pot_pos:
                                action = "INTERACT"
                                # Track interact attempts
                                self.current_macro_state.interact_failures += 1
                                return action, True
                            else:
                                # Need to rotate to face the pot
                                action = self._get_rotation_action(player.orientation, 
                                                               self._get_orientation_to_position(player.position, pot_pos))
                                return action, True
        
        # If not adjacent to a pot with cooked soup, try to get closer
        return self._go_to_pot(state, player_idx)
    
    def _go_to_serve(self, state: OvercookedState, player_idx: int) -> Tuple[str, bool]:
        """Navigate to the serving area using BFS."""
        player = state.players[player_idx]
        serving_locations = self.mdp.get_serving_locations()
        
        # Check termination
        if self._check_termination(state, player_idx):
            self.current_macro_state.is_completed = True
            return "STAY", False
        
        # Check failure
        if self._check_failure(state, player_idx):
            return "STAY", False
        
        # Find the nearest accessible serving location using BFS
        best_target = None
        best_distance = float('inf')
        
        for serve_pos in serving_locations:
            # Find path to this serving location
            dist, path = bfs(self.mdp, state, player_idx)
            distance = dist[serve_pos[1]][serve_pos[0]]
            
            if distance != -1 and distance < best_distance:
                best_distance = distance
                best_target = serve_pos
        
        if best_target is None:
            return "STAY", True  # Continue trying
        
        # Get the next action towards the target
        action = self._get_next_action_towards_target(player.position, best_target, path)
        return action, True
    
    def _serve(self, state: OvercookedState, player_idx: int) -> Tuple[str, bool]:
        """Serve soup at the serving area."""
        player = state.players[player_idx]
        serving_locations = self.mdp.get_serving_locations()
        
        # Check termination
        if self._check_termination(state, player_idx):
            self.current_macro_state.is_completed = True
            return "STAY", False
        
        # Check failure
        if self._check_failure(state, player_idx):
            return "STAY", False
        
        # Check if player is adjacent to a serving location
        for serve_pos in serving_locations:
            if self._is_adjacent(player.position, serve_pos):
                # Check if player is facing the serving area
                facing_pos = self._get_facing_position(player.position, player.orientation)
                if facing_pos == serve_pos:
                    action = "INTERACT"
                    # Track interact attempts
                    self.current_macro_state.interact_failures += 1
                    return action, True
                else:
                    # Need to rotate to face the serving area
                    action = self._get_rotation_action(player.orientation, 
                                                   self._get_orientation_to_position(player.position, serve_pos))
                    return action, True
        
        # If not adjacent, try to get closer
        return self._go_to_serve(state, player_idx)
    
    def _wait_cook(self, state: OvercookedState, player_idx: int) -> Tuple[str, bool]:
        """Wait for soup to cook."""
        # Check termination
        if self._check_termination(state, player_idx):
            self.current_macro_state.is_completed = True
            return "STAY", False
        
        # Check failure
        if self._check_failure(state, player_idx):
            return "STAY", False
        
        return "STAY", True
    
    def _is_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)
    
    def _get_facing_position(self, position: Tuple[int, int], orientation: Tuple[int, int]) -> Tuple[int, int]:
        """Get the position the player is facing."""
        return (position[0] + orientation[0], position[1] + orientation[1])
    
    def _get_orientation_to_position(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Get the orientation needed to face a position."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if dx > 0:
            return (1, 0)  # East
        elif dx < 0:
            return (-1, 0)  # West
        elif dy > 0:
            return (0, 1)  # South
        elif dy < 0:
            return (0, -1)  # North
        else:
            return (0, 0)  # Same position
    
    def _get_rotation_action(self, current_orientation: Tuple[int, int], target_orientation: Tuple[int, int]) -> str:
        """Get action to rotate from current to target orientation."""
        if current_orientation == target_orientation:
            return "STAY"
        
        # Complete rotation logic for all orientations
        if current_orientation == (0, -1):  # North
            if target_orientation == (1, 0):  # East
                return "MOVE_E"  # Turn right
            elif target_orientation == (-1, 0):  # West
                return "MOVE_W"  # Turn left
            elif target_orientation == (0, 1):  # South
                return "MOVE_S"  # Turn around
        elif current_orientation == (0, 1):  # South
            if target_orientation == (1, 0):  # East
                return "MOVE_W"  # Turn left
            elif target_orientation == (-1, 0):  # West
                return "MOVE_E"  # Turn right
            elif target_orientation == (0, -1):  # North
                return "MOVE_N"  # Turn around
        elif current_orientation == (1, 0):  # East
            if target_orientation == (0, -1):  # North
                return "MOVE_N"  # Turn left
            elif target_orientation == (0, 1):  # South
                return "MOVE_S"  # Turn right
            elif target_orientation == (-1, 0):  # West
                return "MOVE_W"  # Turn around
        elif current_orientation == (-1, 0):  # West
            if target_orientation == (0, -1):  # North
                return "MOVE_S"  # Turn right
            elif target_orientation == (0, 1):  # South
                return "MOVE_N"  # Turn left
            elif target_orientation == (1, 0):  # East
                return "MOVE_E"  # Turn around
        
        return "STAY"
    
    def _get_next_action_towards_target(self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                                       path: List[List]) -> str:
        """Get the next action to move towards the target using the path."""
        if current_pos == target_pos:
            return "STAY"
        
        # Reconstruct the path from target to current
        current = target_pos
        path_actions = []
        
        while current != current_pos:
            prev_pos, direction = path[current[1]][current[0]]
            if prev_pos is None:
                break
            path_actions.insert(0, direction)
            current = prev_pos
        
        if not path_actions:
            return "STAY"
        
        # Return the first action in the path
        direction = path_actions[0]
        if direction == Direction.NORTH:
            return "MOVE_N"
        elif direction == Direction.SOUTH:
            return "MOVE_S"
        elif direction == Direction.EAST:
            return "MOVE_E"
        elif direction == Direction.WEST:
            return "MOVE_W"
        else:
            return "STAY"

class MacroActionPolicy:
    """Policy that parses human commands into macro actions with proper state tracking."""
    
    def __init__(self, mdp: OvercookedGridworld):
        self.mdp = mdp
        self.executor = MacroActionExecutor(mdp)
    
    def get_action(self, state: OvercookedState, command: str) -> str:
        """Parse command and return primitive action."""
        macro_action = self._parse_command(command)
        action, should_continue = self.executor.execute_macro_action(macro_action, state)
        return action
    
    def get_macro_status(self) -> Optional[Dict[str, Any]]:
        """Get the current status of the macro action execution."""
        if self.executor.current_macro_state is None:
            return None
        
        state = self.executor.current_macro_state
        return {
            "macro_action": state.macro_action.value,
            "step_count": state.step_count,
            "is_completed": state.is_completed,
            "is_failed": state.is_failed,
            "failure_reason": state.failure_reason,
            "blocked_steps": state.blocked_steps,
            "interact_failures": state.interact_failures
        }
    
    def _parse_command(self, command: str) -> MacroAction:
        """Parse human command into macro action."""
        command = command.lower().strip()
        
        if "onion" in command and "go" in command:
            return MacroAction.GO_TO_ONION
        elif "onion" in command and ("take" in command or "pick" in command or "get" in command):
            return MacroAction.TAKE_ONION
        elif "pot" in command and "go" in command:
            return MacroAction.GO_TO_POT
        elif "pot" in command and ("put" in command or "place" in command):
            return MacroAction.PUT_IN_POT
        elif "dish" in command and "go" in command:
            return MacroAction.GO_TO_DISH
        elif "dish" in command and ("take" in command or "pick" in command or "get" in command):
            return MacroAction.TAKE_DISH
        elif "soup" in command and ("take" in command or "pick" in command or "get" in command):
            return MacroAction.TAKE_SOUP
        elif ("serve" in command and "go" in command) or ("serving" in command and "go" in command):
            return MacroAction.GO_TO_SERVE
        elif "serve" in command:
            return MacroAction.SERVE
        elif "wait" in command or "cook" in command:
            return MacroAction.WAIT_COOK
        elif "stay" in command or "wait" in command:
            return MacroAction.STAY
        else:
            return MacroAction.STAY

def create_macro_action_policy(mdp: OvercookedGridworld) -> MacroActionPolicy:
    """Create a macro action policy for the given MDP."""
    return MacroActionPolicy(mdp)
