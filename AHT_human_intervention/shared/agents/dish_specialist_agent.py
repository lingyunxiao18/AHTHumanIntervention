#!/usr/bin/env python3
"""
Dish Specialist Agent

An agent that only knows how to pick up dishes, take soup from ready pots, and serve soup.
This agent is completely specialized and will not handle onions or pot filling.
"""

import sys
import numpy as np

# Add project root to path
sys.path.append('.')

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction


class DishSpecialistAgent:
    """Agent that specializes only in dish handling and soup serving."""
    
    def __init__(self, mdp, agent_idx=0, agent_name="DishSpec"):
        self.mdp = mdp
        self.agent_idx = agent_idx
        self.agent_name = agent_name
        self.step_count = 0
        self.heuristic = f"{agent_name}: Dish specialist"
        
        # Cache important locations - only dishes, pots (for soup), and serving matter
        self.dish_locations = list(self.mdp.get_dish_dispenser_locations())
        self.pot_locations = list(self.mdp.get_pot_locations())
        self.serve_locations = list(self.mdp.get_serving_locations())
        
        # This agent ignores onion dispensers
        self.ignored_locations = {
            'onion_locations': list(self.mdp.get_onion_dispenser_locations())
        }
        
        print(f"🍽️ {agent_name} initialized: DISH SPECIALIST")
        print(f"   ✅ Knows about: Dishes {self.dish_locations}, Pots {self.pot_locations}, Serving {self.serve_locations}")
        print(f"   ❌ Ignores: Onions, Pot Filling")
    
    def _get_item_name(self, obj):
        """Get the name of an object."""
        if obj is None:
            return None
        return getattr(obj, 'name', str(obj))
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_closest_target(self, current_pos, targets):
        """Get closest target by Manhattan distance."""
        if not targets:
            return None
        
        closest_target = None
        min_distance = float('inf')
        
        for target in targets:
            distance = self._manhattan_distance(current_pos, target)
            if distance < min_distance:
                min_distance = distance
                closest_target = target
        
        return closest_target
    
    def _move_towards_target(self, current_pos, target_pos):
        """Move towards target using simple pathfinding."""
        if current_pos == target_pos:
            return 0  # STAY
        
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Move in the direction that reduces the largest distance component
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 4  # EAST or WEST
        else:
            return 2 if dy > 0 else 3  # SOUTH or NORTH
    
    def _is_adjacent(self, pos1, pos2):
        """Check if two positions are adjacent."""
        return self._manhattan_distance(pos1, pos2) == 1
    
    def _is_facing_target(self, agent_pos, agent_orientation, target_pos):
        """Check if agent is facing the target."""
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        
        # Expected direction to face target
        if dx > 0:
            expected_orientation = (1, 0)  # EAST
        elif dx < 0:
            expected_orientation = (-1, 0)  # WEST
        elif dy > 0:
            expected_orientation = (0, 1)  # SOUTH
        elif dy < 0:
            expected_orientation = (0, -1)  # NORTH
        else:
            return True  # Same position
        
        return agent_orientation == expected_orientation
    
    def _get_direction_to_face(self, from_pos, to_pos):
        """Get the direction action needed to face target."""
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
    
    def _find_ready_pots(self, state):
        """Find pots with ready soup."""
        ready_pots = []
        
        try:
            for pot_pos in self.pot_locations:
                # Check pot state
                pot_obj = None
                for obj_pos, obj in state.objects.items():
                    if obj_pos == pot_pos and hasattr(obj, 'state'):
                        pot_obj = obj
                        break
                
                if pot_obj and hasattr(pot_obj, 'state'):
                    # Pot state is (soup_type, num_items, cook_time)
                    soup_type, num_items, cook_time = pot_obj.state
                    if num_items == 3 and cook_time >= 20:  # Ready soup
                        ready_pots.append(pot_pos)
            
            print(f"[DISH_SPEC] {self.agent_name} found ready pots: {ready_pots}")
            return ready_pots
            
        except Exception as e:
            print(f"[ERROR] Pot analysis failed: {e}")
            return []
    
    def _find_cooking_pots(self, state):
        """Find pots that are currently cooking."""
        cooking_pots = []
        
        try:
            for pot_pos in self.pot_locations:
                pot_obj = None
                for obj_pos, obj in state.objects.items():
                    if obj_pos == pot_pos and hasattr(obj, 'state'):
                        pot_obj = obj
                        break
                
                if pot_obj and hasattr(pot_obj, 'state'):
                    soup_type, num_items, cook_time = pot_obj.state
                    if num_items == 3 and cook_time < 20:  # Cooking
                        cooking_pots.append((pot_pos, 20 - cook_time))  # (position, time_left)
            
            return cooking_pots
            
        except Exception as e:
            print(f"[ERROR] Cooking pot analysis failed: {e}")
            return []
    
    def specialist_decision(self, state):
        """Make specialist decision - only handle dishes and soup."""
        ego = state.players[self.agent_idx]
        ego_item = self._get_item_name(ego.get_object()) if ego.has_object() else None
        
        # ONLY handle dish/soup workflow
        if ego_item == "soup":
            return "SERVE_SOUP"
        elif ego_item == "dish":
            # Check if there are ready pots
            ready_pots = self._find_ready_pots(state)
            if ready_pots:
                return "TAKE_SOUP"
            else:
                # No ready soup, wait or check for cooking
                cooking_pots = self._find_cooking_pots(state)
                if cooking_pots:
                    return "WAIT_FOR_COOKING"
                else:
                    return "NO_SOUP_AVAILABLE"  # Special state - need onions first
        elif ego_item is None:
            return "GET_DISH"
        elif ego_item == "onion":
            # Holding onion - this agent doesn't know what to do with onions!
            return "CONFUSED"  # Special state indicating need for intervention
        else:
            return "CONFUSED"
    
    def execute_specialist_action(self, macro, state):
        """Execute specialized dish-related actions only."""
        ego = state.players[self.agent_idx]
        ego_pos = ego.position
        ego_orientation = ego.orientation
        
        if macro == "GET_DISH":
            if not self.dish_locations:
                print(f"[DISH_SPEC] {self.agent_name}: No dish dispensers available!")
                return 0
            
            target = self._get_closest_target(ego_pos, self.dish_locations)
            print(f"[DISH_SPEC] {self.agent_name} targeting dish dispenser: {target}")
            
            if self._is_adjacent(ego_pos, target):
                if self._is_facing_target(ego_pos, ego_orientation, target):
                    print(f"[DISH_SPEC] {self.agent_name} picking up dish!")
                    return 5  # INTERACT
                else:
                    return self._get_direction_to_face(ego_pos, target)
            else:
                return self._move_towards_target(ego_pos, target)
        
        elif macro == "TAKE_SOUP":
            ready_pots = self._find_ready_pots(state)
            if not ready_pots:
                print(f"[DISH_SPEC] {self.agent_name}: No ready soup available!")
                return 0  # Stay - wait for soup
            
            target = self._get_closest_target(ego_pos, ready_pots)
            print(f"[DISH_SPEC] {self.agent_name} targeting ready pot: {target}")
            
            if self._is_adjacent(ego_pos, target):
                if self._is_facing_target(ego_pos, ego_orientation, target):
                    print(f"[DISH_SPEC] {self.agent_name} taking soup from pot!")
                    return 5  # INTERACT
                else:
                    return self._get_direction_to_face(ego_pos, target)
            else:
                return self._move_towards_target(ego_pos, target)
        
        elif macro == "SERVE_SOUP":
            if not self.serve_locations:
                print(f"[DISH_SPEC] {self.agent_name}: No serving locations available!")
                return 0
            
            target = self._get_closest_target(ego_pos, self.serve_locations)
            print(f"[DISH_SPEC] {self.agent_name} targeting serving counter: {target}")
            
            if self._is_adjacent(ego_pos, target):
                if self._is_facing_target(ego_pos, ego_orientation, target):
                    print(f"[DISH_SPEC] {self.agent_name} serving soup!")
                    return 5  # INTERACT
                else:
                    return self._get_direction_to_face(ego_pos, target)
            else:
                return self._move_towards_target(ego_pos, target)
        
        elif macro == "WAIT_FOR_COOKING":
            # Find a good waiting position near cooking pots
            cooking_pots = self._find_cooking_pots(state)
            if not cooking_pots:
                print(f"[DISH_SPEC] {self.agent_name}: Nothing cooking, getting more dishes")
                return self.execute_specialist_action("GET_DISH", state)
            
            # Wait near the pot that will be ready soonest
            cooking_pots.sort(key=lambda x: x[1])  # Sort by time remaining
            target_pot, time_left = cooking_pots[0]
            print(f"[DISH_SPEC] {self.agent_name} waiting near pot {target_pot} (ready in {time_left} steps)")
            
            # Find adjacent waiting position
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                wait_pos = (target_pot[0] + dx, target_pot[1] + dy)
                if wait_pos != ego_pos:
                    return self._move_towards_target(ego_pos, wait_pos)
            
            return 0  # Stay in place
        
        elif macro == "NO_SOUP_AVAILABLE":
            print(f"[DISH_SPEC] {self.agent_name}: No soup available and no cooking! Need onions in pots first!")
            print(f"[DISH_SPEC] {self.agent_name}: INTERVENTION NEEDED - Onion specialist should fill pots!")
            return 0  # Stay - intervention needed
        
        elif macro == "CONFUSED":
            # Agent is holding something it doesn't understand
            ego_item = self._get_item_name(ego.get_object()) if ego.has_object() else "none"
            print(f"[DISH_SPEC] {self.agent_name}: CONFUSED! Holding {ego_item} but only know dishes/soup!")
            print(f"[DISH_SPEC] {self.agent_name}: INTERVENTION NEEDED - Don't know what to do!")
            return 0  # Stay in place - human intervention required
        
        else:
            print(f"[DISH_SPEC] {self.agent_name}: Unknown macro {macro}")
            return 0
    
    def get_action(self, state):
        """Get specialist action."""
        self.step_count += 1
        
        try:
            ego = state.players[self.agent_idx]
            ego_pos = ego.position
            ego_item = self._get_item_name(ego.get_object()) if ego.has_object() else "none"
            
            # Make specialist decision
            macro_decision = self.specialist_decision(state)
            
            # Execute specialist action
            action = self.execute_specialist_action(macro_decision, state)
            
            # Update heuristic display
            status = "🍽️" if ego_item == "dish" else "🍲" if ego_item == "soup" else "❓" if macro_decision == "CONFUSED" else "⏳" if macro_decision == "WAIT_FOR_COOKING" else "🔍"
            self.heuristic = f"{self.agent_name}: {macro_decision} @ {ego_pos} {status} (holding: {ego_item})"
            
            return action
            
        except Exception as e:
            print(f"[ERROR] Dish specialist agent {self.agent_idx} failed: {e}")
            self.heuristic = f"{self.agent_name}: Error - staying"
            return 0
    
    def action(self, state):
        """Method called by AgentPair."""
        return self.get_action(state)
