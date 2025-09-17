#!/usr/bin/env python3
"""
Greedy Agent

Simple greedy agent that always goes for the closest objective without coordination.
This agent will create conflicts and demonstrate need for intervention.
"""

import sys
import numpy as np

# Add project root to path
sys.path.append('.')

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction


class GreedyAgent:
    """Greedy agent that always goes for closest objective without coordination."""
    
    def __init__(self, mdp, agent_idx=0, agent_name="Greedy"):
        self.mdp = mdp
        self.agent_idx = agent_idx
        self.agent_name = agent_name
        self.step_count = 0
        self.heuristic = f"{agent_name}: Greedy behavior"
        
        # Cache important locations
        self.onion_locations = list(self.mdp.get_onion_dispenser_locations())
        self.dish_locations = list(self.mdp.get_dish_dispenser_locations())
        self.serve_locations = list(self.mdp.get_serving_locations())
        self.pot_locations = list(self.mdp.get_pot_locations())
        
        print(f"🏃 {agent_name} initialized: Greedy behavior (no coordination)")
    
    def _get_item_name(self, obj):
        """Get the name of an object."""
        if obj is None:
            return None
        return getattr(obj, 'name', str(obj))
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_closest_target(self, current_pos, targets):
        """Get closest target by Manhattan distance (greedy, no pathfinding)."""
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
        """Move greedily towards target (no pathfinding, can get stuck)."""
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
    
    def greedy_decision(self, state):
        """Make greedy decision - always go for closest objective."""
        ego = state.players[self.agent_idx]
        ego_item = self._get_item_name(ego.get_object()) if ego.has_object() else None
        
        if ego_item == "soup":
            return "SERVE_SOUP"
        elif ego_item == "dish":
            return "TAKE_SOUP"  # Always try to take soup, even if not ready
        elif ego_item == "onion":
            return "PUT_ONION_IN_POT"
        else:
            return "GET_ONION"  # Default: always get onions first
    
    def execute_greedy_action(self, macro, state):
        """Execute action greedily without coordination."""
        ego = state.players[self.agent_idx]
        ego_pos = ego.position
        ego_orientation = ego.orientation
        
        if macro == "GET_ONION":
            if not self.onion_locations:
                return 0
            target = self._get_closest_target(ego_pos, self.onion_locations)
            if self._is_adjacent(ego_pos, target):
                if self._is_facing_target(ego_pos, ego_orientation, target):
                    return 5  # INTERACT
                else:
                    return self._get_direction_to_face(ego_pos, target)
            else:
                return self._move_towards_target(ego_pos, target)
        
        elif macro == "PUT_ONION_IN_POT":
            if not self.pot_locations:
                return 0
            target = self._get_closest_target(ego_pos, self.pot_locations)
            if self._is_adjacent(ego_pos, target):
                if self._is_facing_target(ego_pos, ego_orientation, target):
                    return 5  # INTERACT
                else:
                    return self._get_direction_to_face(ego_pos, target)
            else:
                return self._move_towards_target(ego_pos, target)
        
        elif macro == "GET_DISH":
            if not self.dish_locations:
                return 0
            target = self._get_closest_target(ego_pos, self.dish_locations)
            if self._is_adjacent(ego_pos, target):
                if self._is_facing_target(ego_pos, ego_orientation, target):
                    return 5  # INTERACT
                else:
                    return self._get_direction_to_face(ego_pos, target)
            else:
                return self._move_towards_target(ego_pos, target)
        
        elif macro == "TAKE_SOUP":
            if not self.pot_locations:
                return 0
            target = self._get_closest_target(ego_pos, self.pot_locations)
            if self._is_adjacent(ego_pos, target):
                if self._is_facing_target(ego_pos, ego_orientation, target):
                    return 5  # INTERACT
                else:
                    return self._get_direction_to_face(ego_pos, target)
            else:
                return self._move_towards_target(ego_pos, target)
        
        elif macro == "SERVE_SOUP":
            if not self.serve_locations:
                return 0
            target = self._get_closest_target(ego_pos, self.serve_locations)
            if self._is_adjacent(ego_pos, target):
                if self._is_facing_target(ego_pos, ego_orientation, target):
                    return 5  # INTERACT
                else:
                    return self._get_direction_to_face(ego_pos, target)
            else:
                return self._move_towards_target(ego_pos, target)
        
        return 0
    
    def get_action(self, state):
        """Get greedy action."""
        self.step_count += 1
        
        try:
            ego = state.players[self.agent_idx]
            ego_pos = ego.position
            ego_item = self._get_item_name(ego.get_object()) if ego.has_object() else "none"
            
            # Make greedy decision
            macro_decision = self.greedy_decision(state)
            
            # Execute action greedily
            action = self.execute_greedy_action(macro_decision, state)
            
            self.heuristic = f"{self.agent_name}: {macro_decision} @ {ego_pos} (holding: {ego_item}) [GREEDY]"
            
            return action
            
        except Exception as e:
            print(f"[ERROR] Greedy agent {self.agent_idx} failed: {e}")
            self.heuristic = f"{self.agent_name}: Error - staying"
            return 0
    
    def action(self, state):
        """Method called by AgentPair."""
        return self.get_action(state)
