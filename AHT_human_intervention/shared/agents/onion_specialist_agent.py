#!/usr/bin/env python3
"""
Onion Specialist Agent

An agent that only knows how to pick up onions and take them to empty pots.
This agent is completely specialized and will not handle dishes, soups, or serving.
"""

import sys
import numpy as np

# Add project root to path
sys.path.append('.')

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction


class OnionSpecialistAgent:
    """Agent that specializes only in onion collection and pot filling."""
    
    def __init__(self, mdp, agent_idx=0, agent_name="OnionSpec"):
        self.mdp = mdp
        self.agent_idx = agent_idx
        self.agent_name = agent_name
        self.step_count = 0
        self.heuristic = f"{agent_name}: Onion specialist"
        
        # Cache important locations - only onions and pots matter to this agent
        self.onion_locations = list(self.mdp.get_onion_dispenser_locations())
        self.pot_locations = list(self.mdp.get_pot_locations())
        
        # This agent ignores dishes, serving, etc.
        self.ignored_locations = {
            'dish_locations': list(self.mdp.get_dish_dispenser_locations()),
            'serve_locations': list(self.mdp.get_serving_locations())
        }
        
        # Anti-stuck mechanism to prevent blocking ego agent
        self.last_positions = []
        self.stuck_counter = 0
        self.max_stuck_steps = 8  # Lower threshold than ego agent to move more frequently
        
        print(f"🧅 {agent_name} initialized: ONION SPECIALIST")
        print(f"   ✅ Knows about: Onions {self.onion_locations}, Pots {self.pot_locations}")
        print(f"   ❌ Ignores: Dishes, Serving, Soups")
        print(f"   🚫 Anti-stuck: {self.max_stuck_steps} steps threshold")
    
    def _get_item_name(self, obj):
        """Get the name of an object."""
        if obj is None:
            return None
        return getattr(obj, 'name', str(obj))
    
    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_neighbors(self, pos):
        """Get valid neighboring positions."""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # SOUTH, NORTH, EAST, WEST
            new_pos = (pos[0] + dx, pos[1] + dy)
            if new_pos in self.mdp.terrain_pos_dict[" "]:  # Check if it's a walkable space
                neighbors.append(new_pos)
        return neighbors
    
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
    
    def _bfs_path(self, start_pos, target_pos):
        """Find shortest path using BFS."""
        from collections import deque
        
        if start_pos == target_pos:
            return [start_pos]
        
        queue = deque([(start_pos, [start_pos])])
        visited = {start_pos}
        
        while queue:
            current_pos, path = queue.popleft()
            
            # Check all 4 directions
            for dx, dy in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
                new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                if new_pos in visited:
                    continue
                
                # Check if position is valid (walkable)
                try:
                    terrain = self.mdp.terrain_mtx
                    if (0 <= new_pos[0] < len(terrain[0]) and 
                        0 <= new_pos[1] < len(terrain) and 
                        terrain[new_pos[1]][new_pos[0]] != 'X'):  # Not a wall
                        
                        new_path = path + [new_pos]
                        if new_pos == target_pos:
                            return new_path
                        
                        visited.add(new_pos)
                        queue.append((new_pos, new_path))
                except:
                    # Fallback: assume position is valid
                    new_path = path + [new_pos]
                    if new_pos == target_pos:
                        return new_path
                    
                    visited.add(new_pos)
                    queue.append((new_pos, new_path))
        
        return None  # No path found
    
    def _move_towards_target(self, current_pos, target_pos):
        """Move towards target using BFS pathfinding."""
        if current_pos == target_pos:
            return 0  # STAY
        
        # Find BFS path
        path = self._bfs_path(current_pos, target_pos)
        
        if not path or len(path) < 2:
            return 0
        
        # Get next position in path
        next_pos = path[1]
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        # Convert to action
        if dx == 1 and dy == 0:
            action = 1  # EAST
        elif dx == 0 and dy == 1:
            action = 2  # SOUTH
        elif dx == 0 and dy == -1:
            action = 3  # NORTH
        elif dx == -1 and dy == 0:
            action = 4  # WEST
        else:
            action = 0  # STAY
        
        return action
    
    def _is_adjacent(self, pos1, pos2):
        """Check if two positions are adjacent."""
        return self._manhattan_distance(pos1, pos2) == 1
    
    def _find_adjacent_positions(self, target_pos):
        """Find all valid adjacent positions to a target (excluding pot positions)."""
        adjacent_positions = []
        
        # Check all 4 directions
        for dx, dy in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
            adj_pos = (target_pos[0] + dx, target_pos[1] + dy)
            
            # Skip if this is a pot position (agents can't walk on pots)
            if adj_pos in self.pot_locations:
                continue
            
            # Check if position is valid (walkable)
            try:
                terrain = self.mdp.terrain_mtx
                if (0 <= adj_pos[0] < len(terrain[0]) and 
                    0 <= adj_pos[1] < len(terrain) and 
                    terrain[adj_pos[1]][adj_pos[0]] != 'X'):  # Not a wall
                    adjacent_positions.append(adj_pos)
            except:
                # Fallback: assume position is valid
                adjacent_positions.append(adj_pos)
        
        return adjacent_positions
    
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
    
    def _find_empty_pot(self, state):
        """Find a pot that has space for more onions."""
        empty_pots = []
        
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
                    # Only include pots that:
                    # 1. Have space for more onions (num_items < 3)
                    # 2. Are NOT ready (cook_time != 20)
                    # REMOVED: cook_time == 0 condition - allow interaction with cooking pots
                    if num_items < 3 and cook_time != 20:  # Has space AND not ready
                        empty_pots.append(pot_pos)
                else:
                    # No pot object found, assume empty
                    empty_pots.append(pot_pos)
            
            return empty_pots
            
        except Exception as e:
            print(f"[ERROR] Pot analysis failed: {e}")
            # Fallback: assume all pots are available
            return self.pot_locations
    
    def _analyze_pot_states(self, state):
        """Analyze pot states for smart decision making."""
        pot_states = {
            'any_pot_ready': False,
            'any_pot_cooking': False,
            'any_pot_has_space': False
        }
        
        for pot_pos in self.pot_locations:
            # Check pot state using same pattern as _find_empty_pot
            pot_obj = None
            for obj_pos, obj in state.objects.items():
                if obj_pos == pot_pos and hasattr(obj, 'state'):
                    pot_obj = obj
                    break
            
            if pot_obj and hasattr(pot_obj, 'state'):
                soup_type, num_items, cook_time = pot_obj.state
                if cook_time == 20:  # Soup is ready
                    pot_states['any_pot_ready'] = True
                elif num_items == 3 and cook_time > 0 and cook_time < 20:  # Cooking (3 onions, actively cooking)
                    pot_states['any_pot_cooking'] = True
                elif num_items < 3:  # Has space for more onions
                    pot_states['any_pot_has_space'] = True
            else:
                # No pot object found, assume empty (has space)
                pot_states['any_pot_has_space'] = True
        
        return pot_states
    
    def specialist_decision(self, state):
        """Make specialist decision - only handle onions and pots."""
        ego = state.players[self.agent_idx]
        ego_item = self._get_item_name(ego.get_object()) if ego.has_object() else None
        
        # Check if all pots are ready/cooking - if so, STOP working
        pot_states = self._analyze_pot_states(state)
        # Only stop if ALL pots are ready OR ALL pots are cooking (no space for more onions)
        if pot_states['any_pot_ready'] and not pot_states['any_pot_has_space']:
            return "WAIT"  # Stop working when all pots are ready
        elif pot_states['any_pot_cooking'] and not pot_states['any_pot_has_space']:
            return "WAIT"  # Stop working when all pots are cooking
        
        # ONLY handle onion workflow
        if ego_item == "onion":
            return "PUT_ONION_IN_POT"
        elif ego_item is None:
            return "GET_ONION"
        else:
            # Holding something else (dish, soup) - this agent doesn't know what to do!
            return "CONFUSED"  # Special state indicating need for intervention
    
    def execute_specialist_action(self, macro, state):
        """Execute specialized onion-related actions only."""
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
            empty_pots = self._find_empty_pot(state)
            if not empty_pots:
                return 0  # Stay - intervention needed
            
            # GREEDY: Prioritize pots that already have onions (to fill them faster)
            partial_pots = []
            truly_empty_pots = []
            
            for pot_pos in empty_pots:
                try:
                    pot_obj = None
                    for obj_pos, obj in state.objects.items():
                        if obj_pos == pot_pos and hasattr(obj, 'state'):
                            pot_obj = obj
                            break
                    
                    if pot_obj and hasattr(pot_obj, 'state'):
                        soup_type, num_items, cook_time = pot_obj.state
                        if num_items > 0:  # Has some onions already
                            partial_pots.append((pot_pos, num_items))
                        else:
                            truly_empty_pots.append(pot_pos)
                    else:
                        truly_empty_pots.append(pot_pos)
                except:
                    truly_empty_pots.append(pot_pos)
            
            # GREEDY STRATEGY: Fill partial pots first (closer to completion)
            if partial_pots:
                # Sort by number of onions (descending - fill the fullest first)
                partial_pots.sort(key=lambda x: x[1], reverse=True)
                target = partial_pots[0][0]
            else:
                target = self._get_closest_target(ego_pos, truly_empty_pots)
            
            # Check if we're adjacent to the pot
            if self._is_adjacent(ego_pos, target):
                if self._is_facing_target(ego_pos, ego_orientation, target):
                    return 5  # INTERACT
                else:
                    return self._get_direction_to_face(ego_pos, target)
            else:
                # Find an adjacent position to the pot, not the pot itself
                adjacent_positions = self._find_adjacent_positions(target)
                if adjacent_positions:
                    # Find the closest adjacent position
                    closest_adjacent = self._get_closest_target(ego_pos, adjacent_positions)
                    return self._move_towards_target(ego_pos, closest_adjacent)
                else:
                    # Fallback: move towards pot directly
                    return self._move_towards_target(ego_pos, target)
        
        elif macro == "WAIT":
            # Stop working - all pots are ready/cooking
            return 0  # Stay in place
        
        elif macro == "CONFUSED":
            # Agent is holding something it doesn't understand
            ego_item = self._get_item_name(ego.get_object()) if ego.has_object() else "none"
            return 0  # Stay in place - human intervention required
        
        else:
            return 0
    
    def get_action(self, state):
        """Get specialist action."""
        self.step_count += 1
        
        try:
            ego = state.players[self.agent_idx]
            ego_pos = ego.position
            ego_item = self._get_item_name(ego.get_object()) if ego.has_object() else "none"
            
            # Track position for anti-stuck mechanism
            self.last_positions.append(ego_pos)
            if len(self.last_positions) > self.max_stuck_steps:
                self.last_positions.pop(0)
            
            # Check if we're stuck (same position for too long)
            if len(self.last_positions) == self.max_stuck_steps and len(set(self.last_positions)) <= 2:
                self.stuck_counter += 1
                print(f"[ONION_STUCK] {self.agent_name} detected stuck at {ego_pos}, counter={self.stuck_counter}")
                
                # Simple stuck resolution: try random move
                if self.stuck_counter >= 5:  # Lower threshold to move more frequently
                    neighbors = self._get_neighbors(ego_pos)
                    if neighbors:
                        random_move = neighbors[self.stuck_counter % len(neighbors)]
                        print(f"[ONION_ANTI-STUCK] {self.agent_name} making random move to {random_move}")
                        # Calculate action to move to random position
                        dx = random_move[0] - ego_pos[0]
                        dy = random_move[1] - ego_pos[1]
                        if dx == 1: action = 1  # EAST
                        elif dx == -1: action = 4  # WEST
                        elif dy == 1: action = 2  # SOUTH
                        elif dy == -1: action = 3  # NORTH
                        else: action = 0  # STAY
                        
                        # Reset stuck counter and positions
                        self.stuck_counter = 0
                        self.last_positions = []
                        
                        # Update heuristic display
                        self.heuristic = f"{self.agent_name}: ANTI-STUCK move to {random_move}"
                        return action
            else:
                # Reset stuck counter if making progress
                if self.stuck_counter > 0:
                    self.stuck_counter = max(0, self.stuck_counter - 1)
            
            # Make specialist decision
            macro_decision = self.specialist_decision(state)
            
            # Execute specialist action
            action = self.execute_specialist_action(macro_decision, state)
            
            # Update heuristic display
            status = "🧅" if ego_item == "onion" else "❓" if macro_decision == "CONFUSED" else "🔍"
            self.heuristic = f"{self.agent_name}: {macro_decision} @ {ego_pos} {status} (holding: {ego_item})"
            
            return action
            
        except Exception as e:
            print(f"[ERROR] Onion specialist agent {self.agent_idx} failed: {e}")
            self.heuristic = f"{self.agent_name}: Error - staying"
            return 0
    
    def action(self, state):
        """Method called by AgentPair."""
        return self.get_action(state)
