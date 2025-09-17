#!/usr/bin/env python3
"""
Macro action agents for generating training data.
Different agents follow different macro action strategies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from pipelines.rl_finetuning.macro_actions.macro_actions import MacroActionPolicy, MacroAction
from typing import List, Dict, Any, Optional
import random

class MacroActionAgent:
    """Base class for agents that use macro actions."""
    
    def __init__(self, mdp: OvercookedGridworld, agent_id: int):
        self.mdp = mdp
        self.agent_id = agent_id
        self.policy = MacroActionPolicy(mdp)
        self.current_macro = None
        self.macro_queue = []
        self.last_macro_status = None  # Track last macro status
        
    def get_action(self, state: OvercookedState) -> str:
        """Get the next action for this agent."""
        raise NotImplementedError
        
    def _execute_macro(self, macro_command: str, state: OvercookedState) -> str:
        """Execute a macro action and return the primitive action."""
        action = self.policy.get_action(state, macro_command)
        status = self.policy.get_macro_status()
        
        # Check if macro status changed (completed or failed)
        if status and self.last_macro_status:
            if (status['is_completed'] and not self.last_macro_status['is_completed']) or \
               (status['is_failed'] and not self.last_macro_status['is_failed']):
                # Macro just completed or failed
                self.current_macro = None
        
        # Update current macro state
        if status:
            if status['is_completed'] or status['is_failed']:
                self.current_macro = None
            else:
                self.current_macro = status['macro_action']
        
        # Store current status for next comparison
        self.last_macro_status = status
        
        return action

class OnionCookAgent(MacroActionAgent):
    """Agent that focuses on onion cooking: go to onion -> take onion -> go to pot -> put in pot."""
    
    def __init__(self, mdp: OvercookedGridworld, agent_id: int):
        super().__init__(mdp, agent_id)
        self.cooking_phase = "go_to_onion"  # Current phase of cooking
        
    def get_action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_id]
        
        # Check current phase and execute appropriate macro
        if self.cooking_phase == "go_to_onion":
            if player.held_object is None:
                # Need to get onion
                action = self._execute_macro("go to onion", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed (either just completed or was already completed)
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        # Macro just completed
                        print(f"🔄 Phase transition: go_to_onion -> take_onion")
                    else:
                        # Macro was already completed (immediate completion)
                        print(f"🔄 Immediate completion: go_to_onion -> take_onion")
                    self.cooking_phase = "take_onion"
                return action
            else:
                # Already have onion, move to next phase
                self.cooking_phase = "go_to_pot"
                
        if self.cooking_phase == "take_onion":
            if player.held_object is None:
                # Take onion
                action = self._execute_macro("take onion", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Phase transition: take_onion -> go_to_pot")
                    else:
                        print(f"🔄 Immediate completion: take_onion -> go_to_pot")
                    self.cooking_phase = "go_to_pot"
                return action
            else:
                # Already have onion, move to next phase
                self.cooking_phase = "go_to_pot"
                
        if self.cooking_phase == "go_to_pot":
            if player.held_object is not None and player.held_object.name == "onion":
                # Go to pot
                action = self._execute_macro("go to pot", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Phase transition: go_to_pot -> put_in_pot")
                    else:
                        print(f"🔄 Immediate completion: go_to_pot -> put_in_pot")
                    self.cooking_phase = "put_in_pot"
                return action
            else:
                # Lost onion, go back to start
                self.cooking_phase = "go_to_onion"
                
        if self.cooking_phase == "put_in_pot":
            if player.held_object is not None and player.held_object.name == "onion":
                # Put onion in pot
                action = self._execute_macro("put in pot", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Phase transition: put_in_pot -> go_to_onion (restart)")
                    else:
                        print(f"🔄 Immediate completion: put_in_pot -> go_to_onion (restart)")
                    self.cooking_phase = "go_to_onion"
                return action
            else:
                # No onion, go back to start
                self.cooking_phase = "go_to_onion"
        
        # Fallback
        return "STAY"

class DishWaiterAgent(MacroActionAgent):
    """Agent that picks up dishes and waits for soup to be ready."""
    
    def __init__(self, mdp: OvercookedGridworld, agent_id: int):
        super().__init__(mdp, agent_id)
        self.waiting_phase = "go_to_dish"
        
    def get_action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_id]
        
        # Check if we have a dish
        if player.held_object is None:
            # Need to get dish
            action = self._execute_macro("go to dish dispenser", state)
            status = self.policy.get_macro_status()
            
            # Check if macro completed
            if status and status['is_completed']:
                if self.last_macro_status and not self.last_macro_status['is_completed']:
                    print(f"🔄 Phase transition: go_to_dish -> take_dish")
                else:
                    print(f"🔄 Immediate completion: go_to_dish -> take_dish")
                self.waiting_phase = "take_dish"
            return action
            
        elif player.held_object.name == "dish":
            # Have dish, check if any pot has cooked soup
            pots = self.mdp.get_pot_locations()
            soup_ready = False
            
            for pot_pos in pots:
                if pot_pos in state.objects:
                    pot_obj = state.objects[pot_pos]
                    if pot_obj.name == "pot" and len(pot_obj.state) >= 4:
                        onions_count, is_cooking, cook_time, is_ready = pot_obj.state
                        if is_ready and onions_count >= 3:  # Cooked soup ready
                            soup_ready = True
                            break
            
            if soup_ready:
                # Soup is ready, go get it
                action = self._execute_macro("go to pot", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Phase transition: go_to_pot -> take_soup")
                    else:
                        print(f"🔄 Immediate completion: go_to_pot -> take_soup")
                    self.waiting_phase = "take_soup"
                return action
            else:
                # No soup ready, wait
                return "STAY"
                
        elif "soup" in str(player.held_object.name):
            # Have soup, go serve it
            action = self._execute_macro("go to serving area", state)
            status = self.policy.get_macro_status()
            
            # Check if macro completed
            if status and status['is_completed']:
                if self.last_macro_status and not self.last_macro_status['is_completed']:
                    print(f"🔄 Phase transition: go_to_serve -> serve")
                else:
                    print(f"🔄 Immediate completion: go_to_serve -> serve")
                self.waiting_phase = "serve"
            return action
            
        # Fallback
        return "STAY"

class SoupServerAgent(MacroActionAgent):
    """Agent that focuses on serving soup: take soup -> go to serve -> serve."""
    
    def __init__(self, mdp: OvercookedGridworld, agent_id: int):
        super().__init__(mdp, agent_id)
        self.serving_phase = "get_soup"
        
    def get_action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_id]
        
        # Check if we have soup
        if player.held_object is None:
            # Need dish first
            action = self._execute_macro("go to dish dispenser", state)
            status = self.policy.get_macro_status()
            if status and status['is_completed']:
                self.serving_phase = "take_dish"
            return action
            
        elif player.held_object.name == "dish":
            # Have dish, check for soup
            pots = self.mdp.get_pot_locations()
            soup_ready = False
            
            for pot_pos in pots:
                if pot_pos in state.objects:
                    pot_obj = state.objects[pot_pos]
                    if pot_obj.name == "pot" and len(pot_obj.state) >= 4:
                        onions_count, is_cooking, cook_time, is_ready = pot_obj.state
                        if is_ready and onions_count >= 3:  # Cooked soup ready
                            soup_ready = True
                            break
            
            if soup_ready:
                # Go to pot to get soup
                action = self._execute_macro("go to pot", state)
                status = self.policy.get_macro_status()
                if status and status['is_completed']:
                    self.serving_phase = "take_soup"
                return action
            else:
                # No soup ready, wait
                return "STAY"
                
        elif "soup" in str(player.held_object.name):
            # Have soup, go serve it
            action = self._execute_macro("go to serving area", state)
            status = self.policy.get_macro_status()
            if status and status['is_completed']:
                self.serving_phase = "serve"
            return action
            
        # Fallback
        return "STAY"

class CoordinatedCookingAgent(MacroActionAgent):
    """Agent that coordinates cooking: alternates between cooking and serving based on pot state."""
    
    def __init__(self, mdp: OvercookedGridworld, agent_id: int):
        super().__init__(mdp, agent_id)
        self.current_task = "cook"
        
    def get_action(self, state: OvercookedState) -> str:
        player = state.players[self.agent_id]
        
        # Check pot states to decide what to do
        pots = self.mdp.get_pot_locations()
        pots_with_onions = 0
        pots_cooking = 0
        pots_ready = 0
        
        for pot_pos in pots:
            if pot_pos in state.objects:
                obj = state.objects[pot_pos]
                if obj.name == "soup":  # Pots contain soup objects, not pot objects
                    soup_type, num_items, cook_time = obj.state
                    if num_items > 0:
                        pots_with_onions += 1
                    if cook_time > 0:  # Cooking
                        pots_cooking += 1
                    if num_items >= 3 and cook_time >= 20:  # Ready soup
                        pots_ready += 1
                    # Check if pot has exactly 3 onions and should start cooking
                    if num_items == 3 and cook_time == 0:
                        # Pot has 3 onions but isn't cooking yet - environment should start it
                        print(f"🔥 Pot at {pot_pos} has 3 onions - should start cooking automatically")
        
        # Decision logic
        if pots_ready > 0 and player.held_object is None:
            # Soup is ready, go get dish and serve
            print(f"🍲 Soup ready - switching to serve mode")
            self.current_task = "serve"
        elif pots_cooking > 0 and player.held_object is None:
            # Pots are cooking, wait or get dish
            print(f"⏳ Pots cooking - switching to wait mode")
            self.current_task = "wait"
        elif pots_with_onions < 2:  # Keep some pots cooking
            # Need more onions, go cook
            self.current_task = "cook"
        
        # Execute current task
        if self.current_task == "cook":
            return self._execute_cooking_task(state)
        elif self.current_task == "serve":
            return self._execute_serving_task(state)
        elif self.current_task == "wait":
            # Check if any soup is ready to serve
            pots = self.mdp.get_pot_locations()
            soup_ready = False
            for pot_pos in pots:
                if pot_pos in state.objects:
                    pot_obj = state.objects[pot_pos]
                    if pot_obj.name == "pot" and len(pot_obj.state) >= 4:
                        onions_count, is_cooking, cook_time, is_ready = pot_obj.state
                        if is_ready and onions_count >= 3:
                            soup_ready = True
                            break
            
            if soup_ready and player.held_object is None:
                # Soup is ready, switch to serving
                print(f"🔄 Soup ready, switching to serve mode")
                self.current_task = "serve"
                return "STAY"
            
            # Check if any pots need more onions
            pots_need_onions = False
            for pot_pos in pots:
                if pot_pos in state.objects:
                    pot_obj = state.objects[pot_pos]
                    if pot_obj.name == "pot" and len(pot_obj.state) >= 4:
                        onions_count, is_cooking, cook_time, is_ready = pot_obj.state
                        if onions_count < 3 and not is_cooking and not is_ready:
                            pots_need_onions = True
                            break
            
            if pots_need_onions and player.held_object is None:
                # Pots need onions, switch back to cooking
                print(f"🔄 Pots need onions, switching to cook mode")
                self.current_task = "cook"
                return "STAY"
            
            # Just wait
            return "STAY"
        
        return "STAY"
    
    def _execute_cooking_task(self, state: OvercookedState) -> str:
        """Execute cooking task: go to onion -> take onion -> go to pot -> put in pot."""
        player = state.players[self.agent_id]
        
        # Track current cooking phase
        if not hasattr(self, 'cooking_phase'):
            self.cooking_phase = "go_to_onion"
        
        # Check if we should switch to serving (soup is ready)
        pots = self.mdp.get_pot_locations()
        soup_ready = False
        for pot_pos in pots:
            if pot_pos in state.objects:
                obj = state.objects[pot_pos]
                if obj.name == "soup":  # Pots contain soup objects
                    soup_type, num_items, cook_time = obj.state
                    if num_items >= 3 and cook_time >= 20:  # Ready soup
                        soup_ready = True
                        break
        
        if soup_ready and player.held_object is None:
            # Soup is ready, switch to serving
            self.current_task = "serve"
            return "STAY"
        
        # Check if we need to start cooking a pot with 3 onions
        pot_needs_cooking = False
        for pot_pos in pots:
            if pot_pos in state.objects:
                obj = state.objects[pot_pos]
                if obj.name == "soup":  # Pots contain soup objects
                    soup_type, num_items, cook_time = obj.state
                    if num_items == 3 and cook_time == 0:  # 3 onions but not cooking
                        pot_needs_cooking = True
                        break
        
        if pot_needs_cooking and player.held_object is None:
            # Pot has 3 onions but isn't cooking - need to interact to start cooking
            print(f"🔥 Pot has 3 onions - need to interact to start cooking")
            self.cooking_phase = "start_cooking"
        
        if self.cooking_phase == "start_cooking":
            # Need to interact with pot to start cooking
            action = self._execute_macro("go to pot", state)
            status = self.policy.get_macro_status()
            
            # Check if macro completed
            if status and status['is_completed']:
                if self.last_macro_status and not self.last_macro_status['is_completed']:
                    print(f"🔄 Cooking phase transition: start_cooking -> interact_with_pot")
                # Now interact with pot to start cooking
                return "INTERACT"
            return action
        
        if self.cooking_phase == "go_to_onion":
            if player.held_object is None:
                action = self._execute_macro("go to onion", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Cooking phase transition: go_to_onion -> take_onion")
                    self.cooking_phase = "take_onion"
                return action
            else:
                # Already have onion, move to next phase
                self.cooking_phase = "go_to_pot"
                
        if self.cooking_phase == "take_onion":
            if player.held_object is None:
                action = self._execute_macro("take onion", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Cooking phase transition: take_onion -> go_to_pot")
                    self.cooking_phase = "go_to_pot"
                return action
            else:
                # Already have onion, move to next phase
                self.cooking_phase = "go_to_pot"
                
        if self.cooking_phase == "go_to_pot":
            if player.held_object and player.held_object.name == "onion":
                action = self._execute_macro("go to pot", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Cooking phase transition: go_to_pot -> put_in_pot")
                    self.cooking_phase = "put_in_pot"
                return action
            else:
                # No onion, restart cycle
                self.cooking_phase = "go_to_onion"
                
        if self.cooking_phase == "put_in_pot":
            if player.held_object and player.held_object.name == "onion":
                # Check if any pot can accept onions
                can_put_onion = False
                for pot_pos in pots:
                    if pot_pos in state.objects:
                        obj = state.objects[pot_pos]
                        if obj.name == "soup":  # Pots contain soup objects
                            soup_type, num_items, cook_time = obj.state
                            if num_items < 3 and cook_time == 0:  # Can add more onions
                                can_put_onion = True
                                break
                    else:
                        # Empty pot can accept onions
                        can_put_onion = True
                        break
                
                if can_put_onion:
                    action = self._execute_macro("put in pot", state)
                    status = self.policy.get_macro_status()
                    
                    # Check if macro completed
                    if status and status['is_completed']:
                        if self.last_macro_status and not self.last_macro_status['is_completed']:
                            print(f"🔄 Cooking phase transition: put_in_pot -> go_to_onion (cycle complete)")
                        self.cooking_phase = "go_to_onion"
                    return action
                else:
                    # All pots are full or cooking, switch to waiting
                    print(f"🔄 All pots full/cooking, switching to wait mode")
                    self.current_task = "wait"
                    # Drop the onion since we can't use it - just return STAY and let environment handle it
                    return "STAY"
            else:
                # No onion, restart cycle
                self.cooking_phase = "go_to_onion"
        
        # Fallback
        return "STAY"
    
    def _execute_serving_task(self, state: OvercookedState) -> str:
        """Execute serving task: go to dish -> take dish -> go to pot -> take soup -> go to serve -> serve."""
        player = state.players[self.agent_id]
        
        # Track current serving phase
        if not hasattr(self, 'serving_phase'):
            self.serving_phase = "go_to_dish"
        
        if self.serving_phase == "go_to_dish":
            if player.held_object is None:
                action = self._execute_macro("go to dish dispenser", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Serving phase transition: go_to_dish -> take_dish")
                    self.serving_phase = "take_dish"
                return action
            else:
                # Already have dish, move to next phase
                self.serving_phase = "go_to_pot"
                
        if self.serving_phase == "take_dish":
            if player.held_object is None:
                action = self._execute_macro("take dish", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Serving phase transition: take_dish -> go_to_pot")
                    self.serving_phase = "go_to_pot"
                return action
            else:
                # Already have dish, move to next phase
                self.serving_phase = "go_to_pot"
                
        if self.serving_phase == "go_to_pot":
            if player.held_object and player.held_object.name == "dish":
                action = self._execute_macro("go to pot", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Serving phase transition: go_to_pot -> take_soup")
                    self.serving_phase = "take_soup"
                return action
            else:
                # No dish, restart cycle
                self.serving_phase = "go_to_dish"
                
        if self.serving_phase == "take_soup":
            if player.held_object and player.held_object.name == "dish":
                action = self._execute_macro("take soup", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Serving phase transition: take_soup -> go_to_serve")
                    self.serving_phase = "go_to_serve"
                return action
            else:
                # No dish, restart cycle
                self.serving_phase = "go_to_dish"
                
        if self.serving_phase == "go_to_serve":
            if player.held_object and "soup" in str(player.held_object.name):
                action = self._execute_macro("go to serving area", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Serving phase transition: go_to_serve -> serve")
                    self.serving_phase = "serve"
                return action
            else:
                # No soup, restart cycle
                self.serving_phase = "go_to_dish"
                
        if self.serving_phase == "serve":
            if player.held_object and "soup" in str(player.held_object.name):
                action = self._execute_macro("serve", state)
                status = self.policy.get_macro_status()
                
                # Check if macro completed
                if status and status['is_completed']:
                    if self.last_macro_status and not self.last_macro_status['is_completed']:
                        print(f"🔄 Serving phase transition: serve -> go_to_dish (cycle complete)")
                    self.serving_phase = "go_to_dish"
                return action
            else:
                # No soup, restart cycle
                self.serving_phase = "go_to_dish"
        
        # Fallback
        return "STAY"

def create_macro_action_agents(mdp: OvercookedGridworld) -> List[MacroActionAgent]:
    """Create a list of macro action agents for training data generation."""
    agents = [
        OnionCookAgent(mdp, 0),
        DishWaiterAgent(mdp, 1)
    ]
    return agents

def create_coordinated_agents(mdp: OvercookedGridworld) -> List[MacroActionAgent]:
    """Create coordinated agents that work together."""
    agents = [
        CoordinatedCookingAgent(mdp, 0),
        SoupServerAgent(mdp, 1)
    ]
    return agents

if __name__ == "__main__":
    # Test the agents
    mdp = OvercookedGridworld.from_layout_name("random3")
    agents = create_macro_action_agents(mdp)
    
    print("Created macro action agents:")
    for i, agent in enumerate(agents):
        print(f"Agent {i}: {type(agent).__name__}")
