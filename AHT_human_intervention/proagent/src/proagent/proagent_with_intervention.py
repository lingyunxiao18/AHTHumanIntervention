#!/usr/bin/env python3
"""
ProAgent with Advanced LLM Intervention Support

This is a self-contained agent that uses AdvancedLLMInterpreter (CoT + Memory) as the planner
while incorporating the controller and verificator logic from the original ProAgent.
"""

import os
import sys
import re
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, Any, List

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import Overcooked primitives
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction

# Import advanced intervention system
intervention_path = os.path.join(PROJECT_ROOT, 'proagent', 'src', 'human_intervention')
if intervention_path not in sys.path:
    sys.path.append(intervention_path)
    from advanced_llm_intervention import (
        AgentMemory,
        AdvancedLLMInterpreter,
        HumanMessage,
        LLMClient,
        PROAGENT_ALLOWED_ML_ACTIONS,
    )

# OpenAI key file is at workspace root (one level up from PROJECT_ROOT)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
openai_key_file = os.path.join(WORKSPACE_ROOT, "openai_key.txt")

# Mapping for low-level override strings to action constants
LOW_LEVEL_OVERRIDE_TO_ACTION = {
    "move_north": Direction.NORTH,
    "move_south": Direction.SOUTH,
    "move_east": Direction.EAST,
    "move_west": Direction.WEST,
    "wait": Action.STAY,  
    "interact": Action.INTERACT,
    "stay": Action.STAY,
}

# ==================== EVENT DETECTION ====================

def detect_environment_events(state, prev_state, stuck_counter, agent_index=0):
    """
    Detect environment events that can trigger intervention pattern reuse.
    
    Args:
        state: Current game state
        prev_state: Previous game state (None if first step)
        stuck_counter: Current stuck counter (number of steps without position change)
        agent_index: Index of the ego agent
        
    Returns:
        events: List of detected event strings
        stuck_counter: Updated stuck counter
    """
    events = []
    
    if state is None or len(state.players) < 2:
        return events, stuck_counter
    
    ego = state.players[agent_index]
    teammate = state.players[1 - agent_index]
    ego_pos = tuple(ego.position)
    mate_pos = tuple(teammate.position)
    
    # --- Event 1: no progress / stuck ---
    if prev_state is not None and len(prev_state.players) > agent_index:
        prev_pos = tuple(prev_state.players[agent_index].position)
        if ego_pos == prev_pos:
            stuck_counter += 1
        else:
            stuck_counter = 0
        
        if stuck_counter >= 3:  # threshold adjustable
            events.append("no_progress")
    else:
        stuck_counter = 0
    
    # --- Event 2: teammate blocking path (adjacent + wanting same tile) ---
    manhattan_dist = abs(ego_pos[0] - mate_pos[0]) + abs(ego_pos[1] - mate_pos[1])
    if manhattan_dist == 1:
        events.append("teammate_blocking")
    
    # --- Event 3: deadlock heuristic (stuck + teammate present) ---
    if "no_progress" in events and "teammate_blocking" in events:
        events.append("deadlock")
    
    # Debug output
    if prev_state is not None:
        prev_pos = tuple(prev_state.players[agent_index].position) if len(prev_state.players) > agent_index else None
        print(f"[EVENT_DEBUG] ego_pos={ego_pos}, prev_pos={prev_pos}, stuck_counter={stuck_counter}, manhattan_dist={manhattan_dist}, events={events}")
    else:
        print(f"[EVENT_DEBUG] prev_state=None, ego_pos={ego_pos}, stuck_counter={stuck_counter}, manhattan_dist={manhattan_dist}, events={events}")
    
    return events, stuck_counter


class ProAgentWithIntervention:
    """
    Self-contained ProAgent that uses AdvancedLLMInterpreter as the planner.
    
        Features:
        - AdvancedLLMInterpreter (CoT + Memory) for high-level planning
        - Controller and verificator logic for medium-level action execution
        - Supports human interventions via memory injection
        - Complete independence from base ProAgent class
    """
    
    def __init__(self, mlam, layout, model='gpt-5-mini', 
                 agent_index=None, history_horizon=8, 
                 enable_cot=True, enable_memory=True, **kwargs):
        """
        Initialize ProAgent with AdvancedLLMInterpreter planner.
        
        Args:
            enable_cot: If True, enable Chain-of-Thought reasoning (default: True)
            enable_memory: If True, enable memory module (default: True)
        """
        self.model = model
        self.mlam = mlam
        self.layout = layout
        self.mdp = self.mlam.mdp
        self.agent_index = agent_index
        self.enable_cot = enable_cot
        self.enable_memory = enable_memory
        
        # OpenAI API key handling
        self.openai_api_keys = []
        self.load_openai_keys()
        self.key_rotation = True
        
        # Agent state
        self.prev_state = None
        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.pot_id_to_pos = []
        self.current_timestep = 0
        
        # Low-level override tracking
        self.low_level_override = None
        self.low_level_override_duration = 0
        
        # Layout prompt generation
        self.layout_prompt = self.generate_layout_prompt()
        
        # Initialize interpreter-based planner
        self.history_horizon = history_horizon
        
        # Initialize LLM client and interpreter
        from openai import OpenAI
        # Initialize OpenAI client with API key loaded via openai_key.txt/env
        self.llm_client = LLMClient(openai_client=OpenAI(api_key=self.openai_api_key()), model=model)
        
        # Always use AdvancedLLMInterpreter (it handles CoT and memory flags internally)
        self.memory = AgentMemory(mdp=self.mdp)
        self.interpreter = AdvancedLLMInterpreter(
            self.llm_client, 
            self.memory, 
            history_horizon=history_horizon,
            enable_cot=enable_cot,
            enable_memory=enable_memory
        )
        cot_status = "ENABLED" if enable_cot else "DISABLED"
        mem_status = "ENABLED" if enable_memory else "DISABLED"
        print(f"🤖 ProAgentWithIntervention initialized (CoT: {cot_status}, Memory: {mem_status})")
        
        # Human intervention tracking
        self._human_inbox: List[str] = []
        self._recent_history: List[Dict[str, Any]] = []
        self._intervention_history: List[str] = []
        self._last_human_intervention_tick = None
        self._last_human_intervention_action = None
        self._last_human_intervention_text = None

        # Debug: keep last interpreter outputs
        self.last_plan: Optional[Dict[str, Any]] = None
        self.last_plan_category: Optional[str] = None
        self.last_plan_rationale: Optional[str] = None
        self.last_intervention_reason: Optional[str] = None
        self.last_chain_of_thought: Optional[str] = None
        
        # Event-based trigger system for proactive intervention reuse
        self._prev_state = None
        self._stuck_counter = 0
    
    # ==================== OpenAI Key Management ====================
    
    def load_openai_keys(self):
        # 1) Environment variable takes precedence
        api_key_env = os.environ.get("OPENAI_API_KEY")
        if api_key_env:
            self.openai_api_keys = [api_key_env.strip()]
            return
        # 2) Try workspace root openai_key.txt
        if os.path.exists(openai_key_file):
            with open(openai_key_file, "r") as f:
                context = f.read()
            self.openai_api_keys = [k for k in context.split('\n') if k.strip()]
            if self.openai_api_keys:
                return
        raise FileNotFoundError("OPENAI_API_KEY not set and openai_key.txt not found in workspace root")

    def openai_api_key(self):
        if self.key_rotation:
            self.update_openai_key()
        return self.openai_api_keys[0]

    def update_openai_key(self):
        self.openai_api_keys.append(self.openai_api_keys.pop(0))
    
    # ==================== Layout and State Management ====================
    
    def generate_layout_prompt(self):
        layout_prompt_dict = {
            "onion_dispenser": " <Onion Dispenser {id}>",
            "dish_dispenser": " <Dish Dispenser {id}>",
            "serving": " <Serving Loc {id}>",
            "pot": " <Pot {id}>",
        }
        layout_prompt = "Here's the layout of the kitchen:"
        for obj_type, prompt_template in layout_prompt_dict.items():
            locations = getattr(self.mdp, f"get_{obj_type}_locations")()
            for obj_id, obj_pos in enumerate(locations):
                layout_prompt += prompt_template.format(id=obj_id) + ","
                if obj_type == "pot":
                    self.pot_id_to_pos.append(obj_pos)
        layout_prompt = layout_prompt[:-1] + ".\n"
        return layout_prompt
    
    def generate_state_prompt(self, state):
        ego = state.players[self.agent_index]
        teammate = state.players[1 - self.agent_index]

        time_prompt = f"Scene {state.timestep}: "
        ego_object = ego.held_object.name if ego.held_object else "nothing"
        teammate_object = teammate.held_object.name if teammate.held_object else "nothing"
        ego_state_prompt = f"<Player {self.agent_index}> holds "
        if ego_object == 'soup':
            ego_state_prompt += f"a dish with {ego_object} and needs to deliver soup.  "
        elif ego_object == 'nothing':
            ego_state_prompt += f"{ego_object}. "
        else:
            ego_state_prompt += f"one {ego_object}. "
        
        teammate_state_prompt = f"<Player {1-self.agent_index}> holds "
        if teammate_object == 'soup':
            teammate_state_prompt += f"a dish with {teammate_object}. "
        elif teammate_object == "nothing":
            teammate_state_prompt += f"{teammate_object}. "
        else:
            teammate_state_prompt += f"one {teammate_object}. "

        kitchen_state_prompt = "Kitchen states: "
        prompt_dict = {
            "empty": "<Pot {id}> is empty; ",
            "cooking": "<Pot {id}> starts cooking, the soup will be ready after {t} timesteps; ",
            "ready": "<Pot {id}> has already cooked the soup; ",
            "1_items": "<Pot {id}> has 1 onion; ",
            "2_items": "<Pot {id}> has 2 onions; ",
            "3_items": "<Pot {id}> has 3 onions and is full; "
        }

        pot_states_dict = self.mdp.get_pot_states(state)   

        # Version 0.0.1: nested structure by soup type
        for key in pot_states_dict.keys():
            if key == "empty":
                for pos in pot_states_dict[key]: 
                    pot_id = self.pot_id_to_pos.index(pos)
                    kitchen_state_prompt += prompt_dict[key].format(id=pot_id)     
            else: # key = 'onion' or 'tomota'
                for soup_key in pot_states_dict[key].keys():
                    # soup_key: ready, cooking, partially_full
                    for pos in pot_states_dict[key][soup_key]:
                        pot_id = self.pot_id_to_pos.index(pos)
                        soup_object = state.get_object(pos)
                        soup_type, num_items, cook_time = soup_object.state
                        if soup_key == "cooking":
                            kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id, t=self.mdp.soup_cooking_time-cook_time)
                        elif soup_key == "partially_full":
                            pass
                        else:
                            kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id)

        return (self.layout_prompt + time_prompt + ego_state_prompt +
                teammate_state_prompt + kitchen_state_prompt)
    
    # ==================== PLANNER: AdvancedLLMInterpreter Integration ====================
    
    def generate_ml_action(self, state, events=None):
        """
        Generates medium-level actions using CoT reasoning + memory (or baseline without CoT/memory).
        
        Args:
            state: Current game state
            events: Optional list of detected events (if None, will detect them here)
        """
        # Use text-based state prompt (same as original ProAgent)
        state_prompt = self.generate_state_prompt(state)
        
        # Include events in prompt if provided (events are detected in action() method)
        if events is None:
            # Fallback: detect events here if not provided
            events, _ = detect_environment_events(
                state, self._prev_state, self._stuck_counter, self.agent_index
            )
        if events:
            state_prompt += f"\nEVENTS: {events}"
            print(f"🔍 Including events in prompt: {events}")
        
        # Get recent history with teammate actions
        recent_history = self._recent_history[-self.history_horizon:]
        
        # Get human intervention if available (interventions work with any config)
        human_text = self._human_inbox.pop(0) if self._human_inbox else ""
        
        # Create human message
        hm = HumanMessage(t=getattr(state, "timestep", 0), text=human_text or "")
        if hm.text.strip():
            print(f"[HUMAN] Consuming intervention at t={hm.t}: '{hm.text}'")
        
        try:
            # Get plan from interpreter (pass state, agent_index, and events for intervention recording)
            plan = self.interpreter.propose_plan(state_prompt, hm, recent_history, state, self.agent_index, events=events)
            
            # Store debug info
            try:
                self.last_plan = {
                    "steps": plan.steps,
                }
                self.last_plan_category = getattr(plan, 'category', None)
                self.last_intervention_reason = getattr(plan, 'intervention_reason', None)
                self.last_chain_of_thought = getattr(plan, 'chain_of_thought', None) if self.enable_cot else ""
                if self.enable_cot or self.enable_memory:
                    print(f"[INTP] category={self.last_plan_category} steps={self.last_plan.get('steps')} intervention_reason={self.last_intervention_reason}")
                    if self.last_chain_of_thought:
                        print(f"[INTP] CoT: {self.last_chain_of_thought}")
                else:
                    print(f"[INTP] steps={self.last_plan.get('steps')}")
            except Exception:
                pass
            
            # Record a memory event when a new plan is computed (skip if memory disabled)
            if self.enable_memory:
                try:
                    t = int(getattr(state, 'timestep', 0))
                    self.memory.write_events([
                        {"t": t,
                         "type": "plan",
                         "steps": list(plan.steps),
                         "category": str(getattr(plan, 'category', '')),
                         "chain_of_thought": str(getattr(plan, 'chain_of_thought', ''))}
                    ])
                except Exception:
                    pass
            
            # Check for low-level override (applies for 1 step, then continues with medium-level plan)
            if hasattr(plan, 'low_level_override') and plan.low_level_override:
                print(f"🎯 Low-level override detected: {plan.low_level_override}")
                self.low_level_override = plan.low_level_override
                self.low_level_override_duration = 1
                # Track human intervention for card learning
                if hm.text.strip():
                    self._last_human_intervention_tick = getattr(state, 'timestep', 0)
                    self._last_human_intervention_action = plan.low_level_override
                    self._last_human_intervention_text = hm.text.strip()
                
            # Process medium-level steps (works with or without low-level override)
            if not plan.steps:
                raise RuntimeError("Interpreter returned empty plan")
            ml_action = plan.steps[0]
            
            print(f"🧠 Interpreter generated plan with {len(plan.steps)} steps")
            print(f"📋 Current ml_action: {ml_action}")
            
            return ml_action
                
        except Exception as e:
            print(f"❌ Interpreter failed: {e}")
            raise
    
    # ==================== VERIFIER ====================
    
    def check_current_ml_action_done(self, state):
        """
        checks if the current ml action is done
        :return: True or False
        """
        player = state.players[self.agent_index]
        if "pickup" in self.current_ml_action:
            pattern = r"pickup(?:[(]|_)(\w+)(?:[)]|)" # fit both pickup(onion) and pickup_onion
            obj_str = re.search(pattern, self.current_ml_action).group(1)
            return player.has_object() and player.get_object().name == obj_str
        
        elif "fill" in self.current_ml_action:
            return player.held_object.name == 'soup'
        
        elif "put" in self.current_ml_action or "place" in self.current_ml_action:
            return not player.has_object()
        
        elif "deliver" in self.current_ml_action:
            return not player.has_object()
        
        elif "wait" in self.current_ml_action:
            return self.time_to_wait == 0

    def validate_current_ml_action(self, state):
        """
        make sure the current_ml_action exists and is valid
        """
        if self.current_ml_action is None:
            return False

        pot_states_dict = self.mdp.get_pot_states(state)
        player = state.players[self.agent_index]
        # Version 0.0.1: nested structure by soup type
        soup_cooking = len(pot_states_dict['onion']['cooking']) > 0
        soup_ready = len(pot_states_dict['onion']['ready']) > 0
        pot_not_full = pot_states_dict["empty"] + pot_states_dict["onion"]['partially_full']
        cookable_pots = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)]

        has_onion = False
        has_dish = False
        has_soup = False
        has_object = player.has_object()
        if has_object:
            has_onion = player.get_object().name == 'onion'
            has_dish = player.get_object().name == 'dish'
            has_soup = player.get_object().name == 'soup'
        empty_counter = self.mdp.get_empty_counter_locations(state)

        if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:   
            flag2 = len(self.find_motion_goals(state)) == 0 
            if flag2: 
                return False 
            return not has_object and len(self.mdp.get_onion_dispenser_locations()) > 0
        elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
            flag2 = len(self.find_motion_goals(state)) == 0 
            if flag2: 
                return False 
            return not has_object and len(self.mdp.get_dish_dispenser_locations()) > 0
        elif "put_onion_in_pot" in self.current_ml_action:
            return has_onion and len(pot_not_full) > 0
        elif "place_obj_on_counter" in self.current_ml_action:
            return has_object and len(empty_counter) > 0
        elif "fill_dish_with_soup" in self.current_ml_action:
            return has_dish and (soup_ready or soup_cooking)
        elif "deliver_soup" in self.current_ml_action:
            return has_soup
        elif "wait" in self.current_ml_action:
            return 0 < int(self.current_ml_action.split('(')[1][:-1]) <= 20
    
    # ==================== CONTROLLER ====================
    
    def find_shared_counters(self, state, mlam):  
        empty_counters = self.mdp.get_empty_counter_locations(state)
        if empty_counters:
            available_plans = mlam._get_ml_actions_for_positions(empty_counters)
            return available_plans
        return []          

    def find_motion_goals(self, state):
        """
        Generates the motion goals for the given medium level action.
        :param state:
        :return:
        """
        am = self.mlam
        motion_goals = []
        player = state.players[self.agent_index]
        pot_states_dict = self.mdp.get_pot_states(state)
        counter_objects = self.mdp.get_counter_objects_dict(
            state, list(self.mdp.terrain_pos_dict["X"])
        )
        try:
            print(f"[DBG] Player{self.agent_index} pos_and_or={player.pos_and_or}")
            print(f"[DBG] Onion dispensers={self.mdp.get_onion_dispenser_locations()}, Dish dispensers={self.mdp.get_dish_dispenser_locations()}, Pots={self.mdp.get_pot_locations()}")
        except Exception:
            pass
        if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:
            # Use shared env's helper name
            raw_goals = am.pickup_onion_actions(state, counter_objects)
            print(f"[DBG] raw_goals(pickup_onion)={raw_goals}")
            motion_goals = raw_goals

        elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
            # Use shared env's helper name
            motion_goals = am.pickup_dish_actions(state, counter_objects)
        elif "put_onion_in_pot" in self.current_ml_action:
            motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
        elif "place_obj_on_counter" in self.current_ml_action:  
            motion_goals = self.find_shared_counters(state, self.mlam)     
            if len(motion_goals) == 0: 
                motion_goals = am.place_obj_on_counter_actions(state)

        elif "start_cooking" in self.current_ml_action:
            # Version 0.0.1: fixed recipe size
            soups_ready_to_cook_key = "{}_items".format(self.mdp.num_items_for_soup)
            soups_ready_to_cook = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)]
            only_pot_states_ready_to_cook = defaultdict(list)
            only_pot_states_ready_to_cook[soups_ready_to_cook_key] = soups_ready_to_cook
            motion_goals = am.start_cooking_actions(only_pot_states_ready_to_cook)
        elif "fill_dish_with_soup" in self.current_ml_action:
            motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
        elif "deliver_soup" in self.current_ml_action:
            motion_goals = am.deliver_soup_actions()
        elif "wait" in self.current_ml_action:
            motion_goals = am.wait_actions(player)
        else:
            raise ValueError("Invalid action: {}".format(self.current_ml_action))

        raw_len = len(motion_goals)
        motion_goals = [
            mg
            for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]
        print(f"[DBG] filtered_goals({self.current_ml_action}) {len(motion_goals)}/{raw_len}: {motion_goals}")

        return motion_goals

    def choose_motion_goal(self, start_pos_and_or, motion_goals, state = None):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal and returns the plan 
        and the corresponding first action on that plan.
        Uses collision-aware pathfinding to avoid teammate collisions.
        """
        (
            chosen_goal,
            chosen_goal_action,
        ) = self.get_lowest_cost_action_and_goal_new(
            start_pos_and_or, motion_goals, state
        )
        return chosen_goal, chosen_goal_action
    
    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
                start_pos_and_or, goal
            )
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def get_lowest_cost_action_and_goal_new(self, start_pos_and_or, motion_goals, state): 
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """   
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:   
            action_plan, plan_cost = self.real_time_planner(
                start_pos_and_or, goal, state
            )     
            try:
                print(f"[DBG] plan goal={goal} cost={plan_cost:.1f} first_action={action_plan}")
            except Exception:
                pass
            if plan_cost < min_cost:
                best_action = action_plan
                min_cost = plan_cost
                best_goal = goal     
        if best_action is None: 
            if np.random.rand() < 0.5:  
                return None, Action.STAY
            else: 
                return self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
        
        # Debug logging for final goal selection
        try:
            print(f"[COLLISION_AVOID] SELECTED GOAL: {best_goal} with cost {min_cost:.1f}")
        except Exception:
            pass
            
        return best_goal, best_action

    def real_time_planner(self, start_pos_and_or, goal, state):
        # Use shared motion planner to compute plan and cost
        try:
            action_plan, plan_nodes, plan_cost = self.mlam.motion_planner.get_plan(start_pos_and_or, goal)
            
            if not action_plan:
                return None, np.Inf
                
            # Check for teammate collisions in the path
            collision_penalty = self._calculate_collision_penalty(action_plan, start_pos_and_or, state)
            
            # Apply collision penalty to the cost
            adjusted_cost = plan_cost + collision_penalty
            
            # Debug logging for collision avoidance
            try:
                if collision_penalty > 0:
                    print(f"[COLLISION_AVOID] Goal {goal}: original_cost={plan_cost:.1f}, collision_penalty={collision_penalty}, adjusted_cost={adjusted_cost:.1f}")
                if collision_penalty == np.Inf:
                    print(f"[COLLISION_AVOID] Goal {goal}: PATH BLOCKED - rejecting goal")
            except Exception:
                pass
            
            # Return infinite cost if path is completely blocked
            if collision_penalty == np.Inf:
                return None, np.Inf
                
            first_action = action_plan[0]
            return first_action, adjusted_cost
            
        except Exception:
            # Fallback: no plan
            return None, np.Inf
    
    def _calculate_collision_penalty(self, action_plan, start_pos_and_or, state):
        """
        Calculate collision penalty for a given action plan.
        Returns penalty cost (higher for more collisions) or np.Inf if completely blocked.
        """
        # Get teammate positions to avoid
        teammate_positions = set()
        for i, player in enumerate(state.players):
            if i != self.agent_index:
                teammate_positions.add(player.position)
        
        if not teammate_positions:
            return 0  # No teammates to avoid
        
        # Simulate the path and check for collisions
        current_pos_and_or = start_pos_and_or
        collision_count = 0
        total_steps = len(action_plan)
        
        for action in action_plan:
            # Calculate next position based on action
            next_pos_and_or = self._apply_action_to_pos_and_or(current_pos_and_or, action)
            
            # Check if next position collides with teammate
            if next_pos_and_or[0] in teammate_positions:
                collision_count += 1
                
            current_pos_and_or = next_pos_and_or
        
        # Calculate penalty based on collision frequency
        if collision_count == 0:
            return 0  # No collisions
        elif collision_count >= total_steps * 0.5:  # More than 50% of path blocked
            return np.Inf  # Path is essentially unusable
        else:
            # Penalty increases with collision frequency
            return collision_count * 10  # Each collision adds 10 cost units
    
    def _apply_action_to_pos_and_or(self, pos_and_or, action):
        """
        Apply an action to a position and orientation, returning new pos_and_or.
        """
        pos, orientation = pos_and_or
        
        if action == Direction.NORTH:
            new_pos = (pos[0], pos[1] - 1)
            new_orientation = (0, -1)
        elif action == Direction.SOUTH:
            new_pos = (pos[0], pos[1] + 1)
            new_orientation = (0, 1)
        elif action == Direction.EAST:
            new_pos = (pos[0] + 1, pos[1])
            new_orientation = (1, 0)
        elif action == Direction.WEST:
            new_pos = (pos[0] - 1, pos[1])
            new_orientation = (-1, 0)
        elif action == Action.INTERACT:
            new_pos = pos  # Position doesn't change
            new_orientation = orientation
        else:  # Action.STAY or unknown
            new_pos = pos
            new_orientation = orientation
            
        return (new_pos, new_orientation)
    
    # ==================== MAIN ACTION METHOD ====================
    
    def action(self, state):
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        # only use to record the teammate ml_action, 
        # if teammate finish ml_action in t-1, it will record in s_t, 
        # otherwise, s_t will just record None,
        # and we here check this information and store it into proagent
        self.current_timestep = getattr(state, 'timestep', 0)
        
        # Detect environment events EVERY STEP (not just when generating new ML action)
        # This allows us to detect deadlocks even when ML action is still in progress
        print(f"[EVENT_CHECK] Before detection: _prev_state={'exists' if self._prev_state is not None else 'None'}, _stuck_counter={self._stuck_counter}")
        events, self._stuck_counter = detect_environment_events(
            state, self._prev_state, self._stuck_counter, self.agent_index
        )
        print(f"[EVENT_CHECK] After detection: events={events}, _stuck_counter={self._stuck_counter}")
        
        # If events detected and we have a matching pattern, trigger proactive intervention
        if events and self.enable_memory:
            # Check for matching intervention patterns
            patterns = self.memory.semantic.get("intervention_patterns", [])
            matching = []
            for p in patterns:
                p_events = p.get("events_triggered", [])
                if p_events and all(e in events for e in p_events):
                    matching.append(p)
            if matching:
                best_pattern = matching[-1]  # Most recent
                if best_pattern.get("low_level_override") and not self.low_level_override:
                    # Auto-apply the pattern's override
                    self.low_level_override = best_pattern["low_level_override"]
                    self.low_level_override_duration = 1  # Apply for 1 step
                    print(f"✅ Auto-triggered pattern: {best_pattern.get('events_triggered')} → {self.low_level_override}")

        # if current ml action does not exist, generate a new one (respect override)
        if self.current_ml_action is None:
            self.current_ml_action = self.generate_ml_action(state, events=events)

        # if the current ml action is in process, Player{self.agent_index} done, else generate a new one
        if self.current_ml_action_steps > 0:
            current_ml_action_done = self.check_current_ml_action_done(state)
            if current_ml_action_done:
                # generate a new ml action
                self.current_ml_action = self.generate_ml_action(state, events=events)

        count = 0
        while not self.validate_current_ml_action(state):
            self.current_ml_action = self.generate_ml_action(state, events=events)
            
            count += 1
            if count > 3:
                self.current_ml_action = "wait(1)"
                self.time_to_wait = 1

        # Check for low-level override first (before any other logic)
        if self.low_level_override and self.low_level_override_duration > 0:
            print(f"🎯 Applying low-level override: {self.low_level_override}")
            chosen_action = LOW_LEVEL_OVERRIDE_TO_ACTION.get(self.low_level_override, Action.STAY)
            self.low_level_override_duration -= 1
            if self.low_level_override_duration <= 0:
                self.low_level_override = None
                print(f"🔄 Low-level override completed")
        elif "wait" in self.current_ml_action:
            self.current_ml_action_steps += 1
            self.time_to_wait -= 1
            player = state.players[self.agent_index]
            # Compute valid primitive actions in a shared-env compatible way
            try:
                lis_actions = self.mdp.get_valid_actions(player)
            except Exception:
                lis_actions = [Action.STAY]
                for d in Direction.ALL_DIRECTIONS:
                    adj_pos = Action.move_in_direction(player.position, d)
                    try:
                        if self.mdp.get_terrain_type_at_pos(adj_pos) == ' ':
                            lis_actions.append(d)
                    except Exception:
                        pass
            chosen_action = lis_actions[np.random.randint(0, len(lis_actions))]
        else:
            possible_motion_goals = self.find_motion_goals(state)    
            # Debug: log dispenser locations and goal count when picking up onion
            try:
                if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:
                    print(f"[DEBUG] Onion dispensers: {self.mdp.get_onion_dispenser_locations()}")
                print(f"[DEBUG] possible_motion_goals({self.current_ml_action}): {len(possible_motion_goals)}")
            except Exception:
                pass
            current_motion_goal, chosen_action = self.choose_motion_goal(
                start_pos_and_or, 
                possible_motion_goals, 
                state
            )
            try:
                print(f"[DEBUG] chosen_action={chosen_action}, current_motion_goal={current_motion_goal}")
            except Exception:
                pass

        # Version 0.0.1: return single action (not tuple)
        self.prev_state = state
        result = chosen_action
        if chosen_action is None:
            self.current_ml_action = "wait(1)"
            self.time_to_wait = 1
            chosen_action = Action.STAY
        self.current_ml_action_steps += 1

        print(f"[DBG] return chosen_action={chosen_action} type={type(chosen_action)}")
        
        # Track history for interpreter
        # DISABLED: _append_history_tick and _infer_teammate_action disabled for testing
        # self._append_history_tick(state, chosen_action)
        
        # Update previous state for event detection (store reference for next step)
        self._prev_state = state
        
        # Version 0.0.1: return single action (not tuple)
        return chosen_action
    
    def _append_history_tick(self, state, action):
        """Append action to recent history for interpreter, including teammate actions."""
        try:
            players = getattr(state, "players", [])
            ego_pos = [int(players[0].position[0]), int(players[0].position[1])] if players else None
            mate_pos = [int(players[1].position[0]), int(players[1].position[1])] if len(players) > 1 else None
            
            # Get teammate's ML action if available
            mate_ml_action = None
            if hasattr(state, 'ml_actions') and len(state.ml_actions) > 1:
                mate_ml_action = state.ml_actions[1 - self.agent_index]
            
            # Get teammate's low-level action if available
            mate_low_level_action = None
            if hasattr(state, 'actions') and len(state.actions) > 1:
                mate_low_level_action = str(state.actions[1 - self.agent_index])
            elif hasattr(state, 'player_actions') and len(state.player_actions) > 1:
                mate_low_level_action = str(state.player_actions[1 - self.agent_index])
            
            # Try to infer teammate's action from state changes
            mate_action_inferred = self._infer_teammate_action(state)
            
            self._recent_history.append({
                "t": int(getattr(state, "timestep", 0)),
                "ego_action": str(action),
                "ego_pos": ego_pos,
                "ego_ml_action": getattr(self, "current_ml_action", None),
                "mate_pos": mate_pos,
                "mate_ml_action": mate_ml_action,
                "mate_low_level_action": mate_low_level_action,
                "mate_action_inferred": mate_action_inferred,
                "mate_holding": str(getattr(getattr(players[1], "held_object", None), "name", "nothing")) if len(players) > 1 else "unknown"
            })
            
            # Keep history bounded
            if len(self._recent_history) > (self.history_horizon * 2):
                self._recent_history = self._recent_history[-(self.history_horizon * 2):]
                
        except Exception as e:
            print(f"⚠️ Error appending history: {e}")
    
    def _infer_teammate_action(self, current_state):
        """Infer teammate's likely action based on state changes."""
        try:
            if not hasattr(self, '_prev_mate_state') or self._prev_mate_state is None:
                self._prev_mate_state = {
                    'pos': None,
                    'holding': None
                }
                return "unknown"
            
            players = getattr(current_state, "players", [])
            if len(players) <= 1:
                return "unknown"
                
            mate = players[1 - self.agent_index]
            current_pos = [int(mate.position[0]), int(mate.position[1])]
            current_holding = str(getattr(getattr(mate, "held_object", None), "name", "nothing"))
            
            prev_pos = self._prev_mate_state.get('pos')
            prev_holding = self._prev_mate_state.get('holding')
            
            # Update previous state
            self._prev_mate_state['pos'] = current_pos
            self._prev_mate_state['holding'] = current_holding
            
            # Infer action based on changes
            if prev_pos and current_pos != prev_pos:
                return "moved"
            elif prev_holding != current_holding:
                if prev_holding == "nothing" and current_holding != "nothing":
                    return f"picked_up_{current_holding}"
                elif prev_holding != "nothing" and current_holding == "nothing":
                    return f"put_down_{prev_holding}"
                else:
                    return "interacted"
            else:
                return "stayed"
                
        except Exception as e:
            return "unknown"
    
    # ==================== PUBLIC API ====================
    
    def reset(self):
        """Reset agent state, including interpreter memory."""
        self.prev_state = None
        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.current_timestep = 0
        self.low_level_override = None
        self.low_level_override_duration = 0
        if self.enable_memory:
            self.memory = AgentMemory(mdp=self.mdp)
        self._human_inbox.clear()
        self._recent_history.clear()
        self._last_human_intervention_tick = None
        self._last_human_intervention_action = None
        self._last_human_intervention_text = None
        # Reset event detection state
        self._prev_state = None
        self._stuck_counter = 0
    def set_agent_index(self, agent_index):
        """Set the agent index."""
        self.agent_index = agent_index
    
    def apply_human_intervention(self, text: str):
        """Apply human intervention text to the agent's inbox and immediately override current action."""
        if text and text.strip():
            self._human_inbox.append(text.strip())
            self._intervention_history.append(text.strip())
            print(f"🎯 Human intervention received: '{text.strip()}'")
            # Note: State information will be captured when the intervention is processed in generate_ml_action
            
            # Immediately override current ML action to process intervention
            self.current_ml_action = None
            self.current_ml_action_steps = 0
            print(f"🔄 Overriding current ML action to process intervention")

    # Compatibility with demo harness API
    def process_human_intervention(self, text: str) -> bool:
        try:
            self.apply_human_intervention(text)
            return True
        except Exception:
            return False

    def get_intervention_history(self) -> List[str]:
        return list(self._intervention_history)
    
    def get_intervention_stats(self) -> Dict[str, Any]:
        """Get statistics about interpreter usage and interventions."""
        processed = len([h for h in self._intervention_history])
        return {
            "total_interventions": processed,
            "history_length": len(self._recent_history),
            # Count episodic entries to reflect recent memory updates visible in UI
            "memory_entries": len(getattr(self.memory, 'episodic', [])),
            "current_ml_action": self.current_ml_action,
            "current_ml_action_steps": self.current_ml_action_steps
        }
