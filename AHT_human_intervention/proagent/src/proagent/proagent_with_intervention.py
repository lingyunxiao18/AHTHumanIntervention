#!/usr/bin/env python3
"""
ProAgent with Advanced LLM Intervention Support

This is a self-contained agent that uses AdvancedLLMInterpreter (CoT + Memory) as the planner
while incorporating the exact controller and verificator logic from the original ProAgent.
"""

import os
import sys
import copy
import itertools
import json
import re
import numpy as np
import pkg_resources
from collections import defaultdict
from typing import Optional, Dict, Any, List

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import Overcooked primitives
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction

# Try to import helpers; fallback to stubs if not available
try:
    from shared.envs.envs.overcooked.overcooked_ai_py.planning.search import get_intersect_counter, query_counter_states
except Exception:
    def get_intersect_counter(*args, **kwargs):
        return []
    def query_counter_states(*args, **kwargs):
        return {}

# Import advanced intervention system
try:
    from ..human_intervention.advanced_llm_intervention import (
        AgentMemory,
        AdvancedLLMInterpreter,
        HumanMessage,
        LLMClient,
        PROAGENT_ALLOWED_ML_ACTIONS,
    )
except ImportError:
    # Fallback for different import contexts
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

cwd = os.getcwd()
openai_key_file = os.path.join(cwd, "openai_key.txt")

NAME_TO_ACTION = {
    "NORTH": Direction.NORTH,
    "SOUTH": Direction.SOUTH,
    "EAST": Direction.EAST,
    "WEST": Direction.WEST,
    "INTERACT": Action.INTERACT,
    "STAY": Action.STAY
}

# Mapping for low-level override strings to action constants
LOW_LEVEL_OVERRIDE_TO_ACTION = {
    "move_north": Direction.NORTH,
    "move_south": Direction.SOUTH,
    "move_east": Direction.EAST,
    "move_west": Direction.WEST,
    "wait": Action.STAY,  # Wait is implemented as STAY
    "interact": Action.INTERACT,
    "stay": Action.STAY,
}


def make_card_from_success(ctx: Dict[str, Any], chosen_action: str, human_text: str = "") -> Dict[str, Any]:
    """
    Create an intervention card from a successful human fix.
    Uses the actual human words to make keyword matching work.
    """
    # Extract keywords from human text for better matching
    human_words = human_text.lower() if human_text else ""
    
    # Create a title from human text, or use a generic one
    if human_words:
        # Extract key phrases: "stuck", "unstuck", "blocked", "same tile", etc.
        if "stuck" in human_words or "unstuck" in human_words:
            title = f"Getting unstuck: {human_text[:40]}"
        elif "teammate" in human_words and ("same" in human_words or "tile" in human_words):
            title = f"Teammate collision: {human_text[:40]}"
        elif "move" in human_words or "step" in human_words:
            title = f"Movement fix: {human_text[:40]}"
        else:
            title = f"Intervention: {human_text[:40]}"
    else:
        title = "Head-on deadlock: yield once"
    
    # Create when_text that includes human's actual words
    if human_text:
        when_text = f"Human said: '{human_text}'. Agent stuck or blocked; no progress for ≥2 ticks."
    else:
        when_text = "Both agents contend for same tile or try to swap; no progress for ≥2 ticks."
    
    return {
        "id": f"card_{int(ctx.get('tick',0))}",
        "title": title,
        "when_text": when_text,
        "human_text": human_text,  # Store original for reference
        "local_clues": [
            "teammate directly ahead or on contested tile",
            "corridor 1-wide or laterals blocked",
            "distance to subgoal unchanged ≥2 ticks"
        ],
        "action_text": f"Take ONE egocentric step ({chosen_action}), then continue plan.",
        "safety": ["never FORWARD into teammate", "cooldown 8 ticks"],
        "enabled_by_human": True,
        "example_trace": ctx.get("mini_trace",""),
    }


class ProAgentWithIntervention:
    """
    Self-contained ProAgent that uses AdvancedLLMInterpreter as the planner.
    
        Features:
        - AdvancedLLMInterpreter (CoT + Memory) for high-level planning
        - Controller and verificator logic for medium-level action execution
        - Supports human interventions via memory injection
        - Complete independence from base ProAgent class
    """
    
    def __init__(self, mlam, layout, model='gpt-4o-mini', 
                 auto_unstuck=False, controller_mode='new', 
                 agent_index=None, history_horizon=8, 
                 use_baseline=False, **kwargs):
        """
        Initialize ProAgent with AdvancedLLMInterpreter planner.
        
        Args:
            use_baseline: If True, use baseline interpreter (no CoT, no memory)
        """
        self.model = model
        self.mlam = mlam
        self.layout = layout
        self.mdp = self.mlam.mdp
        self.agent_index = agent_index
        self.auto_unstuck = auto_unstuck
        self.controller_mode = controller_mode
        self.use_baseline = use_baseline
        
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
        
        if use_baseline:
            # Use baseline interpreter (no CoT, no memory)
            try:
                from ..human_intervention.advanced_llm_intervention_baseline import (
                    BaselineLLMInterpreter,
                    HumanMessage as BaselineHumanMessage
                )
            except ImportError:
                intervention_path = os.path.join(PROJECT_ROOT, 'proagent', 'src', 'human_intervention')
                if intervention_path not in sys.path:
                    sys.path.append(intervention_path)
                from advanced_llm_intervention_baseline import (
                    BaselineLLMInterpreter,
                    HumanMessage as BaselineHumanMessage
                )
            
            # Create minimal memory (not used, but kept for compatibility)
            self.memory = AgentMemory(mdp=self.mdp)
            self.interpreter = BaselineLLMInterpreter(
                self.llm_client,
                memory=None,  # Not used in baseline
                history_horizon=history_horizon
            )
            print(f"🤖 ProAgentWithIntervention initialized with BASELINE interpreter (CoT and memory DISABLED)")
        else:
            # Use full interpreter (CoT + Memory)
            self.memory = AgentMemory(mdp=self.mdp)
            self.interpreter = AdvancedLLMInterpreter(
                self.llm_client, 
                self.memory, 
                history_horizon=history_horizon
            )
            print(f"🤖 ProAgentWithIntervention initialized with interpreter-based planner")
        
        # Human intervention tracking
        self._human_inbox: List[str] = []
        self._recent_history: List[Dict[str, Any]] = []
        self._intervention_history: List[str] = []

        # Stall tracking for Match→Apply
        self._stall_ticks = 0
        self._prev_subgoal_distance = None
        self._cooldowns = {}
        self._last_human_intervention_tick = None
        self._last_human_intervention_action = None
        self._last_human_intervention_text = None

        # Debug: keep last interpreter outputs
        self.last_plan: Optional[Dict[str, Any]] = None
        self.last_plan_category: Optional[str] = None
        self.last_plan_rationale: Optional[str] = None
        self.last_intervention_reason: Optional[str] = None
        self.last_chain_of_thought: Optional[str] = None
    
    # ==================== OpenAI Key Management ====================
    
    def load_openai_keys(self):
        # 1) Environment variable takes precedence
        api_key_env = os.environ.get("OPENAI_API_KEY")
        if api_key_env:
            self.openai_api_keys = [api_key_env.strip()]
            return
        # 2) Try CWD openai_key.txt (repo root during demos)
        if os.path.exists(openai_key_file):
            with open(openai_key_file, "r") as f:
                context = f.read()
            self.openai_api_keys = [k for k in context.split('\n') if k.strip()]
            if self.openai_api_keys:
                return
        # 3) Fallback to proagent/src/openai_key.txt
        alt_key_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "openai_key.txt"))
        if os.path.exists(alt_key_file):
            with open(alt_key_file, "r") as f:
                context = f.read()
            self.openai_api_keys = [k for k in context.split('\n') if k.strip()]
            if self.openai_api_keys:
                return
        raise FileNotFoundError("OPENAI_API_KEY not set and openai_key.txt not found in CWD or proagent/src/")

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

        if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
            for key in pot_states_dict.keys():
                if key == "cooking":
                    for pos in pot_states_dict[key]:
                        pot_id = self.pot_id_to_pos.index(pos)
                        soup_object = state.get_object(pos)
                        kitchen_state_prompt += prompt_dict[key].format(id=pot_id, t=soup_object.cook_time_remaining)
                else:
                    for pos in pot_states_dict[key]:
                        pot_id = self.pot_id_to_pos.index(pos)
                        kitchen_state_prompt += prompt_dict[key].format(id=pot_id) 
        
        elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
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

        intersect_counters = get_intersect_counter(
                                state.players_pos_and_or[self.agent_index], 
                                state.players_pos_and_or[1 - self.agent_index], 
                                self.mdp, 
                                self.mlam
                            )
        counter_states = query_counter_states(self.mdp, state)  

        if self.layout == 'forced_coordination': 
            kitchen_state_prompt += '{} counters can be visited by <Player {}>. Their states are as follows: '.format(len(intersect_counters), self.agent_index)
            count_states = {}  
            for i in intersect_counters:  
                obj_i = 'nothing' 
                if counter_states[i] != ' ': 
                    obj_i = counter_states[i]                
                if obj_i in count_states:  
                    count_states[obj_i] += 1
                else: 
                    count_states[obj_i]  = 1 
            total_obj = ['onion', 'dish']
            for i in count_states:   
                if i == 'nothing': 
                    continue 
                kitchen_state_prompt += f'{count_states[i]} counters have {i}. '   
            for i in total_obj: 
                if i not in count_states:        
                    kitchen_state_prompt += f'No counters have {i}. ' 
        return (self.layout_prompt + time_prompt + ego_state_prompt +
                teammate_state_prompt + kitchen_state_prompt)
    
    # ==================== PLANNER: AdvancedLLMInterpreter Integration ====================
    
    def generate_ml_action(self, state):
        """
        REPLACED: Use AdvancedLLMInterpreter instead of GPT planner.
        Generates medium-level actions using CoT reasoning + memory (or baseline without CoT/memory).
        """
        # Use text-based state prompt (same as original ProAgent)
        state_prompt = self.generate_state_prompt(state)
        
        # Get recent history with teammate actions
        recent_history = self._recent_history[-self.history_horizon:]
        
        # Get human intervention if available (baseline mode: always empty)
        if self.use_baseline:
            human_text = ""  # No interventions in baseline mode
        else:
            human_text = self._human_inbox.pop(0) if self._human_inbox else ""
        
        # Create human message
        if self.use_baseline:
            # Use baseline HumanMessage
            try:
                from ..human_intervention.advanced_llm_intervention_baseline import HumanMessage as BaselineHumanMessage
            except ImportError:
                from advanced_llm_intervention_baseline import HumanMessage as BaselineHumanMessage
            hm = BaselineHumanMessage(t=getattr(state, "timestep", 0), text="")
        else:
            hm = HumanMessage(t=getattr(state, "timestep", 0), text=human_text or "")
            if hm.text.strip():
                print(f"[HUMAN] Consuming intervention at t={hm.t}: '{hm.text}'")
        
        try:
            # Get plan from interpreter (pass state and agent_index for intervention recording)
            plan = self.interpreter.propose_plan(state_prompt, hm, recent_history, state, self.agent_index)
            
            # Store debug info
            try:
                self.last_plan = {
                    "steps": plan.steps,
                }
                self.last_plan_category = getattr(plan, 'category', None)
                self.last_intervention_reason = getattr(plan, 'intervention_reason', None)
                self.last_chain_of_thought = getattr(plan, 'chain_of_thought', None) if not self.use_baseline else ""
                if not self.use_baseline:
                    print(f"[INTP] category={self.last_plan_category} steps={self.last_plan.get('steps')} intervention_reason={self.last_intervention_reason}")
                    if self.last_chain_of_thought:
                        print(f"[INTP] CoT: {self.last_chain_of_thought}")
                else:
                    print(f"[BASELINE] steps={self.last_plan.get('steps')}")
            except Exception:
                pass
            
            # Record a memory event when a new plan is computed (skip in baseline mode)
            if not self.use_baseline:
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
    
    # ==================== VERIFICATOR ====================
    
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
        if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
            soup_cooking = len(pot_states_dict['cooking']) > 0
            soup_ready = len(pot_states_dict['ready']) > 0
            pot_not_full = pot_states_dict["empty"] + self.mdp.get_partially_full_pots(pot_states_dict)
            cookable_pots = self.mdp.get_full_but_not_cooking_pots(pot_states_dict)
        elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
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
        counter_dicts = query_counter_states(self.mdp, state) 

        counter_list  = get_intersect_counter(state.players_pos_and_or[self.agent_index],
                        state.players_pos_and_or[1 - self.agent_index], 
                        self.mdp, 
                        self.mlam
                    )    

        print('counter_list = {}'.format(counter_list))  
        lis = [] 
        for i in counter_list:  
            if counter_dicts[i] == ' ':  
                lis.append(i)       
        available_plans = mlam._get_ml_actions_for_positions(lis)
        return available_plans          

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
            if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
                next_order = list(state.all_orders)[0]
                soups_ready_to_cook_key = "{}_items".format(len(next_order.ingredients))
                soups_ready_to_cook = pot_states_dict[soups_ready_to_cook_key]
            elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
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
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.
        """

        if self.controller_mode == 'new':
            (
                chosen_goal,
                chosen_goal_action,
            ) = self.get_lowest_cost_action_and_goal_new(
                start_pos_and_or, motion_goals, state
            )
        else: 
            (
                chosen_goal,
                chosen_goal_action,
            ) = self.get_lowest_cost_action_and_goal(
                start_pos_and_or, motion_goals
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
    
    # ==================== STALL TRACKING FOR MATCH→APPLY ====================
    
    def stalled_ticks(self) -> int:
        return getattr(self, "_stall_ticks", 0)

    def update_stall_counter(self, d_prev: int, d_now: int) -> None:
        """Update stall counter based on distance to subgoal. Also check position-based stall."""
        # Check if distance is not decreasing (stuck making progress)
        if d_now >= d_prev:
            self._stall_ticks = (getattr(self, "_stall_ticks", 0) + 1)
        else:
            # Distance decreased - progress made!
            self._stall_ticks = 0
            # Reset cooldown when progress is made (distance decreased)
            if hasattr(self, '_cooldowns'):
                self._cooldowns["cards"] = 0

    def is_headon_conflict(self) -> bool:
        try:
            if self.prev_state is None:
                return False
            a0 = tuple(self.prev_state.players[self.agent_index].position)
            b0 = tuple(self.prev_state.players[1 - self.agent_index].position)
            # Best-effort next intents if you already compute them; else approximate via facing
            a1 = a0  # fallback if not available
            b1 = b0
            return (a1 == b1) or (a1 == b0 and b1 == a0)
        except Exception:
            return False

    def local_corridor_width(self) -> int:
        # Simple local lateral-free count (0/1/2), safe fallback if mdp API differs
        try:
            if self.prev_state is None:
                return 0
            ego = self.prev_state.players[self.agent_index]
            x, y = ego.position
            laterals = [(x+1,y),(x-1,y)]
            free = 0
            for u,v in laterals:
                try:
                    if self.mdp.get_terrain_type_at_pos((u,v)) == ' ':
                        free += 1
                except Exception:
                    pass
            return free
        except Exception:
            return 0

    def free_dirs_egocentric(self) -> Dict[str,bool]:
        # Minimal: LEFT/RIGHT/BACK/FORWARD relative to last facing (fallback = grid neighbors)
        # Implemented conservatively to avoid FORWARD into teammate
        res = {"LEFT": True, "RIGHT": True, "BACK": True, "FORWARD": False}
        try:
            if self.prev_state is None:
                return res
            player = self.prev_state.players[self.agent_index]
            # Use mdp.get_valid_actions if present; mark walls/counters as blocked
            valid = set(self.mdp.get_valid_actions(player))
            # Map to egocentric is domain-specific; keep conservative defaults
            res["FORWARD"] = (Action.INTERACT not in valid) and (Direction.NORTH in valid or Direction.SOUTH in valid or Direction.EAST in valid or Direction.WEST in valid)
        except Exception:
            pass
        return res

    def situation_summary(self) -> str:
        free = self.free_dirs_egocentric()
        parts = [
            f"- stalled_ticks = {self.stalled_ticks()}",
            f"- headon_conflict = {int(self.is_headon_conflict())}",
            f"- corridor_width_local = {self.local_corridor_width()}",
            f"- holding = {getattr(self, 'holding_item', 'unknown')}",
            f"- last_move = {getattr(self, 'last_move', 'unknown')}",
            f"- free_moves = " + ",".join([d for d,v in free.items() if v]),
        ]
        return "SITUATION (egocentric):\n" + "\n".join(parts)
    
    def maybe_auto_unstuck(self):
        # Gate: only when stalled and we have at least one human-enabled card
        # DISABLED in baseline mode
        if self.use_baseline:
            return None
            
        stall_count = self.stalled_ticks()
        if stall_count < 2:
            return None
        
        cards_all = self.memory.semantic.get("playbook_cards", [])
        human_enabled = [c for c in cards_all if c.get("enabled_by_human")]
        print(f"[Match→Apply] Checking: stalled={stall_count}, total_cards={len(cards_all)}, human_enabled={len(human_enabled)}")
        
        if not human_enabled:
            print(f"[Match→Apply] No human-enabled cards found")
            return None

        # Pick a small set of candidate cards by keywords
        # Include common human phrases for getting unstuck
        query = ["stuck","unstuck","deadlock","head-on","swap","yield","teammate","same","tile","blocked","move","step"]
        cards = self.memory.topk_cards(query, k=5)
        print(f"[Match→Apply] Query cards with keywords {query}: found {len(cards)} matches")
        if cards:
            print(f"[Match→Apply] Matching cards: {[c.get('id') + ': ' + c.get('title', '')[:50] for c in cards]}")
        if not cards:
            print(f"[Match→Apply] No cards matched keywords")
            return None

        # Build prompt pieces
        situation = self.situation_summary()
        cards_text = "\n\n".join([
            f"[{c['id']}] {c['title']}\nWHEN: {c['when_text']}\nACTION: {c['action_text']}\nSAFETY: {', '.join(c.get('safety',[]))}"
            for c in cards
        ])
        user_prompt = f"CURRENT SITUATION:\n{situation}\n\nINTERVENTION CARDS:\n{cards_text}\n\nDecide and fill the JSON."

        # System section name must match what you appended in advanced_llm_system_rules.txt
        # The match_apply method will automatically load the Match→Apply section from the file
        system_text = "MATCH→APPLY CONTROLLER (CARD MATCHING)"

        try:
            print(f"[Match→Apply] Querying LLM with {len(cards)} cards...")
            result = self.llm_client.match_apply(system=system_text, user=user_prompt)
            print(f"[Match→Apply] LLM response: apply={result.get('apply') if result else None}, matched_card={result.get('matched_card_id') if result else None}")
        except Exception as e:
            print(f"[LLM] Match→Apply exception: {e}")
            import traceback
            traceback.print_exc()
            return None

        if not result or not result.get("apply"):
            print(f"[Match→Apply] LLM decided not to apply (apply={result.get('apply') if result else None})")
            return None

        move = result.get("low_level_override")
        if move not in {"LEFT","RIGHT","BACK","WAIT"}:
            print(f"[Match→Apply] Invalid move from LLM: {move}")
            return None  # safety
        
        print(f"[Match→Apply] ✅ LLM matched card '{result.get('matched_card_id')}': {result.get('similarity_reason', '')}")

        # Cooldown bookkeeping - only set cooldown if we're applying
        # The cooldown will be reset if the action actually helps (position changes/distance decreases)
        self._cooldowns = getattr(self, "_cooldowns", {})
        cooldown_duration = int(result.get("cooldown", 4))  # Reduced from 8 to 4 ticks
        self._cooldowns["cards"] = self.current_timestep + cooldown_duration
        print(f"[Match→Apply] Cooldown set to tick {self._cooldowns['cards']} (duration={cooldown_duration})")

        # IMPORTANT: keep current ML plan (your system contract)
        return {"low_level_override": move, "steps": [self.current_ml_action]}
    
    def _egocentric_to_action(self, ego_dir: str, player) -> Optional[Any]:
        """Convert egocentric direction (LEFT/RIGHT/BACK/WAIT) to actual action."""
        if ego_dir == "WAIT":
            return Action.STAY
        try:
            # Get current orientation
            orientation = player.orientation
            # Map egocentric to cardinal based on facing
            if orientation == Direction.NORTH:
                dir_map = {"LEFT": Direction.WEST, "RIGHT": Direction.EAST, "BACK": Direction.SOUTH}
            elif orientation == Direction.SOUTH:
                dir_map = {"LEFT": Direction.EAST, "RIGHT": Direction.WEST, "BACK": Direction.NORTH}
            elif orientation == Direction.EAST:
                dir_map = {"LEFT": Direction.NORTH, "RIGHT": Direction.SOUTH, "BACK": Direction.WEST}
            elif orientation == Direction.WEST:
                dir_map = {"LEFT": Direction.SOUTH, "RIGHT": Direction.NORTH, "BACK": Direction.EAST}
            else:
                return Action.STAY  # fallback
            return dir_map.get(ego_dir, Action.STAY)
        except Exception:
            return Action.STAY
    
    # ==================== MAIN ACTION METHOD ====================
    
    def action(self, state):
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        # only use to record the teammate ml_action, 
        # if teammate finish ml_action in t-1, it will record in s_t, 
        # otherwise, s_t will just record None,
        # and we here check this information and store it into proagent
        self.current_timestep = getattr(state, 'timestep', 0)

        # if current ml action does not exist, generate a new one (respect override)
        if self.current_ml_action is None:
            self.current_ml_action = self.generate_ml_action(state)

        # if the current ml action is in process, Player{self.agent_index} done, else generate a new one
        if self.current_ml_action_steps > 0:
            current_ml_action_done = self.check_current_ml_action_done(state)
            if current_ml_action_done:
                # generate a new ml action
                self.current_ml_action = self.generate_ml_action(state)

        count = 0
        while not self.validate_current_ml_action(state):
            self.current_ml_action = self.generate_ml_action(state)
            
            count += 1
            if count > 3:
                self.current_ml_action = "wait(1)"
                self.time_to_wait = 1

        # Update stall counter once per tick (compute simple distance to subgoal + position check)
        player = state.players[self.agent_index]
        try:
            # Check position-based stall (same position for multiple ticks)
            current_pos = tuple(player.position)
            if not hasattr(self, '_position_history'):
                self._position_history = []
            self._position_history.append(current_pos)
            if len(self._position_history) > 3:
                self._position_history = self._position_history[-3:]
            
            # Check if stuck in same position
            position_stuck = len(self._position_history) >= 2 and len(set(self._position_history[-2:])) == 1
            
            motion_goals = self.find_motion_goals(state)
            if motion_goals:
                # Simple distance: Manhattan distance to nearest goal
                min_dist = min(
                    abs(mg[0][0] - player.position[0]) + abs(mg[0][1] - player.position[1])
                    for mg in motion_goals
                )
            else:
                min_dist = 0
            subgoal_dist_now = min_dist
            subgoal_dist_prev = self._prev_subgoal_distance if self._prev_subgoal_distance is not None else subgoal_dist_now
            
            # Update stall counter based on distance
            self.update_stall_counter(subgoal_dist_prev, subgoal_dist_now)
            
            # Also check if position hasn't changed (alternative stall signal)
            # If position stuck AND distance not improving, ensure stall counter is high enough
            if position_stuck and subgoal_dist_now >= subgoal_dist_prev and subgoal_dist_now > 0:
                self._stall_ticks = max(self._stall_ticks, 2)
            
            # Reset stall counter if position changed (clear progress signal)
            if not position_stuck:
                # Position changed - this is progress, so reduce stall count
                if self._stall_ticks > 0:
                    self._stall_ticks = max(0, self._stall_ticks - 1)
            
            self._prev_subgoal_distance = subgoal_dist_now
            stall_count = self.stalled_ticks()
            if stall_count >= 2:
                print(f"[Stall] Detected stall: ticks={stall_count}, dist_prev={subgoal_dist_prev:.1f}, dist_now={subgoal_dist_now:.1f}, pos_stuck={position_stuck}, pos={current_pos}")
        except Exception as e:
            print(f"[Stall] Error in stall detection: {e}")
            import traceback
            traceback.print_exc()

        # Write a card when human fix succeeds (progress resumed within 5 ticks)
        # DISABLED in baseline mode (no memory/cards)
        if (not self.use_baseline and
            self._last_human_intervention_tick is not None and 
            self._last_human_intervention_action and 
            self.current_timestep - self._last_human_intervention_tick <= 5 and
            self.stalled_ticks() == 0):  # Success: no longer stalled
            ctx = {
                "tick": self._last_human_intervention_tick,
                "mini_trace": getattr(self, "recent_trace", ""),
            }
            # Convert action to egocentric if needed
            action_map = {
                "move_north": "FORWARD", "move_south": "BACK", 
                "move_east": "RIGHT", "move_west": "LEFT"
            }
            ego_action = action_map.get(self._last_human_intervention_action, "BACK")
            human_text = self._last_human_intervention_text or ""
            card = make_card_from_success(ctx, chosen_action=ego_action, human_text=human_text)
            self.memory.add_intervention_card(card)
            print(f"[Cards] ✅ Learned {card['id']} from human intervention at tick {self._last_human_intervention_tick}")
            print(f"[Cards]    Title: {card['title']}")
            print(f"[Cards]    Human said: '{human_text}'")
            print(f"[Cards]    When: {card['when_text']}")
            print(f"[Cards]    Action: {card['action_text']}")
            print(f"[Cards]    Total cards in memory: {len(self.memory.semantic.get('playbook_cards', []))}")
            # Reset tracking
            self._last_human_intervention_tick = None
            self._last_human_intervention_action = None
            self._last_human_intervention_text = None

        # Try auto "Match→Apply" BEFORE normal low-level controller
        # ONLY trigger when agent is stalled (stuck/deadlock situation)
        # Cooldown is only used to prevent spam when making progress
        # DISABLED in baseline mode (no memory/cards)
        reflex = None
        if not self.use_baseline:
            cooldown_until = self._cooldowns.get("cards", 0)
            stall_count = self.stalled_ticks()
            
            # ONLY try Match→Apply if agent is stalled (stuck/deadlock)
            if stall_count >= 2:
                # Agent is stalled - try Match→Apply (cooldown will reset if action helps)
                if self.current_timestep < cooldown_until:
                    print(f"[Match→Apply] ⚠️ Overriding cooldown (agent stalled: {stall_count} ticks)")
                print(f"[Match→Apply] Attempting auto-unstuck at tick {self.current_timestep} (cooldown until {cooldown_until}, stalled={stall_count})")
                reflex = self.maybe_auto_unstuck()
                if reflex is not None:
                    print(f"[Match→Apply] ✅ Auto-unstuck triggered!")
                else:
                    print(f"[Match→Apply] ❌ Auto-unstuck returned None")
            else:
                # Agent is NOT stalled - skip Match→Apply entirely
                if self.current_timestep < cooldown_until:
                    print(f"[Match→Apply] ⏸️ Skipping (not stalled: {stall_count}, cooldown until {cooldown_until})")
                # No log needed if cooldown expired and not stalled - agent is making progress normally
        
        if reflex is not None:
            # Apply the reflex action
            ego_dir = reflex.get("low_level_override")
            if ego_dir and ego_dir in {"LEFT","RIGHT","BACK","WAIT"}:
                chosen_action = self._egocentric_to_action(ego_dir, player)
                if chosen_action:
                    print(f"[Match→Apply] Auto-unstuck: {ego_dir} -> {chosen_action}")
                    # Keep current ML plan as specified
                    self.current_ml_action = reflex.get("steps", [self.current_ml_action])[0] if reflex.get("steps") else self.current_ml_action
                    # Skip normal action selection and return immediately
                    self.prev_state = state
                    self.current_ml_action_steps += 1
                    if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
                        self._append_history_tick(state, chosen_action)
                        return chosen_action, {}
                    else:
                        self._append_history_tick(state, chosen_action)
                        return chosen_action

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

        # Handle version-specific return format
        if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
            self.prev_state = state
            result = chosen_action, {}
        elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
            self.prev_state = state
            result = chosen_action

        if self.auto_unstuck and chosen_action != Action.INTERACT:
            if (
                    self.prev_state is not None
                    and state.players
                    == self.prev_state.players
            ):
                if self.agent_index == 0:
                    joint_actions = list(
                        itertools.product(Action.ALL_ACTIONS, [Action.STAY])
                    )
                elif self.agent_index == 1:
                    joint_actions = list(
                        itertools.product([Action.STAY], Action.ALL_ACTIONS)
                    )
                else:
                    raise ValueError("Player index not recognized")

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    if j_a != [Action.INTERACT,Action.STAY] and  j_a != [Action.STAY,Action.INTERACT]:
                        # Support both shared mdp (returns 2) and legacy (returns 3)
                        _res = self.mlam.mdp.get_state_transition(state, j_a)
                        if isinstance(_res, tuple) and len(_res) == 3:
                            new_state, _, _ = _res
                        else:
                            new_state, _ = _res
                        if (
                                new_state.players_pos_and_or
                                != self.prev_state.players_pos_and_or
                            ):
                            unblocking_joint_actions.append(j_a)
                unblocking_joint_actions.append([Action.STAY, Action.STAY])
                chosen_action = unblocking_joint_actions[
                    np.random.choice(len(unblocking_joint_actions))
                ][self.agent_index]

        self.prev_state = state
        if chosen_action is None:
            self.current_ml_action = "wait(1)"
            self.time_to_wait = 1
            chosen_action = Action.STAY
        self.current_ml_action_steps += 1

        print(f"[DBG] return chosen_action={chosen_action} type={type(chosen_action)}")
        
        # Track history for interpreter
        self._append_history_tick(state, chosen_action)
        
        if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
            return chosen_action, {}
        elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
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
        if not self.use_baseline:
            self.memory = AgentMemory(mdp=self.mdp)
        self._human_inbox.clear()
        self._recent_history.clear()
        # Reset stall tracking
        self._stall_ticks = 0
        self._prev_subgoal_distance = None
        self._cooldowns = {}
        self._last_human_intervention_tick = None
        self._last_human_intervention_action = None
        self._last_human_intervention_text = None
        self._position_history = []
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
