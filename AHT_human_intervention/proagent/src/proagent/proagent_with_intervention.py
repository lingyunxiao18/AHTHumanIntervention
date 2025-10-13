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

# OpenAI key handling (copied from ProAgent)
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


class ProAgentWithIntervention:
    """
    Self-contained ProAgent that uses AdvancedLLMInterpreter as the planner.
    
    Features:
    - AdvancedLLMInterpreter (CoT + Memory) for high-level planning
    - Exact controller and verificator logic from original ProAgent
    - Supports human interventions via memory injection
    - Complete independence from base ProAgent class
    """
    
    def __init__(self, mlam, layout, model='gpt-4o-mini', 
                 auto_unstuck=False, controller_mode='new', 
                 agent_index=None, outdir=None, history_horizon=8, **kwargs):
        """
        Initialize ProAgent with AdvancedLLMInterpreter planner.
        """
        self.model = model
        self.mlam = mlam
        self.layout = layout
        self.mdp = self.mlam.mdp
        self.agent_index = agent_index
        self.auto_unstuck = auto_unstuck
        self.controller_mode = controller_mode
        self.out_dir = outdir
        
        # OpenAI API key handling (copied from ProAgent)
        self.openai_api_keys = []
        self.load_openai_keys()
        self.key_rotation = True
        
        # Agent state (copied from ProAgent)
        self.prev_state = None
        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.possible_motion_goals = None
        self.pot_id_to_pos = []
        self.current_timestep = 0
        self.teammate_ml_actions_dict = {}
        self.teammate_intentions_dict = {}
        
        # Layout prompt generation (copied from ProAgent)
        self.layout_prompt = self.generate_layout_prompt()
        
        # Initialize interpreter-based planner with MDP for layout facts
        self.memory = AgentMemory(mdp=self.mdp)
        self.history_horizon = history_horizon
        
        # Initialize LLM client and interpreter
        from openai import OpenAI
        # Initialize OpenAI client with API key loaded via openai_key.txt/env
        self.llm_client = LLMClient(openai_client=OpenAI(api_key=self.openai_api_key()), model='gpt-4.1-mini')
        
        self.interpreter = AdvancedLLMInterpreter(
            self.llm_client, 
            self.memory, 
            history_horizon=history_horizon
        )
        
        # Human intervention tracking
        self._human_inbox: List[str] = []
        self._recent_history: List[Dict[str, Any]] = []
        self._intervention_history: List[str] = []

        # Debug: keep last interpreter outputs
        self.last_plan: Optional[Dict[str, Any]] = None
        self.last_plan_category: Optional[str] = None
        self.last_plan_rationale: Optional[str] = None
        
        print(f"🤖 ProAgentWithIntervention initialized with interpreter-based planner")
    
    # ==================== OpenAI Key Management (copied from ProAgent) ====================
    
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
    
    # ==================== Layout and State Management (copied from ProAgent) ====================
    
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
        Generates medium-level actions using CoT reasoning + memory.
        """
        # Use text-based state prompt (same as original ProAgent)
        state_prompt = self.generate_state_prompt(state)
        
        # Get recent history with teammate actions
        recent_history = self._recent_history[-self.history_horizon:]
        
        # Get human intervention if available
        human_text = self._human_inbox.pop(0) if self._human_inbox else ""
        
        # Create human message
        hm = HumanMessage(t=getattr(state, "timestep", 0), text=human_text or "")
        if hm.text.strip():
            print(f"[HUMAN] Consuming intervention at t={hm.t}: '{hm.text}'")
        
        try:
            # Get plan from interpreter
            plan = self.interpreter.propose_plan(state_prompt, hm, recent_history)
            # Store debug info
            try:
                self.last_plan = {
                    "steps": plan.steps,  # Direct list of ML_action strings
                    "confidence": getattr(plan, 'confidence', None)
                }
                self.last_plan_category = getattr(plan, 'category', None)
                self.last_plan_rationale = getattr(plan, 'rationale_public', None)
                print(f"[INTP] category={self.last_plan_category} conf={self.last_plan.get('confidence')} steps={self.last_plan.get('steps')} rationale={self.last_plan_rationale}")
            except Exception:
                pass
            # Record a memory event when a new plan is computed
            try:
                t = int(getattr(state, 'timestep', 0))
                self.memory.write_events([
                    {"t": t,
                     "type": "plan",
                     "steps": list(plan.steps),
                     "conf": float(getattr(plan, 'confidence', 0.0)),
                     "category": str(getattr(plan, 'category', '')),
                     "rationale": str(getattr(plan, 'rationale_public', ''))}
                ])
            except Exception:
                pass
            
            # Return first step as current ml_action
            if not plan.steps:
                raise RuntimeError("Interpreter returned empty plan")
            # Use ML action directly from the step
            ml_action = plan.steps[0]
            print(f"🧠 Interpreter generated plan with {len(plan.steps)} steps")
            print(f"📋 Current ml_action: {ml_action}")
            return ml_action
                
        except Exception as e:
            print(f"❌ Interpreter failed: {e}")
            raise
    
    # TODO: when the interpreter fails, generate a failure message and feedback to the interpreter, replan
    
    
    # ==================== VERIFICATOR (copied from ProAgent) ====================
    
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
            soup_cooking = len(pot_states_dict['onion']['cooking'])+len(pot_states_dict['tomato']['cooking']) > 0
            soup_ready = len(pot_states_dict['onion']['ready'])+len(pot_states_dict['tomato']['ready']) > 0
            pot_not_full = pot_states_dict["empty"] + pot_states_dict["onion"]['partially_full'] + pot_states_dict["tomato"]['partially_full']
            cookable_pots = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)]

        has_onion = False
        has_tomato = False
        has_dish = False
        has_soup = False
        has_object = player.has_object()
        if has_object:
            has_onion = player.get_object().name == 'onion'
            has_tomato = player.get_object().name == 'tomato'
            has_dish = player.get_object().name == 'dish'
            has_soup = player.get_object().name == 'soup'
        empty_counter = self.mdp.get_empty_counter_locations(state)

        if self.current_ml_action in ["pickup(onion)", "pickup_onion"]:   
            flag2 = len(self.find_motion_goals(state)) == 0 
            if flag2: 
                return False 
            return not has_object and len(self.mdp.get_onion_dispenser_locations()) > 0
        if self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
            return not has_object and len(self.mdp.get_tomato_dispenser_locations()) > 0
        elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
            flag2 = len(self.find_motion_goals(state)) == 0 
            if flag2: 
                return False 
            return not has_object and len(self.mdp.get_dish_dispenser_locations()) > 0
        elif "put_onion_in_pot" in self.current_ml_action:
            return has_onion and len(pot_not_full) > 0
        elif "put_tomato_in_pot" in self.current_ml_action:
            return has_tomato and len(pot_not_full) > 0
        elif "place_obj_on_counter" in self.current_ml_action:
            return has_object and len(empty_counter) > 0
        elif "fill_dish_with_soup" in self.current_ml_action:
            return has_dish and (soup_ready or soup_cooking)
        elif "deliver_soup" in self.current_ml_action:
            return has_soup
        elif "wait" in self.current_ml_action:
            return 0 < int(self.current_ml_action.split('(')[1][:-1]) <= 20
    
    # ==================== CONTROLLER (copied from ProAgent) ====================
    
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

        elif self.current_ml_action in ["pickup(tomato)", "pickup_tomato"]:
            motion_goals = am.pickup_tomato_actions(state, counter_objects)
        elif self.current_ml_action in ["pickup(dish)", "pickup_dish"]:
            # Use shared env's helper name
            motion_goals = am.pickup_dish_actions(state, counter_objects)
        elif "put_onion_in_pot" in self.current_ml_action:
            motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
        elif "put_tomato_in_pot" in self.current_ml_action:
            motion_goals = am.put_tomato_in_pot_actions(pot_states_dict)
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
                soups_ready_to_cook = pot_states_dict["onion"]['{}_items'.format(self.mdp.num_items_for_soup)] + pot_states_dict["tomato"]['{}_items'.format(self.mdp.num_items_for_soup)]
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
                print(f"[DBG] plan goal={goal} cost={plan_cost} first_action={action_plan}")
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
        return best_goal, best_action

    def real_time_planner(self, start_pos_and_or, goal, state):
        # Use shared motion planner to compute plan and cost
        try:
            action_plan, plan_nodes, plan_cost = self.mlam.motion_planner.get_plan(start_pos_and_or, goal)
            first_action = action_plan[0] if action_plan else None
            return first_action, plan_cost
        except Exception:
            # Fallback: no plan
            return None, np.Inf
    
    # ==================== MAIN ACTION METHOD (copied from ProAgent) ====================
    
    def action(self, state):
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        # only use to record the teammate ml_action, 
        # if teammate finish ml_action in t-1, it will record in s_t, 
        # otherwise, s_t will just record None,
        # and we here check this information and store it into proagent
        self.current_timestep = getattr(state, 'timestep', 0)
        # Shared Overcooked state may not expose ml_actions; guard access
        if hasattr(state, 'ml_actions'):
            if state.ml_actions[1-self.agent_index] is not None:
                self.teammate_ml_actions_dict[str(self.current_timestep-1)] = state.ml_actions[1-self.agent_index]

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

        if "wait" in self.current_ml_action:
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
            if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
                self.prev_state = state
                result = chosen_action, {}
            elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
                self.prev_state = state
                result = chosen_action
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
        self.possible_motion_goals = None
        self.current_timestep = 0
        self.teammate_ml_actions_dict = {}
        self.teammate_intentions_dict = {}
        self.memory = AgentMemory()
        self._human_inbox.clear()
        self._recent_history.clear()
    
    def set_agent_index(self, agent_index):
        """Set the agent index."""
        self.agent_index = agent_index
    
    def apply_human_intervention(self, text: str):
        """Apply human intervention text to the agent's inbox and immediately override current action."""
        if text and text.strip():
            self._human_inbox.append(text.strip())
            self._intervention_history.append(text.strip())
            print(f"🎯 Human intervention received: '{text.strip()}'")
            # Record structured memory event of the human intervention
            try:
                t = int(getattr(self.mdp, 'timestep', getattr(self, 'current_timestep', 0)))
            except Exception:
                t = int(getattr(self, 'current_timestep', 0))
            self.memory.write_events([
                {"t": t,
                 "type": "human_msg",
                 "text": text.strip(),
                 "agent_ml_action": self.current_ml_action,
                 "state": {
                     "ego_pos": [int(self.mdp.state.players[self.agent_index].position[0]), int(self.mdp.state.players[self.agent_index].position[1])] if hasattr(self.mdp, 'state') else None,
                     "mate_pos": [int(self.mdp.state.players[1-self.agent_index].position[0]), int(self.mdp.state.players[1-self.agent_index].position[1])] if hasattr(self.mdp, 'state') else None
                 }}
            ])
            
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
