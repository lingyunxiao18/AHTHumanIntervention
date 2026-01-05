#!/usr/bin/env python3
"""
HINT-Agent: Human-INtervention-enhanced Agent for CrowdNav-AHT

This is a self-contained agent that uses AdvancedLLMInterpreter (CoT + Memory) as the planner
for crowd navigation with waypoint-based actions.
"""

import os
import sys
import json
import numpy as np
from typing import Optional, Dict, Any, List

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import advanced intervention system
try:
    from ..human_intervention.advanced_llm_intervention_crowdnav import (
        AgentMemory,
        AdvancedLLMInterpreter,
        HumanMessage,
        LLMClient,
    )
except ImportError:
    # Fallback for different import contexts
    intervention_path = os.path.join(PROJECT_ROOT, 'hintagent', 'src', 'human_intervention')
    if intervention_path not in sys.path:
        sys.path.append(intervention_path)
    from advanced_llm_intervention_crowdnav import (
        AgentMemory,
        AdvancedLLMInterpreter,
        HumanMessage,
        LLMClient,
    )

# OpenAI key file is at workspace root (one level up from PROJECT_ROOT)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
openai_key_file = os.path.join(WORKSPACE_ROOT, "openai_key.txt")

# Waypoint action directions (0-7)
WAYPOINT_DIRECTIONS = {
    0: "up",
    1: "down", 
    2: "left",
    3: "right",
    4: "up-right",
    5: "up-left",
    6: "down-right",
    7: "down-left"
}


class HINTAgentCrowdNav:
    """
    Self-contained ProAgent for CrowdNav-AHT that uses AdvancedLLMInterpreter as the planner.
    
    Features:
    - AdvancedLLMInterpreter (CoT + Memory) for high-level waypoint planning
    - Supports human interventions via memory injection
    - Waypoint-based action selection (0-7)
    """
    
    def __init__(self, model='gpt-5-mini', agent_index=0, 
                 history_horizon=8, **kwargs):
        """
        Initialize ProAgent with AdvancedLLMInterpreter planner for CrowdNav-AHT.
        """
        self.model = model
        self.agent_index = agent_index
        
        # OpenAI API key handling
        self.openai_api_keys = []
        self.load_openai_keys()
        self.key_rotation = True
        
        # Agent state
        self.prev_observation = None
        self.current_waypoint_action = None
        self.current_waypoint_steps = 0
        self.current_timestep = 0
        
        # Low-level override tracking
        self.low_level_override = None
        self.low_level_override_duration = 0
        
        # Initialize interpreter-based planner
        self.memory = AgentMemory()
        self.history_horizon = history_horizon
        
        # Initialize LLM client and interpreter
        from openai import OpenAI
        self.llm_client = LLMClient(openai_client=OpenAI(api_key=self.openai_api_key()), model=model)
        
        self.interpreter = AdvancedLLMInterpreter(
            self.llm_client, 
            self.memory, 
            history_horizon=history_horizon
        )
        
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
        self.last_intervention_reason: Optional[str] = None
        self.last_chain_of_thought: Optional[str] = None
        
        print(f"🤖 HINTAgentCrowdNav initialized with interpreter-based planner")
    
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
    
    # ==================== State and Observation Management ====================
    
    def parse_observation(self, obs: str) -> Dict[str, Any]:
        """
        Parse text observation into structured state information.
        The observation is a text description from CrowdSimAHT.generate_ob()
        """
        # Extract numeric cache if available (from environment)
        # For now, we'll work with the text observation directly
        # The interpreter will parse it using LLM reasoning
        return {
            "text_observation": obs,
            "parsed": False  # LLM will parse it
        }
    
    def generate_state_prompt(self, obs: str) -> str:
        """
        Generate state prompt from text observation.
        For CrowdNav-AHT, the observation is already a text description.
        """
        return obs
    
    # ==================== Waypoint Action Generation ====================
    
    def generate_waypoint_action(self, obs: str) -> int:
        """
        Generate next waypoint action using the interpreter.
        
        Args:
            obs: Text observation from environment
        
        Returns:
            Waypoint action index (0-7)
        """
        # Check for human interventions
        human_message = None
        if self._human_inbox:
            human_message = self._human_inbox.pop(0)
            print(f"🎯 Processing human intervention: '{human_message}'")
        
        # Generate state prompt
        state_prompt = self.generate_state_prompt(obs)
        
        # Build recent history
        recent_history = self._recent_history[-self.history_horizon:]
        
        # Call interpreter
        try:
            plan = self.interpreter.interpret(
                state_prompt=state_prompt,
                recent_history=recent_history,
                human_message=human_message
            )
        except Exception as e:
            print(f"❌ Interpreter error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: continue toward goal (action 0 = up)
            plan = None
        
        if plan is None:
            # Fallback action
            waypoint_action = 0
        else:
            # Store plan for debugging
            self.last_plan = plan.to_dict()
            self.last_plan_category = plan.category
            self.last_intervention_reason = plan.intervention_reason
            self.last_chain_of_thought = plan.chain_of_thought
            
            # Check for low-level override
            if plan.low_level_override is not None:
                self.low_level_override = plan.low_level_override
                self.low_level_override_duration = 1
                waypoint_action = plan.low_level_override
                print(f"🔄 Low-level override: {waypoint_action} ({WAYPOINT_DIRECTIONS[waypoint_action]})")
            else:
                # Use first step from plan
                if plan.steps:
                    waypoint_action = plan.steps[0]
                    self.current_waypoint_action = waypoint_action
                    self.current_waypoint_steps = 0
                else:
                    # Fallback
                    waypoint_action = 0
            
            # Record in history
            self._recent_history.append({
                "t": self.current_timestep,
                "waypoint_action": waypoint_action,
                "plan": plan.to_dict(),
                "human_message": human_message,
            })
            
            # Record in episodic memory
            self.memory.write_events([{
                "type": "plan",
                "t": self.current_timestep,
                "steps": plan.steps,
                "category": plan.category,
                "human_message": human_message,
            }])
        
        return waypoint_action
    
    # ==================== Main Action Method ====================
    
    def action(self, obs: str) -> int:
        """
        Main action method called by environment.
        
        Args:
            obs: Text observation from CrowdSimAHT environment
        
        Returns:
            Waypoint action index (0-7)
        """
        self.current_timestep += 1
        
        # Check for low-level override first
        if self.low_level_override is not None and self.low_level_override_duration > 0:
            self.low_level_override_duration -= 1
            if self.low_level_override_duration <= 0:
                self.low_level_override = None
            return self.low_level_override
        
        # Generate new waypoint action if needed
        if self.current_waypoint_action is None:
            self.current_waypoint_action = self.generate_waypoint_action(obs)
            self.current_waypoint_steps = 0
        
        # For now, we'll generate a new action each step
        # In a more sophisticated version, we could track waypoint progress
        # and only generate new actions when waypoint is reached
        waypoint_action = self.generate_waypoint_action(obs)
        
        self.prev_observation = obs
        
        return waypoint_action
    
    # ==================== Human Intervention Interface ====================
    
    def apply_human_intervention(self, text: str):
        """Apply human intervention text to the agent's inbox and immediately override current action."""
        if text and text.strip():
            self._human_inbox.append(text.strip())
            self._intervention_history.append(text.strip())
            print(f"🎯 Human intervention received: '{text.strip()}'")
            
            # Immediately override current waypoint action to process intervention
            self.current_waypoint_action = None
            self.current_waypoint_steps = 0
            print(f"🔄 Overriding current waypoint action to process intervention")

    def process_human_intervention(self, text: str) -> bool:
        """Compatibility with demo harness API"""
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
            "memory_entries": len(getattr(self.memory, 'episodic', [])),
            "current_waypoint_action": self.current_waypoint_action,
            "current_waypoint_steps": self.current_waypoint_steps
        }
    
    def set_agent_index(self, agent_index):
        """Set the agent index."""
        self.agent_index = agent_index

