#!/usr/bin/env python3
"""
HINT-Agent: Human-INtervention-enhanced Agent for CrowdNav-AHT

This is a self-contained agent that uses AdvancedLLMInterpreter (CoT + Memory) as the planner
for crowd navigation with waypoint-based actions.
"""

import os
import sys
import json
import re
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, List, Union, Tuple

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
    Self-contained HINTAgent for CrowdNav-AHT that uses AdvancedLLMInterpreter as the planner.
    
    Features:
    - AdvancedLLMInterpreter (CoT + Memory) for high-level waypoint planning
    - Supports human interventions via memory injection
    - Waypoint-based action selection (0-7)
    """
    
    def __init__(self, model='gpt-5-mini', agent_index=0, 
                 history_horizon=2, **kwargs):
        """
        Initialize HINTAgent with AdvancedLLMInterpreter planner for CrowdNav-AHT.
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
        self.low_level_override_action: Optional[Union[int, Tuple[float, float]]] = None
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
        self.last_chain_of_thought: Optional[str] = None
        
        # Event detection and intervention outcome tracking
        self.cycle_dist_thresh = 0.1
        self.cycle_steps = 3
        self.no_progress_window = 5
        self.cyclic_success_window = 3
        self.lack_progress_success_window = 5
        self.no_progress_delta = 0.5
        self.success_progress_delta = 0.5
        self._pos_history = deque(maxlen=8)
        self._stuck_counter = 0
        self._goal_history = deque(maxlen=self.no_progress_window)
        self._pending_intervention = None
        
        print(f"🤖 HINTAgentCrowdNav initialized with interpreter-based planner")
    
    # ==================== OpenAI Key Management ====================
    
    def load_openai_keys(self):
        # 1) Environment variable takes precedence
        api_key_env = os.environ.get("OPENAI_API_KEY")
        if api_key_env:
            self.openai_api_keys = [api_key_env.strip()]
            return
        raise FileNotFoundError("OPENAI_API_KEY not set in environment")

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
        Generate a semantic, egocentric scene abstraction (psi_t).
        """
        psi = self._build_semantic_state(obs)
        return psi

    def _build_semantic_state(self, obs: str) -> str:
        pos = self._extract_robot_pos(obs)
        if pos is None:
            return obs
        rx, ry = pos
        lines = []
        lines.append("Ego at origin. ")

        tpos, tvel = self._extract_teammate(obs)
        if tpos:
            rel = self._rel_desc((rx, ry), tpos)
            lines.append(f"Teammate {rel}. ")

        goals = self._extract_meeting_locations(obs)
        if goals:
            rels = [self._rel_desc((rx, ry), g) for g in goals]
            lines.append(f"Meeting points: {rels[0]} and {rels[1]}. ")

        rgoal = self._extract_robot_goal(obs)
        if rgoal:
            lines.append(f"Current target {self._rel_desc((rx, ry), rgoal)}. ")

        others = self._extract_other_agents(obs)
        if others:
            nearest = sorted(others, key=lambda p: abs(p[0]-rx)+abs(p[1]-ry))[:3]
            rels = [self._rel_desc((rx, ry), p) for p in nearest]
            lines.append("Nearest agents: " + "; ".join(rels) + ". ")

        return "".join(lines).strip()

    def _rel_desc(self, from_pos, to_pos):
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        dist = abs(dx) + abs(dy)
        if dist == 0:
            return "at same spot"
        horiz = "east" if dx > 0 else "west"
        vert = "north" if dy > 0 else "south"
        if dx == 0:
            return f"{abs(dy):.1f}m {vert}"
        if dy == 0:
            return f"{abs(dx):.1f}m {horiz}"
        return f"{abs(dx):.1f}m {horiz}, {abs(dy):.1f}m {vert} (dist {dist:.1f})"

    def _extract_teammate(self, obs: str):
        match = re.search(r"teammate is located at position x = ([\-0-9\.]+) meters, y = ([\-0-9\.]+) meters", obs)
        if not match:
            return None, None
        try:
            return (float(match.group(1)), float(match.group(2))), None
        except Exception:
            return None, None

    def _extract_meeting_locations(self, obs: str):
        match = re.search(r"two candidate meeting locations at x = ([\-0-9\.]+), y = ([\-0-9\.]+) and x = ([\-0-9\.]+), y = ([\-0-9\.]+)", obs)
        if not match:
            return None
        try:
            m1 = (float(match.group(1)), float(match.group(2)))
            m2 = (float(match.group(3)), float(match.group(4)))
            return [m1, m2]
        except Exception:
            return None

    def _best_combined_distance(self, obs: str) -> Optional[float]:
        pos = self._extract_robot_pos(obs)
        teammate_pos, _ = self._extract_teammate(obs)
        meetings = self._extract_meeting_locations(obs)
        if not pos or not teammate_pos or not meetings:
            return None
        combined = []
        for m in meetings:
            ego_d = float(np.hypot(m[0] - pos[0], m[1] - pos[1]))
            mate_d = float(np.hypot(m[0] - teammate_pos[0], m[1] - teammate_pos[1]))
            combined.append(ego_d + mate_d)
        return min(combined) if combined else None

    def _extract_other_agents(self, obs: str):
        agents = []
        for match in re.finditer(r"Agent \d+ is at position x = ([\-0-9\.]+), y = ([\-0-9\.]+)", obs):
            try:
                agents.append((float(match.group(1)), float(match.group(2))))
            except Exception:
                continue
        return agents

    def _format_history(self, recent_history):
        if not recent_history:
            return "None"
        items = []
        for h in recent_history[-self.history_horizon:]:
            items.append(str(h))
        return ", ".join(items)

    def _encode_context(self, psi_text, recent_history, human_text, events):
        history_text = self._format_history(recent_history)
        human_part = human_text.strip() if human_text and human_text.strip() else "None"
        events_part = ", ".join(events) if events else "None"
        return f"STATE: {psi_text}\nHISTORY: {history_text}\nHUMAN: {human_part}\nEVENTS: {events_part}"
    
    # ==================== Waypoint Action Generation ====================
    
    def generate_waypoint_action(self, obs: str) -> int:
        """
        Generate next waypoint action using the interpreter.
        
        Args:
            obs: Text observation from environment
        
        Returns:
            Waypoint action index (0-7)
        """
        # Detect events (progress / cycles)
        events = self._detect_events(obs)

        # Check for human interventions
        human_message = None
        if self._human_inbox:
            human_message = self._human_inbox.pop(0)
            print(f"🎯 Processing human intervention: '{human_message}'")
            if human_message and self._pending_intervention is None:
                self._pending_intervention = {
                    "timestamp": self.current_timestep,
                    "start_t": self.current_timestep,
                    "events": list(events or []),
                    "saw_cyclic_start": "cyclic_behavior" in (events or []),
                    "start_pos": self._extract_robot_pos(obs),
                    "start_best_dist": self._best_combined_distance(obs),
                }
        
        # Generate semantic abstraction and full context prompt
        psi_text = self.generate_state_prompt(obs)
        
        # Build recent history
        recent_history = self._recent_history[-self.history_horizon:]
        state_prompt = self._encode_context(psi_text, recent_history, human_message or "", events)
        
        # Call interpreter
        try:
            plan = self.interpreter.interpret(
                state_prompt=state_prompt,
                recent_history=recent_history,
                human_message=human_message,
                events=events,
                timestep=self.current_timestep,
                psi_text=psi_text
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
            self.last_chain_of_thought = plan.chain_of_thought
            
            # Check for low-level override
            if plan.low_level_override is not None:
                self.low_level_override_duration = 1
                if isinstance(plan.low_level_override, dict):
                    vx = float(plan.low_level_override.get("vx", 0.0))
                    vy = float(plan.low_level_override.get("vy", 0.0))
                    self.low_level_override_action = (vx, vy)
                    waypoint_action = self.low_level_override_action
                else:
                    self.low_level_override_action = int(plan.low_level_override)
                    waypoint_action = self.low_level_override_action
            else:
                # Use first step from plan
                if plan.steps:
                    waypoint_action = plan.steps[0]
                    self.current_waypoint_action = waypoint_action
                    self.current_waypoint_steps = 0
                else:
                    # Fallback
                    waypoint_action = 0
            
            # Record encoded history as prior psi_t
            self._recent_history.append(psi_text)
            
            # Record in episodic memory
            self.memory.write_events([{
                "type": "plan",
                "t": self.current_timestep,
                "steps": plan.steps,
                "category": plan.category,
                "human_message": human_message,
            }])

        # Update intervention outcome if pending
        self._update_intervention_outcome(events, obs)
        
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
        if self.low_level_override_action is not None and self.low_level_override_duration > 0:
            self.low_level_override_duration -= 1
            if self.low_level_override_duration <= 0:
                self.low_level_override_action = None
            return self.low_level_override_action
        
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

    def _extract_robot_pos(self, obs: str):
        match = re.search(r"robot is at position x = ([\-0-9\.]+) meters, y = ([\-0-9\.]+)", obs)
        if not match:
            return None
        try:
            return float(match.group(1)), float(match.group(2))
        except Exception:
            return None

    def _extract_robot_goal(self, obs: str):
        match = re.search(r"robot's current target is at x = ([\-0-9\.]+) meters, y = ([\-0-9\.]+)", obs)
        if not match:
            match = re.search(r"common goal .* x = ([\-0-9\.]+) meters, y = ([\-0-9\.]+)", obs)
        if not match:
            return None
        try:
            return float(match.group(1)), float(match.group(2))
        except Exception:
            return None

    def _detect_events(self, obs: str) -> List[str]:
        events = []
        pos = self._extract_robot_pos(obs)
        teammate_pos, _ = self._extract_teammate(obs)
        meetings = self._extract_meeting_locations(obs)
        if pos is None:
            return events
        self._pos_history.append(pos)
        if len(self._pos_history) >= 2:
            prev = self._pos_history[-2]
            dist = np.hypot(pos[0] - prev[0], pos[1] - prev[1])
            if dist < self.cycle_dist_thresh:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
            if self._stuck_counter >= self.cycle_steps:
                if "cyclic_behavior" not in events:
                    events.append("cyclic_behavior")
        if pos and meetings and teammate_pos:
            combined = []
            for m in meetings:
                ego_d = float(np.hypot(m[0] - pos[0], m[1] - pos[1]))
                mate_d = float(np.hypot(m[0] - teammate_pos[0], m[1] - teammate_pos[1]))
                combined.append(ego_d + mate_d)
            min_combined = min(combined) if combined else None
            if min_combined is not None:
                self._goal_history.append(min_combined)
                if len(self._goal_history) >= self.no_progress_window:
                    if (self._goal_history[0] - self._goal_history[-1]) < self.no_progress_delta:
                        events.append("lack_of_progress")
        return events

    def _get_progress_signature(self, obs: str) -> Dict[str, Any]:
        pos = self._extract_robot_pos(obs)
        goal = self._extract_robot_goal(obs)
        teammate_pos, _ = self._extract_teammate(obs)
        meetings = self._extract_meeting_locations(obs)
        sig = {"pos": pos, "goal_dist": None, "meeting_combined_dists": None}
        if pos and goal:
            sig["goal_dist"] = float(np.hypot(goal[0] - pos[0], goal[1] - pos[1]))
        if pos and teammate_pos and meetings:
            combined = []
            for m in meetings:
                ego_d = float(np.hypot(m[0] - pos[0], m[1] - pos[1]))
                mate_d = float(np.hypot(m[0] - teammate_pos[0], m[1] - teammate_pos[1]))
                combined.append(ego_d + mate_d)
            sig["meeting_combined_dists"] = combined
        return sig

    def _progress_improved(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> bool:
        if not baseline or not current:
            return False
        base_meet = baseline.get("meeting_combined_dists")
        curr_meet = current.get("meeting_combined_dists")
        if base_meet and curr_meet and len(base_meet) == len(curr_meet):
            for b, c in zip(base_meet, curr_meet):
                if c < b - self.success_progress_delta:
                    return True
        return False

    def _update_intervention_outcome(self, current_events: Optional[List[str]], obs: Optional[str] = None):
        if not self._pending_intervention:
            return
        start_t = self._pending_intervention.get("start_t", self.current_timestep)
        steps_since = max(0, int(self.current_timestep) - int(start_t))
        events = current_events or []
        saw_cyclic_start = bool(self._pending_intervention.get("saw_cyclic_start"))

        if saw_cyclic_start and steps_since <= self.cyclic_success_window and obs:
            start_pos = self._pending_intervention.get("start_pos")
            cur_pos = self._extract_robot_pos(obs)
            if start_pos and cur_pos:
                dist = float(np.hypot(cur_pos[0] - start_pos[0], cur_pos[1] - start_pos[1]))
                if dist > self.cycle_dist_thresh:
                    print(f"[INTERVENTION] success (cycle resolved) at t={self.current_timestep} for intervention {self._pending_intervention['timestamp']}")
                    self.interpreter.commit_intervention_pattern(self._pending_intervention["timestamp"])
                    self._pending_intervention = None
                    return

        if steps_since <= self.lack_progress_success_window and obs:
            start_best = self._pending_intervention.get("start_best_dist")
            curr_best = self._best_combined_distance(obs)
            if start_best is not None and curr_best is not None:
                if (start_best - curr_best) >= self.success_progress_delta:
                    print(f"[INTERVENTION] success (meeting progress) at t={self.current_timestep} for intervention {self._pending_intervention['timestamp']}")
                    self.interpreter.commit_intervention_pattern(self._pending_intervention["timestamp"])
                    self._pending_intervention = None
                    return

        if steps_since >= self.lack_progress_success_window:
            if not saw_cyclic_start or steps_since >= self.cyclic_success_window:
                print(f"[INTERVENTION] failure at t={self.current_timestep} for intervention {self._pending_intervention['timestamp']}")
                self.interpreter.discard_intervention_pattern(self._pending_intervention["timestamp"])
                self._pending_intervention = None
    
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

