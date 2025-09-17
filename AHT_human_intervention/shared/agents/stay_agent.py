#!/usr/bin/env python3
"""
Stay Agent

An agent that simply stays at a fixed position and does nothing.
This agent is used to test if the ego agent can work independently.
"""

import sys

# Add project root to path
sys.path.append('.')

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction


class StayAgent:
    """Agent that stays at a fixed position and does nothing."""
    
    def __init__(self, mdp, agent_idx=0, agent_name="StayAgent"):
        self.mdp = mdp
        self.agent_idx = agent_idx
        self.agent_name = agent_name
        self.step_count = 0
        self.heuristic = f"{agent_name}: Staying at (6,1)"
        
        # Fixed position to stay at
        self.stay_position = (6, 1)
        
        print(f"🏠 {agent_name} initialized: STAY AGENT")
        print(f"   📍 Will stay at position: {self.stay_position}")
        print(f"   🚫 Will not perform any actions")
    
    def get_action(self, state):
        """Always return STAY action (0)."""
        self.step_count += 1
        
        try:
            ego = state.players[self.agent_idx]
            ego_pos = ego.position
            
            # Update heuristic display
            self.heuristic = f"{self.agent_name}: Staying at {ego_pos} (step {self.step_count})"
            
            # Always stay - do nothing
            return 0  # STAY action
            
        except Exception as e:
            print(f"[ERROR] Stay agent {self.agent_idx} failed: {e}")
            self.heuristic = f"{self.agent_name}: Error - staying"
            return 0
    
    def action(self, state):
        """Method called by AgentPair."""
        return self.get_action(state)
