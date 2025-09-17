#!/usr/bin/env python3
"""
Random Agent

Simple agent that takes random actions. Useful for testing intervention scenarios.
"""

import random
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction


class RandomAgent:
    """Agent that takes random actions."""
    
    def __init__(self, mdp, agent_idx=0, agent_name="Random"):
        self.mdp = mdp
        self.agent_idx = agent_idx
        self.agent_name = agent_name
        self.step_count = 0
        self.heuristic = f"{agent_name}: Random actions"
        
        print(f"🎲 {agent_name} initialized: Random action agent")
    
    def get_action(self, state):
        """Get random action."""
        self.step_count += 1
        
        # Random action from available actions
        actions = [0, 1, 2, 3, 4, 5]  # STAY, EAST, SOUTH, NORTH, WEST, INTERACT
        action = random.choice(actions)
        
        self.heuristic = f"{self.agent_name}: Random action {action}"
        return action
    
    def action(self, state):
        """Method called by AgentPair."""
        return self.get_action(state)
