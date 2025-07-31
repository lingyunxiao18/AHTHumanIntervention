#!/usr/bin/env python

import torch
import torch.nn.functional as F
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.agents.agent import Agent
from AHT_human_intervention.intervention_LLM_module import process_command

from language_conditioned_policy import (
    build_env_prompt,
    LangConditionedPolicy,
    tokenize,
    VOCAB,
    MAX_LEN,
)

class PretrainedLangAgent(Agent):
    """Agent that uses a pretrained language-conditioned policy."""
    
    def __init__(self, model_path, layout_name="random3", agent_idx=0, device="cpu"):
        self.device = torch.device(device)
        self.agent_idx = agent_idx
        
        # Load MDP
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        
        # Get model dimensions
        example_state = torch.FloatTensor(
            self.mdp.lossless_state_encoding(self.mdp.get_standard_start_state())[0].flatten()
        )
        state_dim = example_state.numel()
        num_actions = len(self.mdp.action_idx_to_name)
        
        # Initialize model
        self.model = LangConditionedPolicy(
            state_dim=state_dim,
            vocab_size=len(VOCAB),
            text_dim=128,
            hidden_dim=256,
            nhead=4,
            num_layers=2,
            max_len=MAX_LEN,
            num_actions=num_actions,
        )
        
        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        # Current command (can be set externally)
        self.current_command = ""
        
    def set_command(self, command):
        """Set the language command for the agent."""
        self.current_command = command
        
    def action(self, state):
        """Get action based on current state and language command."""
        if not self.current_command:
            # Fallback to stay if no command given
            return self.mdp.action_name_to_idx['stay']
            
        # Process state
        state_vec = torch.FloatTensor(
            self.mdp.lossless_state_encoding(state)[self.agent_idx].flatten()
        ).unsqueeze(0).to(self.device)
        
        # Tokenize command
        token_ids = tokenize(self.current_command, VOCAB, MAX_LEN).unsqueeze(0).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            logits = self.model(state_vec, token_ids)
            probs = F.softmax(logits, dim=-1)
            action_idx = torch.argmax(probs, dim=-1).item()
            
        return action_idx

def main():
    """Example usage of the pretrained model."""
    
    # Initialize agent with pretrained model
    model_path = "lang_policy_pretrained.pt"  # Path to your trained model
    agent = PretrainedLangAgent(model_path, layout_name="random3", agent_idx=0)
    
    # Set up environment
    env = OvercookedEnv.from_layout_name("random3")
    state = env.get_standard_start_state()
    
    # Example commands and actions
    test_commands = [
        "pick up the onion",
        "move to the pot", 
        "serve the soup",
        "go to the dish dispenser"
    ]
    
    print("Testing pretrained language-conditioned policy:")
    print("=" * 50)
    
    for i, command in enumerate(test_commands):
        agent.set_command(command)
        action_idx = agent.action(state)
        action_name = agent.mdp.action_idx_to_name[action_idx]
        
        print(f"Command: '{command}'")
        print(f"Predicted action: {action_name} (index: {action_idx})")
        print("-" * 30)

if __name__ == "__main__":
    main() 