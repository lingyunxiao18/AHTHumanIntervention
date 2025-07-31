import torch
from AHT_human_intervention.language.shared_lang_agent import SharedLangAgent

# Dummy/mock MDP for demonstration
class DummyMDP:
    def __init__(self):
        self.action_idx_to_name = {0: 'stay', 1: 'up', 2: 'down', 3: 'left', 4: 'right', 5: 'interact'}
        self.action_name_to_idx = {v: k for k, v in self.action_idx_to_name.items()}
    def lossless_state_encoding(self, state):
        return [torch.zeros(10) for _ in range(2)]
    def get_standard_start_state(self):
        return None

if __name__ == "__main__":
    mdp = DummyMDP()
    # Create agent and (optionally) train policy head here
    agent = SharedLangAgent(mdp, agent_idx=0)
    # Save the policy head
    torch.save(agent.model.policy_head.state_dict(), "policy_head.pt")
    print("Policy head state_dict saved to policy_head.pt")
    # Load the policy head into a new agent
    agent2 = SharedLangAgent(mdp, agent_idx=0, model_path="policy_head.pt")
    print("Policy head state_dict loaded into new agent.") 