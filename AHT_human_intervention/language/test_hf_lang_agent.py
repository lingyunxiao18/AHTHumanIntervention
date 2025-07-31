import torch
from AHT_human_intervention.language.shared_lang_agent import SharedLangAgent

# Dummy/mock MDP and state for testing
class DummyMDP:
    def __init__(self):
        self.action_idx_to_name = {0: 'stay', 1: 'up', 2: 'down', 3: 'left', 4: 'right', 5: 'interact'}
        self.action_name_to_idx = {v: k for k, v in self.action_idx_to_name.items()}
    def lossless_state_encoding(self, state):
        # Return a list of dummy state vectors (one per agent)
        return [torch.zeros(10) for _ in range(2)]
    def get_standard_start_state(self):
        return None

class DummyState:
    def __init__(self):
        self.mdp = DummyMDP()
        self.num_agents = 2
        self.agent_positions = [(0,0), (1,1)]
        self.holding = [None, None]
        self.pot_positions = [(2,2)]
        self.pot_cooking = [None]
        self.pot_cooking_tick_left = [0]
        self.counters = {(3,3): []}
        self.history = []

if __name__ == "__main__":
    # Replace DummyMDP/DummyState with your real OvercookedGridworld and state for real tests
    mdp = DummyMDP()
    state = DummyState()
    agent = SharedLangAgent(mdp, agent_idx=0)
    agent.set_command("Help me deliver the soup quickly!")
    action_idx = agent.act(state)
    print(f"Action index returned: {action_idx}")
    print("✅ SharedLangAgent HuggingFace test completed successfully.") 