# test_smoke_lang_policy.py

import torch
import torch.nn.functional as F

from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from policy.language_conditioned_policy import LangConditionedPolicy, tokenize, build_env_prompt, vocab, max_len

if __name__ == "__main__":
    # 1. Instantiate the Overcooked environment (use any layout name you have)
    mdp = OvercookedEnv.from_layout_name("simple_layout")
    start_state = mdp.get_standard_start_state()

    # 2. Prepare state vector
    #    lossless_state_encoding returns a dict {agent_index: np.array}, so pick one agent (e.g. 0)
    state_array = mdp.lossless_state_encoding(start_state)[0].flatten()
    state_dim = state_array.size
    state_vec = torch.FloatTensor(state_array).unsqueeze(0)  # [1, state_dim]

    # 3. Instantiate your language-conditioned policy
    num_actions = len(mdp.action_idx_to_name)
    policy = LangConditionedPolicy(
        state_dim=state_dim,
        vocab_size=len(vocab),
        text_dim=128,
        hidden_dim=256,
        nhead=4,
        num_layers=2,
        max_len=max_len,
        num_actions=num_actions
    )

    # 4. Build prompt + tokenize with empty command
    cmd = ""
    prompt = build_env_prompt(start_state) + "\nHuman: " + cmd
    token_ids = tokenize(prompt)             # [max_len]
    token_ids = token_ids.unsqueeze(0)       # [1, max_len]

    # 5. Forward pass
    logits = policy(state_vec, token_ids)    # [1, num_actions]
    probs = F.softmax(logits, dim=-1)        # [1, num_actions]

    # 6. Assert valid distribution
    assert probs.shape == (1, num_actions), f"Expected shape (1, {num_actions}), got {probs.shape}"
    sum_prob = probs.sum().item()
    assert abs(sum_prob - 1.0) < 1e-5, f"Probabilities sum to {sum_prob}, not 1.0"

    print("✅ Smoke test passed: forward() returns a valid action distribution.")
    print("Action probabilities:", probs.detach().cpu().numpy())
