import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

# NOTE: Requires 'transformers' and 'torch' packages. Install with:
# pip install transformers torch

# Maximum token sequence length for language inputs
MAX_LEN = 50

# ----------------------------------------------------------------------------
# 1. Environment-to-text encoder
# ----------------------------------------------------------------------------
def build_env_prompt(state):
    """
    Convert an Overcooked state into a structured text prompt for LLM grounding.
    """
    lines = []
    # Environment header and static layout
    layout_name = getattr(state.mdp, 'layout_name', state.mdp.__class__.__name__)
    lines.append(f"Kitchen layout: {layout_name}")

    # Pots and cooking status
    if hasattr(state, 'pot_positions') and hasattr(state, 'pot_cooking_tick_left'):
        for i, pos in enumerate(state.pot_positions):
            cooking = state.pot_cooking[i] if state.pot_cooking[i] is not None else 'nothing'
            time_left = state.pot_cooking_tick_left[i]
            lines.append(f"Pot {i} at {pos}, cooking {cooking}, time left {time_left}s")

    # Agents
    for agent_idx in range(state.num_agents):
        pos = state.agent_positions[agent_idx]
        holding = state.holding[agent_idx] if state.holding[agent_idx] is not None else 'nothing'
        lines.append(f"Agent{agent_idx} at {pos}, holding {holding}")

    # Ingredient counters
    for counter_pos, counter_items in state.counters.items():
        items_str = ', '.join(counter_items) if counter_items else 'empty'
        lines.append(f"Counter at {counter_pos}: {items_str}")

    # Optional history/context
    history = getattr(state, 'history', [])
    if history:
        for speaker, msg in history[-4:]:
            lines.append(f"{speaker}: {msg}")

    return "\n".join(lines)

# ----------------------------------------------------------------------------
# 2. HuggingFace-based Language-conditioned policy network
# ----------------------------------------------------------------------------
class HuggingFaceLangConditionedPolicy(nn.Module):
    """
    π(a | s, ℓ): encodes state s and language ℓ to produce action logits.
    Uses DistilBERT for language encoding.
    """
    def __init__(self, state_dim: int, num_actions: int, text_dim: int = 768, hidden_dim: int = 256, freeze_bert: bool = True):
        super().__init__()
        # State encoder: simple 2-layer MLP
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Text encoder: DistilBERT
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        if freeze_bert:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        # Fusion & policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state_vec: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # state_vec: [B, state_dim]
        # input_ids, attention_mask: [B, T]
        s_emb = self.state_encoder(state_vec)  # [B, hidden_dim]
        bert_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        l_emb = bert_out.last_hidden_state[:, 0, :]  # [B, text_dim] (CLS token)
        x = torch.cat([s_emb, l_emb], dim=-1)  # [B, hidden+text]
        logits = self.policy_head(x)  # [B, num_actions]
        return logits

# ----------------------------------------------------------------------------
# 3. Tokenizer helper
# ----------------------------------------------------------------------------
def get_hf_tokenizer():
    """Returns a DistilBERT tokenizer (singleton pattern)."""
    return DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
