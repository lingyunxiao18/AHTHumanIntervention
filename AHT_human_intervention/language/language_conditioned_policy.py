import torch
import torch.nn as nn

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
    lines.append("=== Environment ===")
    layout_name = getattr(state.mdp, 'layout_name', state.mdp.__class__.__name__)
    lines.append(f"Kitchen layout: {layout_name}")

    # Pots and cooking status
    if hasattr(state, 'pot_positions') and hasattr(state, 'pot_cooking_tick_left'):
        for i, pos in enumerate(state.pot_positions):
            cooking = state.pot_cooking[i] if state.pot_cooking[i] is not None else 'nothing'
            time_left = state.pot_cooking_tick_left[i]
            lines.append(f"Pot {i} at {pos}, cooking {cooking}, time left {time_left}s")

    # Agents
    lines.append("=== Agents ===")
    for agent_idx in range(state.num_agents):
        pos = state.agent_positions[agent_idx]
        holding = state.holding[agent_idx] if state.holding[agent_idx] is not None else 'nothing'
        lines.append(f"Agent{agent_idx} at {pos}, holding {holding}")

    # Ingredient counters
    lines.append("=== Ingredient Counters ===")
    for counter_pos, counter_items in state.counters.items():
        items_str = ', '.join(counter_items) if counter_items else 'empty'
        lines.append(f"Counter at {counter_pos}: {items_str}")

    # Optional history/context
    history = getattr(state, 'history', [])
    if history:
        lines.append("=== History ===")
        for speaker, msg in history[-4:]:
            lines.append(f"{speaker}: {msg}")

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 2. Language-conditioned policy network
# ----------------------------------------------------------------------------
class LangConditionedPolicy(nn.Module):
    """
    π(a | s, ℓ): encodes state s and language ℓ to produce action logits.
    """
    def __init__(self,
                 state_dim: int,
                 vocab_size: int,
                 text_dim: int = 128,
                 hidden_dim: int = 256,
                 nhead: int = 4,
                 num_layers: int = 2,
                 max_len: int = MAX_LEN,
                 num_actions: int = 6):
        super().__init__()
        # State encoder: simple 2-layer MLP
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Text encoder: embedding + positional + Transformer
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.pos_embed   = nn.Embedding(max_len, text_dim)
        encoder_layer   = nn.TransformerEncoderLayer(d_model=text_dim, nhead=nhead)
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Fusion & policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, state_vec: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        # state_vec: [B, state_dim]
        # token_ids: [B, T]
        s_emb = self.state_encoder(state_vec)  # [B, hidden_dim]

        # text encoding
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
        t = self.token_embed(token_ids) + self.pos_embed(positions)  # [B, T, text_dim]
        t = t.transpose(0,1)                                        # [T, B, text_dim]
        t_enc = self.text_encoder(t)                                # [T, B, text_dim]
        l_emb = t_enc.mean(dim=0)                                   # [B, text_dim]

        # fuse and predict
        x = torch.cat([s_emb, l_emb], dim=-1)                    # [B, hidden+text]
        logits = self.policy_head(x)                             # [B, num_actions]
        return logits


# ----------------------------------------------------------------------------
# 3. Tokenizer
# ----------------------------------------------------------------------------
def tokenize(text: str, vocab: dict, max_len: int = MAX_LEN) -> torch.LongTensor:
    tokens = text.lower().split()[:max_len]
    ids = [vocab.get(tok, vocab.get("<unk>")) for tok in tokens]
    # pad if shorter than max_len
    ids += [vocab.get("<pad>")] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ----------------------------------------------------------------------------
# 4. Vocabulary placeholder (populate with your tokens)
# ----------------------------------------------------------------------------
VOCAB = {
    "<pad>": 0,
    "<unk>": 1,
}
