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
    try:
        lines = []
        
        # Environment header - try to get layout name from various sources
        layout_name = "unknown"
        if hasattr(state, 'mdp') and hasattr(state.mdp, 'layout_name'):
            layout_name = state.mdp.layout_name
        elif hasattr(state, 'layout_name'):
            layout_name = state.layout_name
        lines.append(f"Kitchen layout: {layout_name}")

        # Player positions and orientations
        if hasattr(state, 'players') and state.players:
            for i, player in enumerate(state.players):
                pos = player.position
                orientation = player.orientation
                holding = player.get_object().name if player.has_object() else 'nothing'
                lines.append(f"Agent{i} at {pos}, facing {orientation}, holding {holding}")

        # Objects in the environment
        if hasattr(state, 'objects') and state.objects:
            objects_by_type = {}
            for obj in state.objects:
                obj_type = obj.name
                if obj_type not in objects_by_type:
                    objects_by_type[obj_type] = []
                objects_by_type[obj_type].append(f"{obj_type} at {obj.position}")
            
            for obj_type, locations in objects_by_type.items():
                lines.append(f"{obj_type.capitalize()}: {', '.join(locations)}")

        # Order information
        if hasattr(state, 'order_list') and state.order_list:
            lines.append(f"Orders: {state.order_list}")

        # Timestep
        if hasattr(state, 'timestep'):
            lines.append(f"Timestep: {state.timestep}")

        # Optional history/context
        history = getattr(state, 'history', [])
        if history:
            for speaker, msg in history[-4:]:
                lines.append(f"{speaker}: {msg}")

        return "\n".join(lines)
    except Exception as e:
        print(f"[ERROR] Error building environment prompt: {e}")
        return "Kitchen layout: unknown\nAgent0 at (0,0), facing (0,0), holding nothing"

# ----------------------------------------------------------------------------
# 2. HuggingFace-based Language-conditioned policy network
# ----------------------------------------------------------------------------
class HuggingFaceLangConditionedPolicy(nn.Module):
    """
    π(a | s, ℓ): encodes state s and language ℓ to produce action logits.
    Uses DistilBERT for language encoding.
    """
    def __init__(self, state_dim: int, num_actions: int, text_dim: int = 768, hidden_dim: int = 256, freeze_bert: bool = True):
        try:
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
        except Exception as e:
            print(f"[ERROR] Error initializing HuggingFaceLangConditionedPolicy: {e}")
            raise

    def forward(self, state_vec: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # state_vec: [B, state_dim]
        # input_ids, attention_mask: [B, T]
        try:
            s_emb = self.state_encoder(state_vec)  # [B, hidden_dim]
            bert_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            l_emb = bert_out.last_hidden_state[:, 0, :]  # [B, text_dim] (CLS token)
            x = torch.cat([s_emb, l_emb], dim=-1)  # [B, hidden+text]
            logits = self.policy_head(x)  # [B, num_actions]
            return logits
        except Exception as e:
            print(f"[ERROR] Error in HuggingFaceLangConditionedPolicy.forward(): {e}")
            # Return zero logits as fallback
            batch_size = state_vec.shape[0] if state_vec.dim() > 0 else 1
            num_actions = self.policy_head[-1].out_features
            return torch.zeros(batch_size, num_actions, device=state_vec.device)

# ----------------------------------------------------------------------------
# 3. Tokenizer helper
# ----------------------------------------------------------------------------
def get_hf_tokenizer():
    """Returns a DistilBERT tokenizer (singleton pattern)."""
    try:
        return DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    except Exception as e:
        print(f"[ERROR] Error loading DistilBERT tokenizer: {e}")
        raise
