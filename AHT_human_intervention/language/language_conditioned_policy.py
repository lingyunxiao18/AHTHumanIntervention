import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast
from state_to_text import describe_state

# NOTE: Requires 'transformers' and 'torch' packages. Install with:
# pip install transformers torch

# Maximum token sequence length for language inputs
MAX_LEN = 128  # Increased for longer text descriptions

# ----------------------------------------------------------------------------
# 1. Text-based state encoder
# ----------------------------------------------------------------------------
def build_text_state_description(mdp, state, description_type: str = "english") -> str:
    """
    Convert an Overcooked state into a text description using state_to_text module.
    
    Args:
        mdp: The Overcooked MDP object
        state: The current state
        description_type: 'ctx', 'english', or 'both'
    
    Returns:
        Text description of the state
    """
    try:
        return describe_state(mdp, state, mode=description_type)
    except Exception as e:
        print(f"[ERROR] Error building text state description: {e}")
        return "Kitchen state: unknown"

def build_combined_text_input(mdp, state, human_command: str = "", description_type: str = "english") -> str:
    """
    Combine state description with human command into a single text input.
    
    Args:
        mdp: The Overcooked MDP object
        state: The current state
        human_command: Human command/instruction (can be empty string)
        description_type: Type of state description to use
    
    Returns:
        Combined text input for the policy network
    """
    state_desc = build_text_state_description(mdp, state, description_type)
    
    if human_command.strip():
        # Combine state description with human command
        combined_text = f"State: {state_desc}\nCommand: {human_command}"
    else:
        # Just use state description if no command
        combined_text = f"State: {state_desc}"
    
    return combined_text

# ----------------------------------------------------------------------------
# 2. Text-based Language-conditioned policy network
# ----------------------------------------------------------------------------
class TextBasedLangConditionedPolicy(nn.Module):
    """
    π(a | s_text, ℓ): encodes state as text s_text and language command ℓ to produce action logits.
    Uses DistilBERT for encoding both state description and human commands.
    """
    def __init__(self, num_actions: int, text_dim: int = 768, hidden_dim: int = 256, 
                 freeze_bert: bool = False, description_type: str = "english"):
        try:
            super().__init__()
            
            # Text encoder: DistilBERT for encoding combined text input
            self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
            if freeze_bert:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            
            # Tokenizer for text processing
            self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            
            # Store description type
            self.description_type = description_type
            
            # Policy head: maps text embeddings to action logits
            self.policy_head = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_actions)
            )
            
            print(f"[INFO] Text-based policy initialized with {num_actions} actions")
            print(f"[INFO] Using {description_type} state descriptions")
            print(f"[INFO] BERT frozen: {freeze_bert}")
            
        except Exception as e:
            print(f"[ERROR] Error initializing TextBasedLangConditionedPolicy: {e}")
            raise

    def encode_text(self, text_inputs: list) -> tuple:
        """
        Encode a list of text inputs using DistilBERT tokenizer.
        
        Args:
            text_inputs: List of text strings to encode
            
        Returns:
            Tuple of (input_ids, attention_mask) tensors
        """
        try:
            # Tokenize the text inputs
            encoded = self.tokenizer(
                text_inputs,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors='pt'
            )
            
            return encoded['input_ids'], encoded['attention_mask']
            
        except Exception as e:
            print(f"[ERROR] Error encoding text: {e}")
            # Return dummy tensors as fallback
            batch_size = len(text_inputs)
            dummy_ids = torch.zeros(batch_size, MAX_LEN, dtype=torch.long)
            dummy_mask = torch.zeros(batch_size, MAX_LEN, dtype=torch.long)
            return dummy_ids, dummy_mask

    def forward(self, mdp, states, human_commands: list = None) -> torch.Tensor:
        """
        Forward pass through the text-based policy network.
        
        Args:
            mdp: The Overcooked MDP object
            states: List of state objects or state text strings
            human_commands: List of human commands (can be None or empty strings)
            
        Returns:
            Action logits tensor [batch_size, num_actions]
        """
        try:
            batch_size = len(states)
            
            # Handle human commands
            if human_commands is None:
                human_commands = [""] * batch_size
            
            # Build combined text inputs
            text_inputs = []
            for i, state in enumerate(states):
                command = human_commands[i] if i < len(human_commands) else ""
                
                # Check if state is already a string (pre-generated state text)
                if isinstance(state, str):
                    state_desc = state
                else:
                    # Generate state description from state object
                    state_desc = build_text_state_description(mdp, state, self.description_type)
                
                # Combine state description with command
                if command.strip():
                    combined_text = f"State: {state_desc}\nCommand: {command}"
                else:
                    combined_text = f"State: {state_desc}"
                
                text_inputs.append(combined_text)
            
            # Encode text inputs
            input_ids, attention_mask = self.encode_text(text_inputs)
            
            # Move to same device as model
            device = next(self.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Get BERT embeddings
            bert_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_embeddings = bert_output.last_hidden_state[:, 0, :]  # Use CLS token [B, text_dim]
            
            # Pass through policy head
            logits = self.policy_head(text_embeddings)  # [B, num_actions]
            
            return logits
            
        except Exception as e:
            print(f"[ERROR] Error in TextBasedLangConditionedPolicy.forward(): {e}")
            # Return zero logits as fallback
            batch_size = len(states) if states else 1
            num_actions = self.policy_head[-1].out_features
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            return torch.zeros(batch_size, num_actions, device=device)

    def get_action_probs(self, mdp, states, human_commands: list = None) -> torch.Tensor:
        """
        Get action probabilities from the policy.
        
        Args:
            mdp: The Overcooked MDP object
            states: List of state objects
            human_commands: List of human commands
            
        Returns:
            Action probabilities tensor [batch_size, num_actions]
        """
        logits = self.forward(mdp, states, human_commands)
        return torch.softmax(logits, dim=-1)

    def get_action(self, mdp, state, human_command: str = "", temperature: float = 1.0) -> int:
        """
        Get a single action from the policy.
        
        Args:
            mdp: The Overcooked MDP object
            state: Single state object
            human_command: Human command string
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Action index
        """
        probs = self.get_action_probs(mdp, [state], [human_command])
        
        if temperature == 0:
            # Greedy action
            return probs.argmax(dim=-1).item()
        else:
            # Sample with temperature
            scaled_probs = (probs / temperature).softmax(dim=-1)
            return torch.multinomial(scaled_probs, 1).item()

# ----------------------------------------------------------------------------
# 3. Legacy compatibility - keep the old class for backward compatibility
# ----------------------------------------------------------------------------
class HuggingFaceLangConditionedPolicy(nn.Module):
    """
    Legacy policy network that uses matrix states + text commands.
    Kept for backward compatibility.
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
            
            # Calculate actual input dimension for policy head
            actual_state_dim = hidden_dim  # This is what state_encoder outputs
            actual_text_dim = text_dim     # This is what BERT outputs (768)
            policy_input_dim = actual_state_dim + actual_text_dim
            
            print(f"[INFO] Policy head input dimension: {policy_input_dim} (state: {actual_state_dim} + text: {actual_text_dim})")
            
            # Fusion & policy head
            self.policy_head = nn.Sequential(
                nn.Linear(policy_input_dim, hidden_dim),
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
            # Ensure state_vec requires gradients
            if not state_vec.requires_grad:
                state_vec = state_vec.detach().requires_grad_(True)
            
            s_emb = self.state_encoder(state_vec)  # [B, hidden_dim]
            
            # Get BERT output and extract the CLS token representation
            bert_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            l_emb = bert_out.last_hidden_state[:, 0, :]  # [B, text_dim] (CLS token)
            
            # Concatenate state and text embeddings
            x = torch.cat([s_emb, l_emb], dim=-1)  # [B, hidden_dim + text_dim]
            
            # Pass through policy head
            logits = self.policy_head(x)  # [B, num_actions]
            
            return logits
            
        except Exception as e:
            print(f"[ERROR] Error in HuggingFaceLangConditionedPolicy.forward(): {e}")
            # Return zero logits as fallback
            batch_size = state_vec.shape[0] if state_vec.dim() > 0 else 1
            num_actions = self.policy_head[-1].out_features
            return torch.zeros(batch_size, num_actions, device=state_vec.device, requires_grad=True)

# ----------------------------------------------------------------------------
# 4. Tokenizer helper
# ----------------------------------------------------------------------------
def get_hf_tokenizer():
    """Returns a DistilBERT tokenizer (singleton pattern)."""
    try:
        return DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    except Exception as e:
        print(f"[ERROR] Error loading DistilBERT tokenizer: {e}")
        raise
