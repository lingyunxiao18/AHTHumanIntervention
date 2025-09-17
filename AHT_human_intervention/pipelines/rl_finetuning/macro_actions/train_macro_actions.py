#!/usr/bin/env python3
"""
Training script for macro action policy.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
import time
import os
from typing import List, Tuple, Dict, Any

from pipelines.rl_finetuning.macro_actions.macro_actions import MacroAction, create_macro_action_policy
from shared.utils.state_to_text import describe_state

# ----------------------------------------------------------------------------
# Macro Action Dataset
# ----------------------------------------------------------------------------

class MacroActionDataset(Dataset):
    """Dataset for training macro action policy."""
    
    def __init__(self, mdp, data: List[dict]):
        self.mdp = mdp
        self.data = data
        print(f"[INFO] Created macro action dataset with {len(data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'state_text': item['state_text'],
            'command': item.get('human_command', ''),
            'macro_action': item['macro_action'],
            'primitive_action': item['primitive_action']
        }

def create_macro_action_training_data():
    """Create training data for macro actions."""
    print("🎯 CREATING MACRO ACTION TRAINING DATA")
    print("=" * 60)
    
    # Load existing micro-task data
    with open("generated_data/micro_task_training/policy_training_data.json", "r") as f:
        micro_task_data = json.load(f)
    
    with open("generated_data/micro_task_training/policy_training_data_no_commands.json", "r") as f:
        no_commands_data = json.load(f)
    
    all_micro_data = micro_task_data + no_commands_data
    print(f"Loaded {len(all_micro_data)} micro-task samples")
    
    # Create MDP for state conversion
    from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    mdp = OvercookedGridworld.from_layout_name("random3")
    
    # Create macro action policy
    macro_policy = create_macro_action_policy(mdp)
    
    # Generate macro action training data
    macro_training_data = []
    
    # Define macro action commands and their indices
    macro_commands = [
        "go to onion dispenser",
        "take onion",
        "go to pot",
        "put onion in pot",
        "go to dish dispenser", 
        "take dish",
        "go to serving area",
        "serve soup",
        "wait for cooking",
        "stay"
    ]
    
    # Create mapping from macro action names to indices
    macro_action_to_idx = {
        "GO_TO_ONION": 0,
        "TAKE_ONION": 1,
        "GO_TO_POT": 2,
        "PUT_IN_POT": 3,
        "GO_TO_DISH": 4,
        "TAKE_DISH": 5,
        "GO_TO_SERVE": 6,
        "SERVE": 7,
        "WAIT_COOK": 8,
        "STAY": 9
    }
    
    print("Generating macro action samples...")
    
    for i, micro_sample in enumerate(all_micro_data):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(all_micro_data)} samples")
        
        # For each micro-task sample, create multiple macro action samples
        for command in macro_commands:
            # Get macro action from command
            macro_action, args = macro_policy._parse_command(command)
            
            # Convert macro action to index
            macro_action_idx = macro_action_to_idx.get(macro_action.value, 9)  # Default to STAY
            
            # Create macro training sample
            macro_sample = {
                'state_text': micro_sample['state_text'],
                'human_command': command,
                'macro_action': macro_action_idx,  # Use integer index
                'primitive_action': micro_sample['action']  # Use original action as target
            }
            
            macro_training_data.append(macro_sample)
    
    print(f"Generated {len(macro_training_data)} macro action samples")
    
    # Save macro training data
    os.makedirs("generated_data/macro_action_training", exist_ok=True)
    
    with open("generated_data/macro_action_training/macro_training_data.json", "w") as f:
        json.dump(macro_training_data, f, indent=2)
    
    print(f"Saved macro training data to generated_data/macro_action_training/")
    
    return macro_training_data

# ----------------------------------------------------------------------------
# Macro Action Policy Network
# ----------------------------------------------------------------------------

class MacroActionPolicyNetwork(nn.Module):
    """Neural network for macro action policy."""
    
    def __init__(self, num_macro_actions=10, text_dim=256, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        self.num_macro_actions = num_macro_actions
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Text encoder (using BERT)
        from transformers import DistilBertModel
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze BERT
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(768, text_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(text_dim)
        )
        
        # Macro action head
        self.macro_head = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_macro_actions)
        )
        
        # Initialize weights
        self._init_weights()
        
        print(f"[INFO] Macro action policy initialized with {num_macro_actions} macro actions")
        print(f"[INFO] Text dim: {text_dim}, Hidden dim: {hidden_dim}")
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, texts, commands=None):
        """Forward pass."""
        # BERT encoding
        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        
        # Tokenize
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        
        # BERT forward pass
        with torch.no_grad():
            bert_output = self.text_encoder(**inputs)
            text_features = bert_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Text projection
        text_features = self.text_projection(text_features)
        
        # Macro action head
        macro_logits = self.macro_head(text_features)
        
        return macro_logits
    
    def get_macro_action(self, texts, commands=None, temperature=0.0):
        """Get macro action prediction."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(texts, commands)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
            else:
                action = logits.argmax(dim=-1)
        return action.item() if action.dim() == 0 else action.cpu().numpy()

# ----------------------------------------------------------------------------
# Training Functions
# ----------------------------------------------------------------------------

def custom_collate_fn(batch):
    """Custom collate function for macro action data."""
    state_texts = [item['state_text'] for item in batch]
    commands = [item['command'] for item in batch]
    macro_actions = torch.tensor([item['macro_action'] for item in batch], dtype=torch.long)
    primitive_actions = torch.tensor([item['primitive_action'] for item in batch], dtype=torch.long)
    
    return {
        'state_text': state_texts,
        'command': commands,
        'macro_action': macro_actions,
        'primitive_action': primitive_actions
    }

def train_macro_action_policy(policy: nn.Module, 
                             train_loader: DataLoader,
                             val_loader: DataLoader = None,
                             num_epochs: int = 3,
                             learning_rate: float = 1e-5,
                             device: str = "cpu",
                             output_dir: str = None):
    """Train the macro action policy network."""
    
    # Move model to device
    policy = policy.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"Training macro action policy:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")
    
    # Training loop
    for epoch in range(num_epochs):
        policy.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            state_texts = batch['state_text']
            commands = batch['command']
            target_macro_actions = batch['macro_action'].to(device)
            
            # Forward pass
            macro_logits = policy.forward(state_texts, commands)
            
            # Compute loss
            loss = criterion(macro_logits, target_macro_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Compute accuracy
            predictions = macro_logits.argmax(dim=-1)
            correct = (predictions == target_macro_actions).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += target_macro_actions.size(0)
            
            # Print progress
            if batch_idx % 50 == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = total_correct / total_samples
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                      f"Loss: {current_loss:.4f}, Acc: {current_acc:.3f}")
        
        # End of epoch
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.3f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if output_dir:
            epoch_path = os.path.join(output_dir, f"macro_epoch_{epoch+1}.pt")
            torch.save({
                'model': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': avg_acc
            }, epoch_path)
            print(f"[epoch] saved -> {epoch_path}")

# ----------------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------------

def main():
    """Main training function for macro actions."""
    print("=== Macro Action Policy Training ===\n")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create macro action training data
    print("1. Creating macro action training data...")
    macro_training_data = create_macro_action_training_data()
    
    # Split data
    random.shuffle(macro_training_data)
    split_idx = int(0.8 * len(macro_training_data))
    train_data = macro_training_data[:split_idx]
    val_data = macro_training_data[split_idx:]
    
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Validation samples: {len(val_data)}")
    
    # Create MDP
    from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    mdp = OvercookedGridworld.from_layout_name("random3")
    
    # Create datasets
    train_dataset = MacroActionDataset(mdp, train_data)
    val_dataset = MacroActionDataset(mdp, val_data)
    
    # Create macro action policy network
    print("2. Creating macro action policy network...")
    policy = MacroActionPolicyNetwork(num_macro_actions=10)
    
    print(f"   - Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"   - Using Apple M4 GPU (MPS)")
    else:
        device = "cpu"
        print(f"   - Using CPU")
    
    # Create output directory
    output_dir = "trained_policies/macro_action_policy"
    os.makedirs(output_dir, exist_ok=True)
    print(f"   - Policy will be saved to: {output_dir}")
    
    # Create data loaders
    batch_size = 16
    num_workers = 0 if device == "mps" else 2
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"   - Batch size: {batch_size}")
    print(f"   - Workers: {num_workers}")
    
    # Train the policy
    print("3. Starting macro action training...")
    train_macro_action_policy(
        policy=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,
        learning_rate=1e-5,
        device=device,
        output_dir=output_dir
    )
    
    print("\n=== Macro Action Training Complete ===")

if __name__ == "__main__":
    main()
