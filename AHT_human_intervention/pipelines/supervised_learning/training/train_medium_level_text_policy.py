#!/usr/bin/env python3
"""
Training script for the medium-level text-based policy network.
This demonstrates how to train the MediumLevelTextBasedPolicy using
state descriptions and human commands to output medium-level actions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
from typing import List, Tuple, Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))


from language_conditioned_policy import TextBasedLangConditionedPolicy
from state_to_text import describe_state
from shared.envs.envs.overcooked.overcooked_ai_py.planning.planners import MediumLevelActionManager

class MediumLevelTextPolicyDataset(Dataset):
    """
    Dataset for training the medium-level text-based policy network.
    Contains (state_description, human_command, medium_level_action) tuples.
    """
    
    def __init__(self, mdp, data: List[dict]):
        """
        Args:
            mdp: The Overcooked MDP object
            data: List of dictionaries containing 'state_text', 'command', 'medium_level_action'
        """
        self.mdp = mdp
        self.data = data
        
        print(f"[INFO] Created medium-level dataset with {len(data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'state_text': item['state_text'],
            'command': item['human_command'],
            'medium_level_action': item['medium_level_action']
        }

class MediumLevelTextBasedPolicy(TextBasedLangConditionedPolicy):
    """
    Medium-level text-based policy that outputs medium-level action indices.
    Inherits from TextBasedLangConditionedPolicy but adapts for medium-level actions.
    """
    
    def __init__(self, mdp, text_dim: int = 768, hidden_dim: int = 256, 
                 freeze_bert: bool = False, description_type: str = "english"):
        # Initialize medium-level action manager
        ml_params = {
            "wait_allowed": True,
            "counter_drop": mdp.get_counter_locations(),
            "counter_pickup": mdp.get_counter_locations(),
            "same_motion_goals": True,
            "start_orientations": False,
            "counter_goals": mdp.get_counter_locations()
        }
        self.ml_action_manager = MediumLevelActionManager(mdp, ml_params)
        
        # Estimate number of medium-level actions (will be dynamic)
        # We'll use a reasonable upper bound
        estimated_num_actions = 50  # This will be updated dynamically
        
        super().__init__(
            num_actions=estimated_num_actions,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            freeze_bert=freeze_bert,
            description_type=description_type
        )
        
        self.mdp = mdp
        print(f"[INFO] Medium-level policy initialized with estimated {estimated_num_actions} actions")
    
    def get_available_actions(self, state, player_idx: int = 0) -> List[Tuple]:
        """Get available medium-level actions for the current state and player."""
        player = state.players[player_idx]
        return self.ml_action_manager.get_medium_level_actions(state, player)
    
    def forward(self, mdp, states, human_commands: list = None) -> torch.Tensor:
        """
        Forward pass through the medium-level text-based policy network.
        
        Args:
            mdp: The Overcooked MDP object
            states: List of state objects or state text strings
            human_commands: List of human commands (can be None or empty strings)
            
        Returns:
            Action logits tensor [batch_size, max_num_actions]
        """
        try:
            batch_size = len(states)
            
            # Handle human commands
            if human_commands is None:
                human_commands = [""] * batch_size
            
            # Build combined text inputs
            text_inputs = []
            max_num_actions = 0
            
            for i, state in enumerate(states):
                command = human_commands[i] if i < len(human_commands) else ""
                
                # Check if state is already a string (pre-generated state text)
                if isinstance(state, str):
                    state_desc = state
                else:
                    # Generate state description from state object
                    state_desc = describe_state(mdp, state, self.description_type)
                
                # Combine state description with command
                if command.strip():
                    combined_text = f"State: {state_desc}\nCommand: {command}"
                else:
                    combined_text = f"State: {state_desc}"
                
                text_inputs.append(combined_text)
                
                # Get number of available actions for this state
                if not isinstance(state, str):
                    available_actions = self.get_available_actions(state, 0)  # Assume player 0
                    max_num_actions = max(max_num_actions, len(available_actions))
            
            # Update policy head if needed
            if max_num_actions > self.policy_head[-1].out_features:
                print(f"[INFO] Updating policy head to {max_num_actions} actions")
                self.policy_head = nn.Sequential(
                    nn.Linear(self.text_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.hidden_dim, max_num_actions)
                )
            
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
            logits = self.policy_head(text_embeddings)  # [B, max_num_actions]
            
            return logits
            
        except Exception as e:
            print(f"[ERROR] Error in MediumLevelTextBasedPolicy.forward(): {e}")
            # Return zero logits as fallback
            batch_size = len(states) if states else 1
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
            return torch.zeros(batch_size, 50, device=device)  # Default to 50 actions
    
    def get_action(self, mdp, state, human_command: str = "", temperature: float = 1.0, player_idx: int = 0) -> int:
        """
        Get a single medium-level action from the policy.
        
        Args:
            mdp: The Overcooked MDP object
            state: Single state object
            human_command: Human command string
            temperature: Sampling temperature (higher = more random)
            player_idx: Player index (0 or 1)
            
        Returns:
            Medium-level action index
        """
        # Get available actions for this state
        available_actions = self.get_available_actions(state, player_idx)
        
        if len(available_actions) == 0:
            print("[WARNING] No available medium-level actions, returning 0")
            return 0
        
        # Get action probabilities
        probs = self.get_action_probs(mdp, [state], [human_command])
        
        # Mask probabilities to only available actions
        masked_probs = torch.zeros_like(probs)
        masked_probs[0, :len(available_actions)] = probs[0, :len(available_actions)]
        
        # Normalize
        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
        
        if temperature == 0:
            # Greedy action
            return masked_probs.argmax(dim=-1).item()
        else:
            # Sample with temperature
            scaled_probs = (masked_probs / temperature).softmax(dim=-1)
            return torch.multinomial(scaled_probs, 1).item()

def create_synthetic_medium_level_training_data(num_samples: int = 1000) -> List[dict]:
    """
    Create synthetic training data for medium-level actions.
    In practice, you would use real trajectory data from your environment.
    
    Args:
        num_samples: Number of training samples to generate
        
    Returns:
        List of training data dictionaries
    """
    data = []
    
    # Define some common commands and their associated medium-level actions
    command_action_pairs = [
        ("", 0),  # No command -> WAIT
        ("pick up onion", 1),  # PICKUP_ONION
        ("grab dish", 2),  # PICKUP_DISH
        ("cook soup", 3),  # PUT_ONION_IN_POT
        ("deliver soup", 4),  # DELIVER_SOUP
        ("place on counter", 5),  # PLACE_ON_COUNTER
        ("wait", 0),  # WAIT
    ]
    
    for i in range(num_samples):
        # Create a dummy state description
        state_text = f"Kitchen state {i}: Players at positions, pots cooking, objects on counters"
        
        # Randomly select a command-action pair
        command, action = random.choice(command_action_pairs)
        
        data.append({
            'state_text': state_text,
            'human_command': command,
            'medium_level_action': action
        })
    
    return data

def custom_medium_level_collate_fn(batch):
    """Custom collate function to handle medium-level text-based data."""
    state_texts = [item['state_text'] for item in batch]
    commands = [item['command'] for item in batch]
    medium_level_actions = torch.tensor([item['medium_level_action'] for item in batch], dtype=torch.long)
    
    return {
        'state_text': state_texts,
        'command': commands,
        'medium_level_action': medium_level_actions
    }

def train_medium_level_text_policy(policy: MediumLevelTextBasedPolicy, 
                                 train_loader: DataLoader,
                                 val_loader: DataLoader = None,
                                 num_epochs: int = 2,
                                 learning_rate: float = 1e-4,
                                 device: str = "cpu",
                                 output_dir: str = None):
    """
    Train the medium-level text-based policy network.
    
    Args:
        policy: The policy network to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on
    """
    
    # Move policy to device
    policy = policy.to(device)
    
    # Setup optimizer and loss function
    optimizer = optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        policy.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(train_loader):
            state_texts = batch['state_text']
            commands = batch['command']
            target_actions = batch['medium_level_action'].to(device)
            
            # Forward pass
            logits = policy.forward(policy.mdp, state_texts, commands)
            
            # Calculate loss (only for valid actions)
            max_target = target_actions.max().item()
            if max_target < logits.shape[1]:
                loss = criterion(logits[:, :max_target+1], target_actions)
            else:
                # Pad logits if needed
                padded_logits = torch.zeros(logits.shape[0], max_target+1, device=device)
                padded_logits[:, :logits.shape[1]] = logits
                loss = criterion(padded_logits, target_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct_predictions += (predictions == target_actions).sum().item()
            total_predictions += target_actions.size(0)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {correct_predictions/total_predictions:.3f}")
        
        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Avg Loss: {avg_loss:.4f}, "
              f"Accuracy: {accuracy:.3f}")
        
        # Validation (if provided)
        if val_loader is not None:
            policy.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    state_texts = batch['state_text']
                    commands = batch['command']
                    target_actions = batch['medium_level_action'].to(device)
                    
                    logits = policy.forward(policy.mdp, state_texts, commands)
                    
                    # Calculate loss (only for valid actions)
                    max_target = target_actions.max().item()
                    if max_target < logits.shape[1]:
                        loss = criterion(logits[:, :max_target+1], target_actions)
                    else:
                        # Pad logits if needed
                        padded_logits = torch.zeros(logits.shape[0], max_target+1, device=device)
                        padded_logits[:, :logits.shape[1]] = logits
                        loss = criterion(padded_logits, target_actions)
                    
                    val_loss += loss.item()
                    predictions = logits.argmax(dim=-1)
                    val_correct += (predictions == target_actions).sum().item()
                    val_total += target_actions.size(0)
            
            val_avg_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            print(f"Validation - Loss: {val_avg_loss:.4f}, Accuracy: {val_accuracy:.3f}")
    
    print("Training completed!")
    
    # Save the trained policy
    if output_dir:
        import os
        policy_path = os.path.join(output_dir, "medium_level_text_policy.pt")
        torch.save({
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': num_epochs,
            'loss': val_avg_loss if val_loader else total_loss / len(train_loader),
            'accuracy': val_accuracy if val_loader else correct_predictions / total_predictions,
            'mdp': policy.mdp,
            'text_dim': policy.text_dim,
            'hidden_dim': policy.hidden_dim
        }, policy_path)
        print(f"Policy saved to: {policy_path}")

def main():
    """Main training function."""
    print("=== Medium-Level Text-Based Policy Training ===\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load real training data
    print("1. Loading real training data...")
    
    # Load data with commands
    with open("generated_data/trajectories_medium_level_with_commands/policy_training_data.json", "r") as f:
        data_with_commands = json.load(f)
    print(f"   - Loaded {len(data_with_commands)} samples with commands")
    
    # Load data without commands
    with open("generated_data/trajectories_medium_level_no_commands/policy_training_data.json", "r") as f:
        data_without_commands = json.load(f)
    print(f"   - Loaded {len(data_without_commands)} samples without commands")
    
    # Combine datasets
    all_data = data_with_commands + data_without_commands
    print(f"   - Total samples: {len(all_data)}")
    
    # Split into train/val (80/20)
    random.shuffle(all_data)
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Validation samples: {len(val_data)}")
    
    # Create MDP for the policy
    from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedGridworld
    mdp = OvercookedGridworld.from_layout_name("random3")
    
    # Create datasets
    train_dataset = MediumLevelTextPolicyDataset(mdp, train_data)
    val_dataset = MediumLevelTextPolicyDataset(mdp, val_data)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_medium_level_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_medium_level_collate_fn)
    
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print()
    
    # Initialize medium-level policy network
    print("2. Initializing medium-level policy network...")
    policy = MediumLevelTextBasedPolicy(
        mdp=mdp,
        text_dim=768,
        hidden_dim=256,
        freeze_bert=False,  # Allow fine-tuning
        description_type="english"
    )
    
    print(f"   - Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print()
    
    # Train the policy
    print("3. Starting training...")
    
    # Device selection for Apple M4 GPU
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Metal Performance Shaders
        print(f"   - Using Apple M4 GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"   - Using NVIDIA GPU (CUDA)")
    else:
        device = "cpu"
        print(f"   - Using CPU")
    
    print(f"   - Device: {device}")
    
    # Create output directory for trained policy
    import os
    output_dir = "trained_policies/medium_level_text_policy"
    os.makedirs(output_dir, exist_ok=True)
    print(f"   - Policy will be saved to: {output_dir}")
    
    train_medium_level_text_policy(
        policy=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,
        learning_rate=1e-4,
        device=device,
        output_dir=output_dir
    )
    
    # Test the trained policy
    print("\n4. Testing trained policy...")
    policy.eval()
    
    test_commands = [
        "",
        "Pick up an onion",
        "Cook some soup",
        "Deliver the soup",
        "Wait here"
    ]
    
    # Create test state
    from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
    test_state = OvercookedState.from_player_positions(
        ((1, 1), (3, 3)), 
        order_list=["any"]
    )
    
    with torch.no_grad():
        for command in test_commands:
            action_idx = policy.get_action(mdp, test_state, command, temperature=0.0)
            available_actions = policy.get_available_actions(test_state, 0)
            if action_idx < len(available_actions):
                action_pos, action_orient = available_actions[action_idx]
                print(f"   Command: '{command}' -> Action: Go to {action_pos} facing {action_orient}")
            else:
                print(f"   Command: '{command}' -> Action: Invalid index {action_idx}")
    
    print("\n=== Medium-Level Training Complete ===")

if __name__ == "__main__":
    main()
