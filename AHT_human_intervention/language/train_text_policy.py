#!/usr/bin/env python3
"""
Training script for the text-based policy network.
This demonstrates how to train the TextBasedLangConditionedPolicy using
state descriptions and human commands instead of matrix states.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json
from typing import List, Tuple, Dict, Any

from language_conditioned_policy import TextBasedLangConditionedPolicy
from state_to_text import describe_state

class TextPolicyDataset(Dataset):
    """
    Dataset for training the text-based policy network.
    Contains (state_description, human_command, action) tuples.
    """
    
    def __init__(self, mdp, data: List[dict]):
        """
        Args:
            mdp: The Overcooked MDP object
            data: List of dictionaries containing 'state_text', 'command', 'action'
        """
        self.mdp = mdp
        self.data = data
        
        print(f"[INFO] Created dataset with {len(data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'state_text': item['state_text'],
            'command': item['human_command'],
            'action': item['action']
        }

def create_synthetic_training_data(num_samples: int = 1000) -> Tuple[List, List[str], List[int]]:
    """
    Create synthetic training data for demonstration purposes.
    In practice, you would use real trajectory data from your environment.
    
    Args:
        num_samples: Number of training samples to generate
        
    Returns:
        Tuple of (states, commands, actions)
    """
    states = []
    commands = []
    actions = []
    
    # Define some common commands and their associated actions
    command_action_pairs = [
        ("", 0),  # No command -> STAY
        ("move north", 1),  # MOVE_N
        ("move south", 2),  # MOVE_S
        ("move east", 3),   # MOVE_E
        ("move west", 4),   # MOVE_W
        ("interact", 5),    # INTERACT
        ("cook soup", 5),   # INTERACT
        ("pick up dish", 5), # INTERACT
        ("deliver soup", 5), # INTERACT
    ]
    
    for i in range(num_samples):
        # Create a dummy state (in practice, use real states from environment)
        state = create_dummy_state()
        states.append(state)
        
        # Randomly select a command-action pair
        command, action = random.choice(command_action_pairs)
        commands.append(command)
        actions.append(action)
    
    return states, commands, actions

def create_dummy_state():
    """Create a dummy state for synthetic data generation."""
    class DummyPlayer:
        def __init__(self, position, orientation, holding=None):
            self.position = position
            self.orientation = orientation
            self._holding = holding
        
        def has_object(self):
            return self._holding is not None
        
        def get_object(self):
            if self._holding:
                return DummyObject(self._holding)
            return None
    
    class DummyObject:
        def __init__(self, name):
            self.name = name
            self.position = (0, 0)
            self.state = ("onion", 2, 1)
    
    class DummyState:
        def __init__(self):
            self.players = [
                DummyPlayer((3, 2), (0, -1), "onion"),
                DummyPlayer((5, 1), (1, 0), None)
            ]
            self.objects = {
                (2, 3): DummyObject("soup"),
                (4, 4): DummyObject("dish"),
            }
    
    return DummyState()

def create_dummy_mdp():
    """Create a dummy MDP for synthetic data generation."""
    class DummyMDP:
        def __init__(self):
            self.layout_name = "simple_layout"
        
        def get_pot_locations(self):
            return [(2, 3), (6, 3)]
        
        def get_onion_dispenser_locations(self):
            return [(1, 1)]
        
        def get_dish_dispenser_locations(self):
            return [(7, 1)]
        
        def get_serving_locations(self):
            return [(4, 6)]
    
    return DummyMDP()

def custom_collate_fn(batch):
    """Custom collate function to handle text-based data."""
    state_texts = [item['state_text'] for item in batch]
    commands = [item['command'] for item in batch]
    actions = torch.tensor([item['action'] for item in batch], dtype=torch.long)
    
    return {
        'state_text': state_texts,
        'command': commands,
        'action': actions
    }

def train_text_policy(policy: TextBasedLangConditionedPolicy, 
                     train_loader: DataLoader,
                     val_loader: DataLoader = None,
                     num_epochs: int = 2,
                     learning_rate: float = 1e-4,
                     device: str = "cpu",
                     output_dir: str = None):
    """
    Train the text-based policy network.
    
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
            target_actions = batch['action'].to(device)
            
            # Forward pass
            logits = policy.forward(policy.mdp, state_texts, commands)
            
            # Calculate loss
            loss = criterion(logits, target_actions)
            
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
                    target_actions = batch['action'].to(device)
                    
                    logits = policy.forward(policy.mdp, state_texts, commands)
                    loss = criterion(logits, target_actions)
                    
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
        policy_path = os.path.join(output_dir, "text_policy.pt")
        torch.save({
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': num_epochs,
            'loss': val_avg_loss if val_loader else total_loss / len(train_loader),
            'accuracy': val_accuracy if val_loader else correct_predictions / total_predictions,
            'mdp': policy.mdp,
            'num_actions': policy.num_actions,
            'text_dim': policy.text_dim,
            'hidden_dim': policy.hidden_dim
        }, policy_path)
        print(f"Policy saved to: {policy_path}")

def main():
    """Main training function."""
    print("=== Text-Based Policy Training with Real Data ===\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load real training data
    print("1. Loading real training data...")
    
    # Load data with commands
    with open("generated_data/trajectories_with_commands/policy_training_data.json", "r") as f:
        data_with_commands = json.load(f)
    print(f"   - Loaded {len(data_with_commands)} samples with commands")
    
    # Load data without commands
    with open("generated_data/trajectories_no_commands/policy_training_data.json", "r") as f:
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
    mdp = create_dummy_mdp()  # We'll use the same layout as training data
    
    # Create datasets
    train_dataset = TextPolicyDataset(mdp, train_data)
    val_dataset = TextPolicyDataset(mdp, val_data)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print()
    
    # Initialize policy network
    print("2. Initializing policy network...")
    num_actions = 6
    policy = TextBasedLangConditionedPolicy(
        num_actions=num_actions,
        text_dim=768,
        hidden_dim=256,
        freeze_bert=False,  # Allow fine-tuning
        description_type="english"
    )
    
    # Set the MDP for the policy
    policy.mdp = mdp
    
    print(f"   - Number of actions: {num_actions}")
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
    output_dir = "trained_policies/text_policy"
    os.makedirs(output_dir, exist_ok=True)
    print(f"   - Policy will be saved to: {output_dir}")
    
    train_text_policy(
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
        "Go to the onion dispenser and pick up an onion",
        "You need a dish to pick up the soup",
        "Prioritize cooking onions}"
    ]
    
    test_state = create_dummy_state()
    
    with torch.no_grad():
        for command in test_commands:
            action_idx = policy.get_action(mdp, test_state, command, temperature=0.0)
            action_names = ["STAY", "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "INTERACT"]
            print(f"   Command: '{command}' -> Action: {action_names[action_idx]}")
    
    print("\n=== Training Demo Complete ===")

if __name__ == "__main__":
    main()
