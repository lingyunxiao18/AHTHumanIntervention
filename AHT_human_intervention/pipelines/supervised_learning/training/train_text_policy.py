#!/usr/bin/env python3
"""
Training script for the text-based policy network.
This demonstrates how to train the TextBasedLangConditionedPolicy using
state descriptions and human commands instead of matrix states.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import random
import json
import signal
import sys
import time
import os
from typing import List, Tuple, Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipelines.supervised_learning.training.language_conditioned_policy import TextBasedLangConditionedPolicy
from shared.utils.state_to_text import describe_state

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def save_ckpt(policy, optimizer, epoch, batch_idx, path):
    """Save checkpoint with model and optimizer state."""
    torch.save({
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
    }, path)
    print(f"[ckpt] saved -> {path}")

# ----------------------------------------------------------------------------
# Dataset and data loading
# ----------------------------------------------------------------------------

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

def create_balanced_sampler(dataset, target_distribution=None):
    """
    Create a WeightedRandomSampler to balance action distribution.
    
    Target distribution:
    - Horizontal moves (E/W): ~50% (25% each)
    - Vertical moves (N/S): ~25% (12.5% each) 
    - INTERACT: ~20%
    - STAY: ~5%
    """
    if target_distribution is None:
        # Define target distribution based on kitchen layout
        target_distribution = {
            0: 0.05,   # STAY: 5%
            1: 0.125,  # MOVE_N: 12.5%
            2: 0.125,  # MOVE_S: 12.5%
            3: 0.25,   # MOVE_E: 25%
            4: 0.25,   # MOVE_W: 25%
            5: 0.20    # INTERACT: 20%
        }
    
    # Get current action distribution
    actions = [item['action'] for item in dataset.data]
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    total_samples = len(actions)
    print(f"Current action distribution:")
    action_names = ["STAY", "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "INTERACT"]
    for action_idx in range(6):
        count = action_counts.get(action_idx, 0)
        percentage = count / total_samples
        print(f"  {action_names[action_idx]}: {count} ({percentage:.3f})")
    
    # Calculate weights to achieve target distribution
    weights = []
    for action in actions:
        current_prob = action_counts[action] / total_samples
        target_prob = target_distribution[action]
        
        # Weight = target_prob / current_prob (inverse of current frequency)
        weight = target_prob / current_prob if current_prob > 0 else 1.0
        weights.append(weight)
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    print(f"Created balanced sampler with {len(weights)} samples")
    return sampler

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
                     output_dir: str = None,
                     gradient_accumulation_steps: int = 8,
                     checkpoint_every: int = 200):
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
    
    # Import os for checkpointing
    import os
    
    # Setup optimizer and loss function
    optimizer = optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Setup checkpointing
    last_save = time.time()
    def handle_sigint(signum, frame):
        if output_dir:
            save_ckpt(policy, optimizer, epoch, batch_idx, os.path.join(output_dir, "interrupt.pt"))
        print("\n[ckpt] interrupt saved; exiting")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)
    
    # Training loop
    for epoch in range(num_epochs):
        policy.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in enumerate(train_loader):
            state_texts = batch['state_text']
            commands = batch['command']
            target_actions = batch['action'].to(device)
            
            # Forward pass
            logits = policy.forward(policy.mdp, state_texts, commands)
            
            # Calculate loss with gradient accumulation
            loss = criterion(logits, target_actions) / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation: only step every N batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * gradient_accumulation_steps
            predictions = logits.argmax(dim=-1)
            correct_predictions += (predictions == target_actions).sum().item()
            total_predictions += target_actions.size(0)
            
            # Progress reporting
            if batch_idx % 100 == 0:
                current_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                      f"Loss: {loss.item() * gradient_accumulation_steps:.4f}, "
                      f"Acc: {current_acc:.3f}")
            
            # Checkpointing and validation
            if (batch_idx + 1) % checkpoint_every == 0 or time.time() - last_save > 600:
                if output_dir:
                    save_ckpt(policy, optimizer, epoch, batch_idx, 
                             os.path.join(output_dir, f"step_{epoch+1}_{batch_idx+1}.pt"))
                last_save = time.time()
                
                # Run validation after checkpointing
                if val_loader is not None:
                    policy.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for val_batch in val_loader:
                            state_texts = val_batch['state_text']
                            commands = val_batch['command']
                            target_actions = val_batch['action'].to(device)
                            
                            logits = policy.forward(policy.mdp, state_texts, commands)
                            loss = criterion(logits, target_actions)
                            
                            val_loss += loss.item()
                            predictions = logits.argmax(dim=-1)
                            val_correct += (predictions == target_actions).sum().item()
                            val_total += target_actions.size(0)
                    
                    val_avg_loss = val_loss / len(val_loader)
                    val_accuracy = val_correct / val_total
                    
                    print(f"[VAL] Epoch {epoch+1}, Batch {batch_idx+1} - "
                          f"Val Loss: {val_avg_loss:.4f}, Val Acc: {val_accuracy:.3f}")
                    
                    # Switch back to training mode
                    policy.train()
        
        # Final optimizer step if needed
        if batch_idx % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
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
            'num_actions': 6,  # Fixed: use the actual number of actions
            'text_dim': 256,  # Fixed: use the actual text dimension
            'hidden_dim': 128  # Fixed: use the actual hidden dimension
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
    with open("generated_data/micro_task_training/policy_training_data.json", "r") as f:
        data_with_commands = json.load(f)
    print(f"   - Loaded {len(data_with_commands)} samples with commands")
    
    # Load data without commands
    with open("generated_data/micro_task_training/policy_training_data_no_commands.json", "r") as f:
        data_without_commands = json.load(f)
    print(f"   - Loaded {len(data_without_commands)} samples without commands")
    
    # Combine datasets and take a smaller subset for faster training
    all_data = data_with_commands + data_without_commands
    print(f"   - Total samples: {len(all_data)}")
    
    # Take only a subset for faster training (e.g., 10% of the data)
    max_samples = 100000  # Full training dataset
    if len(all_data) > max_samples:
        random.shuffle(all_data)
        all_data = all_data[:max_samples]
        print(f"   - Using subset of {len(all_data)} samples for faster training")
    
    # Split into train/val (80/20)
    random.shuffle(all_data)
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"   - Training samples: {len(train_data)}")
    print(f"   - Validation samples: {len(val_data)}")
    
    # Create MDP for the policy
    from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    mdp = OvercookedGridworld.from_layout_name("random3")  # Use the same layout as training data
    
    # Create datasets
    train_dataset = TextPolicyDataset(mdp, train_data)
    val_dataset = TextPolicyDataset(mdp, val_data)
    
    # Initialize policy network
    print("4. Initializing policy network...")
    num_actions = 6
    policy = TextBasedLangConditionedPolicy(
        num_actions=num_actions,
        text_dim=256,  # Reduced from 768 for faster training
        hidden_dim=128,  # Reduced from 256 for faster training
        freeze_bert=True,  # Allow fine-tuning
        description_type="english"
    )
    
    # Set the MDP for the policy
    policy.mdp = mdp
    
    print(f"   - Number of actions: {num_actions}")
    print(f"   - Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print()
    
    # Train the policy
    print("5. Starting training...")
    
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
    
    # Create balanced sampler for training data
    print("3. Creating balanced data sampler...")
    train_sampler = create_balanced_sampler(train_dataset)
    
    # Create data loaders with custom collate function
    # Optimized for smaller dataset
    batch_size = 32  # Smaller batch size for 10k samples
    num_workers = 0 if device == "mps" else 2  # Reduced workers for smaller dataset
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,  # Use balanced sampler instead of shuffle
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory for MPS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory for MPS
    )
    
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Effective batch size: {batch_size * 4} (with gradient accumulation)")
    print(f"   - Workers: {num_workers}")
    print()
    
    train_text_policy(
        policy=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,  # Full training epochs
        learning_rate=1e-4,
        device=device,
        output_dir=output_dir,
        gradient_accumulation_steps=4,  # Reduced from 8 for faster training (32 * 4 = 128 effective batch size)
        checkpoint_every=50  # Checkpoint every 50 batches
    )
    
    # Test the trained policy
    print("\n6. Testing trained policy...")
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
