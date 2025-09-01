#!/usr/bin/env python3
"""
Train Language-Conditioned Policy on Trajectory-Based Dataset

This script trains the HuggingFaceLangConditionedPolicy on the improved
trajectory-based training data with 77 unique commands and 528 examples.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from pathlib import Path
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime

from language_conditioned_policy import HuggingFaceLangConditionedPolicy

class TrajectoryDataset(Dataset):
    """Dataset for trajectory-based training data."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_file, 'r') as f:
            self.examples = json.load(f)
        
        print(f"📊 Loaded {len(self.examples)} training examples")
        
        # Analyze data
        self._analyze_data()
    
    def _analyze_data(self):
        """Analyze the training data."""
        commands = [ex['command'] for ex in self.examples]
        actions = [ex['action'] for ex in self.examples]
        
        unique_commands = len(set(commands))
        unique_actions = len(set(actions))
        
        print(f"💬 Unique commands: {unique_commands}")
        print(f"🎮 Unique actions: {unique_actions}")
        
        # Action distribution
        action_dist = Counter(actions)
        print(f"\n🎯 Action distribution:")
        action_names = {0: 'STAY', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'INTERACT'}
        for action, freq in sorted(action_dist.items()):
            action_name = action_names.get(action, f"Unknown_{action}")
            print(f"  {action_name}: {freq} times ({freq/len(self.examples)*100:.1f}%)")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Tokenize command
        encoding = self.tokenizer(
            ex['command'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Extract features
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        state_features = torch.tensor(ex['state_features'], dtype=torch.float32)
        action = torch.tensor(ex['action'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'state_features': state_features,
            'action': action,
            'command': ex['command']  # For debugging
        }

def train_policy(model, train_loader, val_loader, num_epochs=20, learning_rate=3e-5):
    """Train the language-conditioned policy."""
    
    # Use MPS (Apple GPU) if available, otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🚀 Training on Apple GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Training on NVIDIA GPU (CUDA)")
    else:
        device = torch.device('cpu')
        print(f"🚀 Training on CPU")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            state_features = batch['state_features'].to(device)
            actions = batch['action'].to(device) # Changed from 'actions' to 'action'
            
            optimizer.zero_grad()
            
            # Forward pass - Note: model expects (state_features, input_ids, attention_mask)
            logits = model(state_features, input_ids, attention_mask)
            loss = criterion(logits, actions)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            train_total += actions.size(0)
            train_correct += (predicted == actions).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100*train_correct/train_total:.1f}%")
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                state_features = batch['state_features'].to(device)
                actions = batch['action'].to(device) # Changed from 'actions' to 'action'
                
                logits = model(state_features, input_ids, attention_mask)
                loss = criterion(logits, actions)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                val_total += actions.size(0)
                val_correct += (predicted == actions).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"\n📊 Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.1f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.1f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'trajectory_policy_best.pth')
            print(f"  💾 Saved best model (val_loss: {best_val_loss:.4f})")
        
        print("-" * 50)
    
    # Save final model
    torch.save(model.state_dict(), 'trajectory_policy_final.pth')
    print(f"💾 Saved final model")
    
    return train_losses, val_losses

def test_predictions(model, val_loader, device, num_test=5):
    """Test model predictions on validation examples."""
    model.eval()
    
    print(f"\n🧪 Testing predictions on {num_test} examples:")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_test:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            state_features = batch['state_features'].to(device)
            actions = batch['action'].to(device) # Changed from 'actions' to 'action'
            
            # Get predictions - Note: model expects (state_features, input_ids, attention_mask)
            logits = model(state_features, input_ids, attention_mask)
            _, predicted = torch.max(logits, 1)
            
            # Show results
            for j in range(min(2, len(actions))):  # Show first 2 examples from batch
                command = batch['command'][j]
                true_action = actions[j].item()
                pred_action = predicted[j].item()
                
                action_names = {0: 'STAY', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT', 5: 'INTERACT'}
                true_name = action_names.get(true_action, f"Unknown_{true_action}")
                pred_name = action_names.get(pred_action, f"Unknown_{pred_action}")
                
                print(f"  Example {i*len(actions) + j + 1}:")
                print(f"    Command: \"{command}\"")
                print(f"    True: {true_name} ({true_action})")
                print(f"    Pred: {pred_name} ({pred_action})")
                print(f"    {'✅' if true_action == pred_action else '❌'}")
                print()

def main():
    """Main training function."""
    print("🚀 TRAJECTORY-BASED POLICY TRAINING")
    print("=" * 60)
    
    # Find the most recent trajectory data file
    data_dir = Path("pretraining_data")
    
    # Check for all types of trajectory files
    trajectory_files = list(data_dir.glob("trajectory_direct_command_*.json"))
    enhanced_files = list(data_dir.glob("enhanced_trajectory_direct_command_*.json"))
    large_files = list(data_dir.glob("large_enhanced_trajectory_*.json"))
    massive_files = list(data_dir.glob("massive_enhanced_trajectory_*.json"))
    all_trajectory_files = list(data_dir.glob("*trajectory*.json"))
    
    # Combine all trajectory files
    all_files = trajectory_files + enhanced_files + large_files + massive_files
    all_files = [f for f in all_files if not f.name.endswith('_stats.json')]
    
    if not all_files:
        print("❌ No trajectory data files found!")
        return
    
    # Prioritize massive datasets, then large datasets, then enhanced datasets
    if massive_files:
        latest_file = max(massive_files, key=lambda f: f.stat().st_mtime)
        print(f"📁 Using massive dataset: {latest_file.name} (size: {latest_file.stat().st_size} bytes)")
    elif large_files:
        latest_file = max(large_files, key=lambda f: f.stat().st_mtime)
        print(f"📁 Using large dataset: {latest_file.name} (size: {latest_file.stat().st_size} bytes)")
    elif enhanced_files:
        latest_file = max(enhanced_files, key=lambda f: f.stat().st_mtime)
        print(f"📁 Using enhanced dataset: {latest_file.name} (size: {latest_file.stat().st_size} bytes)")
    elif trajectory_files:
        latest_file = max(trajectory_files, key=lambda f: f.stat().st_mtime)
        print(f"📁 Using trajectory dataset: {latest_file.name} (size: {latest_file.stat().st_size} bytes)")
    else:
        # Fallback to any trajectory file
        latest_file = max(all_files, key=lambda f: f.stat().st_mtime)
        print(f"📁 Using fallback dataset: {latest_file.name} (size: {latest_file.stat().st_size} bytes)")
    
    # Initialize tokenizer and model
    print("\n🔧 Initializing model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Model parameters
    state_dim = 20  # State features dimension
    text_dim = 768  # BERT output dimension
    hidden_dim = 256  # State encoder hidden dimension
    num_actions = 6  # STAY, UP, DOWN, LEFT, RIGHT, INTERACT
    
    model = HuggingFaceLangConditionedPolicy(
        state_dim=state_dim,
        text_dim=text_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions
    )
    
    print(f"✅ Model initialized:")
    print(f"   Text dimension: {text_dim}")
    print(f"   Hidden dimension: {hidden_dim}")
    print(f"   Number of actions: {num_actions}")
    
    # Create dataset and dataloaders
    print("\n📊 Creating dataset and dataloaders...")
    dataset = TrajectoryDataset(str(latest_file), tokenizer)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"   Train set: {len(train_dataset)} examples")
    print(f"   Validation set: {len(val_dataset)} examples")
    
    # Create dataloaders with optimized batch size for GPU
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        batch_size = 32  # Larger batch size for GPU
        num_workers = 4   # Parallel data loading
    else:
        batch_size = 16  # Smaller batch size for CPU
        num_workers = 0  # No parallel loading for CPU
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"   Batch size: {batch_size}")
    print(f"   Workers: {num_workers}")
    
    # Train the model
    print("\n🎯 Starting training...")
    
    # Adjust training parameters based on dataset size
    if len(dataset) > 300000:  # Massive dataset
        num_epochs = 1  # Fewer epochs for massive datasets
        learning_rate = 1e-5  # Lower learning rate for stability
        print(f"📊 Massive dataset detected ({len(dataset)} examples), using {num_epochs} epochs")
    elif len(dataset) > 50000:  # Large dataset
        num_epochs = 15  # Fewer epochs for large datasets
        learning_rate = 2e-5  # Slightly lower learning rate
        print(f"📊 Large dataset detected ({len(dataset)} examples), using {num_epochs} epochs")
    else:
        num_epochs = 25  # More epochs for smaller datasets
        learning_rate = 3e-5
    
    train_losses, val_losses = train_policy(
        model, train_loader, val_loader, 
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    
    # Test predictions
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    test_predictions(model, val_loader, device, num_test=5)
    
    print(f"\n🎉 Training completed!")
    print(f"   Best model saved as: trajectory_policy_best.pth")
    print(f"   Final model saved as: trajectory_policy_final.pth")

if __name__ == "__main__":
    main()
