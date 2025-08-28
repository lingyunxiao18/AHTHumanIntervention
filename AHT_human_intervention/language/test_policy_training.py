#!/usr/bin/env python3
"""Test script to train the language-conditioned policy on existing pretraining data."""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path

# Import the policy
from language_conditioned_policy import HuggingFaceLangConditionedPolicy

class PretrainingDataset(Dataset):
    """Dataset for the existing pretraining data."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the data
        with open(data_file, 'r') as f:
            self.examples = json.load(f)
        
        print(f"Loaded {len(self.examples)} examples from {data_file}")
        
        # Filter out examples with missing data
        self.examples = [ex for ex in self.examples if 
                        ex.get('state_text') and 
                        ex.get('command') and 
                        ex.get('action') is not None and
                        ex.get('state_features')]
        
        print(f"Filtered to {len(self.examples)} valid examples")
        
        # Show some statistics
        actions = [ex['action'] for ex in self.examples]
        unique_actions = set(actions)
        print(f"Action distribution: {dict(zip(unique_actions, [actions.count(a) for a in unique_actions]))}")
        
        # Show intervention types
        intervention_types = [ex['intervention_type'] for ex in self.examples]
        unique_types = set(intervention_types)
        print(f"Intervention types: {unique_types}")
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize the command
        command_tokens = self.tokenizer(
            example['command'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert action to tensor
        action = torch.tensor(example['action'], dtype=torch.long)
        
        # Extract state features
        state_features = torch.tensor(example['state_features'], dtype=torch.float32)
        
        return {
            'command_input_ids': command_tokens['input_ids'].squeeze(0),
            'command_attention_mask': command_tokens['attention_mask'].squeeze(0),
            'state_features': state_features,
            'action': action,
            'intervention_type': example['intervention_type'],
            'intervention_category': example['intervention_category']
        }

def train_policy_on_data(data_file: str, num_epochs: int = 5, batch_size: int = 16):
    """Train the language-conditioned policy on the pretraining data."""
    
    print("🚀 TESTING LANGUAGE-CONDITIONED POLICY TRAINING")
    print("=" * 60)
    
    # Initialize tokenizer
    print("1. Initializing tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print("✅ Tokenizer initialized")
    
    # Create dataset
    print("2. Creating dataset...")
    dataset = PretrainingDataset(data_file, tokenizer)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"   Training examples: {len(train_dataset)}")
    print(f"   Validation examples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize policy
    print("3. Initializing policy...")
    state_dim = 20  # Based on the state_features we saw
    num_actions = 6  # 0-5 for Overcooked actions
    policy = HuggingFaceLangConditionedPolicy(
        state_dim=state_dim,
        num_actions=num_actions,
        text_dim=768,
        hidden_dim=256,
        freeze_bert=False
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = policy.to(device)
    print(f"✅ Policy initialized on {device}")
    
    # Training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training loop
    print(f"4. Training for {num_epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        # Training
        policy.train()
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move batch to device
            command_input_ids = batch['command_input_ids'].to(device)
            command_attention_mask = batch['command_attention_mask'].to(device)
            state_features = batch['state_features'].to(device)
            actions = batch['action'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = policy(state_features, command_input_ids, command_attention_mask)
            
            # Compute loss
            loss = criterion(logits, actions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == actions).sum().item()
            total_predictions += actions.size(0)
        
        # Validation
        policy.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                command_input_ids = batch['command_input_ids'].to(device)
                command_attention_mask = batch['command_attention_mask'].to(device)
                state_features = batch['state_features'].to(device)
                actions = batch['action'].to(device)
                
                # Forward pass
                logits = policy(state_features, command_input_ids, command_attention_mask)
                
                # Compute loss
                loss = criterion(logits, actions)
                
                # Statistics
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == actions).sum().item()
                val_total += actions.size(0)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct_predictions / total_predictions
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'training_history': {
                    'train_loss': avg_train_loss,
                    'train_acc': train_acc,
                    'val_loss': avg_val_loss,
                    'val_acc': val_acc
                }
            }, 'test_policy_best.pth')
            print(f"New best model saved! Val Acc: {val_acc:.4f}")
    
    print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.4f}")
    
    # Test some predictions
    print("\n5. Testing predictions...")
    test_predictions(policy, val_loader, device, tokenizer)
    
    return policy

def test_predictions(policy, val_loader, device, tokenizer):
    """Test some predictions on validation data."""
    policy.eval()
    
    # Get a few examples (handle small validation sets)
    batch = next(iter(val_loader))
    batch_size = batch['command_input_ids'].size(0)
    num_test = min(3, batch_size)  # Don't exceed batch size
    
    command_input_ids = batch['command_input_ids'][:num_test].to(device)
    command_attention_mask = batch['command_attention_mask'][:num_test].to(device)
    state_features = batch['state_features'][:num_test].to(device)
    true_actions = batch['action'][:num_test]
    
    with torch.no_grad():
        logits = policy(state_features, command_input_ids, command_attention_mask)
        predictions = torch.argmax(logits, dim=1)
    
    # Show results
    action_names = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "INTERACT"}
    
    for i in range(num_test):
        command = tokenizer.decode(command_input_ids[i], skip_special_tokens=True)
        pred_action = predictions[i].item()
        true_action = true_actions[i].item()
        
        print(f"\nExample {i+1}:")
        print(f"Command: {command}")
        print(f"Predicted: {pred_action} ({action_names.get(pred_action, 'UNKNOWN')})")
        print(f"True: {true_action} ({action_names.get(true_action, 'UNKNOWN')})")
        print(f"Correct: {'✅' if pred_action == true_action else '❌'}")

def main():
    """Main function to test policy training."""
    
    # Find the largest dataset
    data_dir = Path("pretraining_data")
    data_files = list(data_dir.glob("*.json"))
    
    if not data_files:
        print("❌ No pretraining data files found!")
        return
    
    # Use the largest file
    largest_file = max(data_files, key=lambda f: f.stat().st_size)
    print(f"Using dataset: {largest_file}")
    
    try:
        # Train the policy
        policy = train_policy_on_data(
            data_file=str(largest_file),
            num_epochs=5,
            batch_size=16
        )
        
        print(f"\n🎉 Policy training test completed successfully!")
        print(f"Model saved as: test_policy_best.pth")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
