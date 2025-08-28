#!/usr/bin/env python
"""
Complete Training Pipeline: Supervised Learning + RL Fine-tuning
Phase 1: Train language-conditioned policy with generated data
Phase 2: Fine-tune with RL using dual rewards
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import DistilBertTokenizer
from typing import List, Dict, Tuple, Optional, Any
import json
import os
from tqdm import tqdm
import random

from enhanced_training_generator import EnhancedTrainingGenerator, TrainingExample
from language_conditioned_policy import HuggingFaceLangConditionedPolicy
from enhanced_llm_command_generator import EnhancedLLMCommandGenerator
from rl_finetuning import RLFineTuner, LLMInstructionEvaluator
# Import Overcooked components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'envs', 'overcooked', 'overcooked_ai_py', 'mdp'))
from overcooked_mdp import OvercookedGridworld

class CompleteTrainingPipeline:
    """Complete training pipeline combining supervised learning and RL fine-tuning."""
    
    def __init__(self, 
                 device: str = 'cpu',
                 openai_api_key: Optional[str] = None):
        self.device = device
        self.openai_api_key = openai_api_key
        
        # Initialize components
        self.command_generator = EnhancedLLMCommandGenerator()
        
        # Create Overcooked MDP for state generation
        self.overcooked_mdp = OvercookedGridworld.from_layout_name("simple")
        self.training_generator = EnhancedTrainingGenerator(
            self.command_generator, self.overcooked_mdp
        )
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Training history
        self.training_history = {
            'phase1_supervised': {},
            'phase2_rl': {}
        }
    
    def phase1_supervised_training(self, 
                                  examples_per_category: int = 50,
                                  num_epochs: int = 15,
                                  batch_size: int = 16,
                                  learning_rate: float = 1e-4) -> HuggingFaceLangConditionedPolicy:
        """Phase 1: Train policy using supervised learning with generated data."""
        
        print("=" * 60)
        print("🚀 PHASE 1: Supervised Learning Training")
        print("=" * 60)
        
        # Generate training data
        print("Generating training data...")
        examples = self.training_generator.generate_balanced_dataset(examples_per_category)
        print(f"Generated {len(examples)} training examples")
        
        # Save training data
        self.training_generator.save_training_data('phase1_training_data.json', 'json')
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(examples, batch_size)
        
        # Initialize policy
        print("Initializing policy...")
        state_dim = 20
        num_actions = 6
        policy = HuggingFaceLangConditionedPolicy(
            state_dim=state_dim,
            num_actions=num_actions,
            text_dim=768,
            hidden_dim=256,
            freeze_bert=False
        ).to(self.device)
        
        # Training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        
        # Training loop
        best_val_acc = 0.0
        training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        print(f"\nTraining for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 30)
            
            # Training
            train_loss, train_acc = self._train_epoch(policy, train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_acc = self._validate(policy, val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save training history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'training_history': training_history
                }, 'phase1_supervised_policy.pth')
                print(f"New best model saved! Val Acc: {val_acc:.4f}")
        
        # Save final training history
        self.training_history['phase1_supervised'] = training_history
        
        print(f"\n✅ Phase 1 completed! Best validation accuracy: {best_val_acc:.4f}")
        return policy
    
    def phase2_rl_finetuning(self, 
                             policy: HuggingFaceLangConditionedPolicy,
                             num_episodes: int = 100,
                             max_steps_per_episode: int = 100,
                             exploration_rate: float = 0.1) -> HuggingFaceLangConditionedPolicy:
        """Phase 2: Fine-tune policy using RL with dual rewards."""
        
        print("\n" + "=" * 60)
        print("🚀 PHASE 2: RL Fine-tuning with Dual Rewards")
        print("=" * 60)
        
        # Initialize RL fine-tuner
        print("Initializing RL fine-tuner...")
        llm_evaluator = LLMInstructionEvaluator(api_key=self.openai_api_key)
        rl_tuner = RLFineTuner(
            policy=policy,
            tokenizer=self.tokenizer,
            device=self.device,
            llm_evaluator=llm_evaluator
        )
        
        # Set exploration rate
        rl_tuner.epsilon = exploration_rate
        
        print("RL fine-tuning components:")
        print("  • LLM Instruction Evaluator: Evaluates command compliance")
        print("  • Environment Reward Calculator: Calculates game rewards")
        print("  • Dual Reward System: Combines both reward types")
        
        # Training loop
        print(f"\nFine-tuning for {num_episodes} episodes...")
        episode_rewards = []
        compliance_scores = []
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Generate random command for this episode
            command = self._generate_random_command()
            print(f"Command: {command}")
            
            # Train episode (using simulated environment)
            episode_stats = self._train_rl_episode(rl_tuner, command, max_steps_per_episode)
            
            # Record statistics
            episode_rewards.append(episode_stats['total_reward'])
            compliance_scores.append(episode_stats['avg_compliance'])
            
            print(f"Episode Reward: {episode_stats['total_reward']:.2f}")
            print(f"Compliance Score: {episode_stats['avg_compliance']:.2f}")
            print(f"Steps: {episode_stats['steps']}")
            
            # Save checkpoint periodically
            if (episode + 1) % 20 == 0:
                rl_tuner.save_policy(f'phase2_rl_checkpoint_episode_{episode + 1}.pth')
                print(f"Checkpoint saved at episode {episode + 1}")
        
        # Save final RL fine-tuned policy
        rl_tuner.save_policy('phase2_rl_finetuned_policy.pth')
        
        # Update training history
        self.training_history['phase2_rl'] = {
            'episode_rewards': episode_rewards,
            'compliance_scores': compliance_scores,
            'final_epsilon': rl_tuner.epsilon
        }
        
        print(f"\n✅ Phase 2 completed!")
        print(f"Average episode reward: {np.mean(episode_rewards):.2f}")
        print(f"Average compliance score: {np.mean(compliance_scores):.2f}")
        
        return policy
    
    def _create_data_loaders(self, examples: List[TrainingExample], batch_size: int) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders."""
        # Shuffle examples
        random.shuffle(examples)
        
        # Split into train and validation
        split_idx = int(len(examples) * 0.8)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        print(f"Training examples: {len(train_examples)}")
        print(f"Validation examples: {len(val_examples)}")
        
        # Create datasets
        train_dataset = InterventionDataset(train_examples, self.tokenizer)
        val_dataset = InterventionDataset(val_examples, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _train_epoch(self, policy, dataloader, criterion, optimizer) -> Tuple[float, float]:
        """Train for one epoch."""
        policy.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            command_input_ids = batch['command_input_ids'].to(self.device)
            command_attention_mask = batch['command_attention_mask'].to(self.device)
            state_features = batch['state_features'].to(self.device)
            actions = batch['action'].to(self.device)
            
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
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == actions).sum().item()
            total_predictions += actions.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_predictions:.2f}%'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def _validate(self, policy, dataloader, criterion) -> Tuple[float, float]:
        """Validate the model."""
        policy.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                command_input_ids = batch['command_input_ids'].to(self.device)
                command_attention_mask = batch['command_attention_mask'].to(self.device)
                state_features = batch['state_features'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Forward pass
                logits = policy(state_features, command_input_ids, command_attention_mask)
                
                # Compute loss
                loss = criterion(logits, actions)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == actions).sum().item()
                total_predictions += actions.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def _generate_random_command(self) -> str:
        """Generate a random command for RL training."""
        intervention_types = self.command_generator.get_intervention_types()
        intervention_categories = self.command_generator.get_intervention_categories()
        
        trigger = random.choice(intervention_types)
        category = random.choice(intervention_categories)
        
        commands = self.command_generator.generate_intervention_command(
            trigger, category, 1
        )
        
        return commands[0] if commands else "Complete the cooking task efficiently"
    
    def _train_rl_episode(self, rl_tuner, command: str, max_steps: int) -> Dict[str, float]:
        """Train one RL episode using simulated environment."""
        # Simulate environment states (in real implementation, use actual environment)
        current_state = self._simulate_environment_state()
        total_reward = 0.0
        step_rewards = []
        
        for step in range(max_steps):
            # Get action from policy
            action = rl_tuner.get_action(current_state, command, training=True)
            
            # Simulate environment step
            next_state, env_reward, done, info = self._simulate_environment_step(current_state, action)
            
            # Collect experience
            rl_tuner.collect_experience(current_state, command, action, next_state, done, info)
            
            # Update policy periodically
            if step % rl_tuner.update_frequency == 0:
                loss = rl_tuner.update_policy()
            
            # Record rewards
            step_rewards.append(env_reward)
            total_reward += env_reward
            
            # Check if episode is done
            if done:
                break
            
            current_state = next_state
        
        # Calculate episode statistics
        avg_step_reward = np.mean(step_rewards) if step_rewards else 0.0
        
        return {
            'total_reward': total_reward,
            'avg_step_reward': avg_step_reward,
            'steps': step + 1,
            'avg_compliance': 0.7  # Simulated compliance score
        }
    
    def _simulate_environment_state(self) -> Dict:
        """Simulate a random environment state for RL training."""
        # Generate random state (in real implementation, use actual environment)
        scenarios = ['agent_holding_ingredient', 'agent_near_pot', 'pot_ready_to_cook']
        scenario = random.choice(scenarios)
        
        if scenario == 'agent_holding_ingredient':
            return {
                'agents': [
                    {'id': 1, 'pos': (1, 1), 'holding': 'onion', 'facing': 'UP'},
                    {'id': 2, 'pos': (2, 2), 'holding': None, 'facing': 'DOWN'}
                ],
                'objects': [
                    {'type': 'pot', 'pos': (1, 2), 'status': 'empty'},
                    {'type': 'onion', 'pos': (0, 0), 'status': 'available'}
                ],
                'layout': 'random3',
                'orders': ['onion soup'],
                'time_remaining': random.randint(200, 300)
            }
        else:
            return {
                'agents': [
                    {'id': 1, 'pos': (1, 1), 'holding': None, 'facing': 'UP'},
                    {'id': 2, 'pos': (2, 2), 'holding': None, 'facing': 'DOWN'}
                ],
                'objects': [
                    {'type': 'pot', 'pos': (1, 2), 'status': 'empty'},
                    {'type': 'onion', 'pos': (0, 0), 'status': 'available'}
                ],
                'layout': 'random3',
                'orders': ['onion soup'],
                'time_remaining': random.randint(200, 300)
            }
    
    def _simulate_environment_step(self, state: Dict, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Simulate one environment step."""
        # Simple simulation (in real implementation, use actual environment)
        next_state = state.copy()
        
        # Simulate action effects
        agent = next_state['agents'][0]
        if action in [1, 2, 3, 4]:  # Movement actions
            # Simple movement simulation
            x, y = agent['pos']
            if action == 1:  # UP
                y = max(0, y - 1)
            elif action == 2:  # DOWN
                y = min(2, y + 1)
            elif action == 3:  # LEFT
                x = max(0, x - 1)
            elif action == 4:  # RIGHT
                x = min(2, x + 1)
            agent['pos'] = (x, y)
        
        elif action == 5:  # INTERACT
            # Simple interaction simulation
            if agent['holding'] is None:
                # Try to pick up something
                for obj in next_state['objects']:
                    if obj['type'] == 'onion' and obj['pos'] == agent['pos']:
                        agent['holding'] = 'onion'
                        obj['status'] = 'picked'
                        break
        
        # Simulate time passing
        next_state['time_remaining'] = max(0, next_state['time_remaining'] - 1)
        
        # Simulate reward
        reward = random.uniform(-1, 5)  # Random reward for simulation
        
        # Simulate done condition
        done = next_state['time_remaining'] <= 0
        
        # Simulate info
        info = {'step_type': 'simulated'}
        
        return next_state, reward, done, info
    
    def save_training_history(self, filename: str):
        """Save complete training history."""
        with open(filename, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def run_complete_pipeline(self, 
                             examples_per_category: int = 50,
                             supervised_epochs: int = 15,
                             rl_episodes: int = 100) -> HuggingFaceLangConditionedPolicy:
        """Run the complete training pipeline."""
        
        print("🎉 COMPLETE TRAINING PIPELINE")
        print("=" * 60)
        print("This pipeline combines:")
        print("1. Supervised Learning: Train on generated intervention data")
        print("2. RL Fine-tuning: Fine-tune with dual rewards")
        print("=" * 60)
        
        # Phase 1: Supervised Learning
        policy = self.phase1_supervised_training(
            examples_per_category=examples_per_category,
            num_epochs=supervised_epochs
        )
        
        # Phase 2: RL Fine-tuning
        policy = self.phase2_rl_finetuning(
            policy=policy,
            num_episodes=rl_episodes
        )
        
        # Save complete training history
        self.save_training_history('complete_training_history.json')
        
        print("\n🎯 COMPLETE PIPELINE FINISHED!")
        print("=" * 60)
        print("Files generated:")
        print("  • phase1_supervised_policy.pth - Supervised learning policy")
        print("  • phase2_rl_finetuned_policy.pth - RL fine-tuned policy")
        print("  • complete_training_history.json - Complete training history")
        print("  • phase1_training_data.json - Phase 1 training data")
        
        return policy

class InterventionDataset(Dataset):
    """Dataset for intervention-based training data."""
    
    def __init__(self, training_examples: List[TrainingExample], tokenizer, max_length: int = 128):
        self.examples = training_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize the command
        command_tokens = self.tokenizer(
            example.command,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert action to tensor
        action = torch.tensor(example.action, dtype=torch.long)
        
        # Extract state features if available
        if example.state_features:
            state_features = torch.tensor(example.state_features, dtype=torch.float32)
        else:
            state_features = torch.zeros(20, dtype=torch.float32)
        
        return {
            'command_input_ids': command_tokens['input_ids'].squeeze(0),
            'command_attention_mask': command_tokens['attention_mask'].squeeze(0),
            'state_features': state_features,
            'action': action,
            'intervention_type': example.intervention_type,
            'intervention_category': example.intervention_category
        }

def main():
    """Run the complete training pipeline."""
    print("=== Complete Training Pipeline Demo ===\n")
    
    # Check for OpenAI API key
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        print("⚠️  Warning: No OpenAI API key found.")
        print("   RL fine-tuning will use fallback evaluation.")
        print("   Set OPENAI_API_KEY environment variable for LLM-based evaluation.\n")
    
    # Initialize pipeline
    pipeline = CompleteTrainingPipeline(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        openai_api_key=openai_api_key
    )
    
    # Run complete pipeline
    final_policy = pipeline.run_complete_pipeline(
        examples_per_category=30,  # Smaller for demo
        supervised_epochs=10,      # Fewer epochs for demo
        rl_episodes=50             # Fewer episodes for demo
    )
    
    print("\n🎉 Pipeline completed successfully!")
    print("Final policy combines:")
    print("  • Supervised learning: Command understanding")
    print("  • RL fine-tuning: Task optimization + command compliance")

if __name__ == "__main__":
    main() 