#!/usr/bin/env python
"""
RL Fine-tuning for Language-Conditioned Policy
Fine-tunes the pre-trained supervised policy using:
1. Environment rewards (task completion, efficiency)
2. LLM-based instruction compliance rewards
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import random
from collections import deque
import openai
import os

from language_conditioned_policy import HuggingFaceLangConditionedPolicy
from transformers import DistilBertTokenizer

@dataclass
class RLExperience:
    """Experience for RL training."""
    state: torch.Tensor
    command: str
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    metadata: Dict[str, Any]

class LLMInstructionEvaluator:
    """Uses LLM to evaluate how well actions follow human instructions."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.model = model
        if api_key:
            openai.api_key = api_key
        elif 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        else:
            print("Warning: No OpenAI API key found. Using fallback evaluation.")
            self.use_fallback = True
        self.use_fallback = False
    
    def evaluate_instruction_compliance(self, 
                                      action: int, 
                                      command: str, 
                                      game_state: Dict,
                                      action_names: Dict[int, str]) -> float:
        """Evaluate how well an action follows the human instruction."""
        
        if self.use_fallback:
            return self._fallback_evaluation(action, command, game_state, action_names)
        
        try:
            # Create evaluation prompt
            prompt = self._create_evaluation_prompt(action, command, game_state, action_names)
            
            # Get LLM evaluation
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of human-AI instruction compliance in cooking games."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            # Parse response for compliance score
            compliance_score = self._parse_compliance_score(response.choices[0].message.content)
            return compliance_score
            
        except Exception as e:
            print(f"LLM evaluation failed: {e}. Using fallback.")
            return self._fallback_evaluation(action, command, game_state, action_names)
    
    def _create_evaluation_prompt(self, action: int, command: str, game_state: Dict, action_names: Dict[int, str]) -> str:
        """Create prompt for LLM evaluation."""
        action_name = action_names.get(action, "UNKNOWN")
        
        prompt = f"""
Evaluate how well the AI's action follows the human's instruction in this cooking game scenario.

GAME STATE:
{json.dumps(game_state, indent=2)}

HUMAN INSTRUCTION:
"{command}"

AI ACTION TAKEN:
{action_name} (action code: {action})

Rate the compliance from 0.0 to 1.0, where:
- 1.0: Perfect compliance (action directly follows instruction)
- 0.8-0.9: Good compliance (action is appropriate for instruction)
- 0.6-0.7: Moderate compliance (action is somewhat related)
- 0.4-0.5: Poor compliance (action doesn't follow instruction well)
- 0.0-0.3: Very poor compliance (action contradicts instruction)

Provide only the numerical score (0.0 to 1.0):
"""
        return prompt
    
    def _parse_compliance_score(self, response: str) -> float:
        """Parse compliance score from LLM response."""
        try:
            # Extract numerical score
            import re
            numbers = re.findall(r'0\.\d+|1\.0|\d+\.\d+', response)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
            else:
                return 0.5  # Default score
        except:
            return 0.5
    
    def _fallback_evaluation(self, action: int, command: str, game_state: Dict, action_names: Dict[int, str]) -> float:
        """Fallback evaluation when LLM is not available."""
        action_name = action_names.get(action, "UNKNOWN")
        command_lower = command.lower()
        
        # Simple rule-based evaluation
        if "go" in command_lower or "move" in command_lower:
            if action in [1, 2, 3, 4]:  # Movement actions
                return 0.9
            else:
                return 0.3
        
        elif "pick" in command_lower or "grab" in command_lower or "take" in command_lower:
            if action == 5:  # INTERACT action
                return 0.9
            elif action in [1, 2, 3, 4]:  # Moving toward object
                return 0.7
            else:
                return 0.2
        
        elif "put" in command_lower or "drop" in command_lower or "place" in command_lower:
            if action == 5:  # INTERACT action
                return 0.9
            elif action in [1, 2, 3, 4]:  # Moving toward target
                return 0.7
            else:
                return 0.2
        
        elif "cook" in command_lower or "prepare" in command_lower:
            if action == 5:  # INTERACT action
                return 0.9
            elif action in [1, 2, 3, 4]:  # Moving toward cooking equipment
                return 0.7
            else:
                return 0.3
        
        elif "stop" in command_lower or "wait" in command_lower:
            if action == 0:  # STAY action
                return 0.9
            else:
                return 0.2
        
        else:
            # Default moderate score for unclear commands
            return 0.5

class EnvironmentRewardCalculator:
    """Calculates environment-based rewards for RL training."""
    
    def __init__(self):
        self.base_rewards = {
            'ingredient_pickup': 10,
            'cooking_start': 15,
            'cooking_complete': 25,
            'order_complete': 50,
            'efficient_movement': 2,
            'collision_penalty': -5,
            'time_penalty': -1,
            'idle_penalty': -2
        }
    
    def calculate_reward(self, 
                        current_state: Dict, 
                        action: int, 
                        next_state: Dict,
                        action_names: Dict[int, str]) -> float:
        """Calculate environment reward based on state transition."""
        reward = 0.0
        
        # Check for positive events
        if self._ingredient_pickup(current_state, next_state):
            reward += self.base_rewards['ingredient_pickup']
        
        if self._cooking_start(current_state, next_state):
            reward += self.base_rewards['cooking_start']
        
        if self._cooking_complete(current_state, next_state):
            reward += self.base_rewards['cooking_complete']
        
        if self._order_complete(current_state, next_state):
            reward += self.base_rewards['order_complete']
        
        # Check for penalties
        if self._collision_detected(current_state, next_state):
            reward += self.base_rewards['collision_penalty']
        
        if self._time_decreased(current_state, next_state):
            reward += self.base_rewards['time_penalty']
        
        if self._agent_idle(current_state, next_state, action):
            reward += self.base_rewards['idle_penalty']
        
        # Efficiency bonus for purposeful movement
        if self._efficient_movement(current_state, next_state, action):
            reward += self.base_rewards['efficient_movement']
        
        return reward
    
    def _ingredient_pickup(self, current_state: Dict, next_state: Dict) -> bool:
        """Check if ingredient was picked up."""
        current_holding = current_state.get('agents', [{}])[0].get('holding')
        next_holding = next_state.get('agents', [{}])[0].get('holding')
        return current_holding is None and next_holding is not None
    
    def _cooking_start(self, current_state: Dict, next_state: Dict) -> bool:
        """Check if cooking started."""
        current_pots = [obj for obj in current_state.get('objects', []) if obj.get('type') == 'pot']
        next_pots = [obj for obj in next_state.get('objects', []) if obj.get('type') == 'pot']
        
        for current_pot, next_pot in zip(current_pots, next_pots):
            if (current_pot.get('status') == 'empty' and 
                next_pot.get('status') == 'cooking'):
                return True
        return False
    
    def _cooking_complete(self, current_state: Dict, next_state: Dict) -> bool:
        """Check if cooking completed."""
        current_pots = [obj for obj in current_state.get('objects', []) if obj.get('type') == 'pot']
        next_pots = [obj for obj in next_state.get('objects', []) if obj.get('type') == 'pot']
        
        for current_pot, next_pot in zip(current_pots, next_pots):
            if (current_pot.get('status') == 'cooking' and 
                next_pot.get('status') == 'ready'):
                return True
        return False
    
    def _order_complete(self, current_state: Dict, next_state: Dict) -> bool:
        """Check if order was completed."""
        current_orders = len(current_state.get('orders', []))
        next_orders = len(next_state.get('orders', []))
        return current_orders > next_orders
    
    def _collision_detected(self, current_state: Dict, next_state: Dict) -> bool:
        """Check if collision occurred."""
        # Simplified collision detection
        current_agents = current_state.get('agents', [])
        next_agents = next_state.get('agents', [])
        
        if len(current_agents) > 1 and len(next_agents) > 1:
            current_positions = [agent.get('pos') for agent in current_agents]
            next_positions = [agent.get('pos') for agent in next_agents]
            
            # Check if agents moved to same position
            if len(set(next_positions)) < len(next_positions):
                return True
        return False
    
    def _time_decreased(self, current_state: Dict, next_state: Dict) -> bool:
        """Check if time decreased (penalty)."""
        current_time = current_state.get('time_remaining', 0)
        next_time = next_state.get('time_remaining', 0)
        return next_time < current_time
    
    def _agent_idle(self, current_state: Dict, next_state: Dict, action: int) -> bool:
        """Check if agent is idle (penalty)."""
        return action == 0  # STAY action
    
    def _efficient_movement(self, current_state: Dict, next_state: Dict, action: int) -> bool:
        """Check if movement was efficient."""
        if action in [1, 2, 3, 4]:  # Movement actions
            # Check if agent moved toward a goal
            current_pos = current_state.get('agents', [{}])[0].get('pos', (0, 0))
            next_pos = next_state.get('agents', [{}])[0].get('pos', (0, 0))
            
            # Simple efficiency: moving toward objects
            objects = current_state.get('objects', [])
            if objects:
                closest_object = min(objects, key=lambda obj: 
                    abs(obj.get('pos', (0, 0))[0] - current_pos[0]) + 
                    abs(obj.get('pos', (0, 0))[1] - current_pos[1]))
                
                old_distance = (abs(closest_object.get('pos', (0, 0))[0] - current_pos[0]) + 
                               abs(closest_object.get('pos', (0, 0))[1] - current_pos[1]))
                new_distance = (abs(closest_object.get('pos', (0, 0))[0] - next_pos[0]) + 
                               abs(closest_object.get('pos', (0, 0))[1] - next_pos[1]))
                
                return new_distance < old_distance
        return False

class RLFineTuner:
    """Fine-tunes the language-conditioned policy using RL with dual rewards."""
    
    def __init__(self, 
                 policy: HuggingFaceLangConditionedPolicy,
                 tokenizer,
                 device: str = 'cpu',
                 llm_evaluator: Optional[LLMInstructionEvaluator] = None):
        self.policy = policy
        self.tokenizer = tokenizer
        self.device = device
        self.llm_evaluator = llm_evaluator or LLMInstructionEvaluator()
        self.env_reward_calculator = EnvironmentRewardCalculator()
        
        # RL components
        self.optimizer = optim.AdamW(policy.parameters(), lr=1e-5, weight_decay=0.01)
        self.experience_buffer = deque(maxlen=10000)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.batch_size = 32
        self.update_frequency = 4
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'instruction_compliance': [],
            'environment_rewards': [],
            'total_rewards': []
        }
    
    def collect_experience(self, 
                          state: Dict, 
                          command: str, 
                          action: int, 
                          next_state: Dict, 
                          done: bool,
                          metadata: Dict[str, Any] = None):
        """Collect experience for RL training."""
        # Calculate dual rewards
        env_reward = self.env_reward_calculator.calculate_reward(
            state, action, next_state, self.policy.action_names
        )
        
        compliance_reward = self.llm_evaluator.evaluate_instruction_compliance(
            action, command, state, self.policy.action_names
        )
        
        # Combine rewards (you can adjust weights)
        total_reward = env_reward + 2.0 * compliance_reward  # Weight instruction compliance higher
        
        # Create experience
        experience = RLExperience(
            state=torch.tensor(self._extract_state_features(state), dtype=torch.float32),
            command=command,
            action=action,
            reward=total_reward,
            next_state=torch.tensor(self._extract_state_features(next_state), dtype=torch.float32),
            done=done,
            metadata=metadata or {}
        )
        
        self.experience_buffer.append(experience)
    
    def _extract_state_features(self, state: Dict) -> List[float]:
        """Extract numerical features from state."""
        features = []
        
        # Agent positions and states
        for agent in state.get('agents', []):
            features.extend([float(agent.get('pos', (0, 0))[0]), float(agent.get('pos', (0, 0))[1])])
            features.append(1.0 if agent.get('holding') else 0.0)
        
        # Object counts by type
        object_counts = {}
        for obj in state.get('objects', []):
            obj_type = obj.get('type', 'unknown')
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Add object counts to features
        for obj_type in ['onion', 'pot', 'counter', 'stove', 'dish']:
            features.append(float(object_counts.get(obj_type, 0)))
        
        # Time remaining (normalized)
        features.append(float(state.get('time_remaining', 300)) / 300.0)
        
        # Pad to fixed length
        while len(features) < 20:
            features.append(0.0)
        
        return features[:20]
    
    def update_policy(self):
        """Update policy using collected experiences."""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch of experiences
        batch = random.sample(self.experience_buffer, self.batch_size)
        
        # Prepare batch data
        states = torch.stack([exp.state for exp in batch]).to(self.device)
        commands = [exp.command for exp in batch]
        actions = torch.tensor([exp.action for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([exp.next_state for exp in batch]).to(self.device)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.bool).to(self.device)
        
        # Tokenize commands
        command_tokens = self.tokenizer(
            commands,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy(states, command_tokens['input_ids'], command_tokens['attention_mask'])
        
        # Next Q-values (for target calculation)
        with torch.no_grad():
            next_q_values = self.policy(next_states, command_tokens['input_ids'], command_tokens['attention_mask'])
            target_q_values = rewards + self.gamma * next_q_values.max(dim=1)[0] * (~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.gather(1, actions.unsqueeze(1)).squeeze(), target_q_values)
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def get_action(self, state: Dict, command: str, training: bool = True) -> int:
        """Get action from policy with optional exploration."""
        state_features = torch.tensor(self._extract_state_features(state), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Tokenize command
        command_tokens = self.tokenizer(
            [command],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.policy(state_features, command_tokens['input_ids'], command_tokens['attention_mask'])
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            action = random.randint(0, q_values.size(1) - 1)
        else:
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def train_episode(self, 
                     env_simulator, 
                     max_steps: int = 100,
                     command: Optional[str] = None) -> Dict[str, float]:
        """Train for one episode."""
        episode_rewards = []
        instruction_compliance = []
        environment_rewards = []
        
        # Reset environment
        state = env_simulator.reset()
        total_reward = 0.0
        
        for step in range(max_steps):
            # Get action from policy
            action = self.get_action(state, command or "Complete the cooking task efficiently")
            
            # Take action in environment
            next_state, env_reward, done, info = env_simulator.step(action)
            
            # Collect experience
            self.collect_experience(state, command or "Complete the cooking task efficiently", 
                                 action, next_state, done, info)
            
            # Update policy periodically
            if step % self.update_frequency == 0:
                loss = self.update_policy()
            
            # Record rewards
            episode_rewards.append(env_reward)
            total_reward += env_reward
            
            # Check if episode is done
            if done:
                break
            
            state = next_state
        
        # Calculate episode statistics
        avg_env_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        avg_compliance = np.mean(instruction_compliance) if instruction_compliance else 0.5
        
        # Update training history
        self.training_history['episode_rewards'].append(total_reward)
        self.training_history['instruction_compliance'].append(avg_compliance)
        self.training_history['environment_rewards'].append(avg_env_reward)
        self.training_history['total_rewards'].append(total_reward)
        
        return {
            'total_reward': total_reward,
            'avg_env_reward': avg_env_reward,
            'avg_compliance': avg_compliance,
            'steps': step + 1
        }
    
    def save_policy(self, filename: str):
        """Save the fine-tuned policy."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'rl_hyperparameters': {
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'batch_size': self.batch_size
            }
        }, filename)
    
    def load_policy(self, filename: str):
        """Load a fine-tuned policy."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        # Load hyperparameters
        if 'rl_hyperparameters' in checkpoint:
            self.gamma = checkpoint['rl_hyperparameters']['gamma']
            self.epsilon = checkpoint['rl_hyperparameters']['epsilon']
            self.batch_size = checkpoint['rl_hyperparameters']['batch_size']

def main():
    """Demo the RL fine-tuning framework."""
    print("=== RL Fine-tuning Framework Demo ===\n")
    
    # This would require a real environment simulator
    # For demo purposes, we'll show the framework structure
    
    print("🚀 RL Fine-tuning Framework Components:")
    print("1. LLM Instruction Evaluator - Evaluates command compliance")
    print("2. Environment Reward Calculator - Calculates game rewards")
    print("3. RL Fine-tuner - Combines both rewards for policy updates")
    print("4. Dual Reward System - Environment + Instruction compliance")
    
    print("\n📊 Dual Reward Structure:")
    print("Total Reward = Environment Reward + 2.0 × Instruction Compliance Reward")
    print("  • Environment: Task completion, efficiency, game performance")
    print("  • Compliance: How well actions follow human commands")
    
    print("\n🔄 Training Process:")
    print("1. Load pre-trained supervised policy")
    print("2. Collect experiences with dual rewards")
    print("3. Update policy using Q-learning")
    print("4. Fine-tune for both task performance and command following")
    
    print("\n💡 Key Benefits:")
    print("  • Combines supervised learning (command understanding) with RL (task optimization)")
    print("  • Policy learns to follow commands while maximizing game rewards")
    print("  • LLM provides intelligent evaluation of instruction compliance")
    print("  • Balances human intent with environmental optimization")

if __name__ == "__main__":
    main() 