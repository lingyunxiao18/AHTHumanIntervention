# 🎮 RL Fine-tuning Pipeline

This pipeline implements **reinforcement learning fine-tuning** on top of supervised models, supporting both low-level actions and macro actions as outputs.

## 🔄 **Pipeline Flow**

1. **Load Supervised Model** → 2. **RL Fine-tuning** → 3. **Evaluation**

## 📁 **Directory Structure**

```
rl_finetuning/
├── low_level_actions/        # RL training with low-level action outputs
├── macro_actions/            # RL training with macro action outputs
├── training/                 # RL training algorithms and scripts
└── README.md                # This file
```

## 🚀 **Quick Start**

### **Macro Action RL Fine-tuning**
```bash
# 1. Fine-tune macro action policy with RL
cd macro_actions/
python train_macro_actions.py --pretrained_model ../../supervised_learning/training/best_model.pt

# 2. Evaluate RL-finetuned policy
python evaluate_macro_policy.py --model trained_macro_rl.pt
```

### **Low-level Action RL Fine-tuning**
```bash
# 1. Fine-tune low-level action policy with RL
cd low_level_actions/
python train_low_level_rl.py --pretrained_model ../../supervised_learning/training/best_model.pt

# 2. Evaluate RL-finetuned policy
python evaluate_low_level_policy.py --model trained_low_level_rl.pt
```

## 📊 **Components**

### **Macro Actions**
- `macro_actions.py` - Macro action definitions and executor
- `macro_action_agents.py` - Agents that use macro actions
- `train_macro_actions.py` - RL training for macro action policies

### **Low-level Actions**  
- `[future files]` - Low-level action RL training components

### **Training**
- `rl_finetuning.py` - Core RL fine-tuning algorithms
- `[additional RL training scripts]` - Specific RL algorithms

## 🎯 **Key Features**

- **Two Action Levels**: Support for both macro and low-level action spaces
- **Pretrained Initialization**: Start from supervised learning checkpoints
- **Policy Gradient Methods**: REINFORCE, PPO, A2C support
- **Environment Interaction**: Direct environment feedback for policy improvement
- **Multi-agent RL**: Coordination learning between agents

## 🔧 **Configuration**

RL training can be configured for:
- **Action Space**: Macro actions vs low-level actions
- **Reward Shaping**: Task-specific reward functions
- **Exploration**: ε-greedy, entropy regularization
- **Learning Rate Schedules**: Adaptive learning rates
- **Multi-agent Coordination**: Centralized vs decentralized training
