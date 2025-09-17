# 📚 Supervised Learning Pipeline

This pipeline implements **supervised training** on behavioral data for language-conditioned policies.

## 🔄 **Pipeline Flow**

1. **Data Generation** → 2. **Training** → 3. **Evaluation**

## 📁 **Directory Structure**

```
supervised_learning/
├── data_generation/          # Generate training data from trajectories
├── training/                 # Train language-conditioned policies  
├── evaluation/               # Evaluate trained models
└── README.md                # This file
```

## 🚀 **Quick Start**

```bash
# 1. Generate training data
cd data_generation/
python generate_trajectory_data_with_commands.py --layout random3 --episodes 1000

# 2. Train language-conditioned policy
cd ../training/
python train_text_policy.py --data ../data_generation/output.jsonl --epochs 50

# 3. Evaluate trained model
cd ../evaluation/
python evaluate_micro_tasks.py --model ../training/best_model.pt
```

## 📊 **Components**

### **Data Generation**
- `generate_trajectory_data_with_commands.py` - Generate command-trajectory pairs
- `generate_micro_task_data.py` - Generate micro-task training data
- `macro_data_generator.py` - Generate macro-action training data
- `convert_micro_tasks_to_training_data.py` - Convert micro-tasks to training format
- `rebalance_training_data.py` - Balance training data distribution

### **Training**
- `train_text_policy.py` - Train text-based language-conditioned policy
- `train_macro_policy.py` - Train macro-action policy
- `language_conditioned_policy.py` - Policy network architectures
- `coordinated_agent.py` - Multi-agent coordination training

### **Evaluation**
- `evaluate_micro_tasks.py` - Evaluate on micro-task benchmarks
- `rollout_text_policy.py` - Rollout trained policies
- `smoke_eval_online.py` - Quick online evaluation

## 🎯 **Key Features**

- **Behavioral Cloning**: Learn from expert demonstrations
- **Language Conditioning**: Policies that respond to natural language commands
- **Multi-level Actions**: Support for both low-level and macro actions
- **Comprehensive Evaluation**: Multiple evaluation metrics and benchmarks
