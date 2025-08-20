# 🎯 Complete Training Pipeline for Language-Conditioned Policy

## Overview

This document describes the complete training pipeline that combines:
1. **Generated human intervention commands** (3x3 framework)
2. **Object-centric state text descriptions**
3. **Computed reasonable actions**

To create comprehensive training data for your language-conditioned policy.

## 🏗️ Architecture Overview

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Command Generator │    │  State Converter    │    │ Action Computation  │
│   (3x3 Framework)  │    │ (Object-Centric)    │    │   (Logic + A*)      │
└─────────┬───────────┘    └─────────┬───────────┘    └─────────┬───────────┘
          │                          │                          │
          ▼                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Integrated Training Generator                                  │
│  Combines: Command + State Text + Action = Training Example               │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Language-Conditioned Policy Training                          │
│  Input: (State Features, Command Tokens) → Output: Action Probabilities   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 Components

### 1. Command Generator (`command_generator.py`)

**Purpose**: Generates diverse human intervention commands based on the 3x3 framework.

**3x3 Framework**:
- **Axis 1 (Why)**: Agent Performance Correction | Environmental State Update | Teammate Model Update
- **Axis 2 (What)**: Direct Command | Factual Information | General Instruction

**Features**:
- Generates commands for each of the 9 intervention types
- Applies politeness, urgency, and context modifiers
- Creates scenario-based commands (emergency, rush, coordination)
- Supports composite interventions

**Example Commands**:
```python
# Agent Performance Correction - Direct Command
"Go to the onion and pick it up"
"Turn left and move forward"
"Put the onion in the pot now"

# Environmental State Update - Factual Information
"There's a pot right behind you"
"The dish is ready to serve"
"Your teammate dropped an ingredient"
```

### 2. State Converter (`simple_state_converter.py`)

**Purpose**: Converts game states to object-centric text descriptions.

**Features**:
- Converts flat matrix states to structured object representations
- Generates natural language descriptions of game states
- Handles agents, objects, positions, and statuses
- Creates human-readable position descriptions

**Example Output**:
```
"Agent 1 is at the center, facing UP, and holding an onion. 
Agent 2 is at the bottom-right corner, facing DOWN, and not holding anything. 
There is a pot at the bottom center with status 'empty'. 
There is an onion at the top-left corner with status 'available'."
```

### 3. Action Computation (Integrated in Training Generator)

**Purpose**: Computes reasonable actions based on commands and states.

**Logic**:
- **Movement Commands**: Navigate to target objects/positions
- **Interaction Commands**: Move to object, then interact
- **Cooking Commands**: Move to pot, then interact
- **Serving Commands**: Move to counter, then interact

**Action Space**: `[STAY, UP, DOWN, LEFT, RIGHT, INTERACT]`

### 4. Integrated Training Generator (`integrated_training_generator.py`)

**Purpose**: Orchestrates the complete training data generation process.

**Features**:
- Generates diverse game states (12 scenarios)
- Creates balanced datasets across intervention types
- Computes optimal actions for each state-command pair
- Extracts numerical state features
- Saves training data in multiple formats

**Training Example Structure**:
```python
TrainingExample = {
    'state_text': "Agent 1 at center holding onion...",
    'command': "Go to the pot and put the onion in it",
    'action': 5,  # INTERACT
    'intervention_type': "agent_performance_correction",
    'intervention_category': "direct_command",
    'state_features': [1.0, 1.0, 0.0, ...],  # Numerical features
    'metadata': {...}
}
```

### 5. Training Script (`train_with_integrated_data.py`)

**Purpose**: Trains the language-conditioned policy using the generated data.

**Features**:
- Custom dataset class for intervention data
- PyTorch training loop with validation
- Learning rate scheduling and gradient clipping
- Model checkpointing and training history
- Support for both CPU and GPU training

## 🚀 Usage

### Quick Start

1. **Generate Training Data**:
```bash
python integrated_training_generator.py
```

2. **Train the Policy**:
```bash
python train_with_integrated_data.py
```

3. **Or use the shell script**:
```bash
./train_intervention_policy.sh
```

### Customization

#### Modify Command Generation
```python
from command_generator import CommandGenerator

generator = CommandGenerator()

# Generate specific intervention types
commands = generator.generate_intervention_command(
    trigger="agent_performance_correction",
    intervention_type="direct_command",
    num_variations=5
)

# Generate scenario-based commands
emergency_commands = generator.generate_scenario_based_commands("kitchen_emergency", 3)
```

#### Modify State Generation
```python
from integrated_training_generator import IntegratedTrainingGenerator

# Create custom scenarios
custom_scenarios = ['my_scenario_1', 'my_scenario_2']
training_generator = IntegratedTrainingGenerator(command_generator, state_converter)
training_generator.scenarios = custom_scenarios
```

#### Modify Action Computation
```python
# Override action computation logic
def custom_action_logic(command, state):
    # Your custom logic here
    return action

training_generator._compute_reasonable_action = custom_action_logic
```

## 📊 Generated Dataset Statistics

**Example Output**:
```
Total Examples: 810
Intervention Type Distribution:
  - Agent Performance Correction: 270
  - Environmental State Update: 270
  - Teammate Model Update: 270

Intervention Category Distribution:
  - Direct Command: 270
  - Factual Information: 270
  - General Instruction: 270

Action Distribution:
  - UP: 149, DOWN: 164, LEFT: 163, RIGHT: 166
  - INTERACT: 100, STAY: 68

Command Length: Mean=76.82, Std=14.29
```

## 🔄 Training Process

### 1. Data Generation Phase
- Generate diverse game states (12 scenarios)
- Create commands for each intervention type (9 combinations)
- Compute reasonable actions for each state-command pair
- Balance dataset across intervention categories

### 2. Training Phase
- Split data into train/validation sets (80/20)
- Tokenize commands and state descriptions
- Extract numerical state features
- Train policy using cross-entropy loss
- Validate on held-out data
- Save best model based on validation accuracy

### 3. Output
- **Trained Policy**: `intervention_policy.pth`
- **Training History**: `training_history.json`
- **Training Data**: `intervention_training_data.json`

## 🎯 Key Benefits

1. **Comprehensive Coverage**: Covers all 9 intervention types systematically
2. **Balanced Dataset**: Equal representation across intervention categories
3. **Realistic Commands**: Human-like intervention language with variations
4. **Optimal Actions**: Computed actions that make sense for commands
5. **Scalable**: Easy to generate more data or modify scenarios
6. **Reproducible**: Deterministic generation process

## 🔧 Advanced Features

### Composite Interventions
```python
# Generate multi-part interventions
composite_commands = generator.generate_composite_intervention(
    triggers=["agent_performance_correction", "environmental_state_update"],
    intervention_types=["direct_command", "factual_information"],
    num_variations=3
)
```

### Custom Scenarios
```python
# Add new game scenarios
training_generator.scenarios.append('new_scenario')
training_generator._create_scenario_state('new_scenario')
```

### Action Logic Customization
```python
# Override action computation for specific command types
def custom_action_logic(command, state):
    if "special_command" in command:
        return special_action
    return default_action_logic(command, state)
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required files are in the same directory
2. **Memory Issues**: Reduce `examples_per_category` for large datasets
3. **Training Divergence**: Adjust learning rate or add gradient clipping
4. **Poor Performance**: Check action computation logic and data quality

### Debug Mode
```python
# Enable verbose output
training_generator.verbose = True
generator.verbose = True
```

## 🔮 Future Enhancements

1. **Real Overcooked States**: Integrate with actual Overcooked environment
2. **A* Pathfinding**: Use actual A* search for optimal action computation
3. **Multi-Agent Scenarios**: Support for more complex team coordination
4. **Dynamic Command Generation**: LLM-based command generation
5. **Reinforcement Learning**: Combine with RL for policy improvement

## 📚 File Structure

```
language/
├── command_generator.py              # 3x3 intervention command generator
├── simple_state_converter.py         # State to text converter
├── integrated_training_generator.py  # Main training data generator
├── train_with_integrated_data.py     # Training script
├── train_intervention_policy.sh      # Training shell script
├── language_conditioned_policy.py    # Policy architecture
└── COMPLETE_TRAINING_PIPELINE.md    # This documentation
```

## 🎉 Conclusion

This training pipeline provides a comprehensive solution for training language-conditioned policies on human intervention data. It systematically covers the 3x3 intervention framework while maintaining realistic game scenarios and optimal action mappings.

The modular design allows for easy customization and extension, making it suitable for research, experimentation, and production use. 