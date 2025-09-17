# 🤖 LLM Baseline Pipeline

This pipeline implements a **pure LLM-based approach** without any neural network training. It uses LLMs as translators from natural language commands to macro actions.

## 🔄 **Pipeline Flow**

1. **LLM Translation** → 2. **Rule-based Fallback** → 3. **Arbitration** → 4. **Execution**

## 📁 **Directory Structure**

```
llm_baseline/
├── core/                     # Core LLM baseline implementation
├── demo/                     # Interactive demos and integration
├── evaluation/               # Evaluation and comparison scripts
└── README.md                # This file
```

## 🚀 **Quick Start**

### **Interactive Demo**
```bash
# Run interactive demo (no training required!)
cd demo/
python play_llm_translator_baseline.py --layout random3 --model gpt-4

# Or run without LLM (HMS-only mode)
python play_llm_translator_baseline.py --layout random3 --hms-only
```

### **Comprehensive Evaluation**
```bash
# Compare LLM baseline vs other approaches
cd evaluation/
python evaluate_llm_baseline.py --episodes 20 --layout random3
```

## 📊 **Components**

### **Core**
- `llm_translator_baseline.py` - **Main implementation** (LLM + HMS + Arbitrator)
- `llm_command_translator.py` - LLM command translation utilities
- `llm_enhanced_command_generator.py` - Enhanced command generation
- `llm_medium_level_command_translator.py` - Medium-level command translation

### **Demo**
- `play_llm_translator_baseline.py` - **Interactive pygame demo**
- `integration_patch.py` - Guide for integrating with existing code

### **Evaluation**
- `evaluate_llm_baseline.py` - **Comprehensive evaluation script**
- `chain_evaluator.py` - Chain-of-thought evaluation

## 🎯 **Key Features**

### **Zero Training Required**
- No neural network training
- Immediate deployment
- Pure translation approach

### **Robust Fallback System**
- **LLM Translator**: Converts commands to macro actions
- **Hand-coded Macro Selector (HMS)**: Transparent rule-based fallback
- **Arbitrator**: Confidence-based decision making
- **Precondition Checking**: Prevents invalid actions

### **Configurable Behavior**
- **Confidence Thresholds**: Tune LLM override sensitivity
- **Multiple LLM Models**: GPT-4, GPT-3.5, etc.
- **HMS-only Mode**: Pure rule-based behavior

## 🔧 **Architecture**

```
[State + Command] → [LLM Translator] → [Arbitrator] → [Macro Executor] → [Action]
                         ↓                  ↑
                  [Hand-coded Macro      [Precondition
                   Selector (HMS)]        Checker]
```

### **Decision Flow**
1. **HMS** always runs and provides baseline macro
2. **LLM** translates command to macro + confidence
3. **Arbitrator** chooses LLM if `confidence ≥ threshold` AND preconditions met
4. **Executor** converts chosen macro to low-level actions

## 📈 **Evaluation Metrics**

- **Performance**: Deliveries, success rate, efficiency
- **Intervention**: Command acceptance rate, resolution latency  
- **Behavior**: LLM vs HMS decision ratios, violation blocks
- **Robustness**: Fallback usage, error handling

## 🎮 **Usage Examples**

### **HMS-only (No LLM)**
```python
from pipelines.llm_baseline.core.llm_translator_baseline import create_hms_only_agent

agent = create_hms_only_agent(mdp, player_id=0)
action = agent.act_low_level(state)
```

### **LLM + HMS Hybrid**
```python
from pipelines.llm_baseline.core.llm_translator_baseline import create_llm_baseline_agent

agent = create_llm_baseline_agent(
    mdp, 
    model_name="gpt-4", 
    confidence_threshold=0.7
)
action = agent.act_low_level(state, "bring onion to left pot")
```

## ✅ **Advantages**

- **No Training Data Required**: Works immediately
- **Interpretable**: Clear decision logic and audit trails
- **Robust**: Graceful degradation when LLM fails
- **Fast**: No inference overhead from large models
- **Flexible**: Easy to modify rules and behavior
