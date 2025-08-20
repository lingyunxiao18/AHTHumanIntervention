# DARPA Funding Review: Language-Conditioned Policy Learning
## AHT Human Intervention Project

---

## 🎯 **Project Overview**

**Objective**: Develop AI agents that can understand and execute natural language commands in collaborative cooking environments

**Key Innovation**: Language-conditioned policies that bridge human intent with robotic execution

**Impact**: Enables seamless human-AI collaboration in complex, dynamic environments

---

## 🏗️ **System Architecture**

### **Core Components**

1. **Environment Interface** - Overcooked cooking simulation
2. **State Encoder** - Converts game state to neural representations  
3. **Language Encoder** - DistilBERT for natural language understanding
4. **Policy Network** - Fuses state and language for action selection
5. **Training Pipeline** - Multi-source data generation and learning

### **Architecture Diagram**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Game State    │───▶│  State Encoder  │───▶│                 │
│   (3x3 Matrix)  │    │   (MLP Layers)  │    │                 │
└─────────────────┘    └─────────────────┘    │                 │
                                              │   Policy Head   │
┌─────────────────┐    ┌─────────────────┐    │                 │
│ Human Command   │───▶│ Language Encoder│───▶│                 │
│  (Text Input)   │    │   (DistilBERT)  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                              │                 │
                                              └─────────┬───────┘
                                                        │
                                              ┌─────────▼───────┐
                                              │   Action Logits │
                                              │   (5 Actions)   │
                                              └─────────────────┘
```

---

## 🔄 **Training Loop Architecture**

### **Phase 1: Baseline Training (No Language)**
- **Purpose**: Establish fundamental state-to-action mapping
- **Data**: Random states + random actions (2000 samples)
- **Goal**: Verify policy can learn basic game mechanics
- **Expected Outcome**: 60-80% action accuracy

### **Phase 2: Language Grounding**
- **Purpose**: Connect language commands to actions
- **Data**: State + language command pairs (1500 samples)
- **Goal**: Understand diverse command phrasings
- **Expected Outcome**: Improved generalization to new commands

### **Phase 3: Optimal Action Learning**
- **Purpose**: Learn high-quality actions from search
- **Data**: A* search-generated optimal actions (1000 samples)
- **Goal**: Improve action efficiency and planning
- **Expected Outcome**: Better task completion rates

### **Phase 4: Combined Training**
- **Purpose**: Integrate all learning sources
- **Data**: Balanced combination of all three sources
- **Goal**: Best overall performance and robustness
- **Expected Outcome**: 85-95% action accuracy with language understanding

---

## 📊 **Training Data Generation Strategy**

### **1. Empty Command Baseline**
```
State: [1,0,0,0,1,0,0,0,1] → Action: MOVE
State: [0,1,0,1,0,0,0,0,1] → Action: INTERACT
State: [1,0,1,0,0,1,0,1,0] → Action: TURN_LEFT
```
**Purpose**: Establish basic policy competence

### **2. LLM-Generated Command Variations**
```
Base: "pick up the onion"
Variations:
- "Please pick up the onion"
- "Could you grab the onion"
- "I need you to collect the onion"
- "It would be helpful if you get the onion"
```
**Purpose**: Improve language generalization

### **3. A* Search Optimal Actions**
```
State: Agent at (0,0), Onion at (1,1), Pot at (2,2)
Optimal Action: MOVE → (1,1) → INTERACT → MOVE → (2,2) → INTERACT
Command: "Pick up the onion and put it in the pot"
```
**Purpose**: Learn optimal behavior patterns

---

## 🧠 **Neural Network Architecture**

### **State Encoder (MLP)**
```
Input: State Vector (N dimensions)
├── Linear(N → 256) + ReLU + Dropout(0.1)
├── Linear(256 → 256) + ReLU + Dropout(0.1)
└── Output: State Features (256 dimensions)
```

### **Language Encoder (DistilBERT)**
```
Input: Tokenized Command (MAX_LEN=50)
├── DistilBERT Base (768 dimensions)
├── CLS Token Extraction
└── Output: Language Features (768 dimensions)
```

### **Policy Head (Fusion + Classification)**
```
Input: [State Features (256) + Language Features (768)]
├── Linear(1024 → 256) + ReLU
├── Linear(256 → 256) + ReLU  
└── Linear(256 → 5 Actions)
```

**Total Parameters**: ~110M (mostly from DistilBERT)

---

## 🎮 **Training Process Flow**

```
┌─────────────────┐
│ 1. Data Gen     │
│   - Empty Cmds  │
│   - LLM Vars    │
│   - A* Search   │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ 2. State Enc    │
│   - MDP State   │
│   - Lossless    │
│   - Flatten     │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ 3. Lang Enc     │
│   - Tokenize    │
│   - BERT        │
│   - Features    │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ 4. Fusion       │
│   - Concat      │
│   - MLP         │
│   - Logits      │
└─────────┬───────┘
          │
┌─────────▼───────┐
│ 5. Training     │
│   - Loss        │
│   - Backprop    │
│   - Update      │
└─────────────────┘
```

---

## 📈 **Performance Metrics & Evaluation**

### **Training Metrics**
- **Loss**: Cross-entropy loss over action classes
- **Accuracy**: Percentage of correct action predictions
- **Validation**: Performance on held-out data

### **Evaluation Scenarios**
1. **Command Understanding**: Test on unseen command phrasings
2. **State Generalization**: Test on new game states
3. **Task Completion**: Success rate on cooking objectives
4. **Human-AI Collaboration**: Natural language intervention success

### **Expected Results**
- **Baseline (No Language)**: 70-80% accuracy
- **Language Training**: 80-90% accuracy  
- **A* Enhanced**: 85-95% accuracy
- **Combined**: 90-98% accuracy

---

## 🔬 **Technical Innovations**

### **1. Multi-Source Data Generation**
- **Empty Commands**: Establishes baseline competence
- **LLM Variations**: Improves language robustness
- **A* Search**: Provides optimal action examples

### **2. Reward Shaping & Event Detection**
- **Task Completion**: Major rewards for order fulfillment
- **Progress Tracking**: Rewards for moving toward goals
- **Efficiency Metrics**: Rewards for optimal actions

### **3. Object-Centric State Representation**
- **Current**: Flat matrix representation
- **Target**: Structured object descriptions
- **Benefits**: Better interpretability and generalization

---

## 🚀 **Implementation Status**

### **✅ Completed**
- [x] Enhanced training pipeline with 3 data sources
- [x] LLM command generation system
- [x] A* search integration for optimal actions
- [x] State-only policy training (baseline)
- [x] RL training with reward shaping

### **🔄 In Progress**
- [ ] Object-centric state conversion
- [ ] Human intervention command generation
- [ ] Multi-agent coordination training

### **📋 Planned**
- [ ] Real-world deployment testing
- [ ] Human-AI collaboration studies
- [ ] Performance optimization

---

## 💰 **Funding Impact & Milestones**

### **Current Phase (Months 1-6)**
- **Goal**: Establish baseline language-conditioned policies
- **Deliverable**: Working training pipeline with 3 data sources
- **Success Metric**: 80%+ action accuracy with language

### **Next Phase (Months 7-12)**
- **Goal**: Deploy in human-AI collaboration scenarios
- **Deliverable**: Interactive cooking environment with natural language
- **Success Metric**: 90%+ task completion with human commands

### **Long-term Vision (Months 13-24)**
- **Goal**: Real-world deployment and generalization
- **Deliverable**: Multi-domain language-conditioned policies
- **Success Metric**: Successful human-AI collaboration in new domains

---

## 🌟 **Key Contributions to AI Research**

### **1. Language Grounding in Embodied AI**
- Novel approach to connecting language with robotic actions
- Multi-source training methodology for robust learning

### **2. Human-AI Collaboration Paradigm**
- Natural language intervention in complex tasks
- Seamless human-AI coordination

### **3. Transfer Learning Framework**
- From simulation to real-world deployment
- Generalization across different domains

---

## 🤝 **Collaboration & Partnerships**

### **Academic Partners**
- **UT Austin**: Core research and development
- **LARG Lab**: Multi-agent systems expertise
- **AI/ML Community**: Open-source contributions

### **Industry Applications**
- **Robotics**: Service robots in kitchens, hospitals
- **Gaming**: AI agents with natural language understanding
- **Education**: Interactive AI tutors and assistants

---

## 📞 **Questions & Discussion**

### **Technical Questions**
- How does the system handle ambiguous commands?
- What's the computational cost of real-time inference?
- How robust is the system to language variations?

### **Deployment Questions**
- What's the timeline for real-world testing?
- How do we ensure safety in human-AI collaboration?
- What are the scalability limitations?

### **Research Questions**
- How does this approach compare to other language grounding methods?
- What are the theoretical guarantees of the training approach?
- How can we extend this to other domains?

---

## 🎯 **Call to Action**

**We seek DARPA support to:**

1. **Scale** the training pipeline for larger environments
2. **Deploy** in real-world human-AI collaboration scenarios  
3. **Extend** to multi-agent coordination and planning
4. **Validate** the approach across different domains

**Expected Impact**: Transform how humans interact with AI systems in complex, dynamic environments

---

*Thank you for your consideration and support of this innovative research!*

**Contact**: [Your Name] - [Your Email]  
**Project**: AHT Human Intervention - Language-Conditioned Policy Learning  
**Institution**: University of Texas at Austin 