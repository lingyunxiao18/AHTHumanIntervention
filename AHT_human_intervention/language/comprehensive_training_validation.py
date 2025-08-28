#!/usr/bin/env python
"""
Comprehensive Training Data Generation and Validation
Generates 10 examples for each of the 6 intervention types and validates action quality.
"""

import sys
import os
from datetime import datetime
import json

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'envs', 'overcooked', 'overcooked_ai_py', 'mdp'))

from enhanced_training_generator import EnhancedTrainingGenerator
from enhanced_llm_command_generator import EnhancedLLMCommandGenerator
from overcooked_mdp import OvercookedGridworld, PlayerState, OvercookedState

def validate_action_reasonableness_simple(command: str, action: int) -> dict:
    """Validate whether the computed action is reasonable for the given command (simplified version)."""
    
    # Action mapping
    action_names = {
        0: "STAY",
        1: "UP", 
        2: "DOWN",
        3: "LEFT",
        4: "RIGHT",
        5: "INTERACT"
    }
    
    action_name = action_names.get(action, f"UNKNOWN({action})")
    
    # Analyze command intent
    command_lower = command.lower()
    
    validation = {
        "action": action,
        "action_name": action_name,
        "command_analysis": {},
        "reasonableness": "unknown",
        "explanation": ""
    }
    
    # Analyze command intent
    if "go" in command_lower or "move" in command_lower or "navigate" in command_lower:
        validation["command_analysis"]["intent"] = "movement"
        
        # Check if movement action is reasonable
        if action in [1, 2, 3, 4]:  # Movement actions
            validation["reasonableness"] = "reasonable"
            validation["explanation"] = f"Command asks for movement, action {action_name} is appropriate"
        elif action == 0:  # Stay
            validation["reasonableness"] = "questionable"
            validation["explanation"] = "Command asks for movement but action is STAY"
        else:  # Interact
            validation["reasonableness"] = "questionable"
            validation["explanation"] = "Command asks for movement but action is INTERACT"
            
    elif "pick" in command_lower or "get" in command_lower or "take" in command_lower:
        validation["command_analysis"]["intent"] = "pickup"
        
        if action == 5:  # Interact
            validation["reasonableness"] = "reasonable"
            validation["explanation"] = "Command asks for pickup, INTERACT action is appropriate"
        elif action in [1, 2, 3, 4]:  # Movement
            validation["reasonableness"] = "reasonable"
            validation["explanation"] = "Command asks for pickup, moving toward target is reasonable"
        else:  # Stay
            validation["reasonableness"] = "questionable"
            validation["explanation"] = "Command asks for pickup but action is STAY"
            
    elif "put" in command_lower or "place" in command_lower or "drop" in command_lower:
        validation["command_analysis"]["intent"] = "placement"
        
        if action == 5:  # Interact
            validation["reasonableness"] = "reasonable"
            validation["explanation"] = "Command asks for placement, INTERACT action is appropriate"
        elif action in [1, 2, 3, 4]:  # Movement
            validation["reasonableness"] = "reasonable"
            validation["explanation"] = "Command asks for placement, moving toward target is reasonable"
        else:  # Stay
            validation["reasonableness"] = "questionable"
            validation["explanation"] = "Command asks for placement but action is STAY"
            
    elif "serve" in command_lower or "deliver" in command_lower:
        validation["command_analysis"]["intent"] = "serving"
        
        if action == 5:  # Interact
            validation["reasonableness"] = "reasonable"
            validation["explanation"] = "Command asks for serving, INTERACT action is appropriate"
        elif action in [1, 2, 3, 4]:  # Movement
            validation["reasonableness"] = "reasonable"
            validation["explanation"] = "Command asks for serving, moving toward serving area is reasonable"
        else:  # Stay
            validation["reasonableness"] = "questionable"
            validation["explanation"] = "Command asks for serving but action is STAY"
            
    elif "cook" in command_lower or "prepare" in command_lower:
        validation["command_analysis"]["intent"] = "cooking"
        
        if action == 5:  # Interact
            validation["reasonableness"] = "reasonable"
            validation["explanation"] = "Command asks for cooking, INTERACT action is appropriate"
        elif action in [1, 2, 3, 4]:  # Movement
            validation["reasonableness"] = "reasonable"
            validation["explanation"] = "Command asks for cooking, moving toward pot is reasonable"
        else:  # Stay
            validation["reasonableness"] = "questionable"
            validation["explanation"] = "Command asks for cooking but action is STAY"
            
    else:
        validation["command_analysis"]["intent"] = "general"
        validation["reasonableness"] = "reasonable"
        validation["explanation"] = "General command, action seems reasonable"
    
    return validation

def generate_comprehensive_training_data():
    """Generate 10 examples for each of the 6 intervention types."""
    print("🎯 COMPREHENSIVE TRAINING DATA GENERATION")
    print("=" * 70)
    print("Generating 10 examples for each of the 6 intervention types:")
    print("- agent_performance_correction: direct_command, factual_information, general_instruction")
    print("- teammate_model_update: direct_command, factual_information, general_instruction")
    print("=" * 70)
    
    # Initialize components
    print("1. Initializing components...")
    command_generator = EnhancedLLMCommandGenerator(layout_name="random3")
    overcooked_mdp = OvercookedGridworld.from_layout_name("random3")
    training_generator = EnhancedTrainingGenerator(command_generator, overcooked_mdp)
    
    print(f"   Layout: {overcooked_mdp.layout_name}")
    print(f"   A* pathfinding: {'✅ Active' if training_generator.mlp else '❌ Inactive'}")
    print(f"   LLM API: {'✅ Active' if not command_generator.use_fallback else '❌ Fallback'}")
    
    # Define the 6 intervention types
    intervention_types = [
        ("agent_performance_correction", "direct_command"),
        ("agent_performance_correction", "factual_information"), 
        ("agent_performance_correction", "general_instruction"),
        ("teammate_model_update", "direct_command"),
        ("teammate_model_update", "factual_information"),
        ("teammate_model_update", "general_instruction")
    ]
    
    all_examples = []
    all_validation_results = []
    
    print("\n2. Generating examples...")
    
    for i, (trigger, intervention_type) in enumerate(intervention_types, 1):
        print(f"\n   {i}. {trigger} - {intervention_type}")
        print(f"      Generating 10 examples...")
        
        examples_for_type = []
        validation_for_type = []
        
        for j in range(10):
            # Generate a single example
            example = training_generator.generate_single_example(trigger, intervention_type)
            examples_for_type.append(example)
            
            # Validate the action
            validation = validate_action_reasonableness_simple(
                example.command, 
                example.action
            )
            validation_for_type.append(validation)
            
            # Show progress
            if (j + 1) % 5 == 0:
                print(f"         Generated {j + 1}/10 examples...")
        
        all_examples.extend(examples_for_type)
        all_validation_results.extend(validation_for_type)
        
        # Summary for this type
        action_counts = {}
        reasonableness_counts = {}
        
        for validation in validation_for_type:
            action = validation['action']
            reasonableness = validation['reasonableness']
            action_counts[action] = action_counts.get(action, 0) + 1
            reasonableness_counts[reasonableness] = reasonableness_counts.get(reasonableness, 0) + 1
        
        print(f"      ✅ Completed {len(examples_for_type)} examples")
        print(f"         Action distribution: {dict(sorted(action_counts.items()))}")
        print(f"         Reasonableness: {dict(reasonableness_counts)}")
    
    # Overall summary
    print(f"\n3. Overall Summary:")
    print(f"   Total examples: {len(all_examples)}")
    
    # Action distribution
    action_counts = {}
    for validation in all_validation_results:
        action = validation['action']
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print(f"   Action distribution:")
    for action, count in sorted(action_counts.items()):
        percentage = (count / len(all_validation_results)) * 100
        print(f"     Action {action}: {count} times ({percentage:.1f}%)")
    
    # Reasonableness analysis
    reasonableness_counts = {}
    for validation in all_validation_results:
        reasonableness = validation['reasonableness']
        reasonableness_counts[reasonableness] = reasonableness_counts.get(reasonableness, 0) + 1
    
    print(f"   Reasonableness analysis:")
    for reasonableness, count in reasonableness_counts.items():
        percentage = (count / len(all_validation_results)) * 100
        print(f"     {reasonableness}: {count} times ({percentage:.1f}%)")
    
    # Intent analysis
    intent_counts = {}
    for validation in all_validation_results:
        intent = validation['command_analysis'].get('intent', 'general')
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"   Command intent analysis:")
    for intent, count in intent_counts.items():
        percentage = (count / len(all_validation_results)) * 100
        print(f"     {intent}: {count} times ({percentage:.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create pretraining_data directory if it doesn't exist
    os.makedirs("pretraining_data", exist_ok=True)
    
    # Save examples
    examples_data = []
    for example in all_examples:
        examples_data.append({
            "state_text": example.state_text,
            "command": example.command,
            "action": example.action,
            "intervention_type": example.intervention_type,
            "intervention_category": example.intervention_category,
            "features": example.state_features
        })
    
    examples_filename = f"pretraining_data/comprehensive_training_data_{timestamp}.json"
    with open(examples_filename, 'w') as f:
        json.dump(examples_data, f, indent=2)
    
    # Save validation results
    validation_filename = f"pretraining_data/comprehensive_validation_{timestamp}.json"
    with open(validation_filename, 'w') as f:
        json.dump(all_validation_results, f, indent=2)
    
    # Save summary statistics
    summary_stats = {
        "timestamp": timestamp,
        "total_examples": len(all_examples),
        "intervention_types": len(intervention_types),
        "examples_per_type": 10,
        "action_distribution": action_counts,
        "reasonableness_distribution": reasonableness_counts,
        "intent_distribution": intent_counts,
        "layout": overcooked_mdp.layout_name,
        "astar_active": training_generator.mlp is not None,
        "llm_api_active": not command_generator.use_fallback
    }
    
    stats_filename = f"pretraining_data/comprehensive_stats_{timestamp}.json"
    with open(stats_filename, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\n4. Files saved in pretraining_data/:")
    print(f"   Examples: comprehensive_training_data_{timestamp}.json")
    print(f"   Validation: comprehensive_validation_{timestamp}.json")
    print(f"   Stats: comprehensive_stats_{timestamp}.json")
    
    return all_examples, all_validation_results, summary_stats

def main():
    """Main function."""
    try:
        examples, validation_results, stats = generate_comprehensive_training_data()
        print(f"\n🎉 Comprehensive training data generation completed!")
        print(f"Generated {len(examples)} examples with action validation.")
        
        # Final quality assessment
        reasonable_count = sum(1 for v in validation_results if v['reasonableness'] == 'reasonable')
        total_count = len(validation_results)
        quality_percentage = (reasonable_count / total_count) * 100
        
        print(f"\n📊 Final Quality Assessment:")
        print(f"   Reasonable actions: {reasonable_count}/{total_count} ({quality_percentage:.1f}%)")
        
        if quality_percentage >= 80:
            print(f"   ✅ Excellent quality!")
        elif quality_percentage >= 60:
            print(f"   ⚠️  Good quality, room for improvement")
        else:
            print(f"   ❌ Poor quality, needs significant improvement")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
