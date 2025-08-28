#!/usr/bin/env python3
"""
Simple pretraining data generator without validation.
Generates training examples for language-conditioned policy in Overcooked environment.
"""

import os
import sys
import json
from datetime import datetime
import argparse
from pathlib import Path
from dataclasses import asdict

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'envs', 'overcooked', 'overcooked_ai_py', 'mdp'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'envs', 'overcooked', 'overcooked_ai_py', 'planning'))

from enhanced_training_generator import EnhancedTrainingGenerator
from enhanced_llm_command_generator import EnhancedLLMCommandGenerator
from overcooked_mdp import OvercookedGridworld


def generate_pretraining_data(num_examples_per_type=10):
    """
    Generate pretraining data for all intervention types.
    
    Args:
        num_examples_per_type: Number of examples to generate for each intervention type
    """
    print("🎯 PRETRAINING DATA GENERATION")
    print("=" * 70)
    print(f"Generating {num_examples_per_type} examples for each of the 6 intervention types:")
    print("- agent_performance_correction: direct_command, factual_information, general_instruction")
    print("- teammate_model_update: direct_command, factual_information, general_instruction")
    print("=" * 70)
    
    # Initialize components
    print("\n1. Initializing components...")
    command_generator = EnhancedLLMCommandGenerator(layout_name="random3")
    overcooked_mdp = OvercookedGridworld.from_layout_name("random3")
    training_generator = EnhancedTrainingGenerator(command_generator, overcooked_mdp)
    print("✅ Training generator initialized")
    
    # Create output directory
    output_dir = Path("pretraining_data")
    output_dir.mkdir(exist_ok=True)
    
    # Define intervention types (6 types total)
    intervention_types = [
        ("agent_performance_correction", "direct_command"),
        ("agent_performance_correction", "factual_information"),
        ("agent_performance_correction", "general_instruction"),
        ("teammate_model_update", "direct_command"),
        ("teammate_model_update", "factual_information"),
        ("teammate_model_update", "general_instruction"),
    ]
    
    print("\n2. Generating examples...")
    all_examples = []
    
    for i, (trigger, intervention_type) in enumerate(intervention_types, 1):
        print(f"\n   {i}. {trigger} - {intervention_type}")
        print(f"      Generating {num_examples_per_type} examples...")
        
        type_examples = []
        for j in range(num_examples_per_type):
            try:
                example = training_generator.generate_single_example(trigger, intervention_type)
                type_examples.append(example)
                all_examples.append(example)
                
                if (j + 1) % 5 == 0 or (j + 1) == num_examples_per_type:
                    print(f"         Generated {j + 1}/{num_examples_per_type} examples...")
                    
            except Exception as e:
                print(f"         ⚠️  Failed to generate example {j + 1}: {e}")
                continue
        
        print(f"      ✅ Completed {len(type_examples)} examples")
        
        # Show action distribution for this type
        actions = [ex.action for ex in type_examples]
        action_dist = {}
        for action in actions:
            action_dist[action] = action_dist.get(action, 0) + 1
        print(f"         Action distribution: {action_dist}")
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save training data
    training_file = output_dir / f"pretraining_data_{timestamp}.json"
    with open(training_file, 'w') as f:
        json.dump([asdict(example) for example in all_examples], f, indent=2)
    
    # Generate and save statistics
    stats = {
        "timestamp": timestamp,
        "total_examples": len(all_examples),
        "intervention_types": len(intervention_types),
        "examples_per_type": num_examples_per_type,
        "action_distribution": {},
        "layout": "random3",
        "astar_active": training_generator.mlp is not None,
        "llm_api_active": os.getenv("OPENAI_API_KEY") is not None
    }
    
    # Calculate action distribution
    for example in all_examples:
        action = str(example.action)
        stats["action_distribution"][action] = stats["action_distribution"].get(action, 0) + 1
    
    stats_file = output_dir / f"pretraining_stats_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n3. Files saved in {output_dir}/:")
    print(f"   Examples: {training_file.name}")
    print(f"   Stats: {stats_file.name}")
    
    print(f"\n🎉 Pretraining data generation completed!")
    print(f"Generated {len(all_examples)} examples total.")
    
    # Show final statistics
    print(f"\n📊 Final Statistics:")
    print(f"   Total examples: {len(all_examples)}")
    print(f"   Intervention types: {len(intervention_types)}")
    print(f"   Examples per type: {num_examples_per_type}")
    print(f"   A* pathfinding: {'✅ Active' if stats['astar_active'] else '❌ Inactive'}")
    print(f"   LLM API: {'✅ Active' if stats['llm_api_active'] else '❌ Fallback'}")
    
    print(f"\n📈 Action Distribution:")
    for action, count in sorted(stats["action_distribution"].items()):
        percentage = (count / len(all_examples)) * 100
        action_names = {"0": "STAY", "1": "UP", "2": "DOWN", "3": "LEFT", "4": "RIGHT", "5": "INTERACT"}
        action_name = action_names.get(action, f"UNKNOWN({action})")
        print(f"   {action_name}: {count} times ({percentage:.1f}%)")
    
    return all_examples, stats


def main():
    """Main function to generate pretraining data."""
    parser = argparse.ArgumentParser(description="Generate pretraining data for language-conditioned policy")
    parser.add_argument("--num", type=int, default=None, help="Number of examples per intervention type")
    parser.add_argument("--total", type=int, default=None, help="Total number of examples across all intervention types (overrides --num)")
    args = parser.parse_args()

    try:
        # If --total is specified, distribute across 6 intervention types
        if args.total is not None:
            per_type = max(1, args.total // 6)
            remainder = args.total % 6
            print(f"Requested total examples: {args.total}. Distributing approximately {per_type} per type (first {remainder} types get +1).")
            # We call generator once with per_type; exact total will be ~ per_type*6.
            examples, stats = generate_pretraining_data(num_examples_per_type=per_type)
        else:
            per_type = args.num if args.num is not None else 10
            examples, stats = generate_pretraining_data(num_examples_per_type=per_type)
        print(f"\n✅ Successfully generated {len(examples)} training examples!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Generation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
