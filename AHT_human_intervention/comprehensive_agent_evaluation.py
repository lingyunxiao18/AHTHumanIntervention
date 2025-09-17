#!/usr/bin/env python3
"""
Comprehensive Agent Evaluation Script

Tests all possible starting location pairs with different agent combinations:
- (SimpleAStar, SimpleAStar)
- (SimpleAStar, OnionSpecialist) 
- (SimpleAStar, RandomAgent)
- (SimpleAStar, StayAgent)

Records soup deliveries, creates histogram, and logs failures.
"""

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import json
from datetime import datetime

# Add project root to path
sys.path.append('.')

# Import Overcooked components
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction

# Import agents
from shared.agents import SimpleAStarAgent, OnionSpecialistAgent, StayAgent, RandomAgent


def convert_action(a):
    """Convert action to proper format."""
    if isinstance(a, (Direction, Action)):
        return a
    if isinstance(a, int):
        mapping = {0: Action.STAY, 1: Direction.EAST, 2: Direction.SOUTH, 
                  3: Direction.NORTH, 4: Direction.WEST, 5: Action.INTERACT}
        return mapping.get(a, Action.STAY)
    return Action.STAY


def run_single_evaluation(agent0_class, agent1_class, start_positions, max_steps=200, verbose=False):
    """Run a single evaluation with given agents and starting positions."""
    
    # Initialize environment
    layout = "random3"
    mdp = OvercookedGridworld.from_layout_name(layout)
    mdp.start_player_positions = start_positions
    
    env = OvercookedEnv(mdp, horizon=max_steps)
    env.reset()
    state = env.state
    
    # Initialize agents
    agent0 = agent0_class(mdp, agent_idx=0, agent_name="Agent0")
    agent1 = agent1_class(mdp, agent_idx=1, agent_name="Agent1")
    
    # Track metrics
    soup_deliveries = 0
    step_count = 0
    failures = []
    prev_agent0_state = None
    
    if verbose:
        print(f"  Starting positions: {start_positions}")
        print(f"  Agent0: {agent0_class.__name__}, Agent1: {agent1_class.__name__}")
    
    try:
        while step_count < max_steps:
            # Get actions from both agents
            action0 = agent0.get_action(state)
            action1 = agent1.get_action(state)
            
            # Convert to proper format
            joint_action = (convert_action(action0), convert_action(action1))
            
            # Step environment
            state, reward, done, info = env.step(joint_action)
            step_count += 1
            
            # Track successful soup deliveries
            if prev_agent0_state is not None:
                prev_holding_soup = prev_agent0_state.get('holding_soup', False)
                current_holding_soup = state.players[0].has_object() and agent0._get_item_name(state.players[0].get_object()) == "soup"
                
                if prev_holding_soup and not current_holding_soup:
                    soup_deliveries += 1
                    if verbose:
                        print(f"    🎉 Soup delivered! Total: {soup_deliveries}")
            
            # Store current state for next comparison
            prev_agent0_state = {
                'holding_soup': state.players[0].has_object() and agent0._get_item_name(state.players[0].get_object()) == "soup"
            }
            
            if done:
                break
                
    except Exception as e:
        failures.append({
            'error': str(e),
            'step': step_count,
            'positions': start_positions,
            'agents': (agent0_class.__name__, agent1_class.__name__)
        })
        if verbose:
            print(f"    ❌ Error at step {step_count}: {e}")
    
    return {
        'soup_deliveries': soup_deliveries,
        'steps': step_count,
        'success': len(failures) == 0,
        'failures': failures,
        'positions': start_positions,
        'agents': (agent0_class.__name__, agent1_class.__name__)
    }


def get_all_valid_position_pairs(mdp):
    """Get all possible pairs of valid starting positions."""
    valid_positions = mdp.get_valid_player_positions()
    pairs = []
    
    for i, pos1 in enumerate(valid_positions):
        for j, pos2 in enumerate(valid_positions):
            if i != j:  # Different positions
                pairs.append((pos1, pos2))
    
    return pairs


def run_comprehensive_evaluation():
    """Run comprehensive evaluation across all agent pairs and starting positions."""
    
    print("🚀 COMPREHENSIVE AGENT EVALUATION")
    print("=" * 60)
    
    # Initialize environment to get valid positions
    layout = "random3"
    mdp = OvercookedGridworld.from_layout_name(layout)
    valid_positions = mdp.get_valid_player_positions()
    position_pairs = get_all_valid_position_pairs(mdp)
    
    print(f"📊 Found {len(valid_positions)} valid positions: {valid_positions}")
    print(f"📊 Testing {len(position_pairs)} position pairs")
    print(f"📊 Testing 4 agent combinations")
    print(f"📊 Total evaluations: {len(position_pairs) * 4}")
    print("=" * 60)
    
    # Define agent combinations
    agent_combinations = [
        (SimpleAStarAgent, SimpleAStarAgent, "A*-A*"),
        (SimpleAStarAgent, OnionSpecialistAgent, "A*-Onion"),
        (SimpleAStarAgent, RandomAgent, "A*-Random"),
        (SimpleAStarAgent, StayAgent, "A*-Stay")
    ]
    
    # Results storage
    all_results = []
    results_by_combination = defaultdict(list)
    all_failures = []
    
    total_evaluations = len(position_pairs) * len(agent_combinations)
    current_evaluation = 0
    
    # Run evaluations
    for agent0_class, agent1_class, combo_name in agent_combinations:
        print(f"\n🤖 Testing {combo_name} combination...")
        combo_results = []
        
        for i, (pos1, pos2) in enumerate(position_pairs):
            current_evaluation += 1
            progress = (current_evaluation / total_evaluations) * 100
            
            if i % 10 == 0:  # Progress update every 10 evaluations
                print(f"  Progress: {progress:.1f}% ({current_evaluation}/{total_evaluations})")
            
            # Run evaluation
            result = run_single_evaluation(
                agent0_class, agent1_class, [pos1, pos2], 
                max_steps=200, verbose=False
            )
            
            combo_results.append(result)
            all_results.append(result)
            results_by_combination[combo_name].append(result)
            
            # Collect failures
            if result['failures']:
                all_failures.extend(result['failures'])
        
        # Print summary for this combination
        soup_counts = [r['soup_deliveries'] for r in combo_results]
        avg_soups = np.mean(soup_counts)
        max_soups = max(soup_counts)
        success_rate = sum(1 for r in combo_results if r['success']) / len(combo_results)
        
        print(f"  ✅ {combo_name} Summary:")
        print(f"     Average soups: {avg_soups:.2f}")
        print(f"     Max soups: {max_soups}")
        print(f"     Success rate: {success_rate:.2%}")
        print(f"     Total evaluations: {len(combo_results)}")
    
    # Generate comprehensive report
    print(f"\n📈 COMPREHENSIVE RESULTS")
    print("=" * 60)
    
    # Overall statistics
    all_soup_counts = [r['soup_deliveries'] for r in all_results]
    overall_avg = np.mean(all_soup_counts)
    overall_max = max(all_soup_counts)
    overall_success_rate = sum(1 for r in all_results if r['success']) / len(all_results)
    
    print(f"🎯 Overall Statistics:")
    print(f"   Total evaluations: {len(all_results)}")
    print(f"   Average soups per run: {overall_avg:.2f}")
    print(f"   Maximum soups in single run: {overall_max}")
    print(f"   Overall success rate: {overall_success_rate:.2%}")
    print(f"   Total failures: {len(all_failures)}")
    
    # Per-combination statistics
    print(f"\n📊 Per-Combination Statistics:")
    for combo_name, combo_results in results_by_combination.items():
        soup_counts = [r['soup_deliveries'] for r in combo_results]
        avg_soups = np.mean(soup_counts)
        std_soups = np.std(soup_counts)
        max_soups = max(soup_counts)
        min_soups = min(soup_counts)
        success_rate = sum(1 for r in combo_results if r['success']) / len(combo_results)
        
        print(f"   {combo_name}:")
        print(f"     Avg: {avg_soups:.2f} ± {std_soups:.2f}")
        print(f"     Range: {min_soups}-{max_soups}")
        print(f"     Success: {success_rate:.2%}")
    
    # Create histogram
    create_histogram(results_by_combination)
    
    # Save detailed results
    save_results(all_results, all_failures)
    
    # Print failure summary
    if all_failures:
        print(f"\n❌ FAILURE SUMMARY:")
        print("=" * 60)
        failure_types = Counter(f['error'] for f in all_failures)
        for error_type, count in failure_types.most_common():
            print(f"   {error_type}: {count} occurrences")
    
    return all_results, all_failures


def create_histogram(results_by_combination):
    """Create histogram of soup delivery results."""
    
    print(f"\n📊 Creating histogram...")
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Soup Delivery Distribution by Agent Combination', fontsize=16)
    
    combinations = list(results_by_combination.keys())
    
    for i, (combo_name, combo_results) in enumerate(results_by_combination.items()):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        soup_counts = [r['soup_deliveries'] for r in combo_results]
        
        # Create histogram
        bins = range(0, max(soup_counts) + 2)
        ax.hist(soup_counts, bins=bins, alpha=0.7, edgecolor='black')
        
        # Add statistics
        avg_soups = np.mean(soup_counts)
        std_soups = np.std(soup_counts)
        
        ax.set_title(f'{combo_name}\nAvg: {avg_soups:.2f} ± {std_soups:.2f}')
        ax.set_xlabel('Soup Deliveries')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add vertical line for mean
        ax.axvline(avg_soups, color='red', linestyle='--', alpha=0.7, label=f'Mean: {avg_soups:.2f}')
        ax.legend()
    
    plt.tight_layout()
    
    # Save histogram
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"agent_evaluation_histogram_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   📊 Histogram saved as: {filename}")
    
    # Also create combined histogram
    plt.figure(figsize=(12, 8))
    for combo_name, combo_results in results_by_combination.items():
        soup_counts = [r['soup_deliveries'] for r in combo_results]
        plt.hist(soup_counts, alpha=0.6, label=combo_name, bins=20)
    
    plt.title('Soup Delivery Distribution - All Combinations')
    plt.xlabel('Soup Deliveries')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    combined_filename = f"agent_evaluation_combined_{timestamp}.png"
    plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
    print(f"   📊 Combined histogram saved as: {combined_filename}")


def save_results(all_results, all_failures):
    """Save detailed results to JSON file."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"agent_evaluation_results_{timestamp}.json"
    
    # Prepare data for JSON serialization
    results_data = {
        'timestamp': timestamp,
        'total_evaluations': len(all_results),
        'total_failures': len(all_failures),
        'results': all_results,
        'failures': all_failures,
        'summary': {
            'overall_avg_soups': np.mean([r['soup_deliveries'] for r in all_results]),
            'overall_max_soups': max([r['soup_deliveries'] for r in all_results]),
            'overall_success_rate': sum(1 for r in all_results if r['success']) / len(all_results)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"   💾 Detailed results saved as: {filename}")


if __name__ == "__main__":
    try:
        results, failures = run_comprehensive_evaluation()
        print(f"\n✅ Evaluation completed successfully!")
        print(f"   Total evaluations: {len(results)}")
        print(f"   Total failures: {len(failures)}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
