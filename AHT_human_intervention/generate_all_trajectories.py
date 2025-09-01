#!/usr/bin/env python3
"""
Generate All Possible Trajectories

This script generates trajectories for all possible starting position combinations
and provides comprehensive analysis of the diversity and performance.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import sys
import time
import itertools

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from trajectory_visualizer import TrajectoryVisualizer, TrajectoryResult

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def analyze_trajectory_diversity(trajectories: List[TrajectoryResult]) -> Dict[str, Any]:
    """Analyze the diversity and performance of trajectories."""
    if not trajectories:
        return {}
    
    # Basic statistics
    total_rewards = [t.total_reward for t in trajectories]
    total_steps = [t.total_steps for t in trajectories]
    completed_trajectories = [t for t in trajectories if t.final_done]
    
    # Starting position analysis
    starting_positions = [t.starting_positions for t in trajectories]
    unique_p0_positions = set(pos[0] for pos in starting_positions)
    unique_p1_positions = set(pos[1] for pos in starting_positions)
    
    # Action distribution analysis
    all_p0_actions = {}
    all_p1_actions = {}
    for trajectory in trajectories:
        for action, count in trajectory.action_distribution["player0"].items():
            all_p0_actions[action] = all_p0_actions.get(action, 0) + count
        for action, count in trajectory.action_distribution["player1"].items():
            all_p1_actions[action] = all_p1_actions.get(action, 0) + count
    
    # Calculate action diversity (entropy)
    def calculate_action_entropy(action_counts):
        total = sum(action_counts.values())
        if total == 0:
            return 0
        entropy = 0
        for count in action_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    p0_entropy = calculate_action_entropy(all_p0_actions)
    p1_entropy = calculate_action_entropy(all_p1_actions)
    
    analysis = {
        "basic_stats": {
            "total_trajectories": len(trajectories),
            "completed_trajectories": len(completed_trajectories),
            "completion_rate": len(completed_trajectories) / len(trajectories),
            "avg_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "min_reward": float(np.min(total_rewards)),
            "max_reward": float(np.max(total_rewards)),
            "avg_steps": float(np.mean(total_steps)),
            "std_steps": float(np.std(total_steps)),
            "min_steps": int(np.min(total_steps)),
            "max_steps": int(np.max(total_steps))
        },
        "starting_positions": {
            "unique_p0_positions": len(unique_p0_positions),
            "unique_p1_positions": len(unique_p1_positions),
            "total_unique_combinations": len(starting_positions),
            "p0_positions": list(unique_p0_positions),
            "p1_positions": list(unique_p1_positions)
        },
        "action_analysis": {
            "player0_total_actions": all_p0_actions,
            "player1_total_actions": all_p1_actions,
            "player0_entropy": float(p0_entropy),
            "player1_entropy": float(p1_entropy),
            "avg_entropy": float((p0_entropy + p1_entropy) / 2)
        },
        "performance_ranking": {
            "top_10_by_reward": sorted(trajectories, key=lambda x: x.total_reward, reverse=True)[:10],
            "top_10_by_steps": sorted(trajectories, key=lambda x: x.total_steps, reverse=True)[:10]
        }
    }
    
    return analysis

def main():
    """Generate all possible trajectories and analyze them."""
    print("🎭 GENERATE ALL POSSIBLE TRAJECTORIES")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = TrajectoryVisualizer(layout_name="random3")
    
    print(f"\n📊 LAYOUT ANALYSIS:")
    print(f"   Layout: {visualizer.layout_name}")
    print(f"   Size: {visualizer.width}x{visualizer.height}")
    print(f"   Valid positions: {len(visualizer.valid_positions)}")
    print(f"   Valid joint positions: {len(visualizer.valid_joint_positions)}")
    
    # Show all valid positions
    print(f"\n📍 All valid positions:")
    for i, pos in enumerate(visualizer.valid_positions):
        print(f"   {i+1:2d}. {pos}")
    
    print(f"\n🎬 Generating ALL {len(visualizer.valid_joint_positions)} trajectories...")
    print("   This may take a while...")
    
    # Generate all trajectories
    start_time = time.time()
    trajectories = visualizer.generate_diverse_trajectories(
        max_trajectories=None,  # Generate all
        max_steps_per_trajectory=100
    )
    end_time = time.time()
    
    print(f"\n⏱️  Generation completed in {end_time - start_time:.2f} seconds")
    
    # Analyze diversity
    print(f"\n🔍 ANALYZING TRAJECTORY DIVERSITY...")
    analysis = analyze_trajectory_diversity(trajectories)
    
    # Print analysis results
    print(f"\n📊 COMPREHENSIVE ANALYSIS RESULTS:")
    print("=" * 60)
    
    # Basic stats
    basic = analysis["basic_stats"]
    print(f"🎯 PERFORMANCE:")
    print(f"   Total trajectories: {basic['total_trajectories']}")
    print(f"   Completed trajectories: {basic['completed_trajectories']} ({basic['completion_rate']:.1%})")
    print(f"   Average reward: {basic['avg_reward']:.3f} ± {basic['std_reward']:.3f}")
    print(f"   Reward range: {basic['min_reward']:.3f} to {basic['max_reward']:.3f}")
    print(f"   Average steps: {basic['avg_steps']:.1f} ± {basic['std_steps']:.1f}")
    print(f"   Steps range: {basic['min_steps']} to {basic['max_steps']}")
    
    # Starting positions
    pos_analysis = analysis["starting_positions"]
    print(f"\n📍 STARTING POSITIONS:")
    print(f"   Unique P0 positions: {pos_analysis['unique_p0_positions']}")
    print(f"   Unique P1 positions: {pos_analysis['unique_p1_positions']}")
    print(f"   Total combinations: {pos_analysis['total_unique_combinations']}")
    
    # Action analysis
    action_analysis = analysis["action_analysis"]
    print(f"\n🎮 ACTION DIVERSITY:")
    print(f"   P0 actions: {action_analysis['player0_total_actions']}")
    print(f"   P1 actions: {action_analysis['player1_total_actions']}")
    print(f"   P0 entropy: {action_analysis['player0_entropy']:.3f}")
    print(f"   P1 entropy: {action_analysis['player1_entropy']:.3f}")
    print(f"   Average entropy: {action_analysis['avg_entropy']:.3f}")
    
    # Top performers
    print(f"\n🏆 TOP PERFORMERS BY REWARD:")
    for i, trajectory in enumerate(analysis["performance_ranking"]["top_10_by_reward"]):
        print(f"   {i+1:2d}. P0={trajectory.starting_positions[0]}, P1={trajectory.starting_positions[1]} "
              f"(reward: {trajectory.total_reward:.3f}, steps: {trajectory.total_steps})")
    
    print(f"\n🏆 TOP PERFORMERS BY STEPS:")
    for i, trajectory in enumerate(analysis["performance_ranking"]["top_10_by_steps"]):
        print(f"   {i+1:2d}. P0={trajectory.starting_positions[0]}, P1={trajectory.starting_positions[1]} "
              f"(steps: {trajectory.total_steps}, reward: {trajectory.total_reward:.3f})")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed analysis
    analysis_filename = f"comprehensive_trajectory_analysis_{timestamp}.json"
    with open(analysis_filename, 'w') as f:
        # Convert trajectories to serializable format
        serializable_trajectories = []
        for trajectory in trajectories:
            serializable_trajectory = {
                "trajectory_id": len(serializable_trajectories),
                "starting_positions": {
                    "player0": trajectory.starting_positions[0],
                    "player1": trajectory.starting_positions[1]
                },
                "total_reward": float(trajectory.total_reward),
                "total_steps": int(trajectory.total_steps),
                "final_done": bool(trajectory.final_done),
                "action_distribution": trajectory.action_distribution
            }
            serializable_trajectories.append(serializable_trajectory)
        
        # Create comprehensive report
        comprehensive_report = {
            "layout_info": {
                "layout_name": visualizer.layout_name,
                "width": int(visualizer.width),
                "height": int(visualizer.height),
                "valid_positions": int(len(visualizer.valid_positions)),
                "valid_joint_positions": int(len(visualizer.valid_joint_positions)),
                "onion_locations": visualizer.onion_locations,
                "pot_locations": visualizer.pot_locations,
                "serving_locations": visualizer.serving_locations,
                "dish_locations": visualizer.dish_locations
            },
            "generation_info": {
                "total_trajectories": int(len(trajectories)),
                "generation_time_seconds": float(end_time - start_time),
                "max_steps_per_trajectory": 100,
                "timestamp": timestamp
            },
            "analysis": convert_numpy_types(analysis),
            "trajectories": serializable_trajectories
        }
        
        json.dump(comprehensive_report, f, indent=2)
    
    print(f"\n💾 Saved comprehensive analysis to: {analysis_filename}")
    
    # Save summary statistics
    summary_filename = f"trajectory_summary_stats_{timestamp}.txt"
    with open(summary_filename, 'w') as f:
        f.write("COMPREHENSIVE TRAJECTORY ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Layout: {visualizer.layout_name}\n")
        f.write(f"Size: {visualizer.width}x{visualizer.height}\n")
        f.write(f"Valid positions: {len(visualizer.valid_positions)}\n")
        f.write(f"Valid joint positions: {len(visualizer.valid_joint_positions)}\n")
        f.write(f"Total trajectories generated: {len(trajectories)}\n")
        f.write(f"Generation time: {end_time - start_time:.2f} seconds\n\n")
        
        f.write("PERFORMANCE STATISTICS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average reward: {basic['avg_reward']:.3f} ± {basic['std_reward']:.3f}\n")
        f.write(f"Reward range: {basic['min_reward']:.3f} to {basic['max_reward']:.3f}\n")
        f.write(f"Average steps: {basic['avg_steps']:.1f} ± {basic['std_steps']:.1f}\n")
        f.write(f"Steps range: {basic['min_steps']} to {basic['max_steps']}\n")
        f.write(f"Completion rate: {basic['completion_rate']:.1%}\n\n")
        
        f.write("STARTING POSITION DIVERSITY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Unique P0 positions: {pos_analysis['unique_p0_positions']}\n")
        f.write(f"Unique P1 positions: {pos_analysis['unique_p1_positions']}\n")
        f.write(f"Total combinations: {pos_analysis['total_unique_combinations']}\n\n")
        
        f.write("ACTION DIVERSITY\n")
        f.write("-" * 30 + "\n")
        f.write(f"P0 action entropy: {action_analysis['player0_entropy']:.3f}\n")
        f.write(f"P1 action entropy: {action_analysis['player1_entropy']:.3f}\n")
        f.write(f"Average entropy: {action_analysis['avg_entropy']:.3f}\n\n")
        
        f.write("TOP 10 BY REWARD\n")
        f.write("-" * 30 + "\n")
        for i, trajectory in enumerate(analysis["performance_ranking"]["top_10_by_reward"]):
            f.write(f"{i+1:2d}. P0={trajectory.starting_positions[0]}, P1={trajectory.starting_positions[1]} "
                   f"(reward: {trajectory.total_reward:.3f}, steps: {trajectory.total_steps})\n")
        
        f.write("\nTOP 10 BY STEPS\n")
        f.write("-" * 30 + "\n")
        for i, trajectory in enumerate(analysis["performance_ranking"]["top_10_by_steps"]):
            f.write(f"{i+1:2d}. P0={trajectory.starting_positions[0]}, P1={trajectory.starting_positions[1]} "
                   f"(steps: {trajectory.total_steps}, reward: {trajectory.total_reward:.3f})\n")
    
    print(f"💾 Saved summary statistics to: {summary_filename}")
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"   Generated {len(trajectories)} trajectories")
    print(f"   Explored {len(visualizer.valid_joint_positions)} starting position combinations")
    print(f"   Analysis saved to: {analysis_filename}")
    print(f"   Summary saved to: {summary_filename}")

if __name__ == "__main__":
    main()
