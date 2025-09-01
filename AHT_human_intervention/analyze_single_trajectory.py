#!/usr/bin/env python3
"""
Analyze Single Trajectory

This script analyzes a single trajectory in detail to understand why
there's no reward and the task wasn't completed.
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

def analyze_single_trajectory_detailed(trajectory: TrajectoryResult, visualizer: TrajectoryVisualizer):
    """Analyze a single trajectory in detail."""
    print(f"🔍 DETAILED TRAJECTORY ANALYSIS")
    print("=" * 60)
    print(f"Starting positions: P0={trajectory.starting_positions[0]}, P1={trajectory.starting_positions[1]}")
    print(f"Total steps: {trajectory.total_steps}")
    print(f"Total reward: {trajectory.total_reward}")
    print(f"Final done: {trajectory.final_done}")
    print("=" * 60)
    
    # Analyze the first few steps in detail
    print(f"\n📋 FIRST 10 STEPS ANALYSIS:")
    print("-" * 60)
    
    for i, step in enumerate(trajectory.trajectory_steps[:10]):
        print(f"\n🎮 STEP {i}:")
        print(f"   P0: {step.player0_pos} {step.player0_orientation} holding {step.player0_holding}")
        print(f"   P1: {step.player1_pos} {step.player1_orientation} holding {step.player1_holding}")
        print(f"   Actions: P0={step.action0}, P1={step.action1}")
        print(f"   Reward: {step.reward}")
        print(f"   Done: {step.done}")
        
        # Show objects in the environment
        print(f"   Objects in environment:")
        for obj_pos, obj in step.state.objects.items():
            if obj.name == "soup":
                if hasattr(obj, 'state') and obj.state:
                    soup_type, num_items, cook_time = obj.state
                    status = "READY" if num_items == 3 and cook_time >= 20 else "COOKING"
                    print(f"     Soup at {obj_pos}: {soup_type} soup ({num_items}/3 onions, {cook_time}/20 steps) - {status}")
                else:
                    print(f"     Soup at {obj_pos}: Unknown state")
            else:
                print(f"     {obj.name} at {obj_pos}")
        
        # Show layout visualization for first few steps
        if i < 3:
            print(f"   Layout:")
            print(step.layout_visualization)
    
    # Analyze action patterns
    print(f"\n🎯 ACTION PATTERN ANALYSIS:")
    print("-" * 60)
    
    p0_actions = [step.action0 for step in trajectory.trajectory_steps]
    p1_actions = [step.action1 for step in trajectory.trajectory_steps]
    
    print(f"Player 0 most common actions:")
    action_counts = {}
    for action in p0_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(p0_actions)) * 100
        print(f"   {action}: {count} times ({percentage:.1f}%)")
    
    print(f"\nPlayer 1 most common actions:")
    action_counts = {}
    for action in p1_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(p1_actions)) * 100
        print(f"   {action}: {count} times ({percentage:.1f}%)")
    
    # Analyze movement patterns
    print(f"\n🚶 MOVEMENT PATTERN ANALYSIS:")
    print("-" * 60)
    
    p0_positions = [step.player0_pos for step in trajectory.trajectory_steps]
    p1_positions = [step.player1_pos for step in trajectory.trajectory_steps]
    
    print(f"Player 0 visited positions:")
    unique_positions = set(p0_positions)
    for pos in sorted(unique_positions):
        count = p0_positions.count(pos)
        percentage = (count / len(p0_positions)) * 100
        print(f"   {pos}: {count} times ({percentage:.1f}%)")
    
    print(f"\nPlayer 1 visited positions:")
    unique_positions = set(p1_positions)
    for pos in sorted(unique_positions):
        count = p1_positions.count(pos)
        percentage = (count / len(p1_positions)) * 100
        print(f"   {pos}: {count} times ({percentage:.1f}%)")
    
    # Analyze task completion attempts
    print(f"\n🎯 TASK COMPLETION ANALYSIS:")
    print("-" * 60)
    
    # Check if players ever reached key locations
    onion_locations = visualizer.onion_locations
    pot_locations = visualizer.pot_locations
    serving_locations = visualizer.serving_locations
    dish_locations = visualizer.dish_locations
    
    p0_visited_onions = any(pos in onion_locations for pos in p0_positions)
    p0_visited_pots = any(pos in pot_locations for pos in p0_positions)
    p0_visited_serving = any(pos in serving_locations for pos in p0_positions)
    p0_visited_dishes = any(pos in dish_locations for pos in p0_positions)
    
    p1_visited_onions = any(pos in onion_locations for pos in p1_positions)
    p1_visited_pots = any(pos in pot_locations for pos in p1_positions)
    p1_visited_serving = any(pos in serving_locations for pos in p1_positions)
    p1_visited_dishes = any(pos in dish_locations for pos in p1_positions)
    
    print(f"Player 0 visited key locations:")
    print(f"   Onion dispensers: {p0_visited_onions}")
    print(f"   Pots: {p0_visited_pots}")
    print(f"   Serving areas: {p0_visited_serving}")
    print(f"   Dish dispensers: {p0_visited_dishes}")
    
    print(f"\nPlayer 1 visited key locations:")
    print(f"   Onion dispensers: {p1_visited_onions}")
    print(f"   Pots: {p1_visited_pots}")
    print(f"   Serving areas: {p1_visited_serving}")
    print(f"   Dish dispensers: {p1_visited_dishes}")
    
    # Check for object interactions
    print(f"\n🔧 OBJECT INTERACTION ANALYSIS:")
    print("-" * 60)
    
    p0_holding_changes = []
    p1_holding_changes = []
    
    for i in range(1, len(trajectory.trajectory_steps)):
        prev_step = trajectory.trajectory_steps[i-1]
        curr_step = trajectory.trajectory_steps[i]
        
        if prev_step.player0_holding != curr_step.player0_holding:
            p0_holding_changes.append((i, prev_step.player0_holding, curr_step.player0_holding))
        
        if prev_step.player1_holding != curr_step.player1_holding:
            p1_holding_changes.append((i, prev_step.player1_holding, curr_step.player1_holding))
    
    print(f"Player 0 holding changes:")
    for step_num, prev_holding, curr_holding in p0_holding_changes:
        print(f"   Step {step_num}: {prev_holding} -> {curr_holding}")
    
    print(f"\nPlayer 1 holding changes:")
    for step_num, prev_holding, curr_holding in p1_holding_changes:
        print(f"   Step {step_num}: {prev_holding} -> {curr_holding}")
    
    # Analyze final state
    print(f"\n🏁 FINAL STATE ANALYSIS:")
    print("-" * 60)
    
    final_step = trajectory.trajectory_steps[-1]
    print(f"Final positions: P0={final_step.player0_pos}, P1={final_step.player1_pos}")
    print(f"Final holdings: P0={final_step.player0_holding}, P1={final_step.player1_holding}")
    print(f"Final reward: {final_step.reward}")
    print(f"Final done: {final_step.done}")
    
    print(f"\nFinal objects in environment:")
    for obj_pos, obj in final_step.state.objects.items():
        if obj.name == "soup":
            if hasattr(obj, 'state') and obj.state:
                soup_type, num_items, cook_time = obj.state
                status = "READY" if num_items == 3 and cook_time >= 20 else "COOKING"
                print(f"   Soup at {obj_pos}: {soup_type} soup ({num_items}/3 onions, {cook_time}/20 steps) - {status}")
            else:
                print(f"   Soup at {obj_pos}: Unknown state")
        else:
            print(f"   {obj.name} at {obj_pos}")
    
    # Potential issues analysis
    print(f"\n❌ POTENTIAL ISSUES ANALYSIS:")
    print("-" * 60)
    
    issues = []
    
    if not p0_visited_onions and not p1_visited_onions:
        issues.append("Neither player visited onion dispensers")
    
    if not p0_visited_pots and not p1_visited_pots:
        issues.append("Neither player visited pots")
    
    if not p0_visited_serving and not p1_visited_serving:
        issues.append("Neither player visited serving areas")
    
    if not p0_visited_dishes and not p1_visited_dishes:
        issues.append("Neither player visited dish dispensers")
    
    if not p0_holding_changes and not p1_holding_changes:
        issues.append("No objects were picked up or dropped")
    
    if len(p0_holding_changes) < 2 and len(p1_holding_changes) < 2:
        issues.append("Very few object interactions occurred")
    
    # Check if players got stuck in patterns
    if len(set(p0_positions)) < 5:
        issues.append("Player 0 visited very few different positions")
    
    if len(set(p1_positions)) < 5:
        issues.append("Player 1 visited very few different positions")
    
    print("Identified issues:")
    for issue in issues:
        print(f"   • {issue}")
    
    if not issues:
        print("   • No obvious issues identified")

def main():
    """Analyze a single trajectory in detail."""
    print("🔍 SINGLE TRAJECTORY ANALYSIS")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = TrajectoryVisualizer(layout_name="random3")
    
    # Generate a single trajectory with a specific starting position
    print(f"\n🎬 Generating single trajectory for analysis...")
    
    # Use a starting position that might be interesting
    player0_pos = (1, 1)
    player1_pos = (6, 1)
    
    trajectory = visualizer.generate_trajectory_with_starting_positions(
        player0_pos, player1_pos, max_steps=100
    )
    
    # Analyze the trajectory in detail
    analyze_single_trajectory_detailed(trajectory, visualizer)
    
    # Visualize the trajectory step by step
    print(f"\n🎭 VISUALIZING TRAJECTORY STEP BY STEP:")
    print("=" * 60)
    print("Press Enter to start visualization...")
    input()
    
    # Display the trajectory with animation
    visualizer.display_trajectory(trajectory.trajectory_steps, delay=1.0)
    
    # Save detailed analysis to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_data/trajectories/single_trajectory_analysis_{timestamp}.txt"
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Redirect output to file
    import io
    import contextlib
    
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        analyze_single_trajectory_detailed(trajectory, visualizer)
    
    with open(filename, 'w') as f:
        f.write("SINGLE TRAJECTORY DETAILED ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Starting positions: P0={player0_pos}, P1={player1_pos}\n")
        f.write(f"Total steps: {trajectory.total_steps}\n")
        f.write(f"Total reward: {trajectory.total_reward}\n")
        f.write(f"Final done: {trajectory.final_done}\n\n")
        f.write(output.getvalue())
    
    print(f"\n💾 Detailed analysis saved to: {filename}")

if __name__ == "__main__":
    main()
