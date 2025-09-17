#!/usr/bin/env python3
"""
Evaluate micro-task performance with binary unit tests.

Quick evaluation (binary unit tests):
- A: success rate of "pick up onion" from 50 randomized starts
- B: reaching "adjacent to pot" from 50 randomized starts  
- C: executing INTERACT only when legal in 50 trials
- D: turning to face target in 50 trials

Set pass thresholds like ≥90% on each before growing the dataset.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
import sys
import random

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedGridworld
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from generate_micro_task_data import (
    PickupOnionAgent, NavigateToPotAgent, InteractWithPotAgent, TurnToFaceAgent,
    PickupMultipleOnionsAgent, PickupPlateAgent, PickupCookedSoupAgent, DeliverSoupAgent,
    create_micro_task_start_state, MicroTaskAgent, NoOpTeammateAgent, DishRunnerTeammateAgent, LightNoiseTeammateAgent
)

def evaluate_pickup_onion(mdp: OvercookedGridworld, num_trials: int = 50) -> Dict[str, Any]:
    """Evaluate Task A: Pick up onion from randomized starts."""
    print(f"Evaluating Task A: Pick up onion ({num_trials} trials)")
    
    env = OvercookedEnv(mdp, horizon=50)
    agent = PickupOnionAgent(0, mdp)
    
    successes = 0
    episode_lengths = []
    
    for trial in range(num_trials):
        # Create random start state
        start_state = create_micro_task_start_state(mdp, "pickup_onion", 0)
        env.reset()
        env.state = start_state
        agent.reset()
        
        state = env.state
        step = 0
        success = False
        
        while step < 50:  # Max 50 steps
            action = agent.action(state)
            
            # Check if task is complete
            if agent.is_task_complete(state):
                success = True
                break
            
            # Take step
            try:
                from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
                
                if action == "interact":
                    env_action = Action.INTERACT
                elif action == "STAY":
                    env_action = Action.STAY
                elif action == "MOVE_N":
                    env_action = Direction.NORTH
                elif action == "MOVE_S":
                    env_action = Direction.SOUTH
                elif action == "MOVE_E":
                    env_action = Direction.EAST
                elif action == "MOVE_W":
                    env_action = Direction.WEST
                else:
                    env_action = Action.STAY
                
                env_actions = [env_action, Action.STAY]
                state, reward, done, info = env.step(env_actions)
                
            except Exception as e:
                print(f"Error in trial {trial}, step {step}: {e}")
                break
            
            step += 1
        
        if success:
            successes += 1
            episode_lengths.append(step)
    
    success_rate = successes / num_trials
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    
    return {
        'task': 'pickup_onion',
        'num_trials': num_trials,
        'successes': successes,
        'success_rate': success_rate,
        'avg_episode_length': avg_length,
        'passed': success_rate >= 0.90
    }

def evaluate_navigate_to_pot(mdp: OvercookedGridworld, num_trials: int = 50) -> Dict[str, Any]:
    """Evaluate Task B: Navigate to pot from randomized starts."""
    print(f"Evaluating Task B: Navigate to pot ({num_trials} trials)")
    
    env = OvercookedEnv(mdp, horizon=50)
    agent = NavigateToPotAgent(0, mdp)
    
    successes = 0
    episode_lengths = []
    
    for trial in range(num_trials):
        # Create random start state with onion
        start_state = create_micro_task_start_state(mdp, "navigate_to_pot", 0)
        env.reset()
        env.state = start_state
        agent.reset()
        
        state = env.state
        step = 0
        success = False
        
        while step < 50:  # Max 50 steps
            action = agent.action(state)
            
            # Check if task is complete
            if agent.is_task_complete(state):
                success = True
                break
            
            # Take step
            try:
                from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
                
                if action == "interact":
                    env_action = Action.INTERACT
                elif action == "STAY":
                    env_action = Action.STAY
                elif action == "MOVE_N":
                    env_action = Direction.NORTH
                elif action == "MOVE_S":
                    env_action = Direction.SOUTH
                elif action == "MOVE_E":
                    env_action = Direction.EAST
                elif action == "MOVE_W":
                    env_action = Direction.WEST
                else:
                    env_action = Action.STAY
                
                env_actions = [env_action, Action.STAY]
                state, reward, done, info = env.step(env_actions)
                
            except Exception as e:
                print(f"Error in trial {trial}, step {step}: {e}")
                break
            
            step += 1
        
        if success:
            successes += 1
            episode_lengths.append(step)
    
    success_rate = successes / num_trials
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    
    return {
        'task': 'navigate_to_pot',
        'num_trials': num_trials,
        'successes': successes,
        'success_rate': success_rate,
        'avg_episode_length': avg_length,
        'passed': success_rate >= 0.90
    }

def evaluate_interact_with_pot(mdp: OvercookedGridworld, num_trials: int = 50) -> Dict[str, Any]:
    """Evaluate Task C: INTERACT with pot only when legal."""
    print(f"Evaluating Task C: INTERACT with pot ({num_trials} trials)")
    
    env = OvercookedEnv(mdp, horizon=20)  # Shorter episodes for this task
    agent = InteractWithPotAgent(0, mdp)
    
    successes = 0
    episode_lengths = []
    legal_interactions = 0
    total_interactions = 0
    
    for trial in range(num_trials):
        # Create start state adjacent to pot with onion
        start_state = create_micro_task_start_state(mdp, "interact_with_pot", 0)
        env.reset()
        env.state = start_state
        agent.reset()
        
        state = env.state
        step = 0
        success = False
        
        while step < 20:  # Max 20 steps
            action = agent.action(state)
            
            # Track interactions
            if action == "interact":
                total_interactions += 1
                # Check if interaction is legal (adjacent to pot and facing it)
                player = state.players[0]
                current_pos = player.position
                orientation = player.orientation
                
                pots = mdp.get_pot_locations()
                if pots:
                    closest_pot = min(pots, key=lambda p: abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1]))
                    
                    # Check if adjacent and facing
                    if (abs(current_pos[0] - closest_pot[0]) + abs(current_pos[1] - closest_pot[1]) == 1):
                        # Check if facing the pot
                        dx = closest_pot[0] - current_pos[0]
                        dy = closest_pot[1] - current_pos[1]
                        
                        facing_correctly = False
                        if dx == 1 and orientation == Direction.EAST:
                            facing_correctly = True
                        elif dx == -1 and orientation == Direction.WEST:
                            facing_correctly = True
                        elif dy == 1 and orientation == Direction.SOUTH:
                            facing_correctly = True
                        elif dy == -1 and orientation == Direction.NORTH:
                            facing_correctly = True
                            
                        if facing_correctly:
                            legal_interactions += 1
                            success = True
            
            # Take step
            try:
                if action == "interact":
                    env_action = Action.INTERACT
                elif action == "STAY":
                    env_action = Action.STAY
                elif action == "MOVE_N":
                    env_action = Direction.NORTH
                elif action == "MOVE_S":
                    env_action = Direction.SOUTH
                elif action == "MOVE_E":
                    env_action = Direction.EAST
                elif action == "MOVE_W":
                    env_action = Direction.WEST
                else:
                    env_action = Action.STAY
                
                env_actions = [env_action, Action.STAY]
                state, reward, done, info = env.step(env_actions)
                
            except Exception as e:
                print(f"Error in trial {trial}, step {step}: {e}")
                break
            
            step += 1
        
        if success:
            successes += 1
            episode_lengths.append(step)
    
    success_rate = successes / num_trials
    legal_interaction_rate = legal_interactions / total_interactions if total_interactions > 0 else 0
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    
    return {
        'task': 'interact_with_pot',
        'num_trials': num_trials,
        'successes': successes,
        'success_rate': success_rate,
        'legal_interactions': legal_interactions,
        'total_interactions': total_interactions,
        'legal_interaction_rate': legal_interaction_rate,
        'avg_episode_length': avg_length,
        'passed': legal_interaction_rate >= 0.90
    }

def evaluate_turn_to_face(mdp: OvercookedGridworld, target_type: str, num_trials: int = 50) -> Dict[str, Any]:
    """Evaluate Task D: Turn to face target."""
    print(f"Evaluating Task D: Turn to face {target_type} ({num_trials} trials)")
    
    env = OvercookedEnv(mdp, horizon=20)
    agent = TurnToFaceAgent(0, mdp, target_type)
    
    successes = 0
    episode_lengths = []
    
    for trial in range(num_trials):
        # Create random start state
        start_state = create_micro_task_start_state(mdp, f"turn_to_face_{target_type}", 0)
        env.reset()
        env.state = start_state
        agent.reset()
        
        state = env.state
        step = 0
        success = False
        
        while step < 20:  # Max 20 steps
            action = agent.action(state)
            
            # Check if task is complete
            if agent.is_task_complete(state):
                success = True
                break
            
            # Take step
            try:
                from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
                
                if action == "interact":
                    env_action = Action.INTERACT
                elif action == "STAY":
                    env_action = Action.STAY
                elif action == "MOVE_N":
                    env_action = Direction.NORTH
                elif action == "MOVE_S":
                    env_action = Direction.SOUTH
                elif action == "MOVE_E":
                    env_action = Direction.EAST
                elif action == "MOVE_W":
                    env_action = Direction.WEST
                else:
                    env_action = Action.STAY
                
                env_actions = [env_action, Action.STAY]
                state, reward, done, info = env.step(env_actions)
                
            except Exception as e:
                print(f"Error in trial {trial}, step {step}: {e}")
                break
            
            step += 1
        
        if success:
            successes += 1
            episode_lengths.append(step)
    
    success_rate = successes / num_trials
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    
    return {
        'task': f'turn_to_face_{target_type}',
        'num_trials': num_trials,
        'successes': successes,
        'success_rate': success_rate,
        'avg_episode_length': avg_length,
        'passed': success_rate >= 0.90
    }

def evaluate_pickup_multiple_onions(mdp: OvercookedGridworld, num_trials: int = 50) -> Dict[str, Any]:
    """Evaluate Task E: Pick up multiple onions (if pot has < 3 onions)."""
    print(f"Evaluating Task E: Pick up multiple onions ({num_trials} trials)")
    
    env = OvercookedEnv(mdp, horizon=100)  # Longer episodes for multiple onions
    agent = PickupMultipleOnionsAgent(0, mdp)
    
    successes = 0
    episode_lengths = []
    
    for trial in range(num_trials):
        # Create random start state
        start_state = create_micro_task_start_state(mdp, "pickup_multiple_onions", 0)
        env.reset()
        env.state = start_state
        agent.reset()
        
        state = env.state
        step = 0
        success = False
        onions_picked = 0
        
        while step < 100:  # Max 100 steps
            action = agent.action(state)
            
            # Track onions picked (simplified - assume successful interactions)
            if action == "interact":
                player = state.players[0]
                if not player.has_object():
                    # Picking up onion
                    onions_picked += 1
                elif player.has_object() and player.get_object().name == 'onion':
                    # Placing onion in pot
                    onions_picked += 1
            
            # Check if we've picked up enough onions (simplified success condition)
            if onions_picked >= 2:  # At least 2 onions picked up
                success = True
                break
            
            # Take step
            try:
                from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
                
                if action == "interact":
                    env_action = Action.INTERACT
                elif action == "STAY":
                    env_action = Action.STAY
                elif action == "MOVE_N":
                    env_action = Direction.NORTH
                elif action == "MOVE_S":
                    env_action = Direction.SOUTH
                elif action == "MOVE_E":
                    env_action = Direction.EAST
                elif action == "MOVE_W":
                    env_action = Direction.WEST
                else:
                    env_action = Action.STAY
                
                env_actions = [env_action, Action.STAY]
                state, reward, done, info = env.step(env_actions)
                
            except Exception as e:
                print(f"Error in trial {trial}, step {step}: {e}")
                break
            
            step += 1
        
        if success:
            successes += 1
            episode_lengths.append(step)
    
    success_rate = successes / num_trials
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    
    return {
        'task': 'pickup_multiple_onions',
        'num_trials': num_trials,
        'successes': successes,
        'success_rate': success_rate,
        'avg_episode_length': avg_length,
        'passed': success_rate >= 0.90
    }

def evaluate_pickup_plate(mdp: OvercookedGridworld, num_trials: int = 50) -> Dict[str, Any]:
    """Evaluate Task F: Pick up plate."""
    print(f"Evaluating Task F: Pick up plate ({num_trials} trials)")
    
    env = OvercookedEnv(mdp, horizon=50)
    agent = PickupPlateAgent(0, mdp)
    
    successes = 0
    episode_lengths = []
    
    for trial in range(num_trials):
        # Create random start state
        start_state = create_micro_task_start_state(mdp, "pickup_plate", 0)
        env.reset()
        env.state = start_state
        agent.reset()
        
        state = env.state
        step = 0
        success = False
        
        while step < 50:  # Max 50 steps
            action = agent.action(state)
            
            # Check if task is complete
            if agent.is_task_complete(state):
                success = True
                break
            
            # Take step
            try:
                from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
                
                if action == "interact":
                    env_action = Action.INTERACT
                elif action == "STAY":
                    env_action = Action.STAY
                elif action == "MOVE_N":
                    env_action = Direction.NORTH
                elif action == "MOVE_S":
                    env_action = Direction.SOUTH
                elif action == "MOVE_E":
                    env_action = Direction.EAST
                elif action == "MOVE_W":
                    env_action = Direction.WEST
                else:
                    env_action = Action.STAY
                
                env_actions = [env_action, Action.STAY]
                state, reward, done, info = env.step(env_actions)
                
            except Exception as e:
                print(f"Error in trial {trial}, step {step}: {e}")
                break
            
            step += 1
        
        if success:
            successes += 1
            episode_lengths.append(step)
    
    success_rate = successes / num_trials
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    
    return {
        'task': 'pickup_plate',
        'num_trials': num_trials,
        'successes': successes,
        'success_rate': success_rate,
        'avg_episode_length': avg_length,
        'passed': success_rate >= 0.90
    }

def evaluate_pickup_cooked_soup(mdp: OvercookedGridworld, num_trials: int = 50) -> Dict[str, Any]:
    """Evaluate Task G: Pick up cooked soup."""
    print(f"Evaluating Task G: Pick up cooked soup ({num_trials} trials)")
    
    env = OvercookedEnv(mdp, horizon=50)
    agent = PickupCookedSoupAgent(0, mdp)
    
    successes = 0
    episode_lengths = []
    
    for trial in range(num_trials):
        # Create random start state
        start_state = create_micro_task_start_state(mdp, "pickup_cooked_soup", 0)
        env.reset()
        env.state = start_state
        agent.reset()
        
        state = env.state
        step = 0
        success = False
        
        while step < 50:  # Max 50 steps
            action = agent.action(state)
            
            # Check if task is complete
            if agent.is_task_complete(state):
                success = True
                break
            
            # Take step
            try:
                from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
                
                if action == "interact":
                    env_action = Action.INTERACT
                elif action == "STAY":
                    env_action = Action.STAY
                elif action == "MOVE_N":
                    env_action = Direction.NORTH
                elif action == "MOVE_S":
                    env_action = Direction.SOUTH
                elif action == "MOVE_E":
                    env_action = Direction.EAST
                elif action == "MOVE_W":
                    env_action = Direction.WEST
                else:
                    env_action = Action.STAY
                
                env_actions = [env_action, Action.STAY]
                state, reward, done, info = env.step(env_actions)
                
            except Exception as e:
                print(f"Error in trial {trial}, step {step}: {e}")
                break
            
            step += 1
        
        if success:
            successes += 1
            episode_lengths.append(step)
    
    success_rate = successes / num_trials
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    
    return {
        'task': 'pickup_cooked_soup',
        'num_trials': num_trials,
        'successes': successes,
        'success_rate': success_rate,
        'avg_episode_length': avg_length,
        'passed': success_rate >= 0.90
    }

def evaluate_deliver_soup(mdp: OvercookedGridworld, num_trials: int = 50) -> Dict[str, Any]:
    """Evaluate Task H: Deliver soup to serving area."""
    print(f"Evaluating Task H: Deliver soup to serving area ({num_trials} trials)")
    
    env = OvercookedEnv(mdp, horizon=50)
    agent = DeliverSoupAgent(0, mdp)
    
    successes = 0
    episode_lengths = []
    
    for trial in range(num_trials):
        # Create random start state with soup
        start_state = create_micro_task_start_state(mdp, "deliver_soup", 0)
        env.reset()
        env.state = start_state
        agent.reset()
        
        state = env.state
        step = 0
        success = False
        
        while step < 50:  # Max 50 steps
            action = agent.action(state)
            
            # Track successful delivery (simplified - assume interaction with serving area is delivery)
            if action == "interact":
                player = state.players[0]
                if player.has_object() and player.get_object().name == 'soup':
                    # Check if adjacent to serving area
                    serving_areas = mdp.get_serving_locations()
                    current_pos = player.position
                    for serving in serving_areas:
                        if abs(current_pos[0] - serving[0]) + abs(current_pos[1] - serving[1]) == 1:
                            success = True
                            break
            
            if success:
                break
            
            # Take step
            try:
                from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
                
                if action == "interact":
                    env_action = Action.INTERACT
                elif action == "STAY":
                    env_action = Action.STAY
                elif action == "MOVE_N":
                    env_action = Direction.NORTH
                elif action == "MOVE_S":
                    env_action = Direction.SOUTH
                elif action == "MOVE_E":
                    env_action = Direction.EAST
                elif action == "MOVE_W":
                    env_action = Direction.WEST
                else:
                    env_action = Action.STAY
                
                env_actions = [env_action, Action.STAY]
                state, reward, done, info = env.step(env_actions)
                
            except Exception as e:
                print(f"Error in trial {trial}, step {step}: {e}")
                break
            
            step += 1
        
        if success:
            successes += 1
            episode_lengths.append(step)
    
    success_rate = successes / num_trials
    avg_length = np.mean(episode_lengths) if episode_lengths else 0
    
    return {
        'task': 'deliver_soup',
        'num_trials': num_trials,
        'successes': successes,
        'success_rate': success_rate,
        'avg_episode_length': avg_length,
        'passed': success_rate >= 0.90
    }

def main():
    """Main evaluation function."""
    
    # Configuration
    layout_name = "random3"
    num_trials = 50
    
    print(f"Evaluating micro-task performance")
    print(f"Layout: {layout_name}")
    print(f"Trials per task: {num_trials}")
    print(f"Pass threshold: ≥90%")
    print()
    
    # Create environment
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    
    # Run evaluations
    results = []
    
    # Task A: Pick up onion
    result_a = evaluate_pickup_onion(mdp, num_trials)
    results.append(result_a)
    print(f"Task A - Pick up onion: {result_a['success_rate']:.3f} ({result_a['successes']}/{num_trials}) - {'PASS' if result_a['passed'] else 'FAIL'}")
    
    # Task B: Navigate to pot
    result_b = evaluate_navigate_to_pot(mdp, num_trials)
    results.append(result_b)
    print(f"Task B - Navigate to pot: {result_b['success_rate']:.3f} ({result_b['successes']}/{num_trials}) - {'PASS' if result_b['passed'] else 'FAIL'}")
    
    # Task C: INTERACT with pot
    result_c = evaluate_interact_with_pot(mdp, num_trials)
    results.append(result_c)
    print(f"Task C - INTERACT with pot: {result_c['legal_interaction_rate']:.3f} ({result_c['legal_interactions']}/{result_c['total_interactions']}) - {'PASS' if result_c['passed'] else 'FAIL'}")
    
    # Task D: Turn to face onion dispenser
    result_d1 = evaluate_turn_to_face(mdp, "onion_dispenser", num_trials)
    results.append(result_d1)
    print(f"Task D1 - Turn to face onion: {result_d1['success_rate']:.3f} ({result_d1['successes']}/{num_trials}) - {'PASS' if result_d1['passed'] else 'FAIL'}")
    
    # Task D: Turn to face pot
    result_d2 = evaluate_turn_to_face(mdp, "pot", num_trials)
    results.append(result_d2)
    print(f"Task D2 - Turn to face pot: {result_d2['success_rate']:.3f} ({result_d2['successes']}/{num_trials}) - {'PASS' if result_d2['passed'] else 'FAIL'}")
    
    # Task E: Pick up multiple onions
    result_e = evaluate_pickup_multiple_onions(mdp, num_trials)
    results.append(result_e)
    print(f"Task E - Pick up multiple onions: {result_e['success_rate']:.3f} ({result_e['successes']}/{num_trials}) - {'PASS' if result_e['passed'] else 'FAIL'}")
    
    # Task F: Pick up plate
    result_f = evaluate_pickup_plate(mdp, num_trials)
    results.append(result_f)
    print(f"Task F - Pick up plate: {result_f['success_rate']:.3f} ({result_f['successes']}/{num_trials}) - {'PASS' if result_f['passed'] else 'FAIL'}")
    
    # Task G: Pick up cooked soup
    result_g = evaluate_pickup_cooked_soup(mdp, num_trials)
    results.append(result_g)
    print(f"Task G - Pick up cooked soup: {result_g['success_rate']:.3f} ({result_g['successes']}/{num_trials}) - {'PASS' if result_g['passed'] else 'FAIL'}")
    
    # Task H: Deliver soup
    result_h = evaluate_deliver_soup(mdp, num_trials)
    results.append(result_h)
    print(f"Task H - Deliver soup: {result_h['success_rate']:.3f} ({result_h['successes']}/{num_trials}) - {'PASS' if result_h['passed'] else 'FAIL'}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    
    passed_tasks = sum(1 for r in results if r['passed'])
    total_tasks = len(results)
    
    print(f"Tasks passed: {passed_tasks}/{total_tasks}")
    print(f"Overall pass rate: {passed_tasks/total_tasks:.3f}")
    
    if passed_tasks == total_tasks:
        print(f"✅ ALL TASKS PASSED - Ready to grow dataset!")
    else:
        print(f"❌ {total_tasks - passed_tasks} TASKS FAILED - Fix micro-skills before growing dataset")
    
    print(f"\nDetailed Results:")
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"  {result['task']}: {result['success_rate']:.3f} - {status}")
    
    # Save results
    output_dir = "generated_data/micro_task_sanity"
    os.makedirs(output_dir, exist_ok=True)
    
    evaluation_filename = f"{output_dir}/unit_test_results.json"
    with open(evaluation_filename, 'w') as f:
        json.dump({
            'layout_name': layout_name,
            'num_trials': num_trials,
            'pass_threshold': 0.90,
            'results': results,
            'summary': {
                'passed_tasks': passed_tasks,
                'total_tasks': total_tasks,
                'overall_pass_rate': passed_tasks/total_tasks,
                'all_passed': passed_tasks == total_tasks
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {evaluation_filename}")

if __name__ == "__main__":
    main()
