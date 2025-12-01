#!/usr/bin/env python3
"""
Baseline ProAgent demo without CoT or memory modules.
Runs vanilla agent on five_by_five layout for 5 trials with random starts.
Tracks successful soup deliveries across 200 steps.
"""
import os
import sys
import argparse
import numpy as np
import pygame
import time
from typing import Dict, Any
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROAGENT_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'proagent', 'src'))
# Ensure ProAgent src takes precedence so we import the right utils
if PROAGENT_SRC in sys.path:
    sys.path.remove(PROAGENT_SRC)
sys.path.insert(0, PROAGENT_SRC)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from shared.envs.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from shared.agents.onion_specialist_agent import OnionSpecialistAgent

# Use ProAgent harness utilities and agent
import utils as pro_utils  # type: ignore
from proagent.proagent_with_intervention import ProAgentWithIntervention


def convert_action(a):
    """Convert various action formats to standard Action/Direction."""
    if isinstance(a, (Direction, Action)):
        return a
    if isinstance(a, int):
        mapping = {0: Action.STAY, 1: Direction.EAST, 2: Direction.SOUTH,
                   3: Direction.NORTH, 4: Direction.WEST, 5: Action.INTERACT}
        return mapping.get(a, Action.STAY)
    if isinstance(a, str):
        s = a.strip().lower()
        if s == 'interact':
            return Action.INTERACT
        if s == 'stay':
            return Action.STAY
        if s == 'north':
            return Direction.NORTH
        if s == 'south':
            return Direction.SOUTH
        if s == 'east':
            return Direction.EAST
        if s == 'west':
            return Direction.WEST
    if isinstance(a, tuple) and len(a) == 2:
        # Convert tuple elements to ints in case they're floats or other numeric types
        try:
            a_int = (int(a[0]), int(a[1]))
        except (ValueError, TypeError):
            a_int = a
        vec_map = {
            (1, 0): Direction.EAST,
            (0, 1): Direction.SOUTH,
            (0, -1): Direction.NORTH,
            (-1, 0): Direction.WEST,
        }
        result = vec_map.get(a_int, Action.STAY)
        return result
    return Action.STAY


def run_single_trial(trial_num: int, layout_name: str, layout_key: str, horizon: int = 200, 
                     verbose: bool = False, visualize: bool = False, 
                     screen=None, viz=None, font=None, clock=None, quiet: bool = True) -> Dict[str, Any]:
    """
    Run a single trial and return results.
    
    Returns:
        dict with keys: soup_deliveries, steps, success
    """
    # Suppress output if quiet mode
    if quiet and not visualize:
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        stdout_ctx = redirect_stdout(stdout_capture)
        stderr_ctx = redirect_stderr(stderr_capture)
        stdout_ctx.__enter__()
        stderr_ctx.__enter__()
    
    try:
        # Create MDP and environment
        mdp = OvercookedGridworld.from_layout_name(layout_name)
        start_fn = mdp.get_random_start_state_fn(random_start_pos=True)
        env = OvercookedEnv(mdp, start_state_fn=start_fn, horizon=horizon)
        env.reset()
        
        # Create agents
        # Player0: Baseline ProAgent (use_baseline=True disables CoT and memory)
        base_agent = pro_utils.make_agent(
            'ProAgent', mdp, layout_key, 
            model='gpt-4o-mini', 
            retrival_method='recent_k', 
            K=10, 
            prompt_level='l2-ap', 
            belief_revision=True, 
            auto_unstuck=False
        )
        
        p0 = ProAgentWithIntervention(
            base_agent.mlam, 
            layout_key,
            model='gpt-4o-mini',
            retrival_method='recent_k',
            K=10,
            prompt_level='l2-ap',
            belief_revision=True,
            auto_unstuck=False,
            use_baseline=True  # Enable baseline mode (no CoT, no memory)
        )
        p0.set_agent_index(0)
        
        # Player1: OnionSpecialistAgent
        p1 = OnionSpecialistAgent(mdp, agent_idx=1, agent_name='OnionSpec')
    except Exception as e:
        if quiet and not visualize:
            stdout_ctx.__exit__(None, None, None)
            stderr_ctx.__exit__(None, None, None)
        raise
    
    # Track deliveries
    soup_deliveries = 0
    prev_holding_soup = False
    step_count = 0
    failures = []
    paused = False
    
    state = env.state
    
    # Visualization helper function
    def wrap_text(text, font_obj, max_width):
        if not text:
            return [""]
        words = text.split()
        lines = []
        current = []
        for w in words:
            test_line = (" ".join(current + [w])).strip()
            if font_obj and font_obj.size(test_line)[0] <= max_width or not current:
                current.append(w)
            else:
                lines.append(" ".join(current))
                current = [w]
        if current:
            lines.append(" ".join(current))
        return lines
    
    try:
        while step_count < horizon:
            # Handle pygame events if visualizing
            if visualize and screen is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return {
                            'soup_deliveries': soup_deliveries,
                            'steps': step_count,
                            'success': len(failures) == 0,
                            'failures': failures,
                            'trial': trial_num
                        }
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return {
                                'soup_deliveries': soup_deliveries,
                                'steps': step_count,
                                'success': len(failures) == 0,
                                'failures': failures,
                                'trial': trial_num
                            }
                
                if paused:
                    # Still render when paused
                    screen.fill((255, 255, 255))
                    if viz is not None:
                        img = viz.render_state(state, mdp.terrain_mtx)
                        screen.blit(img, (10, 10))
                    
                    # Render UI
                    info_lines = [
                        f'Trial: {trial_num}',
                        f'Step: {step_count}/{horizon}',
                        'Controls: SPACE=pause/resume, ESC=quit',
                        f'Deliveries: {soup_deliveries}',
                        '',
                        'PAUSED'
                    ]
                    
                    if viz is not None:
                        text_panel_x = img.get_width() + 30
                    else:
                        text_panel_x = 400
                    text_panel_y = 10
                    text_panel_width = max(500, (screen.get_width() - text_panel_x - 20) if screen else 500)
                    
                    for i, line in enumerate(info_lines):
                        if font is not None:
                            txt = font.render(line, True, (0, 0, 0))
                            screen.blit(txt, (text_panel_x, text_panel_y + i * 22))
                    
                    pygame.display.flip()
                    if clock is not None:
                        clock.tick(10)  # Slower when paused
                    continue
            
            if paused:
                continue
            
            # Get actions
            res0 = p0.action(state)
            # Handle different return formats from action() method
            # The action might be:
            # 1. A tuple like ((-1, 0), {}) - (action_tuple, info_dict) from version 1.1.0
            # 2. A tuple like (-1, 0) - action tuple directly (direction vector)
            # 3. A Direction/Action object
            # 4. An integer
            if isinstance(res0, tuple):
                if len(res0) == 2 and isinstance(res0[1], dict):
                    # Format: (action, info_dict)
                    a0 = res0[0]
                elif len(res0) == 2 and isinstance(res0[0], (int, float)) and isinstance(res0[1], (int, float)):
                    # Format: (x, y) - this IS the action (direction vector)
                    a0 = res0
                else:
                    # Single element tuple or other tuple format
                    a0 = res0[0] if len(res0) == 1 else res0
            else:
                a0 = res0
            
            a1 = p1.action(state)
            
            # Convert actions
            a0_conv = convert_action(a0)
            a1_conv = convert_action(a1)
            joint = (a0_conv, a1_conv)
            
            # Check current state before stepping
            current_holding_soup = (
                state.players[0].has_object() and 
                state.players[0].get_object().name == 'soup'
            )
            
            # Step environment
            # env.step() returns: (next_state, timestep_sparse_reward, done, env_info)
            # timestep_sparse_reward is the sum of sparse rewards (delivery reward = 20 for five_by_five)
            state, sparse_reward, done, info = env.step(joint)
            step_count += 1
            
            # Track deliveries using sparse reward (delivery reward is 20 for five_by_five layout)
            # The sparse reward is only given when soup is delivered
            if sparse_reward > 0:
                soup_deliveries += 1
            
            # Update previous state for next iteration
            next_holding_soup = (
                state.players[0].has_object() and 
                state.players[0].get_object().name == 'soup'
            )
            prev_holding_soup = next_holding_soup
            
            # Render visualization if enabled
            if visualize and screen is not None and viz is not None:
                screen.fill((255, 255, 255))
                img = viz.render_state(state, mdp.terrain_mtx)
                screen.blit(img, (10, 10))
                
                # UI information
                info_lines = [
                    f'Trial: {trial_num}',
                    f'Step: {step_count}/{horizon}',
                    'Controls: SPACE=pause/resume, ESC=quit',
                    f'Deliveries: {soup_deliveries}',
                    '',
                    f'Player0 Position: {state.players[0].position}',
                    f'Player0 Holding: {state.players[0].get_object().name if state.players[0].has_object() else "nothing"}',
                    f'Player1 Position: {state.players[1].position}',
                    f'Player1 Holding: {state.players[1].get_object().name if state.players[1].has_object() else "nothing"}',
                ]
                
                # Add current ML action if available
                try:
                    current_ml = getattr(p0, 'current_ml_action', None)
                    if current_ml:
                        info_lines.append(f'Current ML Action: {current_ml}')
                except Exception:
                    pass
                
                # Add plan steps if available
                try:
                    last_plan = getattr(p0, 'last_plan', None)
                    if last_plan and last_plan.get('steps'):
                        steps_str = ', '.join(last_plan.get('steps', [])[:3])
                        if len(last_plan.get('steps', [])) > 3:
                            steps_str += '...'
                        info_lines.append(f'Plan Steps: {steps_str}')
                except Exception:
                    pass
                
                # Render text panel
                text_panel_x = img.get_width() + 30
                text_panel_y = 10
                text_panel_width = max(500, screen.get_width() - text_panel_x - 20)
                
                wrapped_lines = []
                for line in info_lines:
                    if font is not None:
                        wrapped_lines.extend(wrap_text(line, font, text_panel_width))
                
                max_lines = 35
                start_line = max(0, len(wrapped_lines) - max_lines)
                visible_lines = wrapped_lines[start_line:]
                
                for i, line in enumerate(visible_lines):
                    if font is not None:
                        txt = font.render(line, True, (0, 0, 0))
                        screen.blit(txt, (text_panel_x, text_panel_y + i * 22))
                
                pygame.display.flip()
                if clock is not None:
                    clock.tick(1.5)  # Control frame rate
                else:
                    time.sleep(0.1)  # Small delay if no clock
            
            if done:
                break
                
    except Exception as e:
        failures.append({
            'error': str(e),
            'step': step_count,
            'trial': trial_num
        })
    finally:
        # Restore output if it was suppressed
        if quiet and not visualize:
            stdout_ctx.__exit__(None, None, None)
            stderr_ctx.__exit__(None, None, None)
    
    return {
        'soup_deliveries': soup_deliveries,
        'steps': step_count,
        'success': len(failures) == 0,
        'failures': failures,
        'trial': trial_num
    }


def main():
    parser = argparse.ArgumentParser(description='Baseline ProAgent Demo (No CoT, No Memory)')
    parser.add_argument('--layout', type=str, default='five_by_five', help='Layout name')
    parser.add_argument('--horizon', type=int, default=200, help='Max steps per trial')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of trials to run')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    parser.add_argument('--visualize', action='store_true', help='Enable pygame visualization (only for single trial)')
    args = parser.parse_args()
    
    layout_name = args.layout
    # Map layout key to actual layout name if needed
    if layout_name in pro_utils.NEW_LAYOUTS:
        layout_name = pro_utils.NEW_LAYOUTS[layout_name]
    
    # Suppress all output during execution unless verbose or visualize
    quiet_mode = not args.verbose and not args.visualize
    
    # Initialize pygame if visualization is enabled
    screen = None
    viz = None
    font = None
    clock = None
    if args.visualize:
        pygame.init()
        screen = pygame.display.set_mode((1400, 800))
        pygame.display.set_caption(f'Baseline ProAgent Demo [{layout_name}] (No CoT, No Memory)')
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)
        viz = StateVisualizer()
    
    # Run trials
    results = []
    for trial in range(1, args.num_trials + 1):
        # Determine layout key for agent creation
        layout_key_for_agent = args.layout if args.layout in pro_utils.NEW_LAYOUTS else layout_name
        # Only visualize first trial if visualization is enabled
        should_visualize = args.visualize and trial == 1
        result = run_single_trial(
            trial, layout_name, layout_key_for_agent, args.horizon, args.verbose,
            visualize=should_visualize, screen=screen, viz=viz, font=font, clock=clock,
            quiet=quiet_mode
        )
        results.append(result)
    
    # Print only delivery counts
    deliveries = [r['soup_deliveries'] for r in results]
    for r in results:
        print(f"Trial {r['trial']}: {r['soup_deliveries']} deliveries")
    
    # Clean up pygame if it was initialized
    if args.visualize and pygame.get_init():
        pygame.quit()


if __name__ == '__main__':
    main()

