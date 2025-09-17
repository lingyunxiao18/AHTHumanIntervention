#!/usr/bin/env python3
"""
LLM Human Intervention Demo - Oscillation Testing

Test Simple Hand Coded Agent (Player 0) with Onion Specialist (Player 1) with LLM-based human intervention.
This demo focuses on testing oscillation scenarios where agents might get stuck in repetitive behaviors.
Press 'p' to pause and input natural language commands that are processed by OpenAI GPT-4.
"""

import os
import sys
import pygame
import numpy as np
import random
import time
from collections import deque

# Ensure module imports work when run from repo root or directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import LLM intervention system
from pipelines.llm_baseline.core.llm_intervention_system import LLMInterventionSystem, InterventionResult

# Import Overcooked components
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from shared.envs.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# Import agents
from shared.agents import SimpleHandcodedAgent, SimpleHandcodedAgentHumanGuidance

def convert_action(a):
    """Convert action to proper format."""
    if isinstance(a, (Direction, Action)):
        return a
    if isinstance(a, int):
        mapping = {0: Action.STAY, 1: Direction.EAST, 2: Direction.SOUTH, 
                  3: Direction.NORTH, 4: Direction.WEST, 5: Action.INTERACT}
        return mapping.get(a, Action.STAY)
    return Action.STAY

def detect_oscillation(position_history, window_size=10):
    """Detect if agent is oscillating between positions."""
    if len(position_history) < window_size:
        return False, None
    
    # Check last window_size positions
    recent_positions = list(position_history)[-window_size:]
    
    # Look for back-and-forth patterns
    if len(recent_positions) >= 4:
        # Check for A-B-A-B pattern
        if (recent_positions[-4] == recent_positions[-2] and 
            recent_positions[-3] == recent_positions[-1] and
            recent_positions[-4] != recent_positions[-3]):
            return True, f"Oscillating between {recent_positions[-4]} and {recent_positions[-3]}"
        
        # Check for longer oscillation patterns
        if len(recent_positions) >= 6:
            if (recent_positions[-6] == recent_positions[-3] and 
                recent_positions[-5] == recent_positions[-2] and
                recent_positions[-4] == recent_positions[-1]):
                return True, f"3-step oscillation detected"
    
    return False, None

def main():
    """Main demo with LLM intervention focused on oscillation detection."""
    print("🔄 LLM HUMAN INTERVENTION DEMO - OSCILLATION TESTING")
    print("Player 0: Simple A* Agent (Dumber) - LLM INTERVENTION TARGET")
    print("Player 1: Onion Specialist Agent (Autonomous)")
    print("Controls: 'p' = Pause & Intervention, SPACE = Pause, ESC = Quit")
    print("=" * 70)
    
    # Use schelling layout which is highly symmetric and induces oscillating behavior
    layouts = ["schelling"]
    layout_index = 0
    
    def initialize_layout(layout_name):
        """Initialize environment with given layout."""
        print(f"\n🎯 Initializing layout: {layout_name}")
        mdp = OvercookedGridworld.from_layout_name(layout_name)
        
        # Show the actual start positions parsed from the layout
        print(f"🎯 Using start positions: {mdp.start_player_positions}")
        env = OvercookedEnv(mdp, horizon=500)
        env.reset()
        state = env.state
        
        # Initialize agents
        agent0 = SimpleHandcodedAgentHumanGuidance(mdp, agent_idx=0, agent_name="SimpleHandcodedAgentHumanGuidance")
        agent1 = SimpleHandcodedAgentHumanGuidance(mdp, agent_idx=1, agent_name="SimpleHandcodedAgentHumanGuidance")
        
        return mdp, env, state, agent0, agent1
    
    # Initialize first layout
    layout = layouts[layout_index]
    mdp, env, state, agent0, agent1 = initialize_layout(layout)
    
    # Initialize LLM intervention system
    intervention_system = LLMInterventionSystem()
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((1400, 900))
    pygame.display.set_caption("LLM Human Intervention Demo - Oscillation Testing")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)
    
    # Initialize visualizer
    visualizer = StateVisualizer()
    
    # Game state
    step_count = 0
    running = True
    paused = False
    intervention_mode = False
    intervention_text = ""
    last_intervention = None
    soup_deliveries = 0
    
    # Oscillation detection
    player0_position_history = deque(maxlen=20)
    player1_position_history = deque(maxlen=20)
    oscillation_detected = False
    oscillation_info = ""
    oscillation_step = -1
    
    while running and step_count < 300:  # Longer horizon for oscillation testing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if intervention_mode:
                    if event.key == pygame.K_RETURN:
                        # Process intervention
                        if intervention_text.strip():
                            print(f"\n👤 Human: '{intervention_text}'")
                            
                            # Get SimpleHandCoded agent state for context (Player 0 - the "dumb" agent)
                            agent_state = agent0.get_intervention_state(state)
                            
                            # Add oscillation context to the state
                            if oscillation_detected:
                                agent_state['oscillation_detected'] = True
                                agent_state['oscillation_info'] = oscillation_info
                            
                            # Process with LLM
                            result = intervention_system.process_intervention(intervention_text, agent_state)
                            
                            # Convert InterventionResult to dict for compatibility
                            if result:
                                result_dict = {
                                    'action_override': result.action_override,
                                    'macro_override': result.macro_override,
                                    'duration': result.duration,
                                    'reasoning': result.reasoning
                                }
                                print(f"🤖 LLM Intervention Result: {result_dict}")
                                
                                # Apply intervention to SimpleHandCoded agent (Player 0) only if LLM succeeded
                                agent0.apply_intervention(result_dict)
                                last_intervention = {
                                    'command': intervention_text,
                                    'result': result_dict,
                                    'step': step_count
                                }
                                
                                # Reset oscillation detection after intervention
                                oscillation_detected = False
                                oscillation_info = ""
                                oscillation_step = -1
                            else:
                                print("❌ Intervention failed - LLM could not process command")
                                last_intervention = {
                                    'command': intervention_text,
                                    'result': None,
                                    'step': step_count
                                }
                        
                        intervention_mode = False
                        intervention_text = ""
                        paused = False
                    elif event.key == pygame.K_BACKSPACE:
                        intervention_text = intervention_text[:-1]
                    else:
                        if event.unicode:
                            intervention_text += event.unicode
                        elif event.key == pygame.K_SPACE:
                            intervention_text += " "
                else:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        # Enter intervention mode
                        intervention_mode = True
                        intervention_text = ""
                        paused = True
                        print(f"\n🎤 INTERVENTION MODE: Type your command and press ENTER")
        
        if not paused and not intervention_mode:
            # Get actions from both agents
            action0 = agent0.get_action(state)
            action1 = agent1.get_action(state)
            
            # Convert to proper format
            joint_action = (convert_action(action0), convert_action(action1))
            
            # Step environment
            state, reward, done, info = env.step(joint_action)
            step_count += 1
            
            # Track positions for oscillation detection
            player0_position_history.append(state.players[0].position)
            player1_position_history.append(state.players[1].position)
            
            # Detect oscillations
            if not oscillation_detected:
                is_oscillating, osc_info = detect_oscillation(player0_position_history)
                if is_oscillating:
                    oscillation_detected = True
                    oscillation_info = f"Player 0: {osc_info}"
                    oscillation_step = step_count
                    print(f"🔄 OSCILLATION DETECTED at step {step_count}: {oscillation_info}")
            
            # Debug first 11 steps: show positions and actions
            if step_count <= 10:
                p0 = state.players[0].position
                p1 = state.players[1].position
                print(f"[t={step_count}] P0 pos={p0} act={action0} | P1 pos={p1} act={action1}")
            
            # Add to intervention history
            ego_item = agent0._get_item_name(state.players[0].get_object()) if state.players[0].has_object() else "none"
            mate_item = agent0._get_item_name(state.players[1].get_object()) if state.players[1].has_object() else "none"
            
            # Check if there was an intervention in this step
            current_intervention = None
            current_intervention_result = None
            if last_intervention and last_intervention['step'] == step_count - 1:
                current_intervention = last_intervention['command']
                current_intervention_result = last_intervention['result']
            
            intervention_system.add_to_history(
                step=step_count,
                ego_action=action0,
                mate_action=action1,
                ego_pos=state.players[0].position,
                mate_pos=state.players[1].position,
                ego_item=ego_item,
                mate_item=mate_item,
                intervention=current_intervention,
                intervention_result=current_intervention_result
            )
            
            # Track successful soup deliveries
            if hasattr(env, 'prev_agent_state') and env.prev_agent_state:
                prev_holding_soup = env.prev_agent_state.get('holding_soup', False)
                current_holding_soup = state.players[0].has_object() and agent0._get_item_name(state.players[0].get_object()) == "soup"
                
                if prev_holding_soup and not current_holding_soup:
                    soup_deliveries += 1
                    print(f"🎉 SOUP DELIVERED! Successful delivery #{soup_deliveries}")
            
            # Store current state for next comparison
            env.prev_agent_state = {
                'holding_soup': state.players[0].has_object() and agent0._get_item_name(state.players[0].get_object()) == "soup"
            }
            
            # Debug: Print delivery info every 50 steps
            if step_count % 50 == 0:
                print(f"🔍 DEBUG Step {step_count}: Successful deliveries: {soup_deliveries}")
        
        # Render the game
        screen.fill((255, 255, 255))
        
        # Render Overcooked state
        img_surface = visualizer.render_state(state, mdp.terrain_mtx)
        screen.blit(img_surface, (10, 10))
        
        # Render agent info
        info_y = 420
        agent_info = [
            f"Step: {step_count}/300",
            f"Layout: {layout}",
            f"Player 0 (SimpleHandcoded): {agent0.heuristic}",
            f"Player 1 (OnionSpec): {agent1.heuristic}",
            f"Successful Deliveries: {soup_deliveries}",
            "",
            "🔄 LLM Human Intervention Demo - Oscillation Testing",
            "Controls: 'p'=Intervention, SPACE=Pause, ESC=Quit",
            "",
        ]
        
        # Add oscillation status
        if oscillation_detected:
            agent_info.extend([
                f"🔄 OSCILLATION DETECTED (Step {oscillation_step}):",
                f"   {oscillation_info}",
                "   Consider intervention to break the cycle!",
                "",
            ])
        
        # Add intervention status
        if intervention_mode:
            agent_info.extend([
                "🎤 INTERVENTION MODE - Type command:",
                f"'{intervention_text}_'",
                "",
                "Target: SimpleAStar Agent (Player 0)",
            ])
        elif last_intervention:
            cmd = last_intervention.get('command', '')
            step_meta = last_intervention.get('step', '?')
            res = last_intervention.get('result')
            if res:
                cmd_type = res.get('command_type', 'unknown')
                reasoning = (res.get('reasoning') or '')[:50]
                agent_info.extend([
                    f"Last intervention (Step {step_meta}):",
                    f"👤 '{cmd}'",
                    f"🤖 {cmd_type} - {reasoning}...",
                    "",
                ])
            else:
                agent_info.extend([
                    f"Last intervention (Step {step_meta}):",
                    f"👤 '{cmd}'",
                    "🤖 failed to parse/process",
                    "",
                ])
        
        for i, line in enumerate(agent_info):
            if "OSCILLATION" in line:
                color = (255, 140, 0)  # Orange for oscillation warnings
            elif "Player 0" in line:
                color = (0, 100, 0)
            elif "Player 1" in line:
                color = (200, 0, 0)
            elif "🎤" in line:
                color = (0, 0, 200)
            else:
                color = (0, 0, 0)
            text_surface = font.render(line[:80], True, color)  # Truncate long lines
            screen.blit(text_surface, (10, info_y + i * 26))
        
        # Show current agent state for intervention context
        if intervention_mode:
            context_y = info_y + len(agent_info) * 26 + 20
            agent_state = agent0.get_intervention_state(state)
            context_info = [
                "🤖 SimpleAStar Agent Context (Player 0):",
                f"Position: {agent_state['agent_pos']}",
                f"Target: {agent_state['target_pos']}",
                f"Macro: {agent_state['current_macro']}",
                f"Holding: {agent_state['holding_item']}",
            ]
            
            # Add oscillation context if detected
            if oscillation_detected:
                context_info.extend([
                    f"🔄 Oscillation: {oscillation_info}",
                    "Recent positions: " + str(list(player0_position_history)[-5:])
                ])
            
            for i, line in enumerate(context_info):
                text_surface = small_font.render(line, True, (100, 100, 100))
                screen.blit(text_surface, (10, context_y + i * 20))
        
        if paused and not intervention_mode:
            pause_text = font.render("PAUSED - Press SPACE to continue, 'p' for intervention", True, (255, 0, 0))
            screen.blit(pause_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(5)  # Slow down demo for clearer visualization
    
    pygame.quit()
    
    print(f"\n📊 OSCILLATION DEMO COMPLETED")
    print(f"Steps: {step_count}")
    print(f"Successful soup deliveries: {soup_deliveries}")
    if oscillation_detected:
        print(f"Oscillation detected at step {oscillation_step}: {oscillation_info}")
    if last_intervention:
        result_type = "success" if last_intervention['result'] else "failed"
        print(f"Last intervention: '{last_intervention['command']}' ({result_type})")

if __name__ == "__main__":
    main()
