#!/usr/bin/env python3
"""
LLM Human Intervention Demo - Sudden FIRE Hazard

At timestep 10, a FIRE hazard ('F') appears on a random empty corridor tile (' ') and persists for 5 steps (t in [10, 14]).
If any agent steps onto the FIRE tile during its active window, the episode ends immediately (game over).
The env is not modified; FIRE is an overlay-only hazard handled in the demo loop and renderer.

Press 'p' to pause and input natural language commands for Player 0 via the LLM intervention system.
"""

import os
import sys
import pygame
import numpy as np
import random
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


def choose_random_empty_corridor(mdp, state):
    """Pick a random empty ' ' tile that is not currently occupied by a player."""
    empty_tiles = list(mdp.terrain_pos_dict[" "])
    occupied = set(state.player_positions)
    candidates = [p for p in empty_tiles if p not in occupied]
    return random.choice(candidates) if candidates else None


def main():
    """Main demo with LLM intervention focused on sudden FIRE hazard."""
    print("🔥 LLM HUMAN INTERVENTION DEMO - FIRE HAZARD")
    print("Player 0: Simple A* Agent (LLM INTERVENTION TARGET)")
    print("Player 1: Simple A* Agent (Autonomous)")
    print("Controls: 'p' to intervene, SPACE to pause, ESC to quit")
    print("=" * 70)

    # Use a corridor-like layout; schelling works well
    layouts = ["schelling"]
    layout_index = 0

    def initialize_layout(layout_name):
        """Initialize environment with given layout."""
        print(f"\n🎯 Initializing layout: {layout_name}")
        mdp = OvercookedGridworld.from_layout_name(layout_name)
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
    pygame.display.set_caption("LLM Human Intervention Demo - FIRE Hazard")
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

    # FIRE hazard controls (overlay only)
    fire_start_t = 10
    fire_duration = 5
    fire_pos = None

    # Precompute since we do not modify env: for rendering overlay placement
    grid_tile_size = visualizer.tile_size

    while running and step_count < 300:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if intervention_mode:
                    if event.key == pygame.K_RETURN:
                        if intervention_text.strip():
                            print(f"\n👤 Human: '{intervention_text}'")
                            agent_state = agent0.get_intervention_state(state)

                            # Add FIRE context for LLM
                            is_fire_active = fire_pos is not None and (fire_start_t <= step_count <= fire_start_t + fire_duration - 1)
                            agent_state['fire_active'] = bool(is_fire_active)
                            agent_state['fire_pos'] = fire_pos
                            agent_state['fire_window'] = (fire_start_t, fire_start_t + fire_duration - 1)

                            result = intervention_system.process_intervention(intervention_text, agent_state)

                            if result:
                                result_dict = {
                                    'action_override': result.action_override,
                                    'macro_override': result.macro_override,
                                    'duration': result.duration,
                                    'reasoning': result.reasoning
                                }
                                print(f"🤖 LLM Intervention Result: {result_dict}")
                                # Apply ONLY to Player 0
                                agent0.apply_intervention(result_dict)
                                last_intervention = {
                                    'command': intervention_text,
                                    'result': result_dict,
                                    'step': step_count
                                }
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
                        intervention_mode = True
                        intervention_text = ""
                        paused = True
                        print(f"\n🎤 INTERVENTION MODE: Type your command and press ENTER")

        # Activate FIRE at start time
        if fire_pos is None and step_count == fire_start_t:
            fire_pos = choose_random_empty_corridor(mdp, state)
            if fire_pos is not None:
                print(f"🔥 FIRE spawned at {fire_pos} for {fire_duration} steps (t={fire_start_t}..{fire_start_t+fire_duration-1})")

        # Deactivate FIRE after duration
        if fire_pos is not None and step_count > fire_start_t + fire_duration - 1:
            print("🔥 FIRE extinguished")
            fire_pos = None

        if not paused and not intervention_mode:
            # Get actions from both agents
            action0 = agent0.get_action(state)
            action1 = agent1.get_action(state)

            joint_action = (convert_action(action0), convert_action(action1))

            # Step environment
            state, reward, done, info = env.step(joint_action)
            step_count += 1

            # FIRE collision check (overlay logic)
            if fire_pos is not None and (fire_start_t <= step_count - 1 <= fire_start_t + fire_duration - 1):
                p_positions = state.player_positions
                if fire_pos in p_positions:
                    print(f"💥 GAME OVER: Player stepped into FIRE at {fire_pos} (t={step_count})")
                    running = False

            # Early debug prints
            if step_count <= 12:
                p0 = state.players[0].position
                p1 = state.players[1].position
                print(f"[t={step_count}] P0 pos={p0} act={action0} | P1 pos={p1} act={action1}")

            # History for LLM
            ego_item = agent0._get_item_name(state.players[0].get_object()) if state.players[0].has_object() else "none"
            mate_item = agent0._get_item_name(state.players[1].get_object()) if state.players[1].has_object() else "none"

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

        # Render the game
        screen.fill((255, 255, 255))

        # Render Overcooked state
        img_surface = visualizer.render_state(state, mdp.terrain_mtx)
        screen.blit(img_surface, (10, 10))

        # Draw FIRE overlay if active
        overlay_active = fire_pos is not None and (fire_start_t <= step_count <= fire_start_t + fire_duration - 1)
        if overlay_active and fire_pos is not None:
            tile_x = 10 + fire_pos[0] * visualizer.tile_size
            tile_y = 10 + fire_pos[1] * visualizer.tile_size
            # Semi-transparent red rectangle with 'F'
            overlay = pygame.Surface((visualizer.tile_size, visualizer.tile_size), pygame.SRCALPHA)
            overlay.fill((255, 0, 0, 120))
            screen.blit(overlay, (tile_x, tile_y))
            text_surface = font.render("F", True, (255, 255, 255))
            screen.blit(text_surface, (tile_x + visualizer.tile_size // 3, tile_y + visualizer.tile_size // 5))

        # Render HUD/info
        info_y = 420
        agent_info = [
            f"Step: {step_count}/300",
            f"Layout: {layout}",
            f"Player 0 (SimpleHandcoded): {agent0.heuristic}",
            f"Player 1 (SimpleHandcoded): {agent1.heuristic}",
            "",
            "🔥 FIRE Hazard Demo",
            "Controls: 'p'=Intervention, SPACE=Pause, ESC=Quit",
            "",
        ]

        if fire_pos is not None:
            agent_info.extend([
                f"FIRE: Active at {fire_pos} (t={fire_start_t}-{fire_start_t+fire_duration-1})",
                "Avoid stepping onto the 'F' tile!",
                "",
            ])

        if intervention_mode:
            agent_info.extend([
                "🎤 INTERVENTION MODE - Type command:",
                f"'{intervention_text}_'",
                "",
                "Target: Player 0",
            ])
        elif last_intervention:
            cmd = last_intervention.get('command', '')
            step_meta = last_intervention.get('step', '?')
            res = last_intervention.get('result')
            if res:
                reasoning = (res.get('reasoning') or '')[:50]
                agent_info.extend([
                    f"Last intervention (Step {step_meta}):",
                    f"👤 '{cmd}'",
                    f"🤖 {reasoning}...",
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
            color = (0, 0, 0)
            if "FIRE" in line:
                color = (200, 0, 0)
            elif "Player 0" in line:
                color = (0, 100, 0)
            elif "Player 1" in line:
                color = (200, 0, 0)
            elif "🎤" in line:
                color = (0, 0, 200)
            text_surface = font.render(line[:80], True, color)
            screen.blit(text_surface, (10, info_y + i * 26))

        # Context when in intervention mode
        if intervention_mode:
            context_y = info_y + len(agent_info) * 26 + 20
            agent_state = agent0.get_intervention_state(state)
            is_fire_active = fire_pos is not None and (fire_start_t <= step_count <= fire_start_t + fire_duration - 1)
            context_info = [
                "🤖 SimpleAStar Agent Context (Player 0):",
                f"Position: {agent_state['agent_pos']}",
                f"Target: {agent_state['target_pos']}",
                f"Macro: {agent_state['current_macro']}",
                f"Holding: {agent_state['holding_item']}",
                f"FIRE Active: {is_fire_active}",
                f"FIRE Pos: {fire_pos}",
            ]
            for i, line in enumerate(context_info):
                text_surface = small_font.render(line, True, (100, 100, 100))
                screen.blit(text_surface, (10, context_y + i * 20))

        if paused and not intervention_mode:
            pause_text = font.render("PAUSED - Press SPACE to continue, 'p' for intervention", True, (255, 0, 0))
            screen.blit(pause_text, (10, 10))

        pygame.display.flip()
        pygame.event.pump()
        # Slow down demo for clarity
        clock.tick(5)

    pygame.quit()

    print(f"\n📊 FIRE DEMO COMPLETED")
    print(f"Steps: {step_count}")


if __name__ == "__main__":
    main()
