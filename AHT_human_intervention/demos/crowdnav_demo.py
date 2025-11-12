#!/usr/bin/env python3
import os
import sys
import argparse
import pygame
import time
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROAGENT_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'proagent', 'src'))
# Ensure ProAgent src takes precedence so we import the right utils
if PROAGENT_SRC in sys.path:
    sys.path.remove(PROAGENT_SRC)
sys.path.insert(0, PROAGENT_SRC)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Add CrowdNav_AHT to path
CROWDNAV_PATH = os.path.join(PROJECT_ROOT, 'shared', 'envs', 'envs', 'CrowdNav_AHT')
if CROWDNAV_PATH not in sys.path:
    sys.path.insert(0, CROWDNAV_PATH)

from crowd_sim.envs import CrowdSimAHT
from crowd_nav.configs.config import Config
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.orca import ORCA

# Use ProAgent harness utilities and agent
from proagent.proagent_with_intervention_crowdnav import ProAgentWithInterventionCrowdNav


def main():
    parser = argparse.ArgumentParser(description='ProAgentWithIntervention CrowdNav-AHT Demo')
    parser.add_argument('--horizon', type=int, default=200, help='Maximum number of steps')
    parser.add_argument('--seed', type=int, default=6, help='Random seed for environment')
    parser.add_argument('--display', action='store_true', help='Enable matplotlib display')
    args = parser.parse_args()

    # Initialize environment
    config = Config()
    env = CrowdSimAHT()
    env.configure(config)
    env.thisSeed = args.seed
    env.nenv = 1
    env.phase = 'test'

    # Create and set robot
    robot = Robot(config, 'robot')
    robot.policy = ORCA(config)
    env.set_robot(robot)

    # Create ProAgent with intervention
    p0 = ProAgentWithInterventionCrowdNav(model='gpt-4o-mini', agent_index=0)
    p0.set_agent_index(0)

    # Setup matplotlib for rendering if display enabled
    if args.display:
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        plt.ion()
        plt.show()
        env.render_axis = ax

    # Pygame for controls
    pygame.init()
    screen = pygame.display.set_mode((1400, 800))
    pygame.display.set_caption('ProAgentWithIntervention CrowdNav-AHT Demo')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Print helpful information
    print("🤖 ProAgentWithIntervention CrowdNav-AHT Demo Started")
    print("📋 Features:")
    print("   - CoT reasoning with memory")
    print("   - Plan categorization (policy/env/teammate/vague)")
    print("   - Confidence scoring")
    print("   - Structured rationale output")
    print("   - Memory persistence across interventions")
    print("   - Explicit Chain-of-Thought display in pygame window")
    print("")
    print("🎮 Controls:")
    print("   - SPACE: Pause/Resume")
    print("   - P: Start intervention (type command)")
    print("   - M: Print memory state (debug)")
    print("   - ESC: Quit")
    print("")
    print("💡 Example interventions to try:")
    print("   - 'Head toward the goal more directly'")
    print("   - 'Avoid the crowd ahead'")
    print("   - 'Coordinate with your teammate'")
    print("   - 'Take a detour to the left'")
    print("")

    obs = env.reset()
    step = 0
    paused = False
    intervention_mode = False
    intervention_text = ''

    # Simple word-wrapping helper for the right-side info panel
    def wrap_text(text, font_obj, max_width):
        if not text:
            return [""]
        words = text.split()
        lines = []
        current = []
        for w in words:
            test_line = (" ".join(current + [w])).strip()
            if font_obj.size(test_line)[0] <= max_width or not current:
                current.append(w)
            else:
                lines.append(" ".join(current))
                current = [w]
        if current:
            lines.append(" ".join(current))
        return lines

    # Wrap a labeled message into at most two lines, with ellipsis on overflow
    def wrap_two_lines(label, content, font_obj, max_width):
        full = f"{label}{content}" if label else content
        wrapped = wrap_text(full, font_obj, max_width)
        if len(wrapped) <= 2:
            return wrapped
        first = wrapped[0]
        second = " ".join(wrapped[1:])
        if font_obj.size(second)[0] > max_width:
            s = second
            while s and font_obj.size(s + "...")[0] > max_width:
                s = s[:-1]
            second = s + "..."
        else:
            second = second + "..."
        return [first, second]

    # Wrap chain of thought text into multiple lines with proper formatting
    def wrap_multiline_text(label, content, font_obj, max_width, max_lines=6):
        if not content:
            return [f"{label}(empty)"]
        
        lines = content.split('\n')
        result_lines = []
        
        first_line = f"{label}{lines[0]}" if lines else label
        result_lines.extend(wrap_text(first_line, font_obj, max_width))
        
        for line in lines[1:]:
            if line.strip():
                indented_line = f"    {line.strip()}"
                result_lines.extend(wrap_text(indented_line, font_obj, max_width))
        
        if len(result_lines) > max_lines:
            result_lines = result_lines[:max_lines-1]
            result_lines.append("    ...")
        
        return result_lines

    while step < args.horizon:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                if args.display:
                    plt.close('all')
                return
            elif event.type == pygame.KEYDOWN:
                if intervention_mode:
                    if event.key == pygame.K_RETURN:
                        cmd = intervention_text.strip()
                        if cmd:
                            try:
                                print(f"🎯 Applying intervention: '{cmd}'")
                                success = p0.process_human_intervention(cmd)
                                if success:
                                    print(f"✅ Intervention processed: '{cmd}'")
                                    stats = p0.get_intervention_stats()
                                    print(f"📊 Agent stats after intervention: {stats}")
                                else:
                                    print(f"❌ Intervention failed: '{cmd}'")
                            except Exception as e:
                                print(f"❌ Intervention error: {e}")
                                import traceback
                                traceback.print_exc()
                        intervention_mode = False
                        intervention_text = ''
                        paused = False
                    elif event.key == pygame.K_BACKSPACE:
                        intervention_text = intervention_text[:-1]
                    else:
                        if event.unicode:
                            intervention_text += event.unicode
                else:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_p:
                        paused = True
                        intervention_mode = True
                        intervention_text = ''
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        if args.display:
                            plt.close('all')
                        return
                    elif event.key == pygame.K_m:
                        # Debug: print memory state
                        try:
                            print("\n🧠 Agent Memory State:")
                            if hasattr(p0, 'memory'):
                                memory = p0.memory
                                print(f"   Semantic entries: {len(memory.semantic)}")
                                print(f"   Episodic entries: {len(memory.episodic)}")
                                print(f"   Memory view: {memory.prompt_view()}")
                                memory.debug_print_memory()
                            else:
                                print("   No memory object found")
                            print("")
                        except Exception as e:
                            print(f"❌ Memory debug error: {e}")

        if not paused and not intervention_mode:
            # Query agent
            action = p0.action(obs)
            
            # Log per-step actions
            try:
                last_plan = getattr(p0, 'last_plan', None)
                last_intervention_reason = getattr(p0, 'last_intervention_reason', None)
                last_chain_of_thought = getattr(p0, 'last_chain_of_thought', None)
                
                if last_intervention_reason:
                    print(f"[STEP {step}] 🎯 Intervention Reason: {last_intervention_reason}")
                if last_chain_of_thought:
                    print(f"[STEP {step}] 🧠 Chain of Thought: {last_chain_of_thought}")
                if last_plan:
                    print(f"[STEP {step}] Plan Steps: {last_plan.get('steps', [])}")
                print(f"[STEP {step}] Waypoint Action: {action}")
                
                stats = p0.get_intervention_stats()
                print(f"[STEP {step}] Agent Stats: {stats}")
            except Exception as e:
                print(f"[STEP {step}] Error in logging: {e}")

            # Step env
            obs, reward, done, info = env.step(action)
            step += 1
            
            # Brief pause to ensure step is visible
            time.sleep(0.1)

            if done:
                print(f"Episode ended: {info}")
                obs = env.reset()
                step = 0

        # Render environment if display enabled
        if args.display:
            try:
                env.render()
            except Exception:
                pass

        # Render pygame UI
        screen.fill((255, 255, 255))
        
        # UI - Enhanced with interpreter information
        info_lines = [
            f'Step: {step}',
            'Controls: SPACE=pause, P=intervene, M=memory debug, ESC=quit',
            f'Total Interventions: {len(p0.get_intervention_history())}',
        ]

        # Add interpreter status information
        try:
            last_plan = getattr(p0, 'last_plan', None)  
            last_intervention_reason = getattr(p0, 'last_intervention_reason', None)
            last_chain_of_thought = getattr(p0, 'last_chain_of_thought', None)
            
            if last_plan and last_plan.get('steps'):
                steps_str = ', '.join(map(str, last_plan.get('steps', [])[:3]))
                if len(last_plan.get('steps', [])) > 3:
                    steps_str += '...'
                info_lines.append(f'Plan Steps: {steps_str}')
            
            if last_intervention_reason:
                info_lines.append(('INTERVENTION_REASON', last_intervention_reason))
            if last_chain_of_thought:
                info_lines.append(('CHAIN_OF_THOUGHT', last_chain_of_thought))
            
            current_waypoint = getattr(p0, 'current_waypoint_action', None)
            if current_waypoint is not None:
                waypoint_names = {0: "up", 1: "down", 2: "left", 3: "right", 
                                 4: "up-right", 5: "up-left", 6: "down-right", 7: "down-left"}
                info_lines.append(f'Current Waypoint: {current_waypoint} ({waypoint_names.get(current_waypoint, "unknown")})')
                
            # Show teammate model from memory module
            try:
                mem_view = p0.memory.prompt_view() if hasattr(p0, 'memory') else {}
                tm_view = mem_view.get('teammate_model', {}) if isinstance(mem_view, dict) else {}
                tm_desc = tm_view.get('behavior_description')
                if tm_desc:
                    info_lines.append(('TMODEL', tm_desc))
            except Exception:
                pass
                
        except Exception as e:
            info_lines.append(f'Debug Error: {str(e)[:30]}...')
        
        if intervention_mode:
            info_lines.append(f'Intervention: {intervention_text}_')
            
        # Render UI in a right-side panel with word wrapping
        text_panel_x = 10
        text_panel_y = 10
        text_panel_width = screen.get_width() - 20

        # Rebuild lines with controlled wrapping
        wrapped_lines = []
        for line in info_lines:
            if isinstance(line, tuple) and len(line) == 2:
                tag, content = line
                if tag == 'TMODEL':
                    wrapped_lines.extend(wrap_two_lines('Teammate Model: ', content, font, text_panel_width))
                    continue
                if tag == 'INTERVENTION_REASON':
                    wrapped_lines.extend(wrap_two_lines('Intervention Reason: ', content, font, text_panel_width))
                    continue
                if tag == 'CHAIN_OF_THOUGHT':
                    wrapped_lines.extend(wrap_multiline_text('Chain of Thought: ', content, font, text_panel_width, max_lines=15))
                    continue
            wrapped_lines.extend(wrap_text(line if isinstance(line, str) else str(line), font, text_panel_width))

        max_lines = 35
        start_line = max(0, len(wrapped_lines) - max_lines)
        visible_lines = wrapped_lines[start_line:]

        for i, line in enumerate(visible_lines):
            txt = font.render(line, True, (0, 0, 0))
            screen.blit(txt, (text_panel_x, text_panel_y + i * 22))

        pygame.display.flip()
        clock.tick(2)  # 2 FPS for visibility

    pygame.quit()
    if args.display:
        plt.close('all')


if __name__ == '__main__':
    main()

