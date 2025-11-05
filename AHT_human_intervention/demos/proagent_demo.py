#!/usr/bin/env python3
import os
import sys
import argparse
import pygame
import time

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
from shared.agents.dish_specialist_agent import DishSpecialistAgent
from shared.agents.greedy_agent import GreedyAgent
from shared.agents.random_agent import RandomAgent
from shared.agents.simple_handcoded_agent import SimpleHandcodedAgent
from shared.agents.stay_agent import StayAgent

# Use ProAgent harness utilities and agent
import utils as pro_utils  # type: ignore
from proagent.proagent_with_intervention import ProAgentWithIntervention


def convert_action(a):
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
        vec_map = {
            (1, 0): Direction.EAST,
            (0, 1): Direction.SOUTH,
            (0, -1): Direction.NORTH,
            (-1, 0): Direction.WEST,
        }
        return vec_map.get(a, Action.STAY)
    return Action.STAY


def main():
    parser = argparse.ArgumentParser(description='ProAgentWithIntervention Demo')
    parser.add_argument('--layout', type=str, default='counter_circuit', help='Layout key (mapped via NEW_LAYOUTS)')
    parser.add_argument('--layout_name', type=str, default=None, help='Direct Overcooked layout name (overrides --layout)')
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--random_start', action='store_true', help='Randomize player starting positions')
    parser.add_argument(
        '--teammate',
        type=str,
        default='onion_specialist',
        help='Teammate agent type: onion_specialist, dish_specialist, greedy, random, stay, hand_coded'
    )
    args = parser.parse_args()

    layout = args.layout
    if args.layout_name is not None:
        layout_name = args.layout_name
    else:
        layout_name = pro_utils.NEW_LAYOUTS.get(layout, layout)

    mdp = OvercookedGridworld.from_layout_name(layout_name)
    start_fn = mdp.get_random_start_state_fn(random_start_pos=True) if args.random_start else None
    env = OvercookedEnv(mdp, start_state_fn=start_fn, horizon=args.horizon)
    env.reset()

	# Agents: Player0 = ProAgentWithIntervention, Player1 = OnionSpecialistAgent
    # Create base ProAgent first to get MLAM, then wrap with intervention
    base_agent = pro_utils.make_agent('ProAgent', mdp, layout, model='gpt-4.1-mini', retrival_method='recent_k', K=10, prompt_level='l2-ap', belief_revision=True, auto_unstuck=False)
    
    # Create ProAgentWithIntervention using the same MLAM
    p0 = ProAgentWithIntervention(base_agent.mlam, layout, model='gpt-4.1-mini', retrival_method='recent_k', K=10, prompt_level='l2-ap', belief_revision=True, auto_unstuck=False)

    # Map teammate argument to agent constructors
    teammate_key = (args.teammate or 'onion_specialist').strip().lower()
    teammate_map = {
        'onion_specialist': lambda mdp: OnionSpecialistAgent(mdp, agent_idx=1, agent_name='OnionSpec'),
        'dish_specialist': lambda mdp: DishSpecialistAgent(mdp, agent_idx=1, agent_name='DishSpec'),
        'greedy': lambda mdp: GreedyAgent(mdp, agent_idx=1, agent_name='Greedy'),
        'random': lambda mdp: RandomAgent(mdp, agent_idx=1, agent_name='Random'),
        'stay': lambda mdp: StayAgent(mdp, agent_idx=1, agent_name='Stay'),
        'hand_coded': lambda mdp: SimpleHandcodedAgent(mdp, agent_idx=1, agent_name='HandCoded')
    }
    p1_factory = teammate_map.get(teammate_key, teammate_map['onion_specialist'])
    p1 = p1_factory(mdp)
    # Ensure agent indices are set for shared Agent API
    if hasattr(p0, 'set_agent_index'):
        p0.set_agent_index(0)
    if hasattr(p1, 'set_agent_index'):
        p1.set_agent_index(1)

    # Pygame
    pygame.init()
    # Slightly narrower to better fit on screen
    screen = pygame.display.set_mode((1400, 800))
    pygame.display.set_caption(f'ProAgentWithIntervention Demo [{layout_name}] (Player0) + {getattr(p1, "agent_name", type(p1).__name__)} (Player1)')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    viz = StateVisualizer()
    
    # Print helpful information about the new interpreter system
    print("🤖 ProAgentWithIntervention Demo Started")
    print("📋 New Interpreter Features:")
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
    print("   - 'Focus on cooking onions first'")
    print("   - 'Help your teammate by delivering soup'")
    print("   - 'Avoid blocking the center area'")
    print("   - 'Pick up more onions from the left dispenser'")
    print("")

    state = env.state
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
        # merge any excess into the second line and append ellipsis
        first = wrapped[0]
        second = " ".join(wrapped[1:])
        # If second still too wide, we will trim visually with ellipsis
        if font_obj.size(second)[0] > max_width:
            # binary trim to fit
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
        
        # Split content by newlines to preserve CoT structure
        lines = content.split('\n')
        result_lines = []
        
        # Add the label to the first line
        first_line = f"{label}{lines[0]}" if lines else label
        result_lines.extend(wrap_text(first_line, font_obj, max_width))
        
        # Process remaining lines with proper indentation
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                # Add indentation to show it's part of the CoT
                indented_line = f"    {line.strip()}"
                result_lines.extend(wrap_text(indented_line, font_obj, max_width))
        
        # Limit to max_lines and add ellipsis if truncated
        if len(result_lines) > max_lines:
            result_lines = result_lines[:max_lines-1]
            result_lines.append("    ...")
        
        return result_lines

    while step < 200:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if intervention_mode:
                    if event.key == pygame.K_RETURN:
                        cmd = intervention_text.strip()
                        if cmd:
                            # Use the advanced intervention system
                            try:
                                print(f"🎯 Applying intervention: '{cmd}'")
                                success = p0.process_human_intervention(cmd)
                                if success:
                                    print(f"✅ Intervention processed: '{cmd}'")
                                    # Show intervention stats after processing
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
                        return
                    elif event.key == pygame.K_m:
                        # Debug: print memory state
                        try:
                            print("\n🧠 Agent Memory State:")
                            if hasattr(p0, 'memory'):
                                memory = p0.memory
                                print(f"   Semantic entries: {len(memory.semantic)}")
                                print(f"   Episodic entries: {len(memory.episodic)}")
                                if hasattr(memory, 'facts'):
                                    print(f"   Facts: {len(memory.facts)}")
                                print(f"   Memory view: {memory.prompt_view()[:200]}...")
                                # Print detailed memory contents
                                memory.debug_print_memory()
                            else:
                                print("   No memory object found")
                            print("")
                        except Exception as e:
                            print(f"❌ Memory debug error: {e}")

        if not paused and not intervention_mode:
            # Query agents
            res0 = p0.action(state)
            try:
                print(f"[DBG-DEMO] res0_type={type(res0)} res0_repr={repr(res0)}")
            except Exception:
                pass
            if isinstance(res0, tuple):
                first = res0[0]
                if isinstance(first, (Direction, Action)) or (isinstance(first, tuple) and len(first) == 2):
                    a0 = first
                else:
                    a0 = res0
            else:
                a0 = res0
            a1 = p1.action(state)

            # Log per-step actions and current medium-level plan
            try:
                ml = getattr(p0, 'current_ml_action', None)
                # Get new interpreter plan info
                last_plan = getattr(p0, 'last_plan', None)
                
                a0_conv = convert_action(a0)
                a1_conv = convert_action(a1)
                joint = (a0_conv, a1_conv)
                
                # Enhanced logging with new interpreter output
                last_intervention_reason = getattr(p0, 'last_intervention_reason', None)
                if last_intervention_reason:
                    print(f"[STEP {step}] 🎯 Intervention Reason: {last_intervention_reason}")
                last_chain_of_thought = getattr(p0, 'last_chain_of_thought', None)
                if last_chain_of_thought:
                    print(f"[STEP {step}] 🧠 Chain of Thought: {last_chain_of_thought}")
                if last_plan:
                    print(f"[STEP {step}] Interpreter Plan Steps: {last_plan.get('steps', [])}")
                print(f"[STEP {step}] P0_action_raw={a0}, P0_action_conv={a0_conv}, P0_ml_action={ml}")
                
                # Get intervention stats
                stats = p0.get_intervention_stats()
                print(f"[STEP {step}] Agent Stats: {stats}")
            except Exception as e:
                print(f"[STEP {step}] Error in logging: {e}")

            # Step env
            joint = (a0_conv, a1_conv)
            state, reward, done, info = env.step(joint)
            step += 1
            
            # Brief pause to ensure step is visible
            time.sleep(1)

        # Render
        screen.fill((255, 255, 255))
        img = viz.render_state(state, mdp.terrain_mtx)
        screen.blit(img, (10, 10))

        # UI - Enhanced with interpreter information
        info_lines = [
            f'Step: {step}',
            'Controls: SPACE=pause, P=intervene, M=memory debug, ESC=quit',
            f'Total Interventions: {len(p0.get_intervention_history())}',
        ]

        # Teammate model/type
        teammate_name = getattr(p1, 'agent_name', type(p1).__name__)
        info_lines.append(f'Teammate: {teammate_name}')
        
        # Add interpreter status information
        try:
            last_plan = getattr(p0, 'last_plan', None)  
            last_intervention_reason = getattr(p0, 'last_intervention_reason', None)
            last_chain_of_thought = getattr(p0, 'last_chain_of_thought', None)
            
            if last_plan and last_plan.get('steps'):
                steps_str = ', '.join(last_plan.get('steps', [])[:3])  # Show first 3 steps
                if len(last_plan.get('steps', [])) > 3:
                    steps_str += '...'
                info_lines.append(f'Plan Steps: {steps_str}')
            # Rationale removed - using CoT instead
            if last_intervention_reason:
                # Show intervention reason with special formatting
                info_lines.append(('INTERVENTION_REASON', last_intervention_reason))
            if last_chain_of_thought:
                # Show chain of thought with special formatting
                info_lines.append(('CHAIN_OF_THOUGHT', last_chain_of_thought))
            
            # Show current ML action
            current_ml = getattr(p0, 'current_ml_action', None)
            if current_ml:
                info_lines.append(f'Current ML Action: {current_ml}')
                
            
            # Show teammate model from memory module (LLM-authored)
            try:
                mem_view = p0.memory.prompt_view() if hasattr(p0, 'memory') else {}
                tm_view = mem_view.get('teammate_model', {}) if isinstance(mem_view, dict) else {}
                tm_desc = tm_view.get('behavior_description')
                tm_role = tm_view.get('role')
                tm_conf = tm_view.get('role_confidence')
                if tm_desc:
                    # Defer wrapping; insert a tuple marker for later re-wrap with panel width
                    info_lines.append(('TMODEL', tm_desc))
                if tm_role is not None:
                    if tm_conf is not None:
                        info_lines.append(f'Teammate Role: {tm_role} (conf {float(tm_conf):.2f})')
                    else:
                        info_lines.append(f'Teammate Role: {tm_role}')
            except Exception:
                pass
            
        except Exception as e:
            info_lines.append(f'Debug Error: {str(e)[:30]}...')
        
        if intervention_mode:
            info_lines.append(f'Intervention: {intervention_text}_')
        else:
            # Recent intervention history is now shown in the memory debug section above
            pass
            
        # Render UI in a wider right-side panel with word wrapping
        text_panel_x = img.get_width() + 30
        text_panel_y = 10
        text_panel_width = max(500, screen.get_width() - text_panel_x - 20)

        # Rebuild lines with controlled wrapping for long labeled texts (two lines max)
        wrapped_lines = []
        for line in info_lines:
            if isinstance(line, tuple) and len(line) == 2:
                tag, content = line
                if tag == 'TMODEL':
                    wrapped_lines.extend(wrap_two_lines('Teammate Model: ', content, font, text_panel_width))
                    continue
                if tag == 'RATIONAL':
                    wrapped_lines.extend(wrap_two_lines('Rationale: ', content, font, text_panel_width))
                    continue
                if tag == 'INTERVENTION_REASON':
                    wrapped_lines.extend(wrap_two_lines('Intervention Reason: ', content, font, text_panel_width))
                    continue
                if tag == 'CHAIN_OF_THOUGHT':
                    wrapped_lines.extend(wrap_multiline_text('Chain of Thought: ', content, font, text_panel_width, max_lines=15))
                    continue
                if tag == 'INTERVENTION':   
                    wrapped_lines.extend(wrap_two_lines('Intervention: ', content, font, text_panel_width))
                    continue
                if tag == 'MEM':
                    wrapped_lines.extend(wrap_two_lines('Mem: ', content, font, text_panel_width))
                    continue
            wrapped_lines.extend(wrap_text(line if isinstance(line, str) else str(line), font, text_panel_width))

        max_lines = 35
        start_line = max(0, len(wrapped_lines) - max_lines)
        visible_lines = wrapped_lines[start_line:]

        for i, line in enumerate(visible_lines):
            txt = font.render(line, True, (0, 0, 0))
            screen.blit(txt, (text_panel_x, text_panel_y + i * 22))

        pygame.display.flip()
        clock.tick(1.5)

    pygame.quit()


if __name__ == '__main__':
    main()
