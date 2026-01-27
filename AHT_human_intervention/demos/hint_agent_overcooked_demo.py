#!/usr/bin/env python3
import os
import sys
import argparse
import atexit
import pygame
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HINTAGENT_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'hintagent', 'src'))
PROAGENT_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'hintagent', 'proagent', 'src'))
# Ensure HINT-Agent src takes precedence so we import the right modules
if HINTAGENT_SRC in sys.path:
    sys.path.remove(HINTAGENT_SRC)
sys.path.insert(0, HINTAGENT_SRC)
if PROAGENT_SRC not in sys.path:
    sys.path.insert(0, PROAGENT_SRC)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from shared.envs.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from shared.envs.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from shared.envs.envs.overcooked.overcooked_ai_py.planning.planners import MediumLevelPlanner
from shared.agents.onion_specialist_agent import OnionSpecialistAgent
from shared.agents.dish_specialist_agent import DishSpecialistAgent
from shared.agents.greedy_agent import GreedyAgent
from shared.agents.random_agent import RandomAgent
from shared.agents.simple_handcoded_agent import SimpleHandcodedAgent
from shared.agents.stay_agent import StayAgent

# Use HINT-Agent
from hintagent import HINTAgent


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


class TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def setup_stdout_logging(enabled: bool, log_path: str, log_dir: str, prefix: str) -> None:
    if not enabled and not log_path:
        return
    if log_path:
        path = log_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    else:
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"{prefix}_{ts}.log")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_fp = open(path, "a")
    sys.stdout = TeeStream(sys.stdout, log_fp)
    sys.stderr = TeeStream(sys.stderr, log_fp)

    def _cleanup():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            log_fp.close()
        except Exception:
            pass

    atexit.register(_cleanup)
    print(f"📄 Logging stdout/stderr to {path}")


def main():
    parser = argparse.ArgumentParser(description='HINT-Agent Demo')
    parser.add_argument('--layout', type=str, default='counter_circuit', help='Layout key (mapped via NEW_LAYOUTS)')
    parser.add_argument('--layout_name', type=str, default=None, help='Direct Overcooked layout name (overrides --layout)')
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--random_start', action='store_true', default=True,
                        help='Randomize player starting positions (default: True)')
    parser.add_argument('--fixed_start', action='store_true',
                        help='Use fixed player starting positions')
    parser.add_argument(
        '--teammate',
        type=str,
        default='hand_coded',
        help='Teammate agent type: onion_specialist, dish_specialist, greedy, random, stay, hand_coded'
    )
    parser.add_argument(
        '--no_cot',
        action='store_true',
        help='Disable Chain-of-Thought reasoning (ablation study)'
    )
    parser.add_argument(
        '--no_memory',
        action='store_true',
        help='Disable memory module (ablation study)'
    )
    parser.add_argument('--ego_variant', type=str, default=None,
                        help='Override ego variant label (ProAgent, YAY, CoT, HINT)')
    parser.add_argument('--participant_id', type=str, default='P000')
    parser.add_argument('--title_teammate', type=str, default=None,
                        help='Optional teammate label for the window title')
    parser.add_argument('--title_layout', type=str, default=None,
                        help='Optional layout label for the window title')
    parser.add_argument('--log', action='store_true', help='Mirror stdout/stderr to a log file')
    parser.add_argument('--log_path', type=str, default=None, help='Optional log file path')
    parser.add_argument('--log_dir', type=str, default='run_logs', help='Directory for auto log files')
    args = parser.parse_args()

    setup_stdout_logging(args.log, args.log_path, args.log_dir, "overcooked_stdout")

    def infer_ego_variant(no_cot: bool, no_memory: bool) -> str:
        if no_cot and no_memory:
            return "ProAgent"
        if no_cot and not no_memory:
            return "YAY"
        if not no_cot and no_memory:
            return "CoT"
        return "HINT"

    layout = args.layout
    # Layout name mapping
    NEW_LAYOUTS = {
        "cramped_room": "simple",
        "coordination_ring": "random1",
        "forced_coordination": "random0",
        "counter_circuit": "random3",
        "asymmetric_advantages": "unident_s",
    }
    if args.layout_name is not None:
        layout_name = args.layout_name
    else:
        layout_name = NEW_LAYOUTS.get(layout, layout)

    mdp = OvercookedGridworld.from_layout_name(layout_name)
    random_start = args.random_start and not args.fixed_start
    start_fn = mdp.get_random_start_state_fn(random_start_pos=True) if random_start else None
    env = OvercookedEnv(mdp, start_state_fn=start_fn, horizon=args.horizon)
    env.reset()

    # Agents: Player0 = HINTAgent, Player1 = OnionSpecialistAgent
    # Create MLAM directly (no need to create a ProAgent instance)
    MLAM_PARAMS = {
        "start_orientations": False,
        "wait_allowed": True,
        "counter_goals": [],
        "counter_drop": [],
        "counter_pickup": [],
        "same_motion_goals": True,
    }
    counter_locations = mdp.get_counter_locations()
    MLAM_PARAMS["counter_goals"] = counter_locations
    MLAM_PARAMS["counter_drop"] = counter_locations
    MLAM_PARAMS["counter_pickup"] = counter_locations
    mlam = MediumLevelPlanner.from_pickle_or_compute(mdp, MLAM_PARAMS, force_compute=True).ml_action_manager
    
    enable_cot = not args.no_cot
    enable_memory = not args.no_memory
    ego_variant = args.ego_variant or infer_ego_variant(args.no_cot, args.no_memory)

    if ego_variant.lower() == "proagent":
        from proagent.proagent import ProMediumLevelAgent
        p0 = ProMediumLevelAgent(
            mlam,
            layout,
            model='gpt-5-mini',
            debug_mode='N',
            agent_index=0,
        )
        p0.set_agent_index(0)
    else:
        # Create HINTAgent using the MLAM (supports CoT/memory ablations)
        p0 = HINTAgent(
            mlam,
            layout,
            model='gpt-5-mini',
            enable_cot=enable_cot,
            enable_memory=enable_memory,
        )

    # Map teammate argument to agent constructors
    teammate_key = (args.teammate or 'onion_specialist').strip().lower()
    title_layout = args.title_layout or layout_name
    title_teammate = args.title_teammate or teammate_key
    pygame.display.set_caption(
        f"Overcooked | participant={args.participant_id} | ego={ego_variant} | layout={title_layout} | teammate={title_teammate}"
    )
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
    pygame.display.set_caption(f'HINT-Agent Demo [{layout_name}] (Player0) + {getattr(p1, "agent_name", type(p1).__name__)} (Player1)')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    viz = StateVisualizer()
    
    # Print helpful information about the new interpreter system
    print("🤖 HINT-Agent Demo Started")
    print("📋 Interpreter Configuration:")
    print(f"   - Chain-of-Thought (CoT): {'ENABLED' if enable_cot else 'DISABLED'}")
    print(f"   - Memory Module: {'ENABLED' if enable_memory else 'DISABLED'}")
    print("   - Plan categorization (policy/env/teammate/general_hint)")
    print("   - Structured rationale output")
    if enable_memory:
        print("   - Memory persistence across interventions")
    if enable_cot:
        print("   - Explicit Chain-of-Thought display in pygame window")
    print("")
    print("🎮 Controls:")
    print("   - SPACE: Pause/Resume")
    print("   - P: Start intervention (type command)")
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
                                if hasattr(p0, "process_human_intervention"):
                                    success = p0.process_human_intervention(cmd)
                                    if success:
                                        print(f"✅ Intervention processed: '{cmd}'")
                                        if hasattr(p0, "get_intervention_stats"):
                                            stats = p0.get_intervention_stats()
                                            print(f"📊 Agent stats after intervention: {stats}")
                                    else:
                                        print(f"❌ Intervention failed: '{cmd}'")
                                else:
                                    print("ℹ️ Interventions are not supported for ProAgent.")
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
        if not paused and not intervention_mode:
            # Query agents
            res0 = p0.action(state)
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

            if done:
                env.reset()
                state = env.state
                step = 0

        # Render
        screen.fill((255, 255, 255))
        img = viz.render_state(state, mdp.terrain_mtx)
        screen.blit(img, (10, 10))

        # UI - Enhanced with interpreter information
        info_lines = [
            f'Step: {step}',
            'Controls: SPACE=pause, P=intervene, ESC=quit',
            f'Total Interventions: {len(p0.get_intervention_history())}',
            f'Config: Memory={"ON" if enable_memory else "OFF"}',
        ]

        # Teammate model/type
        teammate_name = getattr(p1, 'agent_name', type(p1).__name__)
        info_lines.append(f'Teammate: {teammate_name}')
        
        # Add interpreter status information
        try:
            last_plan = getattr(p0, 'last_plan', None)  
            
            if last_plan and last_plan.get('steps'):
                steps_str = ', '.join(last_plan.get('steps', [])[:3])  # Show first 3 steps
                if len(last_plan.get('steps', [])) > 3:
                    steps_str += '...'
                info_lines.append(f'Plan Steps: {steps_str}')
            
            # Show current ML action
            current_ml = getattr(p0, 'current_ml_action', None)
            if current_ml:
                info_lines.append(f'Current ML Action: {current_ml}')
                
            
            # Show teammate model from memory module (LLM-authored) - only if memory is enabled
            if enable_memory:
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
            # Recent intervention history is shown above
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
                if tag == 'INTERVENTION_REASON':
                    wrapped_lines.extend(wrap_two_lines('Intervention Reason: ', content, font, text_panel_width))
                    continue
                if tag == 'INTERVENTION':   
                    wrapped_lines.extend(wrap_two_lines('Intervention: ', content, font, text_panel_width))
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
