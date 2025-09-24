#!/usr/bin/env python3
import os
import sys
import pygame

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

# Use ProAgent harness utilities and agent
import utils as pro_utils  # type: ignore


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
    layout = 'counter_circuit'
    layout_name = pro_utils.NEW_LAYOUTS[layout]

    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv(mdp, horizon=500)
    env.reset()

	# Agents: Player0 = ProAgent, Player1 = ProAgent
    p0 = pro_utils.make_agent('ProAgent', mdp, layout, model='gpt-4.1-nano', retrival_method='recent_k', K=10, prompt_level='l2-ap', belief_revision=True, auto_unstuck=True)
    p1 = pro_utils.make_agent('ProAgent', mdp, layout, model='gpt-4.1-nano', retrival_method='recent_k', K=10, prompt_level='l2-ap', belief_revision=True, auto_unstuck=True)
    # Ensure agent indices are set for shared Agent API
    if hasattr(p0, 'set_agent_index'):
        p0.set_agent_index(0)
    if hasattr(p1, 'set_agent_index'):
        p1.set_agent_index(1)

    # Pygame
    pygame.init()
    screen = pygame.display.set_mode((1200, 800))
    pygame.display.set_caption('ProAgent Demo (Player0) + ProAgent (Player1)')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    viz = StateVisualizer()

    state = env.state
    step = 0
    paused = False
    intervention_mode = False
    intervention_text = ''

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
                            # Accept a raw ml_action like pickup(onion), put_onion_in_pot(), pickup(dish), fill_dish_with_soup(), deliver_soup(), wait(k)
                            try:
                                p0.apply_human_intervention(cmd)
                            except Exception:
                                pass
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
                planner_out = getattr(p0, 'last_planner_response', None)
                a0_conv = convert_action(a0)
                a1_conv = convert_action(a1)
                joint = (a0_conv, a1_conv)
                print(f"[STEP {step}] LLM={planner_out}")
                print(f"[STEP {step}] P0_action_raw={a0}, P0_action_conv={a0_conv}, P0_ml_action={ml}")
            except Exception:
                pass

            # Step env
            joint = (a0_conv, a1_conv)
            state, reward, done, info = env.step(joint)
            step += 1

        # Render
        screen.fill((255, 255, 255))
        img = viz.render_state(state, mdp.terrain_mtx)
        screen.blit(img, (10, 10))

        # UI
        info_lines = [
            f'Step: {step}',
            'Controls: SPACE pause, p intervene (type ml_action), ESC quit',
        ]
        if intervention_mode:
            info_lines += [f'Intervention: {intervention_text}_']
        for i, line in enumerate(info_lines):
            txt = font.render(line, True, (0, 0, 0))
            screen.blit(txt, (10, 420 + i * 22))

        pygame.display.flip()
        clock.tick(15)

    pygame.quit()


if __name__ == '__main__':
    main()


