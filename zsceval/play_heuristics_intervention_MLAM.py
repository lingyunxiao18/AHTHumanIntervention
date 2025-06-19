import os
import sys
import pygame
import time
import numpy as np

# Overcooked imports
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from zsceval.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

# Planner imports (old Overcooked planner)
from zsceval.envs.overcooked.overcooked_ai_py.planning.planners import (
    MediumLevelActionManager,
    MediumLevelPlanner,
    Heuristic,
    NO_COUNTERS_PARAMS
)

# Intervention (LLM) stub
from zsceval.intervention_LLM_module import process_command

# Utility to wrap long text in the command box

def wrap_text(text, font, max_width):
    words = text.split(" ")
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + " "
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    return lines


def main():
    # 1. Create the MDP
    mdp = OvercookedGridworld.from_layout_name(
        "random3",
        start_order_list=["any"],
        cook_time=5
    )

    # 2. Build the legacy MLAM + Planner + Heuristic
    ml_params = NO_COUNTERS_PARAMS
    mlam = MediumLevelActionManager(mdp, ml_params)  # medium-level action manager 
    mlp = MediumLevelPlanner(mdp, ml_params, ml_action_manager=mlam)
    heuristic_fn = Heuristic(mlam.motion_planner).simple_heuristic

    # 3. Set up the OvercookedEnv
    env = OvercookedEnv(mdp, horizon=1000)
    env.reset()

    # 4. Pygame display setup
    GAME_WIDTH, GAME_HEIGHT = 800, 600
    TEXTBOX_HEIGHT = 100
    screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT + TEXTBOX_HEIGHT))
    pygame.display.set_caption("Overcooked Simulation with MLAM Planner")
    clock = pygame.time.Clock()
    fps = 5

    # Visualization helper
    state_visualizer = StateVisualizer(grid=mdp.terrain_mtx)

    # Text input state
    font = pygame.font.Font(None, 32)
    input_text = ""
    show_textbox = False

    # Planning state
    plan = []
    step_counter = 0

    # Main loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if show_textbox:
                    if event.key == pygame.K_RETURN:
                        # Example: allow "replan" command to clear and recompute plan early
                        user_cmd = input_text.strip().lower()
                        if user_cmd == "replan":
                            plan = []
                        # Pass through to LLM stub if needed
                        # intervention = process_command(user_cmd)
                        input_text = ""
                        show_textbox = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode
                else:
                    if event.key == pygame.K_p:
                        show_textbox = True

        # Render the Overcooked state
        env_surface = state_visualizer.render_state(env.state, grid=None)
        game_surface = pygame.transform.scale(env_surface, (GAME_WIDTH, GAME_HEIGHT))
        screen.blit(game_surface, (0, 0))

        # Render the command textbox
        textbox_rect = pygame.Rect(0, GAME_HEIGHT, GAME_WIDTH, TEXTBOX_HEIGHT)
        pygame.draw.rect(screen, (200, 200, 200), textbox_rect)
        prompt = "Enter command (e.g. 'replan'): " if show_textbox else "Press 'p' to command"
        lines = wrap_text(prompt + input_text, font, GAME_WIDTH - 20)
        y_off = GAME_HEIGHT + 10
        for line in lines:
            line_surf = font.render(line, True, (0, 0, 0))
            screen.blit(line_surf, (10, y_off))
            y_off += line_surf.get_height() + 2

        pygame.display.flip()
        clock.tick(fps)

        # Execute one step of the planner if not in textbox mode
        if not show_textbox:
            # If plan is empty (or just started), compute a new low-level action plan for the next delivery
            if not plan:
                plan = mlp.get_low_level_action_plan(env.state, heuristic_fn, delivery_horizon=1)
                cost = len(plan)
                print(f"[Planner] New plan (cost={cost}): {plan}")

            # Pop the next joint action (an (Action, Action) tuple) and apply it
            joint_action = plan.pop(0)
            next_state, reward, done, info = env.step(joint_action)
            env.state = next_state
            step_counter += 1

            # Reset if episode done
            if done:
                print("Episode done, resetting...")
                env.reset()
                step_counter = 0

if __name__ == "__main__":
    main()
