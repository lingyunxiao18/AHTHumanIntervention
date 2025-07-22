'''
This script visualizes the lossless encoding of the state and verifies that the encoding is correct.
'''

import numpy as np
import pygame
import matplotlib.pyplot as plt
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

np.set_printoptions(precision=2, suppress=True, linewidth=120)

FEATURE_NAMES = [
    "player_0_loc", "player_1_loc",
    "player_0_orientation_N", "player_0_orientation_S", "player_0_orientation_E", "player_0_orientation_W",
    "player_1_orientation_N", "player_1_orientation_S", "player_1_orientation_E", "player_1_orientation_W",
    "pot_loc", "counter_loc", "onion_disp_loc", "dish_disp_loc", "serve_loc",
    "onions_in_pot", "onions_cook_time", "onion_soup_loc", "dishes", "onions"
]

def print_encoding_summary(encoding, feature_names=None, max_layers=10):
    print("Encoding shape:", encoding.shape)
    if feature_names is not None:
        print("Feature names:", feature_names)
    print("--- Slices of encoding (showing up to {} layers) ---".format(max_layers))
    for i in range(min(encoding.shape[-1], max_layers)):
        print(f"Layer {i} ({feature_names[i] if feature_names else ''}):\n", encoding[:, :, i])
        print()

def show_encoding_images(encoding, feature_names=None, max_layers=10):
    n_layers = min(encoding.shape[-1], max_layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(3*n_layers, 3))
    if n_layers == 1:
        axes = [axes]
    for i in range(n_layers):
        ax = axes[i]
        ax.imshow(encoding[:, :, i], cmap='gray', origin='upper')
        if feature_names:
            ax.set_title(feature_names[i], fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    layout = "random3"
    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv(mdp)
    visualizer = StateVisualizer()

    # Get a few states (initial and after a few steps)
    state = mdp.get_standard_start_state()
    states = [state]
    for _ in range(3):
        # Get valid actions for each agent
        valid_actions = mdp.get_actions(state)
        # Pick the first valid action for each agent (or random)
        joint_action = [valid_actions[0][0], valid_actions[1][0]]
        state, _, _, _ = env.step(joint_action)
        states.append(state)

    for i, s in enumerate(states):
        print(f"\n=== State {i} ===")
        # Visualize state in a window
        visualizer.display_rendered_state(s, grid=mdp.terrain_mtx, window_display=True)
        # Get lossless encoding for agent 0
        encoding = mdp.lossless_state_encoding(s)[0]  # shape: (H, W, C)
        print_encoding_summary(encoding, feature_names=FEATURE_NAMES, max_layers=20)  
        show_encoding_images(encoding, feature_names=FEATURE_NAMES, max_layers=20)
        input("Press Enter to continue to next state...")

if __name__ == "__main__":
    main() 