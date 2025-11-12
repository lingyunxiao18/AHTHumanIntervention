# This script sets up a crowd simulation environment, optionally rendering the environment in real-time.
# It initializes the environment configuration, applies random actions to a simulated agent, and renders
# the environment if display is enabled. The script demonstrates a loop to interact with the environment for 2000 steps.

import matplotlib.pyplot as plt
import numpy as np
from crowd_sim.envs import *

if __name__ == '__main__':
    display = True

    from crowd_nav.configs.config import Config
    config = Config()

    # Enable rendering if display mode is active and 'sim' attribute exists in config
    if display and hasattr(config, 'sim'):
        inner_object = getattr(config, 'sim')
        setattr(inner_object, 'render', True)

    # Initialize and configure the crowd simulation environment
    env = CrowdSimAHT()
    env.configure(config)
    env.thisSeed = 6     # Set seed for reproducibility
    env.nenv = 1          # Define single environment instance
    env.phase = 'test'    # Set environment phase to 'test'


    fig, ax = plt.subplots(figsize=(9, 9))  # Create figure for plotting environment state
    ax.set_xlim(-10, 10)                    # Define plot boundaries
    ax.set_ylim(-10, 10)
    ax.set_xlabel('x(m)', fontsize=16)      # Label axes
    ax.set_ylabel('y(m)', fontsize=16)
    plt.ion()                               # Enable interactive plotting
    plt.show()
    env.render_axis = ax                    # Link environment to the plot axis

    obs = env.reset()  # Initialize environment and get initial observation

    done = False       # Track if an episode has ended

    for i in range(2000):
        action = np.random.choice(8)                # Select random action
        obs, reward, done, info = env.step(action)  # Take action, receive next state, reward, and completion flag
        #print(obs)

        if display:                                 # Render environment state if display mode is enabled
            env.render()

        if done:                                    # If episode ends, print info and reset environment
            print(str(info))
            env.reset()

    env.close()  # Close the environment after completing the loop
