# import gym
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.linalg import norm
import copy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs import CrowdSimDict
from crowd_nav.policy.orca import ORCA


# The class for the simulation environment used for training a DSRNN policy

class CrowdSimAHT(CrowdSimDict):

    def __init__(self):
        super().__init__()
        # Waypoint following micro-step controls
        self.wp_timeout_steps = 8   # maximum ORCA sub-steps toward a waypoint
        self.wp_reach_eps = 0.20    # meters; consider waypoint reached within this radius
        self.wp_dist = 3
        self._render_buffer = []
        self.max_text_len = 1024

    # --- inside class CrowdSimAHT(CrowdSimDict) ---
    def set_robot(self, robot):
        self.robot = robot

        # Use fixed-length UTF-8 bytes so obs stays numeric for Gym
        self.observation_space = spaces.Text(max_length=self.max_text_len)

        # Local ORCA driver to move toward waypoints
        self._orca_driver = ORCA(self.config)

        self.action_space = gym.spaces.Discrete(8)
        # Internal cache for exact float states (not part of observation_space)
        self._numeric_cache = {
            "robot": None,  # dict with floats
            "teammate": None,  # dict with floats
            "goal": None,  # (gx, gy)
            "others": None  # list of dicts with floats
        }

    def _encode_text_ob(self, text: str) -> np.ndarray:
        """
        Encode a Python string to fixed-length UTF-8 bytes in a uint8 array.
        """
        b = text.encode("utf-8")[:self.max_text_len]
        arr = np.zeros(self.max_text_len, dtype=np.uint8)
        arr[:len(b)] = np.frombuffer(b, dtype=np.uint8)
        return arr

    def generate_ob(self, reset):
        # Update visibility bookkeeping as before
        visible_humans, num_visibles, human_visibility = self.get_num_human_in_fov()

        # === Gather all floats first (kept internally) ===
        # Robot floats
        r_px = float(self.robot.px)
        r_py = float(self.robot.py)
        r_vx = float(self.robot.vx)
        r_vy = float(self.robot.vy)
        r_theta = float(self.robot.theta)
        r_gx = float(self.robot.gx)
        r_gy = float(self.robot.gy)

        # Update last human states as original code does
        self.update_last_human_states(human_visibility, reset=reset)

        # Pick teammate and others
        teammate_dict = None
        others_list = []
        if len(self.humans) > 0:
            # Assume first human is the teammate
            tm = self.humans[0]
            teammate_dict = {
                "px": float(tm.px),
                "py": float(tm.py),
                "vx": float(tm.vx),
                "vy": float(tm.vy),
            }
            # All remaining humans are background agents
            for j in range(1, len(self.humans)):
                hj = self.humans[j]
                others_list.append({
                    "px": float(hj.px),
                    "py": float(hj.py),
                    "vx": float(hj.vx),
                    "vy": float(hj.vy),
                })
        else:
            teammate_dict = None
            others_list = []

        # Common goal: by default use robot goal (adjust if you track a separate shared goal)
        common_gx, common_gy = r_gx, r_gy

        # Cache all floats internally (not part of observation space)
        self._numeric_cache = {
            "robot": {"px": r_px, "py": r_py, "vx": r_vx, "vy": r_vy, "theta": r_theta},
            "teammate": teammate_dict,
            "goal": {"gx": common_gx, "gy": common_gy},
            "others": others_list
        }

        # === Build the language template with explanations ===
        desc = []

        desc.append(
            f"The robot is at position x = {r_px:.2f} meters, y = {r_py:.2f} meters. "
            f"It is moving with velocity vx = {r_vx:.2f} m/s in x direction and vy = {r_vy:.2f} m/s in y direction. "
            f"The robot is facing an orientation theta = {r_theta:.2f} radians."
        )

        if teammate_dict is not None:
            desc.append(
                f"The teammate is located at position x = {teammate_dict['px']:.2f} meters, y = {teammate_dict['py']:.2f} meters, "
                f"and is moving with velocity vx = {teammate_dict['vx']:.2f} m/s, vy = {teammate_dict['vy']:.2f} m/s."
            )
        else:
            desc.append("There is no teammate in the scene.")

        desc.append(
            f"The common goal for both the robot and teammate is at x = {common_gx:.2f} meters, y = {common_gy:.2f} meters."
        )

        if len(others_list) == 0:
            desc.append("There are no other background agents nearby.")
        else:
            other_sents = []
            for k, od in enumerate(others_list):
                other_sents.append(
                    f"Agent {k + 1} is at position x = {od['px']:.2f}, y = {od['py']:.2f}, "
                    f"moving with velocity vx = {od['vx']:.2f} m/s and vy = {od['vy']:.2f} m/s."
                )
            desc.append(" ".join(other_sents))

        lang_ob_text = " ".join(desc)
        ob = lang_ob_text[:self.max_text_len]
        return ob

    # add all generated humans to the self.humans list
    def generate_random_human_position(self, human_num):
        """
        Generate human position: generate start position on a circle, goal position is at the opposite side
        :param human_num:
        :return:
        """
        # generate the teammate
        self.humans.append(self.generate_circle_crossing_human(teammate=True))
        # generate other background agents
        for i in range(human_num - 1):
            self.humans.append(self.generate_circle_crossing_human())


    # step function
    def step(self, action, update=True):
        """
        Use ORCA to move toward a discrete 2m waypoint. Keep advancing the world
        until the robot reaches the waypoint (within self.wp_reach_eps) or we hit
        the sub-step timeout (self.wp_timeout_steps).
        """
        # Translate discrete action -> waypoint coordinates
        if isinstance(action, (int, np.integer)) or (hasattr(action, 'shape') and getattr(action, 'shape', None) == ()):
            a_idx = int(action)
        else:
            # handle one-hot or array-like by argmax
            try:
                a_idx = int(np.argmax(action))
            except Exception:
                a_idx = int(action)

        wx, wy = self._discrete_action_to_waypoint(a_idx)
        print(wx, wy)

        last_reward = 0.0
        last_done = False
        last_episode_info = {}

        # Repeatedly call ORCA to drive toward (wx, wy) for up to timeout steps
        for _ in range(int(self.wp_timeout_steps)):
            # Record frame after each micro-step
            self._render_buffer.append({
                "robot": (float(self.robot.px), float(self.robot.py), float(self.robot.radius),
                          float(self.robot.theta if getattr(self.robot, "kinematics", "unicycle") == "unicycle"
                                else np.arctan2(self.robot.vy, self.robot.vx))),
                "humans": [(float(h.px), float(h.py), float(h.radius),
                            float(np.arctan2(h.vy, h.vx))) for h in self.humans],
                "waypoint": (float(wx), float(wy)),  # the currently selected waypoint
            })
            # Recompute ORCA action toward current waypoint
            # print(wx, wy)
            orca_action = self._orca_action_toward(wx, wy)

            # Humans act based on their policies
            human_actions = self.get_human_actions()

            # Reward/termination computed on this micro-step
            reward, done, episode_info = self.calc_reward(orca_action)
            last_reward, last_done, last_episode_info = reward, done, episode_info

            # Apply actions
            self.robot.step(orca_action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step

            # Check if robot reached the waypoint (within epsilon)
            if np.hypot(wx - self.robot.px, wy - self.robot.py) <= self.wp_reach_eps:
                break

            # Stop early if environment signals termination (e.g., goal reached/collision)
            if done:
                break

        # Build observation (language-based in this env variant)
        ob = self.generate_ob(reset=False)
        info = {'info': last_episode_info}

        # Optional updates that your base env performs
        if self.random_goal_changing and self.global_time % 5 == 0:
            self.update_human_goals_randomly()

        if self.end_goal_changing:
            for i, human in enumerate(self.humans):
                if np.hypot(human.gx - human.px, human.gy - human.py) < human.radius:
                    if i == 0:
                        self.humans[0].isObstacle = True
                    else:
                        self.update_human_goal(human)


        # Save the currently selected waypoint so render() can show it
        self._last_selected_waypoint = (wx, wy)

        return ob, last_reward, last_done, info

    def _discrete_action_to_waypoint(self, a_idx: int):
        """
        Map a discrete action index (0..7) to a waypoint that is 2 meters from
        the robot's current position on a circle of radius R.
        Ordering:
            0: up (+y)
            1: down (-y)
            2: left (-x)
            3: right (+x)
            4: up-right (+x, +y)
            5: up-left (-x, +y)
            6: down-right (+x, -y)
            7: down-left (-x, -y)
        """
        dirs = {
            0: (0.0, 1.0),  # up
            1: (0.0, -1.0),  # down
            2: (-1.0, 0.0),  # left
            3: (1.0, 0.0),  # right
            4: (1.0, 1.0),  # up-right
            5: (-1.0, 1.0),  # up-left
            6: (1.0, -1.0),  # down-right
            7: (-1.0, -1.0)  # down-left
        }

        a_idx = int(a_idx)
        dx, dy = dirs[a_idx]

        # Normalize diagonals so the step is exactly R
        if dx != 0.0 and dy != 0.0:
            inv = (dx * dx + dy * dy) ** 0.5
            dx /= inv
            dy /= inv

        wx = float(self.robot.px + self.wp_dist * dx)
        wy = float(self.robot.py + self.wp_dist * dy)
        return wx, wy


    def _orca_action_toward(self, gx: float, gy: float):
        """
        Build a JointState with the robot's goal temporarily set to (gx, gy),
        and use the local ORCA driver to compute a safe ActionXY toward it.
        Requires:
            - from crowd_sim.envs.utils.state import JointState
            - self._orca_driver = ORCA(self.config)  (e.g., set in set_robot)
        """
        # Robot full state with temporary goal
        self_state = self.robot.get_full_state()
        self_state.gx = float(gx)
        self_state.gy = float(gy)

        # Observable human states
        human_states = [h.get_observable_state() for h in self.humans]

        # ORCA predict
        state = JointState(self_state, human_states)
        return self._orca_driver.predict(state)


    def render(self, mode='human'):
        import matplotlib.pyplot as plt
        from matplotlib import patches
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
        import matplotlib

        # Pull buffered frames recorded during step() micro-steps
        frames = getattr(self, "_render_buffer", None)

        # If nothing buffered, render the current state once
        if not frames:
            frames = [{
                "robot": (
                    float(self.robot.px), float(self.robot.py), float(self.robot.radius),
                    float(self.robot.theta if getattr(self.robot, "kinematics", "unicycle") == "unicycle"
                          else np.arctan2(self.robot.vy, self.robot.vx))
                ),
                # Each human tuple: (px, py, radius, heading_theta)
                "humans": [
                    (float(h.px), float(h.py), float(h.radius), float(np.arctan2(h.vy, h.vx)))
                    for h in self.humans
                ],
                "waypoint": getattr(self, "_last_selected_waypoint", None),
            }]

        # Bounds: prefer circle_radius; fall back to data-driven bounds
        xlim = (-10, 10)
        ylim = (-10, 10)

        ax = self.render_axis

        # Colors / styles
        robot_color = 'yellow'
        human_edge = 'k'
        waypoint_color = 'blue'
        goal_color = 'red'
        path_color = 'gray'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        # Trails across micro-steps
        r_traj_x, r_traj_y = [], []
        h_trajs = [[] for _ in self.humans]  # list of [xs, ys] per human

        for i, f in enumerate(frames):
            # ax.cla()
            for child in ax.get_children():
                if isinstance(child, matplotlib.legend.Legend):
                    continue  # keep legend
                try:
                    child.remove()
                except Exception:
                    pass
            ax.set_aspect('equal')
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            # Goal
            ax.plot([self.robot.gx], [self.robot.gy], marker='*', color=goal_color,
                    markersize=14, label="Goal" if i == 0 else None)

            # Robot trail up to current frame
            r_px, r_py, r_r, r_theta = f["robot"]
            r_traj_x.append(r_px);
            r_traj_y.append(r_py)
            ax.plot(r_traj_x, r_traj_y, linewidth=1.5, alpha=0.6, color=path_color,
                    label="Agent path" if i == 0 else None)

            # Human trails up to current frame
            for hi, _ in enumerate(self.humans):
                if hi < len(f["humans"]):
                    h_px, h_py, h_r, h_theta = f["humans"][hi]
                    if not h_trajs[hi]:
                        h_trajs[hi] = [[], []]
                    h_trajs[hi][0].append(h_px)
                    h_trajs[hi][1].append(h_py)
                    ax.plot(h_trajs[hi][0], h_trajs[hi][1], linewidth=1, alpha=0.4, color=path_color)

            # --- Draw current states with discs, arrows, and labels ---

            # Robot disc and heading arrow
            robot_disc = plt.Circle((r_px, r_py), r_r, color=robot_color, ec='k', lw=0.5)
            ax.add_patch(robot_disc)
            robot_heading = r_theta if getattr(self.robot, "kinematics", "unicycle") == "unicycle" \
                else np.arctan2(self.robot.vy, self.robot.vx)
            rhx = r_px + r_r * np.cos(robot_heading)
            rhy = r_py + r_r * np.sin(robot_heading)
            ax.add_patch(patches.FancyArrowPatch((r_px, r_py), (rhx, rhy), color='red', arrowstyle=arrow_style))

            # Humans: circle, heading arrow, and number label (1..N)
            for hi, _ in enumerate(self.humans):
                if hi < len(f["humans"]):
                    h_px, h_py, h_r, h_theta = f["humans"][hi]

                    # --- Teammate (0-th human) uses a distinct color ---
                    if hi == 0:
                        human_edge = 'green'  # outline color
                        arrow_color = 'green'  # heading arrow color
                        text_color = 'green'  # label color
                    else:
                        human_edge = 'k'
                        arrow_color = 'black'
                        text_color = 'k'

                    # Circle outline
                    h_disc = plt.Circle((h_px, h_py), h_r, fill=False, ec=human_edge, lw=1.5)
                    ax.add_patch(h_disc)

                    # Heading arrow (length ≈ radius)
                    hhx = h_px + h_r * np.cos(h_theta)
                    hhy = h_py + h_r * np.sin(h_theta)
                    ax.add_patch(
                        patches.FancyArrowPatch(
                            (h_px, h_py), (hhx, hhy),
                            color=arrow_color,
                            arrowstyle=arrow_style,
                            alpha=0.9
                        )
                    )

                    # Number label at the center
                    ax.text(h_px, h_py, str(hi + 1),
                            ha='center', va='center',
                            fontsize=9, color=text_color, weight='bold')

            # Waypoint marker for the current micro-step (if any)
            if f.get("waypoint") is not None:
                wx, wy = f["waypoint"]
                ax.plot([wx], [wy], marker='x', markersize=10, color=waypoint_color,
                        label="Waypoint" if i == 0 else None)

            # Legend (build once; labels may be None on later frames)
            if i == 0:
                legend_handles = [
                    # Goal as a red star
                    Line2D([], [], linestyle='None', marker='*',
                           markerfacecolor=goal_color, markeredgecolor=goal_color,
                           markersize=12, label='Goal'),

                    # Waypoint as a blue 'x'
                    Line2D([], [], linestyle='None', marker='x',
                           color=waypoint_color, markersize=10, label='Waypoint'),

                    # Robot path as a short gray line
                    Line2D([], [], linestyle='-', color=path_color, linewidth=1.5, label='Robot path'),

                    # Robot: yellow filled circle
                    Circle((0, 0), radius=0.3, facecolor=robot_color, edgecolor='k', label='Robot'),

                    # Teammate: empty circle with green outline
                    Circle((0, 0), radius=0.3, fill=False, edgecolor='green', linewidth=1.5, label='Teammate'),

                    # Background agent: empty circle with black outline
                    Circle((0, 0), radius=0.3, fill=False, edgecolor='k', linewidth=1.0, label='Background agent'),
                ]

                ax.legend(handles=legend_handles,
                          loc='upper right', fontsize=10,
                          handlelength=1.2, handletextpad=0.8,
                          borderpad=0.5, framealpha=0.9)
                # ax.legend(loc='upper right', fontsize=10)

            # Pause 0.1s after plotting each micro-step
            plt.pause(0.1)

        # Clear the buffer once rendered so next call shows only new micro-steps
        if hasattr(self, "_render_buffer"):
            self._render_buffer.clear()

