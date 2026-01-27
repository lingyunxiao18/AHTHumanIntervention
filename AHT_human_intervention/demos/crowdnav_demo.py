#!/usr/bin/env python3
import os
import sys
import argparse
import atexit
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import threading
import queue

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
HINTAGENT_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'hintagent', 'src'))
# Ensure HINT-Agent src takes precedence so we import the right modules
if HINTAGENT_SRC in sys.path:
    sys.path.remove(HINTAGENT_SRC)
sys.path.insert(0, HINTAGENT_SRC)
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
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.info import ReachGoal

# Use HINT-Agent
from hintagent import HINTAgentCrowdNav


"""
\subsection{Cooperative Reaching in CrowdNav}
In CrowdNav, two cooperative agents must choose one of two candidate meeting
locations and reach it nearly simultaneously while avoiding collisions with
background agents. Background-agent densities can differ between the two
meeting locations and may change over time, requiring on-the-fly adaptation.
CrowdNav episodes use a fixed 20-step time limit. An episode is a success if
both agents reach the same meeting location with an arrival-time difference of
at most one step; it is a failure if the agents commit to different locations,
if either agent collides with a background agent, or if the episode times out
without both reaching an agreed location.
"""


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


class TextDisplay:
    """Text display manager for matplotlib figure."""
    
    def __init__(self, ax_text):
        self.ax_text = ax_text
        self.text_objects = []
        self.static_lines = []  # Lines that persist (controls, status)
        self.current_step_lines = []  # Current step info (cleared each step)
        
        # Setup text area
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        ax_text.axis('off')
        ax_text.set_facecolor((0.95, 0.95, 0.95))
        
        # Title (persistent)
        self.title_obj = ax_text.text(0.5, 0.98, 'Control Panel', ha='center', va='top', 
                                      fontsize=18, weight='bold', transform=ax_text.transAxes)
    
    def add_static_line(self, text, color='black', weight='normal'):
        """Add a line that persists (like controls, status messages)."""
        self.static_lines.append((text, color, weight))
        self._redraw()
    
    def update_status(self, message, color='blue', weight='normal'):
        """Update the status message (replaces previous status if exists)."""
        # Remove old status messages (those starting with status indicators)
        # Also check for "Paused" to catch intervention typing
        self.static_lines = [line for line in self.static_lines 
                            if not (line[0].startswith(('⏸', '▶', 'Intervention', '🎯', '✅', '❌', '🧠', '🎉', 'Paused', 'Applying', 'Processed', 'Failed', 'Error', 'Memory', 'Episode', 'Reasoning')))]
        # Add new status
        if message:
            self.static_lines.append((message, color, weight))
        self._redraw()
    
    def set_current_step(self, lines):
        """Set the current step information (replaces previous step info)."""
        self.current_step_lines = lines
        self._redraw()
    
    def _redraw(self):
        """Redraw all text lines."""
        # Clear existing text (except title)
        for obj in self.text_objects:
            obj.remove()
        self.text_objects.clear()
        
        # Calculate available space
        y_start = 0.95
        line_height = 0.03  # Slightly larger for readability
        char_width = 0.012  # Approximate character width
        
        # Draw static lines first (from top)
        y_pos = y_start - 0.05  # Start below title
        for text, color, weight in self.static_lines:
            wrapped = self._wrap_text(text, 70)
            for line in wrapped:
                if y_pos < 0.05:
                    break
                txt = self.ax_text.text(0.02, y_pos, line, ha='left', va='top',
                                       fontsize=11, color=color, weight=weight,
                                       transform=self.ax_text.transAxes,
                                       family='monospace')
                self.text_objects.append(txt)
                y_pos -= line_height
        
        # Add separator
        if self.current_step_lines:
            y_pos -= line_height * 0.5
            sep = self.ax_text.text(0.02, y_pos, '-' * 50, ha='left', va='top',
                                   fontsize=9, color='gray',
                                   transform=self.ax_text.transAxes,
                                   family='monospace')
            self.text_objects.append(sep)
            y_pos -= line_height
        
        # Draw current step lines (from top of step section)
        for text, color, weight in self.current_step_lines:
            wrapped = self._wrap_text(text, 70)
            for line in wrapped:
                if y_pos < 0.02:
                    break
                txt = self.ax_text.text(0.02, y_pos, line, ha='left', va='top',
                                       fontsize=11, color=color, weight=weight,
                                       transform=self.ax_text.transAxes,
                                       family='monospace')
                self.text_objects.append(txt)
                y_pos -= line_height
    
    def _wrap_text(self, text, max_chars):
        """Wrap text to fit in display."""
        if not text:
            return ['']
        words = text.split()
        if not words:
            return [text]
        
        lines = []
        current = []
        current_len = 0
        
        for word in words:
            word_len = len(word)
            if current_len + word_len + 1 <= max_chars:
                current.append(word)
                current_len += word_len + 1
            else:
                if current:
                    lines.append(' '.join(current))
                current = [word]
                current_len = word_len
        if current:
            lines.append(' '.join(current))
        
        return lines if lines else [text]




def _clip_position(env, x, y):
    return (
        float(np.clip(x, env.boundary_min, env.boundary_max)),
        float(np.clip(y, env.boundary_min, env.boundary_max))
    )


def _sample_cluster(center, count, radius):
    cx, cy = center
    offsets = np.random.uniform(-radius, radius, size=(count, 2))
    return [(cx + dx, cy + dy) for dx, dy in offsets]


def _sample_start_pair(env, radius, min_sep):
    for _ in range(50):
        angles = np.random.uniform(0.0, 2.0 * np.pi, size=2)
        radii = np.random.uniform(0.0, radius, size=2)
        r0 = (radii[0] * np.cos(angles[0]), radii[0] * np.sin(angles[0]))
        r1 = (radii[1] * np.cos(angles[1]), radii[1] * np.sin(angles[1]))
        if np.hypot(r0[0] - r1[0], r0[1] - r1[1]) >= min_sep:
            return _clip_position(env, r0[0], r0[1]), _clip_position(env, r1[0], r1[1])
    return _clip_position(env, radii[0] * np.cos(angles[0]), radii[0] * np.sin(angles[0])), _clip_position(
        env, radii[1] * np.cos(angles[1]), radii[1] * np.sin(angles[1])
    )


def setup_cooperative_reaching(env, config, meeting_locations, total_humans=None,
                               dense_count=None, sparse_count=2, density_radius=1.5,
                               step_limit=20, robot_start=(-8.0, -8.0),
                               teammate_start=(8.0, -8.0), random_start=True, circle_radius=None):
    env.meeting_locations = meeting_locations
    env.meeting_radius = max(config.robot.radius, config.humans.radius)
    env.step_limit = step_limit
    env._teammate_goal_idx = 1  # default to less dense location

    # Starting positions (random by default)
    if random_start:
        radius = circle_radius if circle_radius is not None else getattr(config.sim, "circle_radius", 6.0)
        min_sep = 2.0 * max(config.robot.radius, config.humans.radius) + 0.5
        (robot_px, robot_py), (teammate_px, teammate_py) = _sample_start_pair(env, radius, min_sep)
    else:
        robot_px, robot_py = _clip_position(env, robot_start[0], robot_start[1])
        teammate_px, teammate_py = _clip_position(env, teammate_start[0], teammate_start[1])
    env.robot.set(
        robot_px, robot_py,
        meeting_locations[0][0], meeting_locations[0][1],
        0.0, 0.0, 0.0
    )

    teammate = Human(config, 'humans')
    teammate.set(
        teammate_px, teammate_py,
        meeting_locations[env._teammate_goal_idx][0], meeting_locations[env._teammate_goal_idx][1],
        0.0, 0.0, 0.0
    )

    # Build background agents with different densities near the two meeting locations
    humans = [teammate]
    if total_humans is None:
        total_humans = getattr(config.sim, "human_num", 1)
    background_count = max(int(total_humans) - 1, 0)
    if background_count > 0:
        if dense_count is None:
            sparse_count = max(int(sparse_count), 0)
            dense_count = max(background_count - sparse_count, 0)
        else:
            dense_count = max(int(dense_count), 0)
            sparse_count = max(int(sparse_count), 0)
    else:
        dense_count = 0
        sparse_count = 0
    dense_positions = _sample_cluster(meeting_locations[0], dense_count, density_radius)
    sparse_positions = _sample_cluster(meeting_locations[1], sparse_count, density_radius)

    for px, py in dense_positions + sparse_positions:
        h = Human(config, 'humans')
        px, py = _clip_position(env, px, py)
        gx, gy = _clip_position(env, px + np.random.uniform(-1.0, 1.0), py + np.random.uniform(-1.0, 1.0))
        h.set(px, py, gx, gy, 0.0, 0.0, 0.0)
        humans.append(h)

    env.humans = humans
    env.human_num = len(humans)
    env.last_human_states = np.zeros((env.human_num, 5))

    # Set the robot's current target to the denser location initially
    env.robot.gx, env.robot.gy = meeting_locations[0]


def update_teammate_goal(env, meeting_locations, style="conservative", density_radius=2.0, switch_margin=1, switch_prob=0.5):
    if not env.humans:
        return
    teammate = env.humans[0]
    others = env.humans[1:]

    counts = [0, 0]
    for human in others:
        for idx, (mx, my) in enumerate(meeting_locations):
            dist = np.hypot(human.px - mx, human.py - my)
            if dist <= density_radius:
                counts[idx] += 1

    # Select goal based on teammate style
    current_idx = getattr(env, "_teammate_goal_idx", 0)
    if style == "aggressive":
        dists = [np.hypot(teammate.px - mx, teammate.py - my) for (mx, my) in meeting_locations]
        desired_idx = int(np.argmin(dists))
    elif style == "switching":
        desired_idx = 1 - current_idx if np.random.random() < switch_prob else current_idx
    else:
        # conservative: prefer the less crowded location
        desired_idx = 0 if counts[0] + switch_margin < counts[1] else 1 if counts[1] + switch_margin < counts[0] else current_idx
    if desired_idx != current_idx:
        env._teammate_goal_idx = desired_idx
        teammate.gx, teammate.gy = meeting_locations[desired_idx]


def update_background_flow(env, meeting_locations, swap_prob=0.1, drift_radius=1.5):
    if len(env.humans) <= 1:
        return
    for human in env.humans[1:]:
        if np.random.random() < swap_prob:
            target_idx = 0 if np.random.random() < 0.5 else 1
            mx, my = meeting_locations[target_idx]
            gx, gy = _clip_position(env, mx + np.random.uniform(-drift_radius, drift_radius),
                                    my + np.random.uniform(-drift_radius, drift_radius))
            human.gx, human.gy = gx, gy


def main():
    parser = argparse.ArgumentParser(description='HINT-Agent CrowdNav-AHT Demo')
    parser.add_argument('--horizon', type=int, default=20, help='Maximum number of steps (default: 20)')
    parser.add_argument('--seed', type=int, default=6, help='Random seed for environment')
    
    # Background agent configuration
    parser.add_argument('--human_num', type=int, default=None, 
                        help='Total number of humans (includes teammate + background agents). Default: 4')
    parser.add_argument('--human_num_range', type=int, default=None,
                        help='Variation range for human_num. Actual number will be (human_num - range) to (human_num + range). Default: 2')
    parser.add_argument('--circle_radius', type=float, default=None,
                        help='Radius (meters) of the circle where agents start. Default: 6.0')
    parser.add_argument('--human_radius', type=float, default=None,
                        help='Radius (meters) of each human agent. Default: 0.3')
    parser.add_argument('--human_v_pref', type=float, default=None,
                        help='Maximum velocity (m/s) of each human agent. Default: 1.0')
    parser.add_argument('--robot_radius', type=float, default=None,
                        help='Radius (meters) of the robot. Default: 0.3')
    parser.add_argument('--robot_v_pref', type=float, default=None,
                        help='Maximum velocity (m/s) of the robot. Default: 1.0')
    parser.add_argument('--dense_count', type=int, default=None,
                        help='Background agents near meeting location 1 (default: total-sparse)')
    parser.add_argument('--sparse_count', type=int, default=2,
                        help='Background agents near meeting location 2 (default: 2)')
    parser.add_argument('--fixed_start', action='store_true',
                        help='Use fixed starting positions (random by default)')
    parser.add_argument('--participant_id', type=str, default='P000')
    parser.add_argument('--ego_variant', type=str, default='HINT',
                        help='Ego variant label (ProAgent, YAY, CoT, HINT)')
    parser.add_argument('--title_layout', type=str, default='cooperative_reaching',
                        help='Layout label for the figure title')
    parser.add_argument('--title_teammate', type=str, default='orca',
                        help='Teammate label for the figure title')
    parser.add_argument('--teammate_style', type=str, default='conservative',
                        choices=['aggressive', 'conservative', 'switching'],
                        help='Teammate style for goal choice (default: conservative)')
    parser.add_argument('--switch_prob', type=float, default=0.5,
                        help='Switching teammate goal flip probability (default: 0.5)')
    parser.add_argument('--no_cot', action='store_true',
                        help='Disable Chain-of-Thought reasoning (ablation)')
    parser.add_argument('--no_memory', action='store_true',
                        help='Disable memory module (ablation)')
    parser.add_argument('--log', action='store_true', help='Mirror stdout/stderr to a log file')
    parser.add_argument('--log_path', type=str, default=None, help='Optional log file path')
    parser.add_argument('--log_dir', type=str, default='run_logs', help='Directory for auto log files')
    
    args = parser.parse_args()

    setup_stdout_logging(args.log, args.log_path, args.log_dir, "crowdnav_stdout")

    # Initialize environment
    config = Config()
    
    # Apply command-line overrides for agent configuration
    if args.human_num is not None:
        config.sim.human_num = args.human_num
        print(f"📊 Setting total humans to {args.human_num}")
    if args.human_num_range is not None:
        config.sim.human_num_range = args.human_num_range
        print(f"📊 Setting human_num_range to {args.human_num_range}")
    if args.circle_radius is not None:
        config.sim.circle_radius = args.circle_radius
        print(f"📊 Setting circle_radius to {args.circle_radius}")
    if args.human_radius is not None:
        config.humans.radius = args.human_radius
        print(f"📊 Setting human radius to {args.human_radius}")
    if args.human_v_pref is not None:
        config.humans.v_pref = args.human_v_pref
        print(f"📊 Setting human max velocity to {args.human_v_pref}")
    if args.robot_radius is not None:
        config.robot.radius = args.robot_radius
        print(f"📊 Setting robot radius to {args.robot_radius}")
    if args.robot_v_pref is not None:
        config.robot.v_pref = args.robot_v_pref
        print(f"📊 Setting robot max velocity to {args.robot_v_pref}")
    
    # Validate configuration
    if config.sim.human_num - config.sim.human_num_range < 1:
        print(f"⚠️  Warning: human_num ({config.sim.human_num}) - human_num_range ({config.sim.human_num_range}) < 1")
        print(f"   Adjusting human_num_range to {config.sim.human_num - 1}")
        config.sim.human_num_range = config.sim.human_num - 1
    
    env = CrowdSimAHT()
    env.configure(config)
    env.thisSeed = args.seed
    env.nenv = 1
    env.phase = 'test'
    env.step_limit = args.horizon
    config.env.time_limit = args.horizon * config.env.time_step

    # Create and set robot
    robot = Robot(config, 'robot')
    robot.policy = ORCA(config)
    env.set_robot(robot)

    # Create HINT-Agent with intervention
    p0 = HINTAgentCrowdNav(
        model='gpt-5-mini',
        agent_index=0,
        enable_cot=not args.no_cot,
        enable_memory=not args.no_memory,
    )
    p0.set_agent_index(0)

    # Setup matplotlib figure with two subplots: visualization and text
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f"CrowdNav | participant={args.participant_id} | ego={args.ego_variant} | layout={args.title_layout} | teammate={args.title_teammate}",
        fontsize=12,
    )
    
    # Disable matplotlib's default keyboard shortcuts to prevent conflicts
    # (e.g., 's' for save, 'h' for home, etc.)
    try:
        # Disable the navigation toolbar's key bindings
        fig.canvas.manager.toolbar = None
    except:
        pass
    
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.6], hspace=0.3, wspace=0.3)
    
    # Left: Environment visualization
    ax_viz = fig.add_subplot(gs[0, 0])
    ax_viz.set_xlim(-10, 10)
    ax_viz.set_ylim(-10, 10)
    ax_viz.set_xlabel('x(m)', fontsize=14)
    ax_viz.set_ylabel('y(m)', fontsize=14)
    ax_viz.set_aspect('equal')
    ax_viz.grid(True, alpha=0.3)
    env.render_axis = ax_viz
    
    # Right: Text display panel
    ax_text = fig.add_subplot(gs[0, 1])
    text_display = TextDisplay(ax_text)
    
    # Input handling
    intervention_text = ''
    paused = False
    status_message = ''
    reasoning_visible = False
    
    def on_key(event):
        nonlocal intervention_text, paused, status_message
        # Prevent matplotlib's default key bindings from firing
        # by handling all keys we care about and ignoring others
        if event.key is None:
            return
        
        # When paused and typing intervention, allow all keys to go to text input
        # But prevent matplotlib shortcuts from triggering
        if paused and event.key and len(event.key) == 1:
            # Allow typing into intervention text (including 's', 'h', etc.)
            # This will be handled by the elif clause below
            pass
        elif event.key in ('s', 'S', 'h', 'H', 'r', 'R', 'c', 'C'):
            # When NOT paused, intercept matplotlib's default shortcuts to prevent them
            # (save, home, refresh, clear) - just ignore these
            return
        
        if event.key == 'p':
            if not paused:
                paused = True
                intervention_text = ''  # Clear any previous intervention text
                status_message = "Paused - Type intervention and press Enter"
                text_display.update_status(status_message, 'blue', 'bold')
                fig.canvas.draw()
        elif event.key == 'escape':
            if paused and intervention_text:
                # Cancel intervention but stay paused
                intervention_text = ''
                status_message = "Paused - Intervention cancelled"
                text_display.update_status(status_message, 'orange')
            else:
                plt.close('all')
                return
            fig.canvas.draw()
        elif event.key == 'enter' and paused:
            # Send intervention and resume
            cmd = intervention_text.strip()
            intervention_text = ''
            paused = False  # Resume after sending intervention
            if cmd:
                try:
                    status_message = f"Applying: '{cmd}'"
                    text_display.update_status(status_message, 'green', 'bold')
                    fig.canvas.draw()
                    success = p0.process_human_intervention(cmd)
                    if success:
                        status_message = f"Processed: '{cmd}' - Resumed"
                        text_display.update_status(status_message, 'green')
                    else:
                        status_message = f"Failed: '{cmd}' - Resumed"
                        text_display.update_status(status_message, 'red')
                except Exception as e:
                    status_message = f"Error: {e} - Resumed"
                    text_display.update_status(status_message, 'red')
            else:
                status_message = "▶ Resumed (no intervention)"
                text_display.update_status(status_message, 'blue')
            fig.canvas.draw()
        elif event.key == 'backspace' and paused:
            intervention_text = intervention_text[:-1]
            status_message = f"Paused - Intervention: {intervention_text}_"
            text_display.update_status(status_message, 'blue')
            fig.canvas.draw()
        elif paused and event.key and len(event.key) == 1:
            # When paused, typing directly enters intervention text
            intervention_text += event.key
            status_message = f"Paused - Intervention: {intervention_text}_"
            text_display.update_status(status_message, 'blue')
            fig.canvas.draw()
        elif event.key == 'm':
            # Memory details hidden for user study
            status_message = "Memory hidden"
            text_display.update_status(status_message, 'purple')
            fig.canvas.draw()
    
    # Connect our key handler with high priority
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Disable matplotlib's default key bindings by unbinding common shortcuts
    # This prevents 's' (save), 'h' (home), etc. from triggering
    try:
        # Get the default keymap and disable problematic keys
        from matplotlib import rcParams
        # Disable save shortcut
        if 'keymap.save' in rcParams:
            rcParams['keymap.save'] = []
        if 'keymap.home' in rcParams:
            rcParams['keymap.home'] = []
        if 'keymap.back' in rcParams:
            rcParams['keymap.back'] = []
        if 'keymap.forward' in rcParams:
            rcParams['keymap.forward'] = []
    except:
        pass
    
    plt.ion()
    plt.show()
    
    # Initial instructions (static)
    text_display.add_static_line("CrowdNav Demo", color='black', weight='bold')
    text_display.add_static_line("Controls: p=pause, ESC=quit", color='gray')
    text_display.add_static_line("When paused: type intervention, press Enter", color='gray')
    text_display.add_static_line("", color='black')  # Spacer

    meeting_locations = [(-2.0, 0.0), (2.0, 0.0)]
    obs = env.reset()
    setup_cooperative_reaching(env, config, meeting_locations, total_humans=config.sim.human_num,
                               dense_count=args.dense_count, sparse_count=args.sparse_count,
                               density_radius=1.5, step_limit=args.horizon,
                               random_start=not args.fixed_start,
                               circle_radius=config.sim.circle_radius)
    step = 0

    while step < args.horizon:
        # Process matplotlib events frequently for responsive keyboard input
        fig.canvas.flush_events()
        
        # When paused, process events more frequently to respond quickly
        if paused:
            # Process events multiple times while paused for better responsiveness
            for _ in range(10):
                fig.canvas.flush_events()
                time.sleep(0.01)  # Small sleep to allow event processing
            continue  # Skip simulation step when paused
        
        if not paused:
            update_teammate_goal(env, meeting_locations, style=args.teammate_style,
                                 density_radius=2.0, switch_margin=1, switch_prob=args.switch_prob)
            update_background_flow(env, meeting_locations, swap_prob=0.1, drift_radius=1.5)
            # Query agent
            if reasoning_visible:
                text_display.update_status("", 'gray')
                reasoning_visible = False
            action = p0.action(obs)

            # Hide step details (CoT, memory, agent identity) for user study
            text_display.set_current_step([])

            # Step env
            obs, reward, done, info = env.step(action)
            step += 1
            
            # Lower frame rate - longer sleep (but check events during sleep)
            # Break sleep into smaller chunks to check for pause events
            for _ in range(10):
                fig.canvas.flush_events()  # Check for events during sleep
                if paused:  # If paused during sleep, break early
                    break
                if not reasoning_visible:
                    text_display.update_status("Reasoning...", 'gray')
                    reasoning_visible = True
                time.sleep(0.05)  # 10 * 0.05 = 0.5 seconds total

            if done:
                status_message = f"Episode ended: {info}"
                text_display.update_status(status_message, 'green', 'bold')
                obs = env.reset()
                setup_cooperative_reaching(env, config, meeting_locations, total_humans=config.sim.human_num,
                                           dense_count=args.dense_count, sparse_count=args.sparse_count,
                                           density_radius=1.5, step_limit=args.horizon,
                                           random_start=not args.fixed_start,
                                           circle_radius=config.sim.circle_radius)
                step = 0
                status_message = "Environment reset. Continuing..."
                text_display.update_status(status_message, 'blue')

        # Render environment
        try:
            env.render()
            fig.canvas.draw()
            plt.pause(0.05)  # Slower update rate
        except Exception as e:
            status_message = f"Render error: {e}"
            text_display.update_status(status_message, 'red')

    plt.close('all')


if __name__ == '__main__':
    main()
