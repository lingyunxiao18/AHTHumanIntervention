#!/usr/bin/env python3
import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import threading
import queue

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
                                      fontsize=14, weight='bold', transform=ax_text.transAxes)
    
    def add_static_line(self, text, color='black', weight='normal'):
        """Add a line that persists (like controls, status messages)."""
        self.static_lines.append((text, color, weight))
        self._redraw()
    
    def update_status(self, message, color='blue', weight='normal'):
        """Update the status message (replaces previous status if exists)."""
        # Remove old status messages (those starting with status indicators)
        # Also check for "Paused" to catch intervention typing
        self.static_lines = [line for line in self.static_lines 
                            if not (line[0].startswith(('⏸', '▶', 'Intervention', '🎯', '✅', '❌', '🧠', '🎉', 'Paused', 'Applying', 'Processed', 'Failed', 'Error', 'Memory', 'Episode')))]
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
        line_height = 0.025  # Slightly smaller for better spacing
        char_width = 0.012  # Approximate character width
        
        # Draw static lines first (from top)
        y_pos = y_start - 0.05  # Start below title
        for text, color, weight in self.static_lines:
            wrapped = self._wrap_text(text, 70)
            for line in wrapped:
                if y_pos < 0.05:
                    break
                txt = self.ax_text.text(0.02, y_pos, line, ha='left', va='top',
                                       fontsize=8, color=color, weight=weight,
                                       transform=self.ax_text.transAxes,
                                       family='monospace')
                self.text_objects.append(txt)
                y_pos -= line_height
        
        # Add separator
        if self.current_step_lines:
            y_pos -= line_height * 0.5
            sep = self.ax_text.text(0.02, y_pos, '-' * 50, ha='left', va='top',
                                   fontsize=7, color='gray',
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
                                       fontsize=8, color=color, weight=weight,
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


def main():
    parser = argparse.ArgumentParser(description='ProAgentWithIntervention CrowdNav-AHT Demo')
    parser.add_argument('--horizon', type=int, default=200, help='Maximum number of steps')
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
    
    args = parser.parse_args()

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

    # Create and set robot
    robot = Robot(config, 'robot')
    robot.policy = ORCA(config)
    env.set_robot(robot)

    # Create ProAgent with intervention
    p0 = ProAgentWithInterventionCrowdNav(model='gpt-5-mini', agent_index=0)
    p0.set_agent_index(0)

    # Setup matplotlib figure with two subplots: visualization and text
    fig = plt.figure(figsize=(16, 9))
    
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
            paused = not paused
            if paused:
                intervention_text = ''  # Clear any previous intervention text
                status_message = "Paused - Type intervention and press Enter"
            else:
                status_message = "Resumed"
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
            try:
                if hasattr(p0, 'memory'):
                    memory = p0.memory
                    status_message = f"Memory: {len(memory.semantic)} semantic, {len(memory.episodic)} episodic"
                else:
                    status_message = "No memory found"
                text_display.update_status(status_message, 'purple')
            except Exception as e:
                status_message = f"Memory error: {e}"
                text_display.update_status(status_message, 'red')
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
    text_display.add_static_line("CrowdNav-AHT Demo", color='black', weight='bold')
    text_display.add_static_line("Controls: p=pause/resume, m=memory, ESC=quit", color='gray')
    text_display.add_static_line("When paused: type intervention, press Enter", color='gray')
    text_display.add_static_line("", color='black')  # Spacer
    
    obs = env.reset()
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
            # Query agent
            action = p0.action(obs)
            
            # Build current step information (replaces previous step)
            step_lines = []
            try:
                last_plan = getattr(p0, 'last_plan', None)
                last_intervention_reason = getattr(p0, 'last_intervention_reason', None)
                last_chain_of_thought = getattr(p0, 'last_chain_of_thought', None)
                
                step_lines.append((f"[STEP {step}]", 'black', 'bold'))
                
                if last_intervention_reason:
                    step_lines.append((f"Reason: {last_intervention_reason}", 'orange', 'normal'))
                
                if last_chain_of_thought:
                    step_lines.append(("Chain of Thought:", 'purple', 'bold'))
                    # Split and display CoT (limit to key parts)
                    cot_lines = last_chain_of_thought.split('\n')
                    for line in cot_lines[:8]:  # Limit to 8 lines
                        if line.strip():
                            step_lines.append((f"  {line.strip()}", 'black', 'normal'))
                    if len(cot_lines) > 8:
                        step_lines.append(("  ...", 'gray', 'normal'))
                
                if last_plan:
                    steps_str = ', '.join(map(str, last_plan.get('steps', [])))
                    step_lines.append((f"Plan: [{steps_str}]", 'blue', 'normal'))
                
                waypoint_names = {0: "up", 1: "down", 2: "left", 3: "right", 
                                 4: "up-right", 5: "up-left", 6: "down-right", 7: "down-left"}
                action_name = waypoint_names.get(action, f"unknown({action})")
                step_lines.append((f"Action: {action} ({action_name})", 'green', 'bold'))
                
            except Exception as e:
                step_lines.append((f"[STEP {step}] Error: {e}", 'red', 'normal'))
            
            # Set current step info (replaces previous)
            text_display.set_current_step(step_lines)

            # Step env
            obs, reward, done, info = env.step(action)
            step += 1
            
            # Lower frame rate - longer sleep (but check events during sleep)
            # Break sleep into smaller chunks to check for pause events
            for _ in range(10):
                fig.canvas.flush_events()  # Check for events during sleep
                if paused:  # If paused during sleep, break early
                    break
                time.sleep(0.05)  # 10 * 0.05 = 0.5 seconds total

            if done:
                status_message = f"Episode ended: {info}"
                text_display.update_status(status_message, 'green', 'bold')
                obs = env.reset()
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
