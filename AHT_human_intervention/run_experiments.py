#!/usr/bin/env python
"""
Interactive Experiment Runner for LLM-based Human Intervention Testing

This script runs the comprehensive experiment design with interactive Pygame visualization,
allowing real-time observation and human intervention during agent interactions.
"""

import os
import sys
import json
import time
import random
import pygame
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import experiment design
from experiment_design import ExperimentDesign, ExperimentScenario, InterventionTrigger, InterventionType

# Import Overcooked components
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.agents.agent import AgentPair
from heuristic_agent import RotateAgent, OnionToPotAgent

# Import the intervention module
from AHT_human_intervention.intervention_LLM_module import process_command

@dataclass
class ExperimentResult:
    """Represents the result of a single experiment"""
    scenario_id: str
    trigger: str
    intervention_type: str
    human_query: str
    expected_heuristic: str
    actual_heuristic: Optional[str]
    success: bool
    response_time: float
    difficulty: str
    notes: str

class InteractiveExperimentRunner:
    def __init__(self, layout_name: str = "random3", horizon: int = 400, num_runs: int = 5):
        self.layout_name = layout_name
        self.horizon = horizon
        self.num_runs = num_runs
        self.results = []
        
        # Initialize environment
        self.mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=["any"], cook_time=5)
        self.env = OvercookedEnv(self.mdp, horizon=horizon)
        
        # Initialize agents
        self.ego_agent = RotateAgent(direction=True)
        self.ego_agent.set_agent_index(0)
        self.ego_agent.set_mdp(self.mdp)
        
        self.confederate = RotateAgent(direction=True)
        self.confederate.set_agent_index(1)
        self.confederate.set_mdp(self.mdp)
        
        self.agent_pair = AgentPair(self.ego_agent, self.confederate, allow_duplicate_agents=True)
        self.agent_pair.set_mdp(self.mdp)
        
        # Initialize visualization
        self.state_visualizer = StateVisualizer(grid=self.mdp.terrain_mtx)
        
        # Load experiment design
        self.design = ExperimentDesign()
        
        # Pygame setup
        self.GAME_WIDTH, self.GAME_HEIGHT = 800, 600
        self.TEXTBOX_HEIGHT = 150
        self.window_size = (self.GAME_WIDTH, self.GAME_HEIGHT + self.TEXTBOX_HEIGHT)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Interactive Experiment Runner: LLM Intervention Testing")
        self.clock = pygame.time.Clock()
        self.fps = 5  # Frames per second
        
        # Font setup
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Interactive state
        self.input_text = ""
        self.show_textbox = False
        self.step_counter = 0
        self.current_scenario = None
        self.intervention_count = 0
        
        # Reset environment
        self.env.reset()
        self.state = self.env.state

    def convert_action(self, action):
        """Convert action tuple to valid action constant"""
        if isinstance(action, tuple) and len(action) == 2:
            mapping = {
                (0, 0): Action.STAY,
                (0, 1): Direction.SOUTH,
                (0, -1): Direction.NORTH,
                (1, 0): Direction.EAST,
                (-1, 0): Direction.WEST,
            }
            return mapping.get(action, action)
        return action

    def wrap_text(self, text, font, max_width):
        """Wrap text into multiple lines"""
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

    def setup_scenario_conditions(self, scenario: ExperimentScenario) -> bool:
        """Setup the environment conditions for a specific scenario"""
        try:
            # Reset environment
            self.env.reset()
            self.state = self.env.state
            self.step_counter = 0
            
            # Apply setup conditions based on scenario
            setup_conditions = scenario.setup_conditions
            
            if setup_conditions.get("teammate_stuck", False):
                # Simulate teammate being stuck
                self.confederate.direction = False
            
            if setup_conditions.get("teammate_onion_only", False):
                # Make teammate focus only on onions
                self.confederate = OnionToPotAgent(direction=True)
                self.confederate.set_agent_index(1)
                self.confederate.set_mdp(self.mdp)
                self.agent_pair = AgentPair(self.ego_agent, self.confederate, allow_duplicate_agents=True)
                self.agent_pair.set_mdp(self.mdp)
            
            return True
            
        except Exception as e:
            print(f"Error setting up scenario conditions: {e}")
            return False

    def run_interactive_scenario(self, scenario: ExperimentScenario) -> ExperimentResult:
        """Run a single scenario with interactive Pygame visualization"""
        print(f"\n=== Running Interactive Scenario: {scenario.scenario_id} ===")
        print(f"Description: {scenario.description}")
        print(f"Expected Heuristic: {scenario.expected_heuristic}")
        print("Press 'p' to intervene, 'r' to reset, 'q' to quit")
        
        self.current_scenario = scenario
        self.intervention_count = 0
        
        # Setup scenario conditions
        if not self.setup_scenario_conditions(scenario):
            return ExperimentResult(
                scenario_id=scenario.scenario_id,
                trigger=scenario.trigger.value,
                intervention_type=scenario.intervention_type.value,
                human_query=scenario.human_query,
                expected_heuristic=scenario.expected_heuristic,
                actual_heuristic=None,
                success=False,
                response_time=0.0,
                difficulty=scenario.difficulty,
                notes="Failed to setup scenario conditions"
            )
        
        # Main interactive loop
        running = True
        while running:
            # Process Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        # Reset scenario
                        self.setup_scenario_conditions(scenario)
                        self.intervention_count = 0
                    elif self.show_textbox:
                        if event.key == pygame.K_RETURN:
                            user_cmd = self.input_text.strip()
                            if user_cmd:
                                # Process intervention
                                start_time = time.time()
                                intervention = process_command(user_cmd)
                                response_time = time.time() - start_time
                                
                                actual_heuristic = intervention.get("ego_agent_new_heuristic", None)
                                
                                # Apply intervention to agent
                                if actual_heuristic:
                                    self.apply_heuristic_to_agent(actual_heuristic)
                                
                                # Record intervention
                                self.intervention_count += 1
                                print(f"Intervention {self.intervention_count}: '{user_cmd}' -> {actual_heuristic}")
                                
                                # Check if this matches expected heuristic
                                success = actual_heuristic == scenario.expected_heuristic
                                if success:
                                    print(f"✓ SUCCESS: Expected '{scenario.expected_heuristic}', got '{actual_heuristic}'")
                                else:
                                    print(f"✗ FAILED: Expected '{scenario.expected_heuristic}', got '{actual_heuristic}'")
                                
                                self.input_text = ""
                                self.show_textbox = False
                                
                                # Return result after first intervention
                                return ExperimentResult(
                                    scenario_id=scenario.scenario_id,
                                    trigger=scenario.trigger.value,
                                    intervention_type=scenario.intervention_type.value,
                                    human_query=user_cmd,
                                    expected_heuristic=scenario.expected_heuristic,
                                    actual_heuristic=actual_heuristic,
                                    success=success,
                                    response_time=response_time,
                                    difficulty=scenario.difficulty,
                                    notes=f"Intervention #{self.intervention_count}"
                                )
                        elif event.key == pygame.K_BACKSPACE:
                            self.input_text = self.input_text[:-1]
                        elif event.key == pygame.K_ESCAPE:
                            self.input_text = ""
                            self.show_textbox = False
                        else:
                            self.input_text += event.unicode
                    else:
                        if event.key == pygame.K_p:
                            self.show_textbox = True

            # Render the environment
            env_surface = self.state_visualizer.render_state(self.env.state, grid=None)
            game_surface = pygame.transform.scale(env_surface, (self.GAME_WIDTH, self.GAME_HEIGHT))
            self.screen.blit(game_surface, (0, 0))

            # Render the textbox area
            textbox_rect = pygame.Rect(0, self.GAME_HEIGHT, self.GAME_WIDTH, self.TEXTBOX_HEIGHT)
            pygame.draw.rect(self.screen, (240, 240, 240), textbox_rect)
            pygame.draw.rect(self.screen, (100, 100, 100), textbox_rect, 2)

            # Render scenario information
            scenario_info = [
                f"Scenario: {scenario.scenario_id}",
                f"Description: {scenario.description}",
                f"Expected: {scenario.expected_heuristic}",
                f"Interventions: {self.intervention_count}",
                f"Step: {self.step_counter}"
            ]
            
            y_offset = self.GAME_HEIGHT + 10
            for info in scenario_info:
                info_surface = self.small_font.render(info, True, (0, 0, 0))
                self.screen.blit(info_surface, (10, y_offset))
                y_offset += info_surface.get_height() + 2

            # Render command input or instructions
            if self.show_textbox:
                text_to_render = "Enter command: " + self.input_text
                instruction = "Press ENTER to submit, ESC to cancel"
            else:
                text_to_render = "Press 'p' to intervene, 'r' to reset, 'q' to quit"
                instruction = "Observe the agents and intervene when needed"
            
            # Wrap and render main text
            lines = self.wrap_text(text_to_render, self.font, self.GAME_WIDTH - 20)
            y_offset = self.GAME_HEIGHT + 80
            for line in lines:
                line_surface = self.font.render(line, True, (0, 0, 0))
                self.screen.blit(line_surface, (10, y_offset))
                y_offset += line_surface.get_height() + 2
            
            # Render instruction
            instruction_surface = self.small_font.render(instruction, True, (100, 100, 100))
            self.screen.blit(instruction_surface, (10, y_offset + 5))

            pygame.display.flip()
            self.clock.tick(self.fps)

            # Advance simulation if not in command input mode
            if not self.show_textbox:
                # Each agent returns (action, info); extract the action and convert if needed
                raw_joint_actions = self.agent_pair.joint_action(self.env.state)
                joint_action = tuple(self.convert_action(a_info[0]) for a_info in raw_joint_actions)
                next_state, reward, done, info = self.env.step(joint_action)
                self.state = next_state
                self.step_counter += 1
                
                if done:
                    print("Episode ended. Resetting environment.")
                    self.env.reset()
                    self.state = self.env.state
                    self.step_counter = 0

        # If we exit without intervention, return a default result
        return ExperimentResult(
            scenario_id=scenario.scenario_id,
            trigger=scenario.trigger.value,
            intervention_type=scenario.intervention_type.value,
            human_query="",
            expected_heuristic=scenario.expected_heuristic,
            actual_heuristic=None,
            success=False,
            response_time=0.0,
            difficulty=scenario.difficulty,
            notes="No intervention made"
        )

    def apply_heuristic_to_agent(self, heuristic: str):
        """Apply the suggested heuristic to the ego agent"""
        if heuristic == "counterclockwise":
            self.ego_agent.direction = False
            print("Applied: counterclockwise movement")
        elif heuristic == "clockwise":
            self.ego_agent.direction = True
            print("Applied: clockwise movement")
        elif heuristic == "place_onion_in_pot":
            # Switch to OnionToPotAgent
            self.ego_agent = OnionToPotAgent(direction=self.ego_agent.direction)
            self.ego_agent.set_agent_index(0)
            self.ego_agent.set_mdp(self.mdp)
            self.agent_pair = AgentPair(self.ego_agent, self.confederate, allow_duplicate_agents=True)
            self.agent_pair.set_mdp(self.mdp)
            print("Applied: place_onion_in_pot strategy")
        elif heuristic == "deliver_soup":
            # Switch to PlateAgent (which focuses on soup delivery)
            from heuristic_agent import PlateAgent
            self.ego_agent = PlateAgent(direction=self.ego_agent.direction)
            self.ego_agent.set_agent_index(0)
            self.ego_agent.set_mdp(self.mdp)
            self.agent_pair = AgentPair(self.ego_agent, self.confederate, allow_duplicate_agents=True)
            self.agent_pair.set_mdp(self.mdp)
            print("Applied: deliver_soup strategy")
        else:
            print(f"Applied: {heuristic} (custom strategy)")

    def run_interactive_experiment(self, scenario_id: str = None):
        """Run interactive experiment with Pygame visualization"""
        pygame.init()
        
        if scenario_id:
            # Run specific scenario
            scenarios = [s for s in self.design.get_all_scenarios() if s.scenario_id == scenario_id]
            if not scenarios:
                print(f"Scenario {scenario_id} not found")
                return
            scenario = scenarios[0]
            result = self.run_interactive_scenario(scenario)
            self.results.append(result)
        else:
            # Run all scenarios interactively
            scenarios = self.design.get_all_scenarios()
            print(f"Running {len(scenarios)} scenarios interactively...")
            
            for i, scenario in enumerate(scenarios, 1):
                print(f"\n{'='*50}")
                print(f"Progress: {i}/{len(scenarios)}")
                print(f"{'='*50}")
                
                result = self.run_interactive_scenario(scenario)
                self.results.append(result)
                
                # Ask if user wants to continue
                if i < len(scenarios):
                    print(f"\nCompleted scenario {scenario.scenario_id}")
                    print("Press 'c' to continue to next scenario, 'q' to quit")
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit()
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_c:
                                    waiting = False
                                elif event.key == pygame.K_q:
                                    pygame.quit()
                                    sys.exit()
                        pygame.time.wait(100)
        
        # Save results
        self.save_results(self.results, "interactive_experiment_results.json")
        self.analyze_results(self.results)
        
        pygame.quit()

    def save_results(self, results: List[ExperimentResult], filename: str = "experiment_results.json"):
        """Save experiment results to JSON file"""
        results_dict = []
        for result in results:
            results_dict.append({
                "scenario_id": result.scenario_id,
                "trigger": result.trigger,
                "intervention_type": result.intervention_type,  # Already a string, no .value needed
                "human_query": result.human_query,
                "expected_heuristic": result.expected_heuristic,
                "actual_heuristic": result.actual_heuristic,
                "success": result.success,
                "response_time": result.response_time,
                "difficulty": result.difficulty,
                "notes": result.notes
            })
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Saved {len(results)} results to {filename}")

    def analyze_results(self, results: List[ExperimentResult]):
        """Analyze and print experiment results"""
        if not results:
            print("No results to analyze")
            return
        
        total_scenarios = len(results)
        successful_scenarios = sum(1 for r in results if r.success)
        success_rate = successful_scenarios / total_scenarios * 100
        
        print(f"\n{'='*60}")
        print("INTERACTIVE EXPERIMENT RESULTS ANALYSIS")
        print(f"{'='*60}")
        print(f"Total scenarios run: {total_scenarios}")
        print(f"Successful scenarios: {successful_scenarios}")
        print(f"Success rate: {success_rate:.2f}%")
        
        # Analyze by condition
        print(f"\n--- Results by Experimental Condition ---")
        for trigger in InterventionTrigger:
            for intervention_type in InterventionType:
                # Compare with string values from ExperimentResult
                condition_results = [r for r in results if r.trigger == trigger.value and r.intervention_type == intervention_type.value]
                if condition_results:
                    condition_success = sum(1 for r in condition_results if r.success)
                    condition_rate = condition_success / len(condition_results) * 100
                    print(f"{trigger.value.upper()} + {intervention_type.value.upper()}: {condition_success}/{len(condition_results)} ({condition_rate:.1f}%)")
        
        # Analyze by difficulty
        print(f"\n--- Results by Difficulty ---")
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_results = [r for r in results if r.difficulty == difficulty]
            if difficulty_results:
                difficulty_success = sum(1 for r in difficulty_results if r.success)
                difficulty_rate = difficulty_success / len(difficulty_results) * 100
                print(f"{difficulty.upper()}: {difficulty_success}/{len(difficulty_results)} ({difficulty_rate:.1f}%)")
        
        # Average response time
        avg_response_time = np.mean([r.response_time for r in results if r.response_time > 0])
        print(f"\nAverage response time: {avg_response_time:.2f}s")

def main():
    """Main function to run interactive experiments"""
    # Configuration
    layout_name = "random3"
    horizon = 400
    num_runs = 5
    
    # Create interactive experiment runner
    runner = InteractiveExperimentRunner(layout_name, horizon, num_runs)
    
    # Run experiments
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            runner.run_interactive_experiment()
        elif sys.argv[1] == "scenario" and len(sys.argv) > 2:
            runner.run_interactive_experiment(scenario_id=sys.argv[2])
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage:")
            print("  python run_experiments.py all                    # Run all scenarios interactively")
            print("  python run_experiments.py scenario <scenario_id> # Run specific scenario")
            print("  python run_experiments.py                        # Show this help")
            return
    else:
        print("Interactive Experiment Runner")
        print("============================")
        print("This will open a Pygame window where you can:")
        print("- Observe agents playing in real-time")
        print("- Press 'p' to intervene with your own commands")
        print("- Press 'r' to reset the current scenario")
        print("- Press 'q' to quit")
        print()
        print("Usage:")
        print("  python run_experiments.py all                    # Run all scenarios interactively")
        print("  python run_experiments.py scenario <scenario_id> # Run specific scenario")
        print()
        print("Example scenario IDs:")
        print("  APC_DC_001, APC_FI_001, ESU_SG_001, etc.")
        return
    
    print("Experiment completed!")

if __name__ == "__main__":
    main() 