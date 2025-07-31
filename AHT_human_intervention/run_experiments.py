#!/usr/bin/env python
"""
Run systematic experiments for the 9 human intervention scenarios.
"""

import os
import sys
import csv
import json
import time
import numpy as np
import torch
from datetime import datetime

# Import experiment scenarios
from experiment_scenarios import EXPERIMENT_SCENARIOS, get_scenarios_by_type, list_scenario_names

# Import Overcooked components
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.agents.agent import AgentPair, StayAgent
from AHT_human_intervention.language.shared_lang_agent import SharedLangAgent

class ExperimentRunner:
    def __init__(self, layout_name="random3", horizon=400, num_runs=5):
        self.layout_name = layout_name
        self.horizon = horizon
        self.num_runs = num_runs
        self.results = []
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"experiment_results_{timestamp}.csv"
        self.setup_logging()
        
    def setup_logging(self):
        """Setup CSV logging for experiment results."""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'scenario_name', 'run_id', 'intervention_type', 'information_type',
                'trigger_step', 'intervention_step', 'episode_reward', 'steps_to_recovery',
                'intervention_success', 'final_heuristic', 'confederate_behavior',
                'total_steps', 'soups_delivered', 'ingredients_cooked'
            ])
    
    def log_result(self, result_dict):
        """Log a single experiment result."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result_dict.get('scenario_name', ''),
                result_dict.get('run_id', 0),
                result_dict.get('intervention_type', ''),
                result_dict.get('information_type', ''),
                result_dict.get('trigger_step', 0),
                result_dict.get('intervention_step', 0),
                result_dict.get('episode_reward', 0),
                result_dict.get('steps_to_recovery', 0),
                result_dict.get('intervention_success', False),
                result_dict.get('final_heuristic', ''),
                result_dict.get('confederate_behavior', ''),
                result_dict.get('total_steps', 0),
                result_dict.get('soups_delivered', 0),
                result_dict.get('ingredients_cooked', 0)
            ])
        self.results.append(result_dict)
    
    def run_single_scenario(self, scenario_name, run_id=0):
        """Run a single scenario once."""
        scenario = EXPERIMENT_SCENARIOS[scenario_name]
        print(f"Running {scenario_name} (run {run_id + 1}/{self.num_runs})")
        
        # Setup environment
        mdp = OvercookedGridworld.from_layout_name(self.layout_name)
        env = OvercookedEnv(mdp, horizon=self.horizon)
        env.reset()
        
        # Setup agents
        ego_agent = SharedLangAgent(mdp, agent_idx=0)
        conf_agent = SharedLangAgent(mdp, agent_idx=1)
        pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
        pair.set_mdp(mdp)
        
        # Track metrics
        episode_reward = 0
        intervention_step = None
        steps_to_recovery = 0
        intervention_success = False
        confederate_switched = False
        pre_intervention_reward = 0
        post_intervention_reward = 0
        
        # Run episode
        for step in range(self.horizon):
            # Confederate behavior changes based on scenario
            if not confederate_switched and step >= scenario['trigger_step'] - 5:
                if scenario['confederate_behavior'] == 'switch_to_stay':
                    conf_agent = StayAgent(mdp, idx=1)
                    pair = AgentPair(ego_agent, conf_agent, allow_duplicate_agents=True)
                    pair.set_mdp(mdp)
                    confederate_switched = True
                    print(f"[LOG] Confederate switched to StayAgent at step {step}")
            
            # Human intervention
            if step == scenario['trigger_step']:
                ego_agent.set_command(scenario['command'])
                intervention_step = step
                pre_intervention_reward = episode_reward
                print(f"[LOG] Intervention: '{scenario['command']}' at step {step}")
            
            # Step environment
            raw = pair.joint_action(env.state)
            ja = tuple(a[0] for a in raw)
            nxt, r, done, _ = env.step(ja)
            env.state = nxt
            episode_reward += r
            
            # Track recovery
            if intervention_step and step > intervention_step:
                if not intervention_success and r > 0:
                    intervention_success = True
                    steps_to_recovery = step - intervention_step
                    post_intervention_reward = episode_reward - pre_intervention_reward
            
            if done:
                break
        
        # Calculate final metrics
        result = {
            'scenario_name': scenario_name,
            'run_id': run_id,
            'intervention_type': scenario['intervention_type'],
            'information_type': scenario['information_type'],
            'trigger_step': scenario['trigger_step'],
            'intervention_step': intervention_step,
            'episode_reward': episode_reward,
            'steps_to_recovery': steps_to_recovery,
            'intervention_success': intervention_success,
            'final_heuristic': getattr(ego_agent, 'heuristic', ''),
            'confederate_behavior': scenario['confederate_behavior'],
            'total_steps': step + 1,
            'soups_delivered': 0,  # TODO: extract from environment state
            'ingredients_cooked': 0  # TODO: extract from environment state
        }
        
        self.log_result(result)
        return result
    
    def run_scenario_multiple_times(self, scenario_name):
        """Run a scenario multiple times and return average results."""
        print(f"\n=== Running {scenario_name} {self.num_runs} times ===")
        results = []
        for run_id in range(self.num_runs):
            result = self.run_single_scenario(scenario_name, run_id)
            results.append(result)
            time.sleep(1)  # Brief pause between runs
        
        # Calculate averages
        avg_reward = np.mean([r['episode_reward'] for r in results])
        avg_recovery_steps = np.mean([r['steps_to_recovery'] for r in results if r['steps_to_recovery'] > 0])
        success_rate = np.mean([r['intervention_success'] for r in results])
        
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Average recovery steps: {avg_recovery_steps:.2f}")
        print(f"Success rate: {success_rate:.2f}")
        
        return results
    
    def run_all_scenarios(self):
        """Run all 9 scenarios."""
        print("Running all 9 experimental scenarios...")
        for scenario_name in list_scenario_names():
            self.run_scenario_multiple_times(scenario_name)
        
        self.print_summary()
    
    def run_scenarios_by_type(self, intervention_type=None, information_type=None):
        """Run scenarios filtered by type."""
        filtered_scenarios = get_scenarios_by_type(intervention_type, information_type)
        print(f"Running {len(filtered_scenarios)} scenarios...")
        
        for scenario_name in filtered_scenarios:
            self.run_scenario_multiple_times(scenario_name)
        
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all results."""
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        
        # Group by intervention type
        for intervention_type in ["Direct Command", "Factual Information", "Strategic Guidance"]:
            type_results = [r for r in self.results if r['intervention_type'] == intervention_type]
            if type_results:
                avg_reward = np.mean([r['episode_reward'] for r in type_results])
                success_rate = np.mean([r['intervention_success'] for r in type_results])
                print(f"{intervention_type}:")
                print(f"  Average reward: {avg_reward:.2f}")
                print(f"  Success rate: {success_rate:.2f}")
                print()

def main():
    # Configuration
    layout_name = "random3"
    horizon = 400
    num_runs = 5
    
    # Create experiment runner
    runner = ExperimentRunner(layout_name, horizon, num_runs)
    
    # Run experiments
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            runner.run_all_scenarios()
        elif sys.argv[1] in list_scenario_names():
            runner.run_scenario_multiple_times(sys.argv[1])
        elif sys.argv[1] == "direct":
            runner.run_scenarios_by_type(intervention_type="Direct Command")
        elif sys.argv[1] == "factual":
            runner.run_scenarios_by_type(intervention_type="Factual Information")
        elif sys.argv[1] == "strategic":
            runner.run_scenarios_by_type(intervention_type="Strategic Guidance")
        else:
            print(f"Unknown scenario: {sys.argv[1]}")
            print(f"Available scenarios: {list_scenario_names()}")
    else:
        print("Usage:")
        print("  python run_experiments.py all                    # Run all scenarios")
        print("  python run_experiments.py <scenario_name>        # Run specific scenario")
        print("  python run_experiments.py direct                 # Run direct command scenarios")
        print("  python run_experiments.py factual                # Run factual information scenarios")
        print("  python run_experiments.py strategic              # Run strategic guidance scenarios")
        print(f"Available scenarios: {list_scenario_names()}")

if __name__ == "__main__":
    main() 