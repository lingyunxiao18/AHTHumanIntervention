#!/usr/bin/env python
"""
Test script for the interactive experiment runner
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_experiments import InteractiveExperimentRunner
from experiment_design import ExperimentDesign

def test_runner_initialization():
    """Test that the runner initializes correctly"""
    print("Testing runner initialization...")
    
    try:
        runner = InteractiveExperimentRunner(layout_name="random3", horizon=400, num_runs=5)
        print("✓ Runner initialized successfully")
        
        # Test that scenarios are loaded
        scenarios = runner.design.get_all_scenarios()
        print(f"✓ Loaded {len(scenarios)} scenarios")
        
        # Test that environment is set up
        print(f"✓ Environment: {runner.layout_name}")
        print(f"✓ Agents: {type(runner.ego_agent).__name__}, {type(runner.confederate).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize runner: {e}")
        return False

def test_scenario_loading():
    """Test that scenarios can be loaded and accessed"""
    print("\nTesting scenario loading...")
    
    try:
        design = ExperimentDesign()
        scenarios = design.get_all_scenarios()
        
        # Test a specific scenario
        test_scenario = None
        for scenario in scenarios:
            if scenario.scenario_id == "APC_DC_001":
                test_scenario = scenario
                break
        
        if test_scenario:
            print(f"✓ Found test scenario: {test_scenario.scenario_id}")
            print(f"  Description: {test_scenario.description}")
            print(f"  Expected heuristic: {test_scenario.expected_heuristic}")
            print(f"  Human query: {test_scenario.human_query}")
        else:
            print("✗ Could not find test scenario APC_DC_001")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load scenarios: {e}")
        return False

def test_heuristic_application():
    """Test that heuristics can be applied to agents"""
    print("\nTesting heuristic application...")
    
    try:
        runner = InteractiveExperimentRunner()
        
        # Test applying different heuristics
        test_heuristics = ["clockwise", "counterclockwise", "place_onion_in_pot", "deliver_soup"]
        
        for heuristic in test_heuristics:
            try:
                runner.apply_heuristic_to_agent(heuristic)
                print(f"✓ Applied heuristic: {heuristic}")
            except Exception as e:
                print(f"✗ Failed to apply heuristic {heuristic}: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test heuristic application: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Interactive Experiment Runner")
    print("====================================")
    
    tests = [
        test_runner_initialization,
        test_scenario_loading,
        test_heuristic_application
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The interactive runner should work correctly.")
        print("\nTo run the interactive experiment:")
        print("  python run_experiments.py scenario APC_DC_001")
        print("  python run_experiments.py all")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 