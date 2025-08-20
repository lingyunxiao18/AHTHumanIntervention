#!/usr/bin/env python
"""
Comprehensive Experiment Design for LLM-based Human Intervention Testing

This experiment tests the performance of the LLM intervention module across:
- 3 Intervention Triggers (Why): Agent Performance Correction, Environmental State Update, Teammate Model Update
- 3 Intervention Types (What): Direct Command, Factual Information, Strategic Guidance
- 3 Agent Types: RotateAgent (clockwise), RotateAgent (counterclockwise), OnionToPotAgent

Total: 9 experimental conditions with multiple scenarios each
"""

import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class InterventionTrigger(Enum):
    AGENT_PERFORMANCE_CORRECTION = "agent_performance_correction"
    ENVIRONMENTAL_STATE_UPDATE = "environmental_state_update"
    TEAMMATE_MODEL_UPDATE = "teammate_model_update"

class InterventionType(Enum):
    DIRECT_COMMAND = "direct_command"
    FACTUAL_INFORMATION = "factual_information"
    STRATEGIC_GUIDANCE = "strategic_guidance"

@dataclass
class ExperimentScenario:
    """Represents a single experimental scenario"""
    scenario_id: str
    trigger: InterventionTrigger
    intervention_type: InterventionType
    description: str
    human_query: str
    expected_heuristic: str
    success_criteria: List[str]
    difficulty: str  # "easy", "medium", "hard"
    setup_conditions: Dict[str, Any]

class ExperimentDesign:
    def __init__(self):
        self.scenarios = self._create_scenarios()
    
    def _create_scenarios(self) -> List[ExperimentScenario]:
        """Create all experimental scenarios"""
        scenarios = []
        
        # 1. AGENT PERFORMANCE CORRECTION + DIRECT COMMAND
        scenarios.extend([
            ExperimentScenario(
                scenario_id="APC_DC_001",
                trigger=InterventionTrigger.AGENT_PERFORMANCE_CORRECTION,
                intervention_type=InterventionType.DIRECT_COMMAND,
                description="Agent is stuck in a loop, needs direct navigation command",
                human_query="Go pick up the onion at the top right",
                expected_heuristic="place_onion_in_pot",
                success_criteria=["Agent moves toward onion dispenser", "Agent picks up onion"],
                difficulty="easy",
                setup_conditions={"agent_stuck": True, "onion_available": True}
            ),
            ExperimentScenario(
                scenario_id="APC_DC_002",
                trigger=InterventionTrigger.AGENT_PERFORMANCE_CORRECTION,
                intervention_type=InterventionType.DIRECT_COMMAND,
                description="Agent is ignoring ready soup, needs delivery command",
                human_query="Take the soup and deliver it to the serving area",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent picks up soup", "Agent moves toward serving area"],
                difficulty="medium",
                setup_conditions={"soup_ready": True, "agent_ignoring_soup": True}
            ),
            ExperimentScenario(
                scenario_id="APC_DC_003",
                trigger=InterventionTrigger.AGENT_PERFORMANCE_CORRECTION,
                intervention_type=InterventionType.DIRECT_COMMAND,
                description="Agent is moving in wrong direction, needs direction correction",
                human_query="Move counterclockwise instead of clockwise",
                expected_heuristic="counterclockwise",
                success_criteria=["Agent changes direction to counterclockwise"],
                difficulty="easy",
                setup_conditions={"agent_direction": "clockwise", "inefficient_movement": True}
            )
        ])
        
        # 2. AGENT PERFORMANCE CORRECTION + FACTUAL INFORMATION
        scenarios.extend([
            ExperimentScenario(
                scenario_id="APC_FI_001",
                trigger=InterventionTrigger.AGENT_PERFORMANCE_CORRECTION,
                intervention_type=InterventionType.FACTUAL_INFORMATION,
                description="Agent doesn't know soup is ready",
                human_query="The soup in the pot is ready to be served",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent moves toward pot", "Agent picks up soup"],
                difficulty="medium",
                setup_conditions={"soup_ready": True, "agent_unaware": True}
            ),
            ExperimentScenario(
                scenario_id="APC_FI_002",
                trigger=InterventionTrigger.AGENT_PERFORMANCE_CORRECTION,
                intervention_type=InterventionType.FACTUAL_INFORMATION,
                description="Agent doesn't know pot is empty and needs onions",
                human_query="The pot is empty and needs onions to start cooking",
                expected_heuristic="place_onion_in_pot",
                success_criteria=["Agent moves toward onion dispenser", "Agent picks up onion"],
                difficulty="medium",
                setup_conditions={"pot_empty": True, "onions_available": True}
            ),
            ExperimentScenario(
                scenario_id="APC_FI_003",
                trigger=InterventionTrigger.AGENT_PERFORMANCE_CORRECTION,
                intervention_type=InterventionType.FACTUAL_INFORMATION,
                description="Agent doesn't know there are no plates available",
                human_query="There are no plates left at the dish dispenser",
                expected_heuristic="place_onion_in_pot",  # Alternative task when plates unavailable
                success_criteria=["Agent switches to onion task", "Agent avoids dish dispenser"],
                difficulty="hard",
                setup_conditions={"no_plates": True, "agent_trying_plates": True}
            )
        ])
        
        # 3. AGENT PERFORMANCE CORRECTION + STRATEGIC GUIDANCE
        scenarios.extend([
            ExperimentScenario(
                scenario_id="APC_SG_001",
                trigger=InterventionTrigger.AGENT_PERFORMANCE_CORRECTION,
                intervention_type=InterventionType.STRATEGIC_GUIDANCE,
                description="Agent is inefficient, needs strategic focus",
                human_query="Focus on placing onions in pots right now",
                expected_heuristic="place_onion_in_pot",
                success_criteria=["Agent prioritizes onion placement", "Agent moves toward onion dispenser"],
                difficulty="medium",
                setup_conditions={"agent_inefficient": True, "onions_needed": True}
            ),
            ExperimentScenario(
                scenario_id="APC_SG_002",
                trigger=InterventionTrigger.AGENT_PERFORMANCE_CORRECTION,
                intervention_type=InterventionType.STRATEGIC_GUIDANCE,
                description="Agent is doing wrong task, needs role clarification",
                human_query="You should be delivering soup while your teammate handles onions",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent switches to soup delivery", "Agent avoids onion tasks"],
                difficulty="hard",
                setup_conditions={"teammate_onion_specialist": True, "soup_available": True}
            ),
            ExperimentScenario(
                scenario_id="APC_SG_003",
                trigger=InterventionTrigger.AGENT_PERFORMANCE_CORRECTION,
                intervention_type=InterventionType.STRATEGIC_GUIDANCE,
                description="Agent is stuck in repetitive behavior, needs strategy change",
                human_query="Try alternating between placing onions and delivering soup",
                expected_heuristic="place_onion_and_deliver_soup",
                success_criteria=["Agent alternates tasks", "Agent shows flexible behavior"],
                difficulty="hard",
                setup_conditions={"agent_repetitive": True, "multiple_tasks_available": True}
            )
        ])
        
        # 4. ENVIRONMENTAL STATE UPDATE + DIRECT COMMAND
        scenarios.extend([
            ExperimentScenario(
                scenario_id="ESU_DC_001",
                trigger=InterventionTrigger.ENVIRONMENTAL_STATE_UPDATE,
                intervention_type=InterventionType.DIRECT_COMMAND,
                description="New onion just appeared, agent needs to grab it",
                human_query="Go get the fresh onion that just appeared",
                expected_heuristic="place_onion_in_pot",
                success_criteria=["Agent moves toward new onion", "Agent picks up onion"],
                difficulty="easy",
                setup_conditions={"new_onion_appeared": True, "agent_unaware": True}
            ),
            ExperimentScenario(
                scenario_id="ESU_DC_002",
                trigger=InterventionTrigger.ENVIRONMENTAL_STATE_UPDATE,
                intervention_type=InterventionType.DIRECT_COMMAND,
                description="Soup just finished cooking, needs immediate pickup",
                human_query="The soup just finished cooking, go pick it up now",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent moves toward pot", "Agent picks up soup"],
                difficulty="medium",
                setup_conditions={"soup_just_finished": True, "agent_unaware": True}
            ),
            ExperimentScenario(
                scenario_id="ESU_DC_003",
                trigger=InterventionTrigger.ENVIRONMENTAL_STATE_UPDATE,
                intervention_type=InterventionType.DIRECT_COMMAND,
                description="Plate dispenser just restocked, agent should get plates",
                human_query="The plate dispenser just restocked, go get a plate",
                expected_heuristic="deliver_soup",  # Plates are for soup delivery
                success_criteria=["Agent moves toward plate dispenser", "Agent picks up plate"],
                difficulty="medium",
                setup_conditions={"plates_restocked": True, "agent_unaware": True}
            )
        ])
        
        # 5. ENVIRONMENTAL STATE UPDATE + FACTUAL INFORMATION
        scenarios.extend([
            ExperimentScenario(
                scenario_id="ESU_FI_001",
                trigger=InterventionTrigger.ENVIRONMENTAL_STATE_UPDATE,
                intervention_type=InterventionType.FACTUAL_INFORMATION,
                description="Agent doesn't know about new environmental state",
                human_query="There's a pot that just finished cooking and is ready to serve",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent moves toward ready pot", "Agent picks up soup"],
                difficulty="medium",
                setup_conditions={"pot_just_finished": True, "agent_unaware": True}
            ),
            ExperimentScenario(
                scenario_id="ESU_FI_002",
                trigger=InterventionTrigger.ENVIRONMENTAL_STATE_UPDATE,
                intervention_type=InterventionType.FACTUAL_INFORMATION,
                description="Agent doesn't know about layout changes",
                human_query="The onion dispenser has been moved to the top area",
                expected_heuristic="place_onion_in_pot",
                success_criteria=["Agent moves toward new onion location", "Agent adapts to change"],
                difficulty="hard",
                setup_conditions={"onion_dispenser_moved": True, "agent_old_knowledge": True}
            ),
            ExperimentScenario(
                scenario_id="ESU_FI_003",
                trigger=InterventionTrigger.ENVIRONMENTAL_STATE_UPDATE,
                intervention_type=InterventionType.FACTUAL_INFORMATION,
                description="Agent doesn't know about time pressure",
                human_query="We're running out of time, the soup needs to be served quickly",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent prioritizes soup delivery", "Agent moves faster"],
                difficulty="medium",
                setup_conditions={"time_pressure": True, "soup_ready": True}
            )
        ])
        
        # 6. ENVIRONMENTAL STATE UPDATE + STRATEGIC GUIDANCE
        scenarios.extend([
            ExperimentScenario(
                scenario_id="ESU_SG_001",
                trigger=InterventionTrigger.ENVIRONMENTAL_STATE_UPDATE,
                intervention_type=InterventionType.STRATEGIC_GUIDANCE,
                description="Environmental change requires strategy adaptation",
                human_query="Since the onion dispenser is now closer, focus on onion tasks",
                expected_heuristic="place_onion_in_pot",
                success_criteria=["Agent prioritizes onions", "Agent adapts to new layout"],
                difficulty="medium",
                setup_conditions={"onion_dispenser_closer": True, "agent_old_strategy": True}
            ),
            ExperimentScenario(
                scenario_id="ESU_SG_002",
                trigger=InterventionTrigger.ENVIRONMENTAL_STATE_UPDATE,
                intervention_type=InterventionType.STRATEGIC_GUIDANCE,
                description="Environmental pressure requires role switching",
                human_query="The kitchen is getting crowded, you should focus on delivery while your teammate cooks",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent switches to delivery", "Agent avoids cooking area"],
                difficulty="hard",
                setup_conditions={"kitchen_crowded": True, "teammate_cooking": True}
            ),
            ExperimentScenario(
                scenario_id="ESU_SG_003",
                trigger=InterventionTrigger.ENVIRONMENTAL_STATE_UPDATE,
                intervention_type=InterventionType.STRATEGIC_GUIDANCE,
                description="Environmental constraints require flexible strategy",
                human_query="With the new layout, you should alternate between tasks more efficiently",
                expected_heuristic="place_onion_and_deliver_soup",
                success_criteria=["Agent shows flexible behavior", "Agent adapts to new layout"],
                difficulty="hard",
                setup_conditions={"layout_changed": True, "agent_rigid": True}
            )
        ])
        
        # 7. TEAMMATE MODEL UPDATE + DIRECT COMMAND
        scenarios.extend([
            ExperimentScenario(
                scenario_id="TMU_DC_001",
                trigger=InterventionTrigger.TEAMMATE_MODEL_UPDATE,
                intervention_type=InterventionType.DIRECT_COMMAND,
                description="Teammate is busy, agent needs to take over their task",
                human_query="Your teammate is stuck, go pick up the onion they were supposed to get",
                expected_heuristic="place_onion_in_pot",
                success_criteria=["Agent moves toward onion", "Agent takes over teammate task"],
                difficulty="medium",
                setup_conditions={"teammate_stuck": True, "onion_needed": True}
            ),
            ExperimentScenario(
                scenario_id="TMU_DC_002",
                trigger=InterventionTrigger.TEAMMATE_MODEL_UPDATE,
                intervention_type=InterventionType.DIRECT_COMMAND,
                description="Teammate is focused on wrong task, agent needs to compensate",
                human_query="Your teammate is only doing onions, you need to handle soup delivery",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent focuses on soup delivery", "Agent compensates for teammate"],
                difficulty="medium",
                setup_conditions={"teammate_onion_only": True, "soup_needs_delivery": True}
            ),
            ExperimentScenario(
                scenario_id="TMU_DC_003",
                trigger=InterventionTrigger.TEAMMATE_MODEL_UPDATE,
                intervention_type=InterventionType.DIRECT_COMMAND,
                description="Teammate is inefficient, agent needs to take over",
                human_query="Your teammate is moving too slowly, go handle the soup delivery yourself",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent takes over soup delivery", "Agent ignores teammate inefficiency"],
                difficulty="hard",
                setup_conditions={"teammate_slow": True, "soup_ready": True}
            )
        ])
        
        # 8. TEAMMATE MODEL UPDATE + FACTUAL INFORMATION
        scenarios.extend([
            ExperimentScenario(
                scenario_id="TMU_FI_001",
                trigger=InterventionTrigger.TEAMMATE_MODEL_UPDATE,
                intervention_type=InterventionType.FACTUAL_INFORMATION,
                description="Agent doesn't know about teammate's capabilities",
                human_query="Your teammate can only handle onions, they can't deliver soup",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent focuses on soup delivery", "Agent lets teammate handle onions"],
                difficulty="medium",
                setup_conditions={"teammate_limited_capability": True, "soup_needs_delivery": True}
            ),
            ExperimentScenario(
                scenario_id="TMU_FI_002",
                trigger=InterventionTrigger.TEAMMATE_MODEL_UPDATE,
                intervention_type=InterventionType.FACTUAL_INFORMATION,
                description="Agent doesn't know about teammate's current state",
                human_query="Your teammate is currently busy with a complex task and can't help with onions",
                expected_heuristic="place_onion_in_pot",
                success_criteria=["Agent handles onions", "Agent doesn't expect teammate help"],
                difficulty="medium",
                setup_conditions={"teammate_busy": True, "onions_needed": True}
            ),
            ExperimentScenario(
                scenario_id="TMU_FI_003",
                trigger=InterventionTrigger.TEAMMATE_MODEL_UPDATE,
                intervention_type=InterventionType.FACTUAL_INFORMATION,
                description="Agent doesn't know about teammate's preferences",
                human_query="Your teammate prefers to work in the top area and avoids the bottom",
                expected_heuristic="place_onion_in_pot",  # Work in bottom area
                success_criteria=["Agent works in bottom area", "Agent respects teammate preferences"],
                difficulty="hard",
                setup_conditions={"teammate_area_preference": True, "bottom_area_available": True}
            )
        ])
        
        # 9. TEAMMATE MODEL UPDATE + STRATEGIC GUIDANCE
        scenarios.extend([
            ExperimentScenario(
                scenario_id="TMU_SG_001",
                trigger=InterventionTrigger.TEAMMATE_MODEL_UPDATE,
                intervention_type=InterventionType.STRATEGIC_GUIDANCE,
                description="Teammate's behavior requires strategic adaptation",
                human_query="Your teammate is very fast at onions, so you should focus on soup delivery",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent focuses on soup delivery", "Agent lets teammate handle onions"],
                difficulty="medium",
                setup_conditions={"teammate_fast_onions": True, "soup_needs_delivery": True}
            ),
            ExperimentScenario(
                scenario_id="TMU_SG_002",
                trigger=InterventionTrigger.TEAMMATE_MODEL_UPDATE,
                intervention_type=InterventionType.STRATEGIC_GUIDANCE,
                description="Teammate's strategy requires coordination",
                human_query="Your teammate is following a clockwise pattern, you should go counterclockwise to avoid conflicts",
                expected_heuristic="counterclockwise",
                success_criteria=["Agent moves counterclockwise", "Agent avoids conflicts"],
                difficulty="hard",
                setup_conditions={"teammate_clockwise": True, "conflicts_occurring": True}
            ),
            ExperimentScenario(
                scenario_id="TMU_SG_003",
                trigger=InterventionTrigger.TEAMMATE_MODEL_UPDATE,
                intervention_type=InterventionType.STRATEGIC_GUIDANCE,
                description="Teammate's limitations require role specialization",
                human_query="Your teammate is better at cooking, so you should handle all the delivery tasks",
                expected_heuristic="deliver_soup",
                success_criteria=["Agent focuses on delivery", "Agent lets teammate cook"],
                difficulty="hard",
                setup_conditions={"teammate_better_cook": True, "delivery_needed": True}
            )
        ])
        
        return scenarios
    
    def get_scenarios_by_condition(self, trigger: InterventionTrigger, intervention_type: InterventionType) -> List[ExperimentScenario]:
        """Get all scenarios for a specific experimental condition"""
        return [s for s in self.scenarios if s.trigger == trigger and s.intervention_type == intervention_type]
    
    def get_all_scenarios(self) -> List[ExperimentScenario]:
        """Get all scenarios"""
        return self.scenarios
    
    def export_to_json(self, filename: str = "experiment_scenarios.json"):
        """Export scenarios to JSON file"""
        scenarios_dict = []
        for scenario in self.scenarios:
            scenarios_dict.append({
                "scenario_id": scenario.scenario_id,
                "trigger": scenario.trigger.value,
                "intervention_type": scenario.intervention_type.value,
                "description": scenario.description,
                "human_query": scenario.human_query,
                "expected_heuristic": scenario.expected_heuristic,
                "success_criteria": scenario.success_criteria,
                "difficulty": scenario.difficulty,
                "setup_conditions": scenario.setup_conditions
            })
        
        with open(filename, 'w') as f:
            json.dump(scenarios_dict, f, indent=2)
        
        print(f"Exported {len(scenarios_dict)} scenarios to {filename}")

def main():
    """Main function to run the experiment design"""
    design = ExperimentDesign()
    
    # Print summary
    print("=== EXPERIMENT DESIGN SUMMARY ===")
    print(f"Total scenarios: {len(design.scenarios)}")
    
    # Print scenarios by condition
    for trigger in InterventionTrigger:
        for intervention_type in InterventionType:
            scenarios = design.get_scenarios_by_condition(trigger, intervention_type)
            print(f"\n{trigger.value.upper()} + {intervention_type.value.upper()}: {len(scenarios)} scenarios")
            for scenario in scenarios:
                print(f"  - {scenario.scenario_id}: {scenario.description}")
    
    # Export to JSON
    design.export_to_json()
    
    return design

if __name__ == "__main__":
    main() 