#!/usr/bin/env python
"""
LLM-Enhanced Command Generator for Human Intervention Commands
Uses GPT to generate semantically similar commands while maintaining the 3x3 framework:
- Axis 1: Trigger for intervention (Why)
- Axis 2: Type of human intervention (What)
"""

import openai
import os
import json
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time

@dataclass
class InterventionType:
    """Represents a type of human intervention."""
    trigger: str  # Axis 1: Why the intervention is needed
    intervention_type: str  # Axis 2: What type of intervention
    description: str
    examples: List[str]

class LLMEnhancedCommandGenerator:
    """Generates diverse human intervention commands using LLM while maintaining 3x3 framework."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-3.5-turbo",
                 use_fallback: bool = True):
        self.model = model
        self.use_fallback = use_fallback
        
        # Set up OpenAI API
        if api_key:
            openai.api_key = api_key
        elif 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        else:
            print("Warning: No OpenAI API key found. Using fallback generation.")
            self.use_fallback = True
        
        # Define the 3x3 intervention framework with hand-coded examples
        self.intervention_framework = {
            # Row 1: Agent Performance Correction
            "agent_performance_correction": {
                "direct_command": InterventionType(
                    trigger="Agent Performance Correction",
                    intervention_type="Direct Command",
                    description="Correct agent mistakes with low-level commands",
                    examples=[
                        "Go to the onion and pick it up",
                        "Turn left and move forward",
                        "Put the onion in the pot now",
                        "Stop what you're doing and serve the dish",
                        "Move to the counter immediately"
                    ]
                ),
                "factual_information": InterventionType(
                    trigger="Agent Performance Correction",
                    intervention_type="Factual Information",
                    description="Provide corrective information about the current state",
                    examples=[
                        "You're holding an onion but the pot is empty",
                        "The dish is ready to serve",
                        "You're facing the wrong direction",
                        "There's a pot right behind you",
                        "The order is almost expired"
                    ]
                ),
                "general_instruction": InterventionType(
                    trigger="Agent Performance Correction",
                    intervention_type="General Instruction",
                    description="Give high-level guidance to correct behavior",
                    examples=[
                        "Focus on completing the current order first",
                        "Prioritize cooking over movement",
                        "Don't waste time on unnecessary actions",
                        "Work more efficiently",
                        "Follow the optimal cooking sequence"
                    ]
                )
            },
            
            # Row 2: Environmental State Update
            "environmental_state_update": {
                "direct_command": InterventionType(
                    trigger="Environmental State Update",
                    intervention_type="Direct Command",
                    description="Command based on superior situational awareness",
                    examples=[
                        "Go to the top-left corner - there's an onion there",
                        "Check the pot on the right - it's about to burn",
                        "Move to the center - your teammate needs help",
                        "Pick up the dish from the counter",
                        "Go to the stove - the burner is free now"
                    ]
                ),
                "factual_information": InterventionType(
                    trigger="Environmental State Update",
                    intervention_type="Factual Information",
                    description="Share information about the environment",
                    examples=[
                        "There's a new order coming in",
                        "The kitchen is getting crowded",
                        "There are onions available in the corner",
                        "The pot on the left is ready",
                        "Your teammate dropped an ingredient"
                    ]
                ),
                "general_instruction": InterventionType(
                    trigger="Environmental State Update",
                    intervention_type="General Instruction",
                    description="Provide strategic guidance based on environment",
                    examples=[
                        "Focus on the left side of the kitchen",
                        "Coordinate with your teammate better",
                        "Watch out for the busy areas",
                        "Use the available space efficiently",
                        "Adapt to the changing kitchen layout"
                    ]
                )
            },
            
            # Row 3: Teammate Model Update
            "teammate_model_update": {
                "direct_command": InterventionType(
                    trigger="Teammate Model Update",
                    intervention_type="Direct Command",
                    description="Command based on teammate knowledge",
                    examples=[
                        "Let your teammate handle the cooking",
                        "Take over serving while they restock",
                        "Move to assist your partner",
                        "Switch roles with your teammate",
                        "Help your teammate with the heavy lifting"
                    ]
                ),
                "factual_information": InterventionType(
                    trigger="Teammate Model Update",
                    intervention_type="Factual Information",
                    description="Share information about teammate state",
                    examples=[
                        "Your teammate is tired and needs a break",
                        "Your partner is better at cooking than serving",
                        "Your teammate can't reach the high shelves",
                        "Your partner is allergic to certain ingredients",
                        "Your teammate prefers the left side of the kitchen"
                    ]
                ),
                "general_instruction": InterventionType(
                    trigger="Teammate Model Update",
                    intervention_type="General Instruction",
                    description="Provide team coordination guidance",
                    examples=[
                        "Work as a team to complete orders",
                        "Coordinate your movements better",
                        "Share the workload evenly",
                        "Communicate your intentions",
                        "Support each other during busy periods"
                    ]
                )
            }
        }
        
        # Fallback templates for when LLM is not available
        self.fallback_templates = {
            "direct_command": [
                "{command}",
                "Please {command}",
                "Could you {command}",
                "I need you to {command}",
                "It would be helpful if you {command}",
                "Can you {command}",
                "Would you mind {command}",
                "I'd like you to {command}",
                "Please {command} right now",
                "I need you to {command} immediately"
            ],
            "factual_information": [
                "{information}",
                "Just so you know, {information}",
                "I noticed that {information}",
                "FYI, {information}",
                "You should be aware that {information}",
                "I want to point out that {information}",
                "Let me tell you that {information}",
                "I see that {information}",
                "I want you to know that {information}",
                "I'm letting you know that {information}"
            ],
            "general_instruction": [
                "{instruction}",
                "I suggest you {instruction}",
                "It would be better if you {instruction}",
                "I recommend that you {instruction}",
                "You should {instruction}",
                "I think you should {instruction}",
                "It might help if you {instruction}",
                "Consider {instruction}",
                "Try to {instruction}",
                "I'd advise you to {instruction}"
            ]
        }
        
        # Modifiers for command variation
        self.urgency_modifiers = [
            "", "quickly", "urgently", "as soon as possible", "right now",
            "immediately", "hurry", "fast", "swiftly", "promptly"
        ]
        
        self.context_modifiers = [
            "", "while you're there", "if possible", "when convenient",
            "at your earliest convenience", "if you have time",
            "when you get a chance", "as soon as you can"
        ]

    def generate_llm_commands(self, 
                             trigger: str, 
                             intervention_type: str, 
                             num_variations: int = 5) -> List[str]:
        """Generate commands using LLM while maintaining framework structure."""
        
        if not self.use_fallback and openai.api_key:
            return self._generate_with_llm(trigger, intervention_type, num_variations)
        else:
            return self._generate_with_fallback(trigger, intervention_type, num_variations)
    
    def _generate_with_llm(self, 
                           trigger: str, 
                           intervention_type: str, 
                           num_variations: int) -> List[str]:
        """Generate commands using GPT while maintaining the 3x3 framework."""
        
        # Get the intervention type details
        if trigger not in self.intervention_framework or intervention_type not in self.intervention_framework[trigger]:
            raise ValueError(f"Unknown trigger: {trigger} or intervention type: {intervention_type}")
        
        intervention = self.intervention_framework[trigger][intervention_type]
        
        # Create LLM prompt
        prompt = self._create_llm_prompt(intervention, num_variations)
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at generating diverse human intervention commands for AI agents in cooking games. Generate natural, varied commands that maintain the same semantic meaning and intervention type."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
                n=1
            )
            
            # Parse response
            generated_commands = self._parse_llm_response(response.choices[0].message.content)
            
            # Ensure we have enough commands
            if len(generated_commands) < num_variations:
                # Fill with fallback if needed
                fallback_commands = self._generate_with_fallback(trigger, intervention_type, num_variations - len(generated_commands))
                generated_commands.extend(fallback_commands)
            
            return generated_commands[:num_variations]
            
        except Exception as e:
            print(f"LLM generation failed: {e}. Using fallback.")
            return self._generate_with_fallback(trigger, intervention_type, num_variations)
    
    def _create_llm_prompt(self, intervention: InterventionType, num_variations: int) -> str:
        """Create a detailed prompt for the LLM."""
        
        prompt = f"""
Generate {num_variations} diverse human intervention commands for the following scenario:

INTERVENTION TYPE: {intervention.trigger} - {intervention.intervention_type}
DESCRIPTION: {intervention.description}

EXISTING EXAMPLES:
{chr(10).join([f"- {example}" for example in intervention.examples])}

REQUIREMENTS:
1. Generate {num_variations} NEW commands (different from the examples above)
2. Maintain the same intervention type and semantic meaning
3. Use natural, human-like language
4. Vary the phrasing, politeness, and urgency
5. Keep commands relevant to cooking game scenarios
6. Ensure each command is actionable and clear

OUTPUT FORMAT:
Return only the commands, one per line, without numbering or bullet points.

Example output format:
Go grab that onion over there
Please collect the vegetable from the corner
Could you pick up the ingredient please
"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse the LLM response into a list of commands."""
        lines = response.strip().split('\n')
        commands = []
        
        for line in lines:
            line = line.strip()
            # Remove common prefixes like "- ", "1. ", "• ", etc.
            line = line.lstrip('- • 1234567890. ')
            
            if line and len(line) > 10:  # Basic validation
                commands.append(line)
        
        return commands
    
    def _generate_with_fallback(self, 
                               trigger: str, 
                               intervention_type: str, 
                               num_variations: int) -> List[str]:
        """Generate commands using fallback templates when LLM is not available."""
        
        if trigger not in self.intervention_framework or intervention_type not in self.intervention_framework[trigger]:
            raise ValueError(f"Unknown trigger: {trigger} or intervention type: {intervention_type}")
        
        intervention = self.intervention_framework[trigger][intervention_type]
        commands = []
        
        for _ in range(num_variations):
            # Select random example
            base_command = random.choice(intervention.examples)
            
            # Apply template
            if intervention_type == "direct_command":
                template = random.choice(self.fallback_templates["direct_command"])
                command = template.format(command=base_command)
            elif intervention_type == "factual_information":
                template = random.choice(self.fallback_templates["factual_information"])
                command = template.format(information=base_command)
            else:  # general_instruction
                template = random.choice(self.fallback_templates["general_instruction"])
                command = template.format(instruction=base_command)
            
            # Add modifiers
            urgency = random.choice(self.urgency_modifiers)
            if urgency:
                command = f"{command} {urgency}"
            
            context = random.choice(self.context_modifiers)
            if context:
                command = f"{command} {context}"
            
            commands.append(command.strip())
        
        return list(set(commands))  # Remove duplicates
    
    def generate_all_intervention_commands(self, num_variations: int = 5) -> Dict:
        """Generate commands for all intervention types using LLM when possible."""
        all_commands = {}
        
        for trigger in self.intervention_framework:
            all_commands[trigger] = {}
            for intervention_type in self.intervention_framework[trigger]:
                print(f"Generating commands for {trigger} - {intervention_type}...")
                commands = self.generate_llm_commands(trigger, intervention_type, num_variations)
                all_commands[trigger][intervention_type] = commands
                
                # Add delay to avoid rate limiting
                if not self.use_fallback:
                    time.sleep(1)
        
        return all_commands
    
    def generate_scenario_based_commands(self, scenario: str, num_variations: int = 5) -> List[str]:
        """Generate scenario-based commands using LLM."""
        
        scenario_prompts = {
            "kitchen_emergency": "Generate commands for kitchen emergency situations like fires, burns, or safety hazards",
            "order_rush": "Generate commands for high-pressure situations with many orders coming in",
            "team_coordination": "Generate commands for coordinating with teammates and managing team dynamics",
            "resource_management": "Generate commands for managing limited resources, ingredients, or equipment"
        }
        
        if scenario not in scenario_prompts:
            return self._generate_with_fallback("agent_performance_correction", "direct_command", num_variations)
        
        if not self.use_fallback and openai.api_key:
            try:
                prompt = f"""
Generate {num_variations} diverse human intervention commands for this cooking game scenario:

SCENARIO: {scenario.replace('_', ' ').title()}
DESCRIPTION: {scenario_prompts[scenario]}

REQUIREMENTS:
1. Generate {num_variations} natural, human-like commands
2. Commands should be appropriate for the scenario
3. Vary the language, politeness, and urgency
4. Keep commands actionable and clear
5. Focus on team coordination and safety

OUTPUT FORMAT:
Return only the commands, one per line, without numbering.
"""
                
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at generating human intervention commands for AI agents in cooking games."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                
                commands = self._parse_llm_response(response.choices[0].message.content)
                return commands[:num_variations]
                
            except Exception as e:
                print(f"LLM generation failed: {e}. Using fallback.")
                return self._generate_with_fallback("agent_performance_correction", "direct_command", num_variations)
        else:
            return self._generate_with_fallback("agent_performance_correction", "direct_command", num_variations)
    
    def get_intervention_types(self) -> List[str]:
        """Get all available intervention types."""
        return list(self.intervention_framework.keys())
    
    def get_intervention_categories(self) -> List[str]:
        """Get all intervention categories."""
        return ["direct_command", "factual_information", "general_instruction"]
    
    def save_generated_commands(self, filename: str, commands: Dict):
        """Save generated commands to file."""
        with open(filename, 'w') as f:
            json.dump(commands, f, indent=2)
    
    def load_generated_commands(self, filename: str) -> Dict:
        """Load generated commands from file."""
        with open(filename, 'r') as f:
            return json.load(f)

def main():
    """Demo the LLM-enhanced command generator."""
    print("=== LLM-Enhanced Command Generator Demo ===\n")
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  No OpenAI API key found. Using fallback generation.")
        print("   Set OPENAI_API_KEY environment variable for LLM generation.\n")
    
    # Initialize generator
    generator = LLMEnhancedCommandGenerator(api_key=api_key)
    
    print("🚀 LLM-Enhanced Command Generation:")
    print("  • Maintains 3x3 intervention framework")
    print("  • Uses GPT for semantic command variations")
    print("  • Falls back to templates when LLM unavailable")
    print("  • Generates diverse, natural language commands\n")
    
    # Show framework structure
    print("📋 3x3 Intervention Framework:")
    print("Axis 1 (Why): Agent Performance Correction | Environmental State Update | Teammate Model Update")
    print("Axis 2 (What): Direct Command | Factual Information | General Instruction\n")
    
    # Generate commands for specific intervention types
    print("🔹 Generating commands for Agent Performance Correction - Direct Command:")
    commands = generator.generate_llm_commands("agent_performance_correction", "direct_command", 3)
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}")
    
    print("\n🔹 Generating commands for Environmental State Update - Factual Information:")
    commands = generator.generate_llm_commands("environmental_state_update", "factual_information", 3)
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}")
    
    # Generate scenario-based commands
    print("\n🚨 Generating Kitchen Emergency Commands:")
    emergency_commands = generator.generate_scenario_based_commands("kitchen_emergency", 3)
    for i, cmd in enumerate(emergency_commands, 1):
        print(f"  {i}. {cmd}")
    
    # Generate all intervention commands
    print("\n📊 Generating All Intervention Commands (LLM-enhanced):")
    print("This may take a moment...")
    all_commands = generator.generate_all_intervention_commands(2)
    
    # Save generated commands
    generator.save_generated_commands('llm_generated_commands.json', all_commands)
    print("\n✅ Generated commands saved to 'llm_generated_commands.json'")
    
    # Show summary
    total_commands = sum(len(types[cat]) for types in all_commands.values() for cat in types.keys())
    print(f"Total commands generated: {total_commands}")
    print(f"Intervention types covered: {len(all_commands)}")
    print(f"Categories per type: {len(next(iter(all_commands.values())))}")

if __name__ == "__main__":
    main() 