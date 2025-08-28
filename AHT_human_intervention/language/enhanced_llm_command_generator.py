#!/usr/bin/env python
"""
Enhanced LLM Command Generator with Layout-Specific Prompts
Refined prompts targeting specific Overcooked layouts to avoid nonsensical commands.
"""

import openai
import os
import json
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time

@dataclass
class LayoutSpecificIntervention:
    """Layout-specific intervention with refined examples."""
    trigger: str
    intervention_type: str
    description: str
    examples: List[str]
    layout_context: str

class EnhancedLLMCommandGenerator:
    """Enhanced command generator with layout-specific prompts and refined examples."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-3.5-turbo",
                 use_fallback: bool = False,  # Default to LLM usage
                 layout_name: str = "random3"):
        self.model = model
        self.use_fallback = use_fallback
        self.layout_name = layout_name
        
        # Set up OpenAI API
        if api_key:
            openai.api_key = api_key
        elif 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        else:
            print("Warning: No OpenAI API key found. Using fallback generation.")
            self.use_fallback = True
        
        # Layout-specific context
        self.layout_contexts = self._get_layout_contexts()
        
        # Enhanced intervention framework with layout-specific examples
        self.intervention_framework = self._create_enhanced_framework()
        
        # Fallback templates
        self.fallback_templates = {
            "direct_command": [
                "{command}",
                "Please {command}",
                "Could you {command}",
                "I need you to {command}",
                "Can you {command}",
                "Would you mind {command}",
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
                "I see that {information}"
            ],
            "general_instruction": [
                "{instruction}",
                "I suggest you {instruction}",
                "It would be better if you {instruction}",
                "I recommend that you {instruction}",
                "You should {instruction}",
                "I think you should {instruction}",
                "Consider {instruction}",
                "Try to {instruction}"
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

    def _get_layout_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Get layout-specific context information."""
        return {
            "random3": {
                "description": "A small kitchen with basic cooking setup",
                "features": [
                    "Two cooking pots in the center",
                    "Onion dispenser on the left side", 
                    "Dish dispenser on the right side",
                    "Serving area at the bottom",
                    "Counters for food preparation",
                    "Two player starting positions"
                ],
                "common_objects": ["onion", "dish", "soup", "pot", "counter"],
                "common_actions": [
                    "pick up onion from dispenser",
                    "put onion in pot",
                    "pick up dish from dispenser", 
                    "serve soup with dish",
                    "move between stations",
                    "coordinate with teammate"
                ],
                "avoid_terms": [
                    "heavy lifting", "shelves", "refrigerator", "freezer",
                    "microwave", "oven", "stove", "knife", "cutting board",
                    "spice rack", "pantry", "storage", "cabinet"
                ]
            },
            "simple": {
                "description": "A very basic kitchen layout",
                "features": [
                    "Single cooking pot",
                    "Onion dispenser",
                    "Dish dispenser", 
                    "Serving area",
                    "Basic counter space"
                ],
                "common_objects": ["onion", "dish", "soup", "pot"],
                "common_actions": [
                    "pick up onion",
                    "cook soup",
                    "pick up dish",
                    "serve soup"
                ],
                "avoid_terms": [
                    "heavy lifting", "shelves", "refrigerator", "freezer",
                    "microwave", "oven", "stove", "knife", "cutting board"
                ]
            }
        }
    
    def _create_enhanced_framework(self) -> Dict[str, Dict[str, LayoutSpecificIntervention]]:
        """Create enhanced intervention framework with layout-specific examples."""
        layout_context = self.layout_contexts.get(self.layout_name, self.layout_contexts["random3"])
        
        return {
            # Row 1: Agent Performance Correction
            "agent_performance_correction": {
                "direct_command": LayoutSpecificIntervention(
                    trigger="Agent Performance Correction",
                    intervention_type="Direct Command",
                    description="Correct agent mistakes with specific cooking actions",
                    examples=[],  # LLM will generate all commands
                    layout_context=layout_context["description"]
                ),
                "factual_information": LayoutSpecificIntervention(
                    trigger="Agent Performance Correction",
                    intervention_type="Factual Information",
                    description="Provide corrective information about cooking state",
                    examples=[
                        "You're holding an onion but the pot is empty",
                        "The soup in the pot is ready to serve",
                        "You're facing away from the onion dispenser",
                        "There's a pot right next to you",
                        "You have an onion but need to cook it first",
                        "The pot is cooking and will be ready soon"
                    ],
                    layout_context=layout_context["description"]
                ),
                "general_instruction": LayoutSpecificIntervention(
                    trigger="Agent Performance Correction",
                    intervention_type="General Instruction",
                    description="Give high-level cooking guidance",
                    examples=[],  # LLM will generate all commands
                    layout_context=layout_context["description"]
                )
            },
            

            
            # Row 3: Teammate Model Update
            "teammate_model_update": {
                "direct_command": LayoutSpecificIntervention(
                    trigger="Teammate Model Update",
                    intervention_type="Direct Command",
                    description="Command based on teammate coordination",
                    examples=[],  # LLM will generate all commands
                    layout_context=layout_context["description"]
                ),
                "factual_information": LayoutSpecificIntervention(
                    trigger="Teammate Model Update",
                    intervention_type="Factual Information",
                    description="Share information about teammate state",
                    examples=[
                        "Your teammate is busy cooking and needs help serving",
                        "Your partner is better at cooking than serving",
                        "Your teammate can't reach the dish dispenser",
                        "Your partner is working on the onion soup order",
                        "Your teammate prefers to handle the cooking station",
                        "Your partner is waiting for you to serve the soup"
                    ],
                    layout_context=layout_context["description"]
                ),
                "general_instruction": LayoutSpecificIntervention(
                    trigger="Teammate Model Update",
                    intervention_type="General Instruction",
                    description="Provide team coordination guidance",
                    examples=[],  # LLM will generate all commands
                    layout_context=layout_context["description"]
                )
            }
        }
    
    def generate_layout_specific_commands(self, 
                                        trigger: str, 
                                        intervention_type: str, 
                                        num_variations: int = 5) -> List[str]:
        """Generate commands using LLM with layout-specific prompts."""
        
        if not self.use_fallback and ('OPENAI_API_KEY' in os.environ or openai.api_key):
            return self._generate_with_enhanced_llm(trigger, intervention_type, num_variations)
        else:
            return self._generate_with_fallback(trigger, intervention_type, num_variations)
    
    def _generate_with_enhanced_llm(self, 
                                   trigger: str, 
                                   intervention_type: str, 
                                   num_variations: int) -> List[str]:
        """Generate commands using GPT with enhanced layout-specific prompts."""
        
        if trigger not in self.intervention_framework or intervention_type not in self.intervention_framework[trigger]:
            raise ValueError(f"Unknown trigger: {trigger} or intervention type: {intervention_type}")
        
        intervention = self.intervention_framework[trigger][intervention_type]
        layout_context = self.layout_contexts[self.layout_name]
        
        # Create enhanced LLM prompt
        prompt = self._create_enhanced_llm_prompt(intervention, layout_context, num_variations)
        
        try:
            # Call OpenAI API (updated for newer version)
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are an expert at generating diverse human intervention commands for AI agents in the Overcooked cooking game. You are specifically working with the '{self.layout_name}' layout. Generate natural, varied commands that are appropriate for this specific kitchen layout and avoid nonsensical terms."},
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
            print(f"Enhanced LLM generation failed: {e}. Using fallback.")
            return self._generate_with_fallback(trigger, intervention_type, num_variations)
    
    def _create_enhanced_llm_prompt(self, intervention: LayoutSpecificIntervention, layout_context: Dict[str, Any], num_variations: int) -> str:
        """Create an enhanced prompt for the LLM with layout-specific context."""
        
        # Create intervention-specific guidance
        if intervention.intervention_type == "Factual Information":
            intervention_guidance = """
CRITICAL: You are providing FACTUAL INFORMATION about the current game state, NOT giving commands or instructions.

FACTUAL INFORMATION should:
- Describe what IS currently happening or what EXISTS in the game
- Use statements like "You're holding...", "The soup is...", "There's a...", "You have..."
- Inform the player about the current state, not tell them what to do
- Be observations about the game world, not actions to take

DO NOT use command language like:
- "Go to...", "Pick up...", "Move to...", "Do this...", "Make sure to..."
- "Let's...", "Time to...", "Head to...", "Grab a...", "Ready to..."
- "Waiting to...", "Almost ready to...", "Available for..."

DO use factual language like:
- "You're holding an onion but the pot is empty"
- "The soup in the pot is ready to serve"
- "There's a pot right next to you"
- "You have an onion but need to cook it first"
- "The pot is cooking and will be ready soon"
- "The dish dispenser is full and ready for serving"
"""
        elif intervention.intervention_type == "Direct Command":
            intervention_guidance = """
You are giving DIRECT COMMANDS for specific actions to take.

DIRECT COMMANDS should:
- Tell the player exactly what action to perform
- Use imperative language like "Go to...", "Pick up...", "Move to...", "Serve..."
- Be clear, actionable instructions
- Focus on immediate, specific tasks
"""
        elif intervention.intervention_type == "General Instruction":
            intervention_guidance = """
You are providing GENERAL INSTRUCTIONS or high-level guidance.

GENERAL INSTRUCTIONS should:
- Give broad guidance or strategy
- Use language like "Focus on...", "Prioritize...", "Work on...", "Don't waste time..."
- Provide high-level direction rather than specific actions
- Guide overall behavior and decision-making
"""
        else:
            intervention_guidance = "Follow the intervention type and examples provided."
        
        prompt = f"""
Generate {num_variations} diverse human intervention commands for the following Overcooked cooking game scenario:

LAYOUT: {self.layout_name.upper()}
LAYOUT DESCRIPTION: {layout_context['description']}
LAYOUT FEATURES: {', '.join(layout_context['features'])}

INTERVENTION TYPE: {intervention.trigger} - {intervention.intervention_type}
DESCRIPTION: {intervention.description}

{intervention_guidance}

EXISTING EXAMPLES:
{chr(10).join([f"- {example}" for example in intervention.examples])}

REQUIREMENTS:
1. Generate {num_variations} NEW commands (different from the examples above)
2. Maintain the same intervention type and semantic meaning as shown in examples
3. Use natural, human-like language appropriate for cooking games
4. Vary the phrasing, politeness, and urgency
5. Focus ONLY on Overcooked cooking game elements: onions, dishes, pots, cooking, serving
6. Use these specific objects and actions: {', '.join(layout_context['common_objects'])} and {', '.join(layout_context['common_actions'])}
7. AVOID these nonsensical terms: {', '.join(layout_context['avoid_terms'])}
8. Ensure each command is appropriate for the {intervention.intervention_type} type
9. Commands should make sense in the context of the {self.layout_name} layout

IMPORTANT GAME MECHANICS:
- Onions and dishes are UNLIMITED (never run out)
- Serving area can hold UNLIMITED soups (no capacity limit)
- No specific orders - just cook onion soups continuously
- No dish washing - dishes are unlimited and reusable
- No chopping - onions go directly into pots
- Focus on cooking efficiency and coordination, not scarcity

OUTPUT FORMAT:
Return only the commands, one per line, without numbering or bullet points.

Example output format:
{self._get_example_output_format(intervention.intervention_type)}
"""
        
        return prompt
    
    def _get_example_output_format(self, intervention_type: str) -> str:
        """Get example output format based on intervention type."""
        if intervention_type == "Factual Information":
            return """You're holding an onion but the pot is empty
The soup in the pot is ready to serve
There's a pot right next to you"""
        elif intervention_type == "Direct Command":
            return """Go grab an onion from the dispenser
Please pick up a dish and serve the soup
Could you move to the pot and start cooking"""
        elif intervention_type == "General Instruction":
            return """Focus on cooking efficiency and coordination
Prioritize keeping both pots busy with cooking
Work together to maximize soup production"""
        else:
            return """Example command 1
Example command 2
Example command 3"""
    
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
        
        # Check if examples are available (should be empty to force LLM usage)
        if not intervention.examples:
            raise ValueError(f"No handcoded examples available for {trigger} - {intervention_type}. LLM generation required.")
        
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
    
    def generate_all_layout_specific_commands(self, num_variations: int = 5) -> Dict:
        """Generate commands for all intervention types with layout-specific context."""
        all_commands = {}
        
        for trigger in self.intervention_framework:
            all_commands[trigger] = {}
            for intervention_type in self.intervention_framework[trigger]:
                print(f"Generating layout-specific commands for {trigger} - {intervention_type}...")
                commands = self.generate_layout_specific_commands(trigger, intervention_type, num_variations)
                all_commands[trigger][intervention_type] = commands
                
                # Add delay to avoid rate limiting
                if not self.use_fallback:
                    time.sleep(1)
        
        return all_commands
    
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
    """Demo the enhanced LLM command generator."""
    print("=== Enhanced LLM Command Generator Demo ===\n")
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  No OpenAI API key found. Using fallback generation.")
        print("   Set OPENAI_API_KEY environment variable for LLM generation.\n")
    
    # Initialize generator for random3 layout
    generator = EnhancedLLMCommandGenerator(api_key=api_key, layout_name="random3")
    
    print("🚀 Enhanced LLM Command Generation:")
    print("  • Layout-specific prompts for Overcooked")
    print("  • Avoids nonsensical terms like 'heavy lifting', 'shelves'")
    print("  • Focuses on cooking game elements: onions, dishes, pots, serving")
    print("  • Uses GPT for semantic command variations")
    print("  • Falls back to templates when LLM unavailable\n")
    
    # Show layout context
    layout_context = generator.layout_contexts["random3"]
    print(f"📋 Layout Context ({generator.layout_name}):")
    print(f"  Description: {layout_context['description']}")
    print(f"  Features: {', '.join(layout_context['features'])}")
    print(f"  Common Objects: {', '.join(layout_context['common_objects'])}")
    print(f"  Avoid Terms: {', '.join(layout_context['avoid_terms'])}\n")
    
    # Generate commands for specific intervention types
    print("🔹 Generating layout-specific commands for Agent Performance Correction - Direct Command:")
    commands = generator.generate_layout_specific_commands("agent_performance_correction", "direct_command", 3)
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}")
    
    print("\n🔹 Generating layout-specific commands for Environmental State Update - Factual Information:")
    commands = generator.generate_layout_specific_commands("environmental_state_update", "factual_information", 3)
    for i, cmd in enumerate(commands, 1):
        print(f"  {i}. {cmd}")
    
    # Generate all intervention commands
    print("\n📊 Generating All Layout-Specific Commands:")
    print("This may take a moment...")
    all_commands = generator.generate_all_layout_specific_commands(2)
    
    # Save generated commands
    generator.save_generated_commands('enhanced_layout_specific_commands.json', all_commands)
    print("\n✅ Enhanced commands saved to 'enhanced_layout_specific_commands.json'")
    
    # Show summary
    total_commands = sum(len(types[cat]) for types in all_commands.values() for cat in types.keys())
    print(f"Total commands generated: {total_commands}")
    print(f"Intervention types covered: {len(all_commands)}")
    print(f"Categories per type: {len(next(iter(all_commands.values())))}")

if __name__ == "__main__":
    main()
