#!/usr/bin/env python
"""
LLM-Enhanced Command Generator for Human Intervention Commands
Uses GPT to generate semantically similar commands while maintaining the 3x3 framework:
- Axis 1: Trigger for intervention (Why)
- Axis 2: Type of human intervention (What)

Now incorporates specific Overcooked game rules for more realistic commands.
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
                 model: str = "gpt-4o-mini",
                 use_fallback: bool = True):
        self.model = model
        self.use_fallback = use_fallback
        
        # Set up OpenAI API
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        elif 'OPENAI_API_KEY' in os.environ:
            self.client = openai.OpenAI()
        else:
            print("Warning: No OpenAI API key found. Using fallback generation.")
            self.use_fallback = True
            self.client = None
        
        # Overcooked game rules for context
        self.overcooked_rules = """
OVERCOOKED GAME RULES:
- Goal: Deliver cooked onion soups to serving areas as quickly as possible
- Soup Recipe: 3 onions + 20 steps cooking time in a pot
- Objects: Onions (O), Dishes (D), Pots (P), Serving areas (S), Counters (X)
- Actions: Move (UP/DOWN/LEFT/RIGHT), Interact (pick up/put down), Stay
- Players can only hold ONE object at a time
- Players must face objects to interact with them
- Soup must be cooked for exactly 20 steps before it can be picked up
- Players need a dish to pick up cooked soup from pots
- Delivering soup to serving areas gives rewards
- Players can place objects on counters (X) temporarily
- Movement is grid-based, one step at a time
- Players cannot pass through walls or other players
- NO chopping, cutting, stirring, washing, or cooking actions needed - just INTERACT
- Pots cannot be carried - they are fixed in place
- Unlimited supply of onions and dishes from dispensers
- No cleaning, organizing, or complex tasks - only basic movement and interaction
- Players cannot communicate with each other directly

EXAMPLE LAYOUT (random0):
XXXXXXX
X  O  X
X P P X
X     X
X D D X
X  S  X
XXXXXXX

Where:
- X = Wall (impassable)
- O = Onion dispenser (unlimited onions)
- D = Dish dispenser (unlimited dishes)  
- P = Pot (fixed, cannot be moved)
- S = Serving area (deliver soup here)
- Space = Walkable area
- Players start in walkable areas

VALID ACTIONS:
- Move UP/DOWN/LEFT/RIGHT to walkable spaces
- INTERACT when facing an object (pick up/put down)
- STAY (do nothing)

INVALID ACTIONS (DO NOT GENERATE):
- Chopping, cutting, stirring, washing
- Carrying pots or moving them
- Cleaning areas
- Communicating with teammates
- Complex cooking procedures
- Multiple object handling
"""
        
        # Define the 3x3 intervention framework with Overcooked-specific examples
        self.intervention_framework = {
            # Row 1: Agent Performance Correction
            "agent_performance_correction": {
                "direct_command": InterventionType(
                    trigger="Agent Performance Correction",
                    intervention_type="Direct Command",
                    description="Correct agent mistakes with low-level commands",
                    examples=[
                        "Go to the onion dispenser and pick up an onion",
                        "Turn left and move to the pot",
                        "Put the onion in the pot now",
                        "Stop what you're doing and pick up a dish",
                        "Move to the counter and place the onion there",
                        "Face the pot and interact with it",
                        "Go to the serving area and deliver the soup",
                        "Pick up the dish from the dispenser",
                        "Move to the onion dispenser on the left",
                        "Put the onion in the pot on the right"
                    ]
                ),
                "factual_information": InterventionType(
                    trigger="Agent Performance Correction",
                    intervention_type="Factual Information",
                    description="Provide corrective information about the current state",
                    examples=[
                        "You're holding an onion but the pot is empty",
                        "The soup in the pot is ready to be picked up",
                        "You're facing the wrong direction to interact",
                        "There's a pot right behind you",
                        "The soup needs 5 more steps to cook",
                        "You need a dish to pick up the soup",
                        "The onion dispenser is on your left",
                        "The serving area is in the top right corner",
                        "You can place the onion on the counter",
                        "The pot is full with 3 onions"
                    ]
                ),
                "general_instruction": InterventionType(
                    trigger="Agent Performance Correction",
                    intervention_type="General Instruction",
                    description="Give high-level guidance to correct behavior",
                    examples=[
                        "Focus on completing the current soup order first",
                        "Prioritize cooking onions over movement",
                        "Don't waste time on unnecessary actions",
                        "Work more efficiently by planning your route to the pot",
                        "Follow the optimal cooking sequence: onion -> pot -> dish -> soup -> serve",
                        "Coordinate with your teammate better",
                        "Watch the cooking timer on the soup",
                        "Use counters to store items temporarily",
                        "Check if soup is ready before getting a dish",
                        "Plan your movements to avoid blocking your teammate"
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
                        "Check the pot on the right - the soup is ready",
                        "Move to the center - your teammate needs help",
                        "Pick up the dish from the counter",
                        "Go to the pot - the soup just finished cooking",
                        "Get the onion from the dispenser - it just restocked",
                        "Move to the serving area - there's space now",
                        "Check the pot - it's about to finish cooking",
                        "Go to the counter - there's a dish waiting",
                        "Move to the onion dispenser - new onions appeared"
                    ]
                ),
                "factual_information": InterventionType(
                    trigger="Environmental State Update",
                    intervention_type="Factual Information",
                    description="Share information about the environment",
                    examples=[
                        "There's a new onion order coming in",
                        "The kitchen layout has changed",
                        "There are onions available in the corner",
                        "The pot on the left is ready",
                        "Your teammate dropped an onion",
                        "The soup just finished cooking",
                        "New onions appeared in the dispenser",
                        "The serving area is free now",
                        "There's a dish on the counter",
                        "The pot is getting full"
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
                        "Adapt to the changing kitchen layout",
                        "Monitor the cooking timers",
                        "Check for new orders",
                        "Watch for restocking events",
                        "Be aware of kitchen congestion",
                        "Plan around environmental changes"
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
                        "Let your teammate handle the cooking and you handle the serving",
                        "Take over serving while they cook",
                        "Move to the pot to assist your partner",
                        "Switch roles with your teammate",
                        "Help your teammate with picking up the plate",
                        "Cover for your teammate while they get onions",
                        "Let your teammate use the pot",
                        "Take the serving role",
                        "Help your teammate with dish pickup",
                        "Coordinate movements with your partner"
                    ]
                ),
                "factual_information": InterventionType(
                    trigger="Teammate Model Update",
                    intervention_type="Factual Information",
                    description="Share information about teammate state",
                    examples=[
                        "Your teammate is focused on cooking",
                        "Your partner is better at serving than cooking",
                        "Your teammate is getting onions",
                        "Your partner is waiting for soup to cook",
                        "Your teammate prefers the left side",
                        "Your partner is handling dish pickup",
                        "Your teammate is delivering soup",
                        "Your partner is restocking onions",
                        "Your teammate is coordinating with you",
                        "Your partner is planning the next move"
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
                        "Support each other during busy periods",
                        "Divide tasks efficiently",
                        "Avoid blocking each other",
                        "Plan together for optimal efficiency",
                        "Help each other when needed",
                        "Coordinate cooking and serving roles"
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
                "You might want to {instruction}"
            ]
        }
        
        print("🚀 LLM-Enhanced Command Generation:")
        print("  • Maintains 3x3 intervention framework")
        print("  • Uses GPT for semantic command variations")
        print("  • Falls back to templates when LLM unavailable")
        print("  • Generates diverse, natural language commands")
        print("  • Incorporates Overcooked game rules")
        
        print("\n📋 3x3 Intervention Framework:")
        print("Axis 1 (Why): Agent Performance Correction | Environmental State Update | Teammate Model Update")
        print("Axis 2 (What): Direct Command | Factual Information | General Instruction")

    def generate_commands_with_llm(self, intervention_type: InterventionType, num_commands: int = 100) -> List[str]:
        """Generate commands using LLM with Overcooked-specific context."""
        if not self.client:
            print(f"❌ No OpenAI client available. Cannot generate commands for {intervention_type.trigger} - {intervention_type.intervention_type}")
            return []
        
        # Generate commands in smaller batches to avoid token limits
        batch_size = 20
        all_commands = []
        
        for batch in range(0, num_commands, batch_size):
            current_batch_size = min(batch_size, num_commands - batch)
            
            prompt = f"""
You are generating human intervention commands for the Overcooked cooking game. 

{self.overcooked_rules}

INTERVENTION CONTEXT:
- Trigger: {intervention_type.trigger}
- Type: {intervention_type.intervention_type}
- Description: {intervention_type.description}

EXAMPLES OF THIS TYPE:
{chr(10).join([f"- {example}" for example in intervention_type.examples])}

TASK: Generate {current_batch_size} diverse, natural language commands that fit this intervention type. 

CRITICAL: Your generated commands must be VERY SIMILAR to the examples above in terms of:
- Language style and tone
- Sentence structure and phrasing
- Level of specificity
- Type of information conveyed
- Overall approach and intent

The examples show the EXACT pattern you should follow. Generate commands that could easily be mixed in with these examples without anyone noticing they were generated separately.

CRITICAL: The commands must match BOTH the trigger AND the intervention type:

TRIGGER CONTEXT:
- AGENT PERFORMANCE CORRECTION: Commands that correct agent mistakes or inefficiencies
- ENVIRONMENTAL STATE UPDATE: Commands based on superior situational awareness or environmental changes
- TEAMMATE MODEL UPDATE: Commands based on teammate knowledge or coordination needs

INTERVENTION TYPE:
- DIRECT COMMAND: Give specific, actionable instructions (e.g., "Move to the onion dispenser", "Interact with the pot")
- FACTUAL INFORMATION: Provide information about the current state (e.g., "The soup needs 5 more steps", "You're holding an onion")
- GENERAL INSTRUCTION: Give high-level guidance or strategy (e.g., "Focus on completing orders", "Coordinate with your teammate")

The commands should be:
1. Specific to Overcooked game mechanics (ONLY move, interact, stay)
2. Natural and varied in language
3. Appropriate for the given trigger AND type
4. Realistic human intervention commands
5. VERY SIMILAR to the examples above in style, tone, and approach
6. ONLY use valid actions (move, interact, stay) - NO invalid actions
7. MATCH THE TRIGGER CONTEXT (agent correction vs environmental awareness vs teammate coordination)
8. MATCH THE INTERVENTION TYPE (direct command vs factual information vs general instruction)
9. FOLLOW THE EXACT PATTERN shown in the examples above
10. BE INDISTINGUISHABLE from the examples in terms of style and approach

CRITICAL: Only generate commands that use the valid actions listed above. Do NOT include any invalid actions like chopping, stirring, carrying pots, cleaning, etc.

Generate only the commands, one per line, without numbering or bullet points.
"""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"You are an expert in generating natural language commands for human-AI interaction in cooking games. Your generated commands must be VERY SIMILAR to the examples provided in the prompt - they should follow the exact same language style, tone, sentence structure, and approach. Generate commands that could easily be mixed in with the examples without anyone noticing they were generated separately. Match BOTH the trigger context (agent correction vs environmental awareness vs teammate coordination) AND the intervention type (direct command vs factual information vs general instruction). Use ONLY valid Overcooked actions: move, interact, stay. Do NOT include invalid actions like chopping, stirring, carrying pots, cleaning, etc."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.9
                )
                
                commands = response.choices[0].message.content.strip().split('\n')
                # Clean up commands
                commands = [cmd.strip() for cmd in commands if cmd.strip() and not cmd.strip().startswith(('-', '•', '1.', '2.', '3.'))]
                
                all_commands.extend(commands)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ LLM generation failed for batch {batch}: {e}")
                # No fallback - skip this batch
                continue
        
        print(f"  ✅ Generated {len(all_commands)} LLM commands")
        return all_commands

    def generate_fallback_commands(self, intervention_type: InterventionType, num_commands: int = 100) -> List[str]:
        """Generate commands using fallback templates."""
        commands = []
        templates = self.fallback_templates.get(intervention_type.intervention_type.lower().replace(" ", "_"), [])
        
        # Use base examples as commands
        base_commands = intervention_type.examples.copy()
        
        # Generate variations using templates
        for template in templates:
            for base_cmd in base_commands:
                if len(commands) >= num_commands:
                    break
                command = template.format(command=base_cmd, information=base_cmd, instruction=base_cmd)
                commands.append(command)
        
        # Add urgency and manner variations
        urgency_words = ["quickly", "hurry", "fast", "immediately", "right now", "asap", "urgently"]
        manner_words = ["carefully", "gently", "precisely", "exactly", "directly", "straight", "efficiently"]
        
        for base_cmd in base_commands:
            if len(commands) >= num_commands:
                break
            for urgency in urgency_words:
                if len(commands) >= num_commands:
                    break
                commands.append(f"{urgency} {base_cmd}")
        
        for base_cmd in base_commands:
            if len(commands) >= num_commands:
                break
            for manner in manner_words:
                if len(commands) >= num_commands:
                    break
                commands.append(f"{manner} {base_cmd}")
        
        # Add more variations if needed
        while len(commands) < num_commands:
            base_cmd = random.choice(base_commands)
            template = random.choice(templates)
            command = template.format(command=base_cmd, information=base_cmd, instruction=base_cmd)
            if command not in commands:
                commands.append(command)
        
        return commands[:num_commands]

    def generate_all_intervention_commands(self, commands_per_type: int = 100) -> Dict[str, Dict[str, List[str]]]:
        """Generate commands for all 9 intervention types."""
        all_commands = {}
        
        print(f"\n📊 Generating All Intervention Commands (LLM-enhanced):")
        print("This may take a moment...")
        
        for trigger, trigger_types in self.intervention_framework.items():
            all_commands[trigger] = {}
            for intervention_type_name, intervention_type in trigger_types.items():
                print(f"Generating commands for {trigger} - {intervention_type_name}...")
                commands = self.generate_commands_with_llm(intervention_type, commands_per_type)
                all_commands[trigger][intervention_type_name] = commands
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
        
        return all_commands

    def save_commands_to_file(self, commands: Dict[str, Dict[str, List[str]]], filename: str = "llm_generated_commands.json"):
        """Save generated commands to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(commands, f, indent=2)
        
        total_commands = sum(len(cmds) for trigger_dict in commands.values() for cmds in trigger_dict.values())
        print(f"\n✅ Generated commands saved to '{filename}'")
        print(f"Total commands generated: {total_commands}")
        print(f"Intervention types covered: {len(commands)}")
        print(f"Categories per type: {len(next(iter(commands.values())))}")

    def demo_generation(self):
        """Demonstrate command generation for a few examples."""
        print("\n=== LLM-Enhanced Command Generator Demo ===\n")
        
        if not self.client:
            print("⚠️  No OpenAI API key found. Using fallback generation.")
            print("   Set OPENAI_API_KEY environment variable for LLM generation.\n")
        
        # Demo for Agent Performance Correction - Direct Command
        apc_dc = self.intervention_framework["agent_performance_correction"]["direct_command"]
        print("🔹 Generating commands for Agent Performance Correction - Direct Command:")
        commands = self.generate_commands_with_llm(apc_dc, 3)
        for i, cmd in enumerate(commands, 1):
            print(f"  {i}. {cmd}")
        
        # Demo for Environmental State Update - Factual Information
        esu_fi = self.intervention_framework["environmental_state_update"]["factual_information"]
        print("\n🔹 Generating commands for Environmental State Update - Factual Information:")
        commands = self.generate_commands_with_llm(esu_fi, 3)
        for i, cmd in enumerate(commands, 1):
            print(f"  {i}. {cmd}")
        
        # Demo for emergency commands
        print("\n🚨 Generating Kitchen Emergency Commands:")
        emergency_commands = [
            "The soup is burning! Turn off the heat immediately!",
            "Quick! The order is about to expire!",
            "Hurry! The customer is waiting!",
            "Emergency! The kitchen is on fire!",
            "Urgent! The soup is about to overflow!"
        ]
        for i, cmd in enumerate(emergency_commands[:3], 1):
            print(f"  {i}. {cmd}")

def main():
    """Main function to run the command generator."""
    generator = LLMEnhancedCommandGenerator()
    
    # Run demo
    generator.demo_generation()
    
    # Generate all commands
    all_commands = generator.generate_all_intervention_commands(commands_per_type=100)
    
    # Save to file
    generator.save_commands_to_file(all_commands)

if __name__ == "__main__":
    main() 