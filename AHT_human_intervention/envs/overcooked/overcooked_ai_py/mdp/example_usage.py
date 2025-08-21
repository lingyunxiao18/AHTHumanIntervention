#!/usr/bin/env python
"""
Example Usage of the Overcooked State to Text Converter
Shows how to integrate the converter with the existing environment.
"""

from overcooked_env import OvercookedEnv
from overcooked_state_converter import create_state_converter


def demonstrate_state_conversion():
    """Demonstrate how to use the state converter with the environment."""
    
    print("=== Overcooked State to Text Converter Example ===\n")
    
    # Create a simple layout
    layout = [
        "X X X X X",
        "X P X X X", 
        "X X X X X",
        "X X X X S"
    ]
    
    try:
        # Create environment using existing layout
        from overcooked_mdp import OvercookedGridworld
        env = OvercookedEnv(lambda: OvercookedGridworld.from_layout_name("simple"))
        
        # Create state converter
        converter = create_state_converter(env.mdp)
        
        print("Environment created successfully!")
        print(f"Layout: {layout}")
        print(f"Grid shape: {mdp.shape}")
        print("\n" + "="*50 + "\n")
        
        # Get initial state
        initial_state = env.state
        print("Initial State Description:")
        print(converter.state_to_text(initial_state))
        print("\n" + "="*50 + "\n")
        
        # Take some actions and show state changes
        print("Taking actions and showing state changes...\n")
        
        # Action: Player 0 moves north
        from overcooked_mdp import Action
        joint_action = (Action.NORTH, Action.STAY)
        
        print(f"Action taken: {joint_action}")
        next_state, reward, done, info = env.step(joint_action)
        
        print("\nNew State Description:")
        print(converter.state_to_text(next_state))
        print(f"\nReward: {reward}")
        print(f"Done: {done}")
        print("\n" + "="*50 + "\n")
        
        # Show summary statistics
        stats = converter.get_summary_stats(next_state)
        print("Current Game Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


def compare_with_lossless_encoding():
    """Compare text description with lossless encoding."""
    
    print("=== Comparison: Text vs Lossless Encoding ===\n")
    
    try:
        from overcooked_mdp import OvercookedGridworld, Action
        
        # Create simple layout
        env = OvercookedEnv(lambda: OvercookedGridworld.from_layout_name("simple"))
        
        # Create converter
        converter = create_state_converter(env.mdp)
        
        # Get state
        state = env.state
        
        print("1. Natural Language Description:")
        print(converter.state_to_text(state))
        print("\n" + "-"*40 + "\n")
        
        print("2. Lossless Encoding Shape:")
        lossless_encoding = mdp.lossless_state_encoding(state)
        print(f"Shape: {lossless_encoding[0].shape}")
        print(f"Features: {lossless_encoding[0].shape[2]} feature channels")
        print(f"Grid: {lossless_encoding[0].shape[0]}x{lossless_encoding[0].shape[1]}")
        
        print("\n3. Traditional state_string representation:")
        print(mdp.state_string(state))
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run the examples."""
    try:
        demonstrate_state_conversion()
        print("\n" + "="*60 + "\n")
        compare_with_lossless_encoding()
        
        print("\n" + "="*60)
        print("Example completed successfully!")
        print("\nKey Benefits of Text Description:")
        print("- Human-readable and interpretable")
        print("- No need to understand feature dimensions")
        print("- Easy to debug and analyze")
        print("- Natural language interface for AI systems")
        print("- Can be used for training language models")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 