#!/usr/bin/env python

import pygame
import sys
import torch
import torch.nn.functional as F
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.agents.agent import AgentPair
from AHT_human_intervention.envs.overcooked.overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from language_conditioned_policy import (
    build_env_prompt,
    LangConditionedPolicy,
    tokenize,
    VOCAB,
    MAX_LEN,
)

class InteractiveLangAgent:
    """Interactive agent that takes language commands via keyboard."""
    
    def __init__(self, model_path, layout_name="random3", agent_idx=0, device="cpu"):
        self.device = torch.device(device)
        self.agent_idx = agent_idx
        self.current_command = ""
        
        # Load MDP
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        
        # Get model dimensions
        example_state = torch.FloatTensor(
            self.mdp.lossless_state_encoding(self.mdp.get_standard_start_state())[0].flatten()
        )
        state_dim = example_state.numel()
        num_actions = len(self.mdp.action_idx_to_name)
        
        # Initialize model
        self.model = LangConditionedPolicy(
            state_dim=state_dim,
            vocab_size=len(VOCAB),
            text_dim=128,
            hidden_dim=256,
            nhead=4,
            num_layers=2,
            max_len=MAX_LEN,
            num_actions=num_actions,
        )
        
        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
    def set_command(self, command):
        """Set the language command for the agent."""
        self.current_command = command
        
    def action(self, state):
        """Get action based on current state and language command."""
        if not self.current_command:
            # Fallback to stay if no command given
            return self.mdp.action_name_to_idx['stay']
            
        # Process state
        state_vec = torch.FloatTensor(
            self.mdp.lossless_state_encoding(state)[self.agent_idx].flatten()
        ).unsqueeze(0).to(self.device)
        
        # Tokenize command
        token_ids = tokenize(self.current_command, VOCAB, MAX_LEN).unsqueeze(0).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            logits = self.model(state_vec, token_ids)
            probs = F.softmax(logits, dim=-1)
            action_idx = torch.argmax(probs, dim=-1).item()
            
        return action_idx

def wrap_text(text, font, max_width):
    """Wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        if font.size(test_line)[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def main():
    """Interactive gameplay with language-conditioned policy."""
    
    # Configuration
    LAYOUT_NAME = "random3"
    MODEL_PATH = "lang_policy_pretrained.pt"
    FPS = 10
    
    # Initialize environment and agents
    env = OvercookedEnv.from_layout_name(LAYOUT_NAME)
    visualizer = StateVisualizer()
    
    # Create language-conditioned agents
    agent0 = InteractiveLangAgent(MODEL_PATH, LAYOUT_NAME, agent_idx=0)
    agent1 = InteractiveLangAgent(MODEL_PATH, LAYOUT_NAME, agent_idx=1)
    
    # Set up agent pair
    pair = AgentPair(agent0, agent1, allow_duplicate_agents=True)
    pair.set_mdp(env.mdp)
    
    # Pygame setup
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    TB_H = 150  # Text box height
    screen = pygame.display.set_mode((WIDTH, HEIGHT + TB_H))
    pygame.display.set_caption("Overcooked: Interactive Language Commands")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 24)
    
    input_text = ""
    show_input = False
    step = 0
    score = 0
    
    print("Controls:")
    print("- Press 'p' to enter a command")
    print("- Press 'Enter' to submit command")
    print("- Press 'Escape' to exit")
    print("- Commands will be applied to both agents")
    
    # Main game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif show_input:
                    if event.key == pygame.K_RETURN:
                        # Submit command
                        command = input_text.strip()
                        if command:
                            agent0.set_command(command)
                            agent1.set_command(command)
                            print(f"Command set: '{command}'")
                        input_text = ""
                        show_input = False
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode
                else:
                    if event.key == pygame.K_p:
                        show_input = True
        
        # Get actions from agents
        joint_action = pair.get_actions(env.state)
        
        # Step environment
        next_state, reward, done, info = env.step(joint_action)
        score += reward
        
        # Render
        surf = visualizer.render_state(env.state, grid=None)
        gs = pygame.transform.scale(surf, (WIDTH, HEIGHT))
        screen.blit(gs, (0, 0))
        
        # Draw text box
        pygame.draw.rect(screen, (240, 240, 240), (0, HEIGHT, WIDTH, TB_H))
        pygame.draw.rect(screen, (100, 100, 100), (0, HEIGHT, WIDTH, TB_H), 2)
        
        # Draw score and step info
        score_text = f"Score: {score} | Step: {step}"
        screen.blit(font.render(score_text, True, (0, 0, 0)), (10, HEIGHT + 10))
        
        # Draw current command
        current_cmd = agent0.current_command if agent0.current_command else "No command set"
        cmd_text = f"Current command: {current_cmd}"
        screen.blit(small_font.render(cmd_text, True, (50, 50, 50)), (10, HEIGHT + 40))
        
        # Draw input prompt
        if show_input:
            prompt_text = f"Enter command: {input_text}"
            screen.blit(font.render(prompt_text, True, (0, 100, 0)), (10, HEIGHT + 70))
        else:
            help_text = "Press 'p' to enter a command"
            screen.blit(small_font.render(help_text, True, (100, 100, 100)), (10, HEIGHT + 70))
        
        # Draw controls
        controls_text = "Controls: 'p'=command, 'Enter'=submit, 'Esc'=exit"
        screen.blit(small_font.render(controls_text, True, (100, 100, 100)), (10, HEIGHT + 100))
        
        pygame.display.flip()
        clock.tick(FPS)
        step += 1

if __name__ == "__main__":
    main() 