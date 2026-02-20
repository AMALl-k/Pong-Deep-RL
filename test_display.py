import pygame
import sys
import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt  # New Import for the chart

# Import custom modules
from config import *
from brain import PongBrain
from memory import ReplayMemory
from trainer import PongTrainer

# ==========================================
# 1. INITIALIZATION & DATA TRACKING
# ==========================================
pygame.init()
pygame.display.set_caption("AI vs Human: Deep Q-Learning")

net = PongBrain()
memory = ReplayMemory(10000)
trainer = PongTrainer(net, lr=LEARNING_RATE, gamma=GAMMA)

# Tracking for the Performance Chart
reward_history = []
current_episode_reward = 0

# Game State Variables
ai_score = 0
human_score = 0
WINNING_SCORE = 10
game_active = True

game_font = pygame.font.SysFont("Arial", 32)
end_font = pygame.font.SysFont("Arial", 50, bold=True)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# ==========================================
# 2. LOAD PROGRESS
# ==========================================
if os.path.exists("pong_ai_model.pth"):
    net.load_state_dict(torch.load("pong_ai_model.pth"))
    net.eval() 
    epsilon = 0.05 
    print("🧠 Trained Brain Loaded.")
else:
    epsilon = 1.0

# ==========================================
# 3. GAME VARIABLES
# ==========================================
ball_x, ball_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
ball_dx, ball_dy = 5, 5
player_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
human_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2

running = True

# Function to save the learning chart
def save_performance_chart(history):
    if len(history) > 5: # Only save if we have enough data
        plt.figure(figsize=(10, 5))
        plt.plot(history, color='green')
        plt.title('AI Learning Progress (Reward over Time)')
        plt.xlabel('Games Played')
        plt.ylabel('Score/Reward')
        plt.grid(True)
        plt.savefig('learning_chart.png')
        print("📈 Chart saved as learning_chart.png")

# ==========================================
# 4. MAIN LOOP
# ==========================================
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game_active:
            # A. State & Brain
            old_state = [player_y, ball_x, ball_y, ball_dx, ball_dy]
            state_tensor = torch.FloatTensor(old_state)
            
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, 3)
            else:
                with torch.no_grad():
                    action = torch.argmax(net(state_tensor)).item()

            # B. Movement
            if action == 1: player_y -= 6
            elif action == 2: player_y += 6

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: human_y -= 7
            if keys[pygame.K_DOWN]: human_y += 7

            player_y = max(0, min(player_y, SCREEN_HEIGHT - PADDLE_HEIGHT))
            human_y = max(0, min(human_y, SCREEN_HEIGHT - PADDLE_HEIGHT))

            # C. Physics
            ball_x += ball_dx
            ball_y += ball_dy
            if ball_y <= 0 or ball_y >= SCREEN_HEIGHT - BALL_SIZE:
                ball_dy *= -1

            # Collisions
            reward = 0
            done = False

            if ball_x <= 10 + PADDLE_WIDTH:
                if player_y < ball_y < player_y + PADDLE_HEIGHT:
                    ball_dx *= -1.1
                    reward = 10
            
            if ball_x >= SCREEN_WIDTH - 20 - PADDLE_WIDTH:
                if human_y < ball_y < human_y + PADDLE_HEIGHT:
                    ball_dx *= -1.1

            # D. Scoring & Reward Tracking
            if ball_x > SCREEN_WIDTH:
                ai_score += 1
                reward = 30
                done = True
            elif ball_x < 0:
                human_score += 1
                reward = -50
                done = True

            current_episode_reward += reward

            if done:
                reward_history.append(current_episode_reward)
                current_episode_reward = 0
                ball_x, ball_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
                ball_dx, ball_dy = 5, 5
                if ai_score >= WINNING_SCORE or human_score >= WINNING_SCORE:
                    game_active = False

            # E. Learning
            new_state = [player_y, ball_x, ball_y, ball_dx, ball_dy]
            memory.push(old_state, action, reward, new_state, done)
            if len(memory) > 32:
                batch = memory.sample(32)
                states, actions, rewards, next_states, dones = zip(*batch)
                trainer.train_step(states, actions, rewards, next_states, dones)
            
            if epsilon > 0.01:
                epsilon *= 0.9999

        # F. Drawing
        screen.fill((30, 30, 35))
        
        # UI
        score_surf = game_font.render(f"AI: {ai_score}  |  YOU: {human_score}", True, (255, 255, 255))
        screen.blit(score_surf, (SCREEN_WIDTH // 2 - 120, 20))

        if not game_active:
            msg = "VICTORY!" if human_score > ai_score else "AI SUPREMACY!"
            end_surf = end_font.render(msg, True, (255, 255, 255))
            screen.blit(end_surf, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 50))
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                ai_score, human_score, game_active = 0, 0, True

        # Entities
        pygame.draw.rect(screen, (255, 255, 255), (ball_x, ball_y, BALL_SIZE, BALL_SIZE))
        pygame.draw.rect(screen, (0, 255, 0), (10, player_y, PADDLE_WIDTH, PADDLE_HEIGHT))
        pygame.draw.rect(screen, (0, 150, 255), (SCREEN_WIDTH - 20, human_y, PADDLE_WIDTH, PADDLE_HEIGHT))

        pygame.display.flip()
        clock.tick(60)

except Exception as e:
    print(f"Error: {e}")

# ==========================================
# 5. SAVE PERFORMANCE & EXIT
# ==========================================
save_performance_chart(reward_history)
torch.save(net.state_dict(), "pong_ai_model.pth")
pygame.quit()