import pygame
import sys
import torch
import numpy as np
import os
import pickle

# Import custom modules
from config import *
from agent.brain import PongBrain
from agent.memory import ReplayMemory
from agent.trainer import PongTrainer

# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
pygame.init()
pygame.display.set_caption("AI vs Human: The Final Showdown")

# Objects
net = PongBrain()
memory = ReplayMemory(10000)
trainer = PongTrainer(net, lr=LEARNING_RATE, gamma=GAMMA)

# UI & Scoring
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
    net.eval() # Play mode
    epsilon = 0.05 # 5% randomness for a "professional" feel
    print("🧠 Trained Brain Loaded. AI is ready.")
else:
    epsilon = 1.0 # Start random if no brain exists
    print("🆕 No saved brain. Starting training mode.")

if os.path.exists("pong_memory.pkl"):
    with open("pong_memory.pkl", "rb") as f:
        memory.buffer = pickle.load(f)

# ==========================================
# 3. GAME VARIABLES
# ==========================================
ball_x, ball_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
ball_dx, ball_dy = 5, 5 # Slightly faster for excitement
player_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
human_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2

running = True

# ==========================================
# 4. MAIN GAME LOOP
# ==========================================
try:
    while running:
        # A. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game_active:
            # B. State Capture
            old_state = [player_y, ball_x, ball_y, ball_dx, ball_dy]
            reward = 0
            done = False

            # C. AI Decision
            state_tensor = torch.FloatTensor(old_state)
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, 3)
            else:
                with torch.no_grad():
                    action = torch.argmax(net(state_tensor)).item()

            # D. Movements
            if action == 1: player_y -= 6
            elif action == 2: player_y += 6

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: human_y -= 7 # Giving human a slight speed edge
            if keys[pygame.K_DOWN]: human_y += 7

            # Paddle Boundaries
            player_y = max(0, min(player_y, SCREEN_HEIGHT - PADDLE_HEIGHT))
            human_y = max(0, min(human_y, SCREEN_HEIGHT - PADDLE_HEIGHT))

            # E. Ball Physics
            ball_x += ball_dx
            ball_y += ball_dy

            if ball_y <= 0 or ball_y >= SCREEN_HEIGHT - BALL_SIZE:
                ball_dy *= -1

            # Collisions
            if ball_x <= 10 + PADDLE_WIDTH:
                if player_y < ball_y < player_y + PADDLE_HEIGHT:
                    ball_dx *= -1.1 # Speed up slightly on hits
                    ball_x = 10 + PADDLE_WIDTH
                    reward = 10

            if ball_x >= SCREEN_WIDTH - 20 - PADDLE_WIDTH:
                if human_y < ball_y < human_y + PADDLE_HEIGHT:
                    ball_dx *= -1.1
                    ball_x = SCREEN_WIDTH - 20 - PADDLE_WIDTH

            # F. Scoring Logic
            if ball_x > SCREEN_WIDTH:
                ai_score += 1
                reward = 30
                done = True
            elif ball_x < 0:
                human_score += 1
                reward = -50 # High penalty for missing
                done = True

            if done:
                ball_x, ball_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
                ball_dx, ball_dy = 5, 5 # Reset speed
                if ai_score >= WINNING_SCORE or human_score >= WINNING_SCORE:
                    game_active = False

            # G. Learning
            new_state = [player_y, ball_x, ball_y, ball_dx, ball_dy]
            memory.push(old_state, action, reward, new_state, done)
            if len(memory) > 32:
                batch = memory.sample(32)
                states, actions, rewards, next_states, dones = zip(*batch)
                trainer.train_step(states, actions, rewards, next_states, dones)

        # H. DRAWING
        screen.fill((30, 30, 35)) # Darker background
        
        # UI
        score_surf = game_font.render(f"AI: {ai_score}  |  YOU: {human_score}", True, (255, 255, 255))
        screen.blit(score_surf, (SCREEN_WIDTH // 2 - 120, 20))

        # Game Over Overlay
        if not game_active:
            msg = "VICTORY! HUMANITY PREVAILS" if human_score > ai_score else "DEFEAT! AI SUPREMACY"
            color = (50, 255, 50) if human_score > ai_score else (255, 50, 50)
            
            end_surf = end_font.render(msg, True, color)
            retry_surf = game_font.render("Press 'R' to Rematch", True, (200, 200, 200))
            
            screen.blit(end_surf, (SCREEN_WIDTH // 2 - 250, SCREEN_HEIGHT // 2 - 50))
            screen.blit(retry_surf, (SCREEN_WIDTH // 2 - 130, SCREEN_HEIGHT // 2 + 30))

            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                ai_score, human_score = 0, 0
                game_active = True

        # Entities
        pygame.draw.rect(screen, (255, 255, 255), (ball_x, ball_y, BALL_SIZE, BALL_SIZE))
        pygame.draw.rect(screen, (0, 255, 0), (10, player_y, PADDLE_WIDTH, PADDLE_HEIGHT)) # AI
        pygame.draw.rect(screen, (0, 150, 255), (SCREEN_WIDTH - 20, human_y, PADDLE_WIDTH, PADDLE_HEIGHT)) # Human

        pygame.display.flip()
        clock.tick(60) # NORMAL PLAY SPEED

except Exception as e:
    print(f"Error occurred: {e}")

# ==========================================
# 5. SAVE & EXIT
# ==========================================
print("💾 Saving brain...")
torch.save(net.state_dict(), "pong_ai_model.pth")
with open("pong_memory.pkl", "wb") as f:
    pickle.dump(memory.buffer, f)

pygame.quit()
sys.exit()