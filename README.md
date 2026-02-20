# 🏓 Pong AI: Deep Reinforcement Learning

This repository contains a self-learning AI agent capable of playing Pong against a human opponent. Unlike traditional game AI that uses pre-written rules, this agent learns through **trial and error** using a Deep Q-Network (DQN).

## 🚀 Project Overview

The agent starts with zero knowledge of the game (random movement) and gradually develops a winning strategy by maximizing rewards.

### 🧠 Core AI Concepts

- **Deep Q-Network (DQN):** A neural network built with **PyTorch** that predicts the best action for any given game state.
- **Experience Replay:** A memory buffer that stores past transitions, allowing the agent to "study" previous games to stabilize learning.

- **Epsilon-Greedy Strategy:** A mechanism to balance **Exploration** (trying new moves) and **Exploitation** (using known winning moves).
- **Bellman Equation:** The mathematical foundation used to calculate the value of future rewards.

### 🛠️ Tech Stack

- **Python 3.x**
- **PyTorch**: For the Neural Network and Gradient Descent.
- **Pygame**: For the 2D physics engine and rendering.
- **NumPy**: For efficient state representation.

## 🎮 How to Use

1. **Train:** Let the AI run at hyper-speed by commenting out the frame rate limit.
2. **Play:** Take control of the blue paddle using the **Arrow Keys** and see if you can beat your own creation!
3. **Save/Load:** The agent's progress is automatically saved to `pong_ai_model.pth`.
