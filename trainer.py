import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PongTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        
        # The Optimizer is the "Teacher" that tweaks the brain's settings
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        # The Loss Function measures how "wrong" the brain was
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        # Convert data to PyTorch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 1. Predict current Q-values (what the brain thinks is best)
        pred = self.model(states)

        # 2. Calculate the "Target" (what actually happened + future potential)
        target = pred.clone()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                # Bellman Equation: Reward + (Gamma * best future move)
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))

            target[idx][actions[idx]] = Q_new

        # 3. Update the brain
        self.optimizer.zero_grad()      # Clear old memories
        loss = self.criterion(target, pred) # Calculate how far off we were
        loss.backward()                 # Find out which "neurons" caused the error
        self.optimizer.step()           # Tweak those neurons slightly