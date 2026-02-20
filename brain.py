import torch
import torch.nn as nn
import torch.nn.functional as F


class PongBrain(nn.Module):
    def __init__(self):
        super(PongBrain, self).__init__()
        # Input: 5 numbers (The state)
        # Hidden Layer: 64 "neurons" (to find patterns)
        self.fc1 = nn.Linear(5, 64)
        # Output: 3 numbers (Move Up, Move Down, Stay)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # Activation function (ReLU) - like a switch that turns neurons on/off
        x = F.relu(self.fc1(x))
        # Return the final 3 numbers (Q-values)
        return self.fc2(x)


print("Brain structure created successfully!")
