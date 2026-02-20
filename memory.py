import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        # deque is a special list that only keeps the last 'capacity' items
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a memory of what happened"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Pick a random group of memories to learn from"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

print("Memory system initialized!")