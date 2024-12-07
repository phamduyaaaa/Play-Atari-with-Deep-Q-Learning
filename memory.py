from collections import deque
import random

class ReplayMemory(object):
    def __init__(self, capacity=1000000):
        self.memory = deque(maxlen=capacity)
    def push(self,frame_stack, action, reward, next_frame_stack, is_done):
        new_tuple = tuple((frame_stack, action, reward, next_frame_stack, is_done))
        self.memory.append(new_tuple)
    def __len__(self):
        return len(self.memory)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
