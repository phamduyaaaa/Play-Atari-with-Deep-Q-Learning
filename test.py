import gymnasium as gym
import ale_py
import torch
from dqn import DQNetwork
import csv
import os
from utils import *
import yaml

if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        # Test configuration
        MODE = config["testing"]["mode"]
        NUM_EPISODES = config["testing"]["num_episodes"]
        ENV = config["env"]
        MODEL_PATH = config["testing"]["model_path"]

    os.makedirs("checkpoints", exist_ok=True)
    # Initialize environment and model
    env = gym.make(ENV, render_mode= MODE)
    NUM_ACTIONS = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()

    os.makedirs("test_results", exist_ok=True)
    with open("test_results/reward_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total Reward"])

        for episode in range(NUM_EPISODES):
            observation, _ = env.reset()
            stack = None
            current_frame_stack, stack = frame_stack(observation, stack)
            current_frame_stack = current_frame_stack.to(device)
            done = False
            total_reward = 0

            while not done:
                with torch.no_grad():
                    q_values = model(current_frame_stack)
                    action = q_values.argmax().item()

                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_frame_stack, stack = frame_stack(next_observation, stack)
                next_frame_stack = next_frame_stack.to(device)
                current_frame_stack = next_frame_stack
                total_reward += reward

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
            writer.writerow([episode + 1, total_reward])

    print("Testing completed. Results saved in 'test_results/reward_data.csv'.")









