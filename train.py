import gymnasium as gym
import ale_py
import torch
from tqdm import tqdm
from dqn import DQNetwork
from memory import ReplayMemory
import csv
import os
from utils import *
import yaml

if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        # Training
        BATCH_SIZE = config["training"]["batch_size"]
        NUM_EPISODES = config["training"]["num_episodes"]
        GAMMA = config["training"]["gamma"]
        MODE = config["training"]["mode"]
        # Epsilon
        EPSILON_START = config["epsilon"]["start"]
        EPSILON_END = config["epsilon"]["end"]
        EPSILON_DECAY = config["epsilon"]["decay"]
        # Model
        UPDATE = config["model"]["update_freq"]
        LEARNING_RATE = config["model"]["lr"]
        # Memory
        CAPACITY = config["memory"]["capacity"]
        ENV = config["env"]

    os.makedirs("checkpoints", exist_ok=True)
    # Initialize environment and model
    env = gym.make(ENV, render_mode= MODE)
    NUM_ACTIONS = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DQNetwork(num_actions=NUM_ACTIONS).to(device)
    target_model = DQNetwork(num_actions=NUM_ACTIONS).to(device)
    target_model.load_state_dict(model.state_dict())  # Sync target network

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    memory = ReplayMemory(capacity=CAPACITY)
    with open("reward_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total Reward", "Loss", "Epsilon"])
        # Khởi tạo thanh tiến trình
        max_reward = float('-inf')  # Biến lưu điểm cao nhất
        progress_bar = tqdm(total=NUM_EPISODES, desc="Training Progress", ncols=100, unit="episode", colour="green", dynamic_ncols=True)
        for episode in range(NUM_EPISODES):
            observation, _ = env.reset()
            stack = None
            current_frame_stack, stack = frame_stack(observation, stack)
            current_frame_stack = current_frame_stack.to(device)

            done = False
            total_reward = 0
            epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))

            while not done:
                q_values = model(current_frame_stack)
                action = epsilon_greedy_policy(env, q_values, epsilon)

                next_observation, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_frame_stack, stack = frame_stack(next_observation, stack)
                next_frame_stack = next_frame_stack.to(device)
                memory.push(current_frame_stack, action, reward, next_frame_stack, done)
                current_frame_stack = next_frame_stack
                total_reward += reward

                # Training step
                if len(memory) > BATCH_SIZE:
                    minibatch = memory.sample(BATCH_SIZE)
                    states, actions, rewards, next_states, dones = zip(*minibatch)

                    states = torch.cat(states).to(device)
                    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                    next_states = torch.cat(next_states).to(device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(device)

                    q_values = model(states).gather(1, actions).squeeze(1)
                    with torch.no_grad():
                        max_next_q_values = target_model(next_states).max(1)[0]
                        target_values = rewards + GAMMA * max_next_q_values * (1 - dones)

                    loss = loss_fn(q_values, target_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()

            if total_reward > max_reward:
                max_reward = total_reward

            cpu_usage, ram_used, ram_total, ram_usage_percent, gpu_memory_used, gpu_memory_total, gpu_usage_percent = get_system_info()

            progress_bar.set_postfix({
                "Reward": total_reward,
                "Max Reward": max_reward,
                "Memory": memory.__len__(),
                "Loss": loss.item() if 'loss' in locals() else None,
                "Epsilon": epsilon,
                "CPU Usage": f"{cpu_usage}%",
                "RAM Usage": f"{ram_used:.2f}GB/{ram_total:.2f}GB",
                "GPU Usage": f"{gpu_memory_used:.2f}MB/{gpu_memory_total:.2f}MB"
            })
            progress_bar.update(1)

            if episode % UPDATE == 0:
                target_model.load_state_dict(model.state_dict())
                model_scripted = torch.jit.script(model)
                model_scripted.save(f'checkpoints/episode_{episode}.pt')
                torch.save({
                    'epoch': episode,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'model_checkpoint.pth')
                torch.cuda.empty_cache()
            writer.writerow([episode, total_reward, loss.item(), epsilon])
    progress_bar.close()








