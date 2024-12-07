# 🎮 Play Atari with Deep Q-Learning

This project aims to replicate the results of the research paper: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602).

## 🔍 Overview

Deep Q-Learning (DQL) is an advanced form of Q-Learning designed to handle the complexities of continuous and high-dimensional environments, such as those found in Atari games. Traditional Q-Learning struggles in these scenarios because it requires discrete states and environments, which are often not feasible in real-world problems.

In this project:
- We explore the differences between **Q-Learning** and **Deep Q-Learning**.
- We implement a DQL-based agent to play Atari games, leveraging a neural network for decision-making.
- We simulate and visualize the learning process and outcomes.

---

## 🤔 What's the Difference Between Q-Learning and Deep Q-Learning?

| **Q-Learning**                                   | **Deep Q-Learning**                                |
|--------------------------------------------------|---------------------------------------------------|
| Requires a **discrete environment**.            | Handles **continuous environments** effectively.  |
| Relies on manually defined states.              | Extracts features automatically using neural networks. |
| Struggles with high-dimensional input (e.g., images). | Uses **image frames** as input by stacking and processing them. |

### More About Q-Learning
For an introduction to Q-Learning, check out my repository: [Play-All-ToyText-with-Q-Learning](https://github.com/phamduyaaaa/Play-All-ToyText-with-Q-Learning).

---

## 🧠 Neural Network Architecture

In Deep Q-Learning, a **convolutional neural network (CNN)** is employed to approximate the Q-function:
- The network takes a stack of **4 consecutive frames** as input, allowing the agent to understand the temporal dynamics of the environment.
- It outputs Q-values for all possible actions, guiding the agent's decision-making process.

---

## 💾 Memory Buffer

To stabilize training, a **replay memory buffer** is used:
- Stores experiences as tuples: **(state, action, reward, next_state, done)**.
- Samples mini-batches of experiences randomly during training, breaking temporal correlations in data.

---

## ⚙️ Hyperparameters

| **Parameter**          | **Value**       | **Description**                                   |
|-------------------------|-----------------|---------------------------------------------------|
| **Batch size**          | `32`           | Number of samples used for each training update. |
| **Learning rate**       | `0.00025`      | Step size for updating network weights.          |
| **Replay buffer size**  | `100,000`      | Maximum number of experiences stored.            |
| **Discount factor (γ)** | `0.999`         | Future reward discounting.                       |
| **Exploration ε**       | `1.0 → 0.1`    | Probability of taking random actions (annealed). |

---

## 🛠️ Implementation Details

The following techniques are used in this project:
1. **Frame Preprocessing**:
   - Convert frames to grayscale to reduce input size.
   - Resize frames to a fixed dimension (e.g., 84x84).
2. **Frame Stacking**:
   - Stack the last 4 frames to capture motion information.
3. **Experience Replay**:
   - Train the neural network using randomly sampled experiences.
4. **Target Network**:
   - Use a separate network to provide stable Q-value targets during training.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/phamduyaaaa/Play-Atari-with-Deep-Q-Learning.git
   cd Play-Atari-with-Deep-Q-Learning
