import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, Counter
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

num_episodes = 100
max_steps_per_episode = 200
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.001
batch_size = 64
memory_size = 10000

replay_memory = deque(maxlen=memory_size)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    with torch.no_grad():
        q_values = policy_net(torch.FloatTensor(state))
        return torch.argmax(q_values).item()

def train_dqn():
    if len(replay_memory) < batch_size:
        return
    minibatch = random.sample(replay_memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0]
        targets = rewards + gamma * max_next_q_values * (1 - dones)

    targets = targets.unsqueeze(1)
    loss = loss_fn(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

action_counts = {0: 0, 1: 0}
action_distribution = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = choose_action(state, epsilon)
        action_counts[action] += 1
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train_dqn()

        if done:
            break

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    action_distribution.append(action_counts.copy())

def plot_action_distribution(action_distribution, num_episodes):
    action_0 = [dist[0] for dist in action_distribution]
    action_1 = [dist[1] for dist in action_distribution]

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_episodes + 1), action_0, label="Action 0", color="blue")
    plt.plot(range(1, num_episodes + 1), action_1, label="Action 1", color="orange")
    plt.title("Action Distribution Over Episodes", fontsize=16)
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Action Count", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

plot_action_distribution(action_distribution, num_episodes)
