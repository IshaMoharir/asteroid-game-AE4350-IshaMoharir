import torch
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
from rl.dqn_agent import DQN
from rl.env import AsteroidsEnv
import os

# -----------------------------
# Load Best Model Path
# -----------------------------
# Reads the path to the best model saved during training.
with open("rl/models/best_model_path.txt", "r") as f:
    MODEL_PATH = f.read().strip()

# -----------------------------
# Evaluation Configuration
# -----------------------------
episodes = 100  # Number of episodes to watch

# -----------------------------
# Environment Setup
# -----------------------------
env = AsteroidsEnv(render_mode=True)
state_dim = len(env.reset())
action_dim = 6

# -----------------------------
# Model Setup & Load Weights
# -----------------------------
model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# -----------------------------
# Evaluation Loop
# -----------------------------
episode_rewards = []
start_time = time.time()

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    print(f" Episode {ep + 1}/{episodes} running...")

    while not done:
        # Handle window events to keep pygame responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Choose an action greedily from the Q-network
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

        # Slow down rendering slightly for visibility
        time.sleep(0.01)

    episode_rewards.append(total_reward)
    print(f" Episode {ep + 1} complete | Reward = {total_reward:.2f}")
    time.sleep(0.5)

# -----------------------------
# Summary Statistics
# -----------------------------
total_time = time.time() - start_time
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)

print("\nEvaluation Summary")
print(f" Episodes: {episodes}")
print(f" Mean Reward: {mean_reward:.2f}")
print(f" Std Dev: {std_reward:.2f}")
print(f" Min Reward: {np.min(episode_rewards):.2f}")
print(f" Max Reward: {np.max(episode_rewards):.2f}")
print(f" Completed in {total_time:.1f} seconds (~{total_time/60:.1f} min)")

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(range(1, episodes + 1), episode_rewards, color='skyblue', edgecolor='black')
plt.axhline(mean_reward, color='red', linestyle='--', label=f"Mean = {mean_reward:.2f}")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Evaluation Rewards per Episode")
plt.legend()
plt.tight_layout()
# plt.show()  # Keep disabled to avoid blocking when running non-interactively

os.makedirs("rl/eval", exist_ok=True)
plt.savefig("rl/eval/eval_rewards_per_episode.png")

# -----------------------------
# Cleanup
# -----------------------------
pygame.quit()
