import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from rl.env import AsteroidsEnv
from rl.train import train_agent

# --- Parameters to sweep ---
reward_per_hit_values = [20.0, 30.0, 40.0]
lr_values = [1e-4, 5e-4, 1e-3]
episodes = 100
runs_per_config = 3

# --- Smoothing helper ---
def smooth(y, box_pts=50):
    box = np.ones(box_pts)/box_pts
    return np.convolve(y, box, mode='same')

# --- Patch environment reward dynamically ---
def patch_env_reward(value):
    original_init = AsteroidsEnv.__init__

    def new_init(self, render_mode=False):
        original_init(self, render_mode=render_mode)
        self.reward_per_hit = value

    AsteroidsEnv.__init__ = new_init

# --- Prepare folders ---
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- Store all results ---
config_results = {}

# --- Run loop ---
for reward_hit in reward_per_hit_values:
    for lr in lr_values:
        label = f"hit{int(reward_hit)}_lr{lr}"
        print(f"\n=== Running config: {label} ===")

        # Patch reward value into the env
        patch_env_reward(reward_hit)

        all_rewards = []

        for run in range(runs_per_config):
            print(f"  → Run {run + 1}/{runs_per_config}")
            avg, _, rewards = train_agent(
                run_id=f"{label}_run{run}",
                episodes=episodes,
                lr=lr
            )
            all_rewards.append(rewards)

        all_rewards = np.array(all_rewards)
        config_results[label] = all_rewards
        np.save(f"results/rewards_{label}.npy", all_rewards)

# --- Smoothed Line Plots (Split by learning rate) ---
fig, axs = plt.subplots(1, len(lr_values), figsize=(6 * len(lr_values), 5), sharey=True)
if len(lr_values) == 1:
    axs = [axs]
colors = ['blue', 'green', 'red']

for i, lr in enumerate(lr_values):
    ax = axs[i]
    ax.set_title(f"Learning Rate: {lr}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True)

    for j, reward_hit in enumerate(reward_per_hit_values):
        label = f"hit{int(reward_hit)}_lr{lr}"
        data = config_results[label]
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        mean_smooth = smooth(mean)

        ax.plot(mean_smooth, label=f"Hit {int(reward_hit)}", color=colors[j])
        ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2, color=colors[j])

    ax.legend()

plt.suptitle("Sensitivity Analysis: Reward per Hit vs Learning Rate")
plt.tight_layout()
plt.savefig("results/sensitivity_split_plot.png")
plt.show()

# --- Final Performance Bar Chart ---
final_scores = []
final_labels = []

for reward_hit in reward_per_hit_values:
    for lr in lr_values:
        label = f"hit{int(reward_hit)}_lr{lr}"
        data = config_results[label]
        avg_final_reward = np.mean(data[:, -200:])  # average over last 200 episodes
        final_scores.append(avg_final_reward)
        final_labels.append(label)

plt.figure(figsize=(10, 5))
bars = plt.bar(final_labels, final_scores, color='skyblue', edgecolor='black')
plt.xticks(rotation=45)
plt.ylabel("Average Reward (Last 200 Episodes)")
plt.title("Final Performance Comparison")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("results/final_comparison.png")
plt.show()
