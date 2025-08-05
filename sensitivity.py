import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from rl.env import AsteroidsEnv
from rl.train import train_agent

# --- Parameters to sweep ---
reward_per_hit_values = [20.0, 30.0, 40.0]
lr_values = [1e-4, 5e-4]
episodes = 3000
runs_per_config = 3

# --- Patch environment reward dynamically ---
def patch_env_reward(value):
    original_init = AsteroidsEnv.__init__

    def new_init(self, render_mode=False):
        original_init(self, render_mode=render_mode)
        self.reward_per_hit = value

    AsteroidsEnv.__init__ = new_init

# --- Run loop ---
os.makedirs("results", exist_ok=True)

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
        np.save(f"results/rewards_{label}.npy", all_rewards)

        # Plot mean reward curve
        mean = np.mean(all_rewards, axis=0)
        std = np.std(all_rewards, axis=0)

        plt.plot(mean, label=label)
        plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)

# --- Final plot ---
plt.title("Sensitivity Analysis: Reward per Hit vs Learning Rate")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/sensitivity_plot.png")
plt.show()
