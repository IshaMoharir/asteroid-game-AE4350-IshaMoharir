import os
import numpy as np
import matplotlib.pyplot as plt
import random
import pygame
import json

from rl.env import AsteroidsEnv
from rl.train import train_agent
from game.bullet import Bullet
from game.config import SHIP_RADIUS, WIDTH, HEIGHT

# --- Search Space (All rewards/penalties + lr) ---
search_space = {
    "lr": [1e-4, 3e-4, 5e-4, 1e-3],
    "reward_per_hit": [20.0, 30.0, 40.0, 50.0],
    "shooting_reward": [0.1, 0.2, 0.3],
    "idle_penalty": [-0.01, -0.03, -0.05],
    "movement_reward": [0.05, 0.08, 0.1],
    "death_penalty": [-10.0, -15.0, -20.0],
    "edge_penalty": [-0.05, -0.1, -0.15],
    "edge_kill_penalty": [-0.2],
    "center_penalty_scale": [0.02, 0.05, 0.1],
    "missed_shot_penalty": [-0.05, -0.1, -0.2],
    "alignment_reward_close": [0.4, 0.5, 0.6],
    "alignment_reward_mid": [0.2, 0.35, 0.4],
    "survival_bonus": [0.02, 0.05, 0.1],
    "repetition_penalty": [-0.05, -0.1],
    "repetition_penalty_threshold": [0.85, 0.9, 0.95]
}

# --- Config ---
sample_configs = 10
runs_per_config = 3
episodes = 1000

# --- Smoothing helper ---
def smooth(y, box_pts=50):
    box = np.ones(box_pts)/box_pts
    return np.convolve(y, box, mode='same')

# --- Patch environment dynamically ---
def patch_env(cfg):
    original_init = AsteroidsEnv.__init__

    def new_init(self, render_mode=False):
        original_init(self, render_mode=render_mode)
        for k, v in cfg.items():
            setattr(self, k, v)

    AsteroidsEnv.__init__ = new_init

    def custom_reward(self, shoot, safe_normalize):
        reward = 0
        alignment_reward = 0
        shooting_reward = 0

        if shoot:
            if len(self.bullets) < 5:
                self.bullets.append(Bullet(self.ship.pos, self.ship.direction))
                self.bullets_fired += 1
                shooting_reward = getattr(self, "shooting_reward", 0.2)
                reward += shooting_reward

        if self.ship.vel.length() < 0.05:
            reward += getattr(self, "idle_penalty", -0.03)
            self.idle_steps += 1
        else:
            reward += getattr(self, "movement_reward", 0.08)

        for b in self.bullets[:]:
            b.update()
            if b.off_screen():
                self.bullets.remove(b)
                continue
            for a in self.asteroids[:]:
                if a.get_rect().collidepoint(b.pos):
                    self.bullets.remove(b)
                    self.asteroids.remove(a)
                    self.asteroids.extend(a.split())
                    reward += getattr(self, "reward_per_hit", 40.0)
                    self.hits_landed += 1
                    break

        ship_rect = pygame.Rect(self.ship.pos.x - SHIP_RADIUS, self.ship.pos.y - SHIP_RADIUS,
                                SHIP_RADIUS * 2, SHIP_RADIUS * 2)
        for a in self.asteroids:
            if ship_rect.colliderect(a.get_rect()):
                reward = getattr(self, "death_penalty", -15.0)
                self.done = True
                self.ship_deaths += 1
                return reward, alignment_reward, shooting_reward

        norm_x = self.ship.pos.x / WIDTH
        norm_y = self.ship.pos.y / HEIGHT
        edge_margin = 0.1
        near_edge = norm_x < edge_margin or norm_x > 1 - edge_margin or norm_y < edge_margin or norm_y > 1 - edge_margin
        if near_edge:
            reward += getattr(self, "edge_penalty", -0.1)
            self.edge_counter += 1
        else:
            self.edge_counter = 0

        if self.edge_counter > 120:
            reward += getattr(self, "edge_kill_penalty", -0.2)
            self.done = True
            return reward, alignment_reward, shooting_reward

        center_dist = abs(norm_x - 0.5) + abs(norm_y - 0.5)
        reward -= getattr(self, "center_penalty_scale", 0.05) * center_dist

        if not shoot:
            ship_dir = safe_normalize(self.ship.direction)
            for asteroid in self.asteroids:
                to_asteroid = safe_normalize(asteroid.pos - self.ship.pos)
                angle = ship_dir.angle_to(to_asteroid)
                if abs(angle) < 15:
                    reward += getattr(self, "missed_shot_penalty", -0.1)
                    break

        if self.asteroids:
            closest = min(self.asteroids, key=lambda a: self.ship.pos.distance_to(a.pos))
            to_asteroid = safe_normalize(closest.pos - self.ship.pos)
            ship_dir = safe_normalize(self.ship.direction)
            angle = ship_dir.angle_to(to_asteroid)
            if abs(angle) < 10:
                alignment_reward = getattr(self, "alignment_reward_close", 0.5)
            elif abs(angle) < 25:
                alignment_reward = getattr(self, "alignment_reward_mid", 0.35)
            reward += alignment_reward

        self.step_counter += 1
        if self.step_counter % 20 == 0:
            reward += getattr(self, "survival_bonus", 0.05)

        if len(self.action_history) == self.history_window:
            most_common = max(set(self.action_history), key=self.action_history.count)
            freq = self.action_history.count(most_common)
            ratio = freq / self.history_window
            if ratio >= getattr(self, "repetition_penalty_threshold", 0.91):
                reward += getattr(self, "repetition_penalty", -0.05)

        return reward, alignment_reward, shooting_reward

    AsteroidsEnv._reward = custom_reward

# --- Sample configs ---
def sample_hyperparams(space, n):
    keys = list(space.keys())
    return [{k: random.choice(space[k]) for k in keys} for _ in range(n)]

# --- Main execution ---
os.makedirs("results_random", exist_ok=True)
configs = sample_hyperparams(search_space, sample_configs)
config_results = {}
label_map = {}

for idx, cfg in enumerate(configs):
    patch_env(cfg)

    short_cfg = f"cfg{idx}"
    full_label = f"{short_cfg}_" + "_".join(f"{k}={v}" for k, v in cfg.items())
    label_map[short_cfg] = full_label

    print(f"\n📦 Config {idx}")
    for k, v in cfg.items():
        print(f"{k:<25} = {v}")
    print("─" * 55)

    all_rewards = []
    for run in range(runs_per_config):
        run_id = f"{short_cfg}_run{run}"
        avg, _, rewards = train_agent(run_id=run_id, episodes=episodes, lr=cfg["lr"])
        all_rewards.append(rewards)

    all_rewards = np.array(all_rewards)
    config_results[short_cfg] = all_rewards
    np.save(f"results_random/rewards_{short_cfg}.npy", all_rewards)

# Save short ↔ full label mapping
with open("results_random/config_label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for i, (short_cfg, data) in enumerate(config_results.items()):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    mean_smooth = smooth(mean)
    axs[0].plot(mean_smooth, label=short_cfg if i < 10 else None)
    axs[1].bar(short_cfg, np.mean(data[:, -100:]), color='skyblue', edgecolor='black')

axs[0].set_title("Smoothed Reward Curves")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")
axs[0].legend(fontsize=8)
axs[0].grid(True)

axs[1].set_title("Final 100-Episode Average Reward")
axs[1].set_ylabel("Average Reward")
axs[1].tick_params(axis='x', rotation=90)
axs[1].grid(True)

plt.tight_layout()
plt.savefig("results_random/random_search_summary.png")
plt.show()
