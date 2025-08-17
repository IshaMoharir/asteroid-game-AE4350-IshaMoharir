import os
import json
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import pygame  # used by env reward patch
from rl.env import AsteroidsEnv
from rl.train import train_agent
from game.bullet import Bullet
from game.config import SHIP_RADIUS, WIDTH, HEIGHT

# -----------------------------
# Global experiment settings
# -----------------------------
# Reproducibility: per-run seeds will be derived from BASE_SEED
BASE_SEED = 12345

# Training budget
EPISODES_BASELINE = 2000      # do a bit more here to stabilise threshold
EPISODES_MAIN = 500           # used for LR sweep and reward sensitivity
SEEDS_PER_SETTING = 3         # >=5 strongly recommended

# Moving-average window for reporting
ROLLING_WINDOW = 100

# Directory for results (timestamped to avoid clobbering)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = os.path.join("results_sensitivity_v2", STAMP)
os.makedirs(ROOT, exist_ok=True)

# -----------------------------
# Helper: smoothing & metrics
# -----------------------------
def moving_average(x, w):
    if len(x) < w:
        return np.array(x, dtype=float)
    return np.convolve(x, np.ones(w)/w, mode="valid")

def auc_of_curve(y):
    # Trapezoidal AUC; assumes uniform episode spacing
    if len(y) < 2:
        return float(y[-1]) if len(y) else 0.0
    return float(np.trapz(y, dx=1.0))

def final_mean(y, tail=100):
    if len(y) == 0:
        return 0.0
    return float(np.mean(y[-tail:]))

def episodes_to_threshold(y, threshold, window=ROLLING_WINDOW):
    """
    Return the first episode index (1-based) at which the rolling mean >= threshold.
    Returns np.inf if never reached.
    """
    roll = moving_average(y, window)
    for i, v in enumerate(roll):
        if v >= threshold:
            # map rolling index back to episode number: the windowed mean ending at i+window
            return i + window
    return math.inf

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# -----------------------------
# Reward patching
# -----------------------------
def patch_env(cfg):
    original_init = AsteroidsEnv.__init__

    def new_init(self, render_mode=False):
        original_init(self, render_mode=render_mode)
        for k, v in cfg.items():
            setattr(self, k, v)

    AsteroidsEnv.__init__ = new_init

    def safe_normalize(v):
        try:
            if v.length() == 0:
                return v
            return v.normalize()
        except Exception:
            return v

    def custom_reward(self, shoot, _safe_normalize=safe_normalize):
        reward = 0
        alignment_reward = 0
        shooting_reward = 0

        # Shooting action
        if shoot:
            if len(self.bullets) < 5:
                self.bullets.append(Bullet(self.ship.pos, self.ship.direction))
                self.bullets_fired += 1
                shooting_reward = getattr(self, "shooting_reward", 0.2)
                reward += shooting_reward

        # Idle vs move
        if self.ship.vel.length() < 0.05:
            reward += getattr(self, "idle_penalty", -0.03)
            self.idle_steps += 1
        else:
            reward += getattr(self, "movement_reward", 0.08)

        # Bullet-asteroid collisions
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

        # Ship-asteroid collision (death)
        ship_rect = pygame.Rect(
            self.ship.pos.x - SHIP_RADIUS, self.ship.pos.y - SHIP_RADIUS,
            SHIP_RADIUS * 2, SHIP_RADIUS * 2
        )
        for a in self.asteroids:
            if ship_rect.colliderect(a.get_rect()):
                reward = getattr(self, "death_penalty", -15.0)
                self.done = True
                self.ship_deaths += 1
                return reward, alignment_reward, shooting_reward

        # Edge penalty
        norm_x = self.ship.pos.x / WIDTH
        norm_y = self.ship.pos.y / HEIGHT
        edge_margin = 0.1
        near_edge = (
            norm_x < edge_margin or norm_x > 1 - edge_margin or
            norm_y < edge_margin or norm_y > 1 - edge_margin
        )
        if near_edge:
            reward += getattr(self, "edge_penalty", -0.1)
            self.edge_counter += 1
        else:
            self.edge_counter = 0

        if self.edge_counter > 120:
            reward += getattr(self, "edge_kill_penalty", -0.2)
            self.done = True
            return reward, alignment_reward, shooting_reward

        # Gentle center pull
        center_dist = abs(norm_x - 0.5) + abs(norm_y - 0.5)
        reward -= getattr(self, "center_penalty_scale", 0.05) * center_dist

        # Missed-shot penalty: if aligned but not shooting
        if not shoot:
            ship_dir = _safe_normalize(self.ship.direction)
            for asteroid in self.asteroids:
                to_asteroid = _safe_normalize(asteroid.pos - self.ship.pos)
                angle = ship_dir.angle_to(to_asteroid)
                if abs(angle) < 15:
                    reward += getattr(self, "missed_shot_penalty", -0.1)
                    break

        # Alignment reward for closest asteroid
        if self.asteroids:
            closest = min(self.asteroids, key=lambda a: self.ship.pos.distance_to(a.pos))
            to_asteroid = _safe_normalize(closest.pos - self.ship.pos)
            ship_dir = _safe_normalize(self.ship.direction)
            angle = ship_dir.angle_to(to_asteroid)
            if abs(angle) < 10:
                alignment_reward = getattr(self, "alignment_reward_close", 0.5)
            elif abs(angle) < 25:
                alignment_reward = getattr(self, "alignment_reward_mid", 0.35)
            reward += alignment_reward

        # Survival bonus every N steps
        self.step_counter += 1
        if self.step_counter % 20 == 0:
            reward += getattr(self, "survival_bonus", 0.05)

        # Repetition penalty
        if len(self.action_history) == self.history_window:
            most_common = max(set(self.action_history), key=self.action_history.count)
            freq = self.action_history.count(most_common)
            ratio = freq / self.history_window
            if ratio >= getattr(self, "repetition_penalty_threshold", 0.91):
                reward += getattr(self, "repetition_penalty", -0.05)

        return reward, alignment_reward, shooting_reward

    AsteroidsEnv._reward = custom_reward

# -----------------------------
# Experiment definitions
# -----------------------------
# Baseline config (≈ your cfg3 style)
BASELINE_CFG = {
    "reward_per_hit": 50.0,
    "shooting_reward": 0.3,
    "idle_penalty": -0.01,
    "movement_reward": 0.1,
    "death_penalty": -10.0,
    "edge_penalty": -0.05,
    "edge_kill_penalty": -0.2,
    "center_penalty_scale": 0.1,
    "missed_shot_penalty": -0.05,
    "alignment_reward_close": 0.4,
    "alignment_reward_mid": 0.2,
    "survival_bonus": 0.02,
    "repetition_penalty": -0.05,
    "repetition_penalty_threshold": 0.85,
}

# Sweep lists
LR_SWEEP = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

# Reward terms to scale (local sensitivity): each will be tested at ×0.5 and ×1.5
REWARD_TERMS_TO_SCALE = [
    "reward_per_hit",
    "idle_penalty",
    "edge_penalty",
    "center_penalty_scale",
    "missed_shot_penalty",
    "alignment_reward_close",
    "survival_bonus",
]

# -----------------------------
# Core runners
# -----------------------------
def run_setting(label, cfg_patch, lr, episodes, seeds, outdir, per_seed_files=None):
    """
    Run a (cfg_patch, lr) with multiple seeds; return dict with metrics + save raw curves.
    """
    os.makedirs(outdir, exist_ok=True)
    patch_env(cfg_patch)

    # Ensure per_seed_files is a list we can append to
    if per_seed_files is None:
        per_seed_files = []

    all_rewards = []  # keep as plain Python list until we know min length

    for sidx in range(seeds):
        seed = BASE_SEED + sidx
        set_all_seeds(seed)
        run_id = f"{label}_seed{seed}"

        avg, _, rewards = train_agent(run_id=run_id, episodes=episodes, lr=lr)
        # Ensure each run is a numeric NumPy array
        rewards = np.asarray(rewards, dtype=np.float64)
        all_rewards.append(rewards)

        npy_path = os.path.join(outdir, f"rewards_{label}_seed{seed}.npy")
        np.save(npy_path, rewards)
        per_seed_files.append(npy_path)

    # Align by min length, then force a real float array (not object)
    min_len = min(len(r) for r in all_rewards)
    aligned = np.stack([r[:min_len] for r in all_rewards], axis=0).astype(np.float64)  # [seeds, episodes]

    mean_curve = aligned.mean(axis=0)
    std_curve = aligned.std(axis=0)  # now dtype is float64, so this is safe

    # Metrics
    roll = moving_average(mean_curve, ROLLING_WINDOW)
    metrics = {
        "label": label,
        "lr": lr,
        "episodes": min_len,
        "auc_mavg": auc_of_curve(roll),
        "final100_mean": final_mean(mean_curve, tail=min(100, min_len)),
        "final100_std": float(aligned[:, -min(100, min_len):].mean(axis=1).std()),
        "per_seed_files": per_seed_files,
    }
    return metrics, mean_curve, std_curve

def plot_mean_with_shaded_std(ax, mean_curve, std_curve, title):
    x = np.arange(len(mean_curve))
    ax.plot(x, moving_average(mean_curve, ROLLING_WINDOW), label="Mean (rolling)")
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, label="±1 std")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True)

# -----------------------------
# Stage A: Baseline
# -----------------------------
def stage_baseline():
    label = "baseline"
    outdir = os.path.join(ROOT, "baseline")
    metrics, mean_curve, std_curve = run_setting(
        label, BASELINE_CFG, lr=1e-3, episodes=EPISODES_BASELINE, seeds=SEEDS_PER_SETTING, outdir=outdir
    )

    # Threshold = median of baseline final-100 rolling mean (per-seed)
    # Recompute per-seed rolling final to calculate threshold robustly
    per_seed_last_ma = []
    for f in metrics["per_seed_files"]:
        r = np.load(f)
        ma = moving_average(r, ROLLING_WINDOW)
        if len(ma) == 0:
            per_seed_last_ma.append(0.0)
        else:
            per_seed_last_ma.append(np.mean(ma[-min(100, len(ma)):]))
    threshold = float(np.median(per_seed_last_ma))

    # Save baseline summary and threshold
    baseline_summary = {
        "metrics": metrics,
        "threshold": threshold,
        "rolling_window": ROLLING_WINDOW,
    }
    with open(os.path.join(outdir, "baseline_summary.json"), "w") as f:
        json.dump(baseline_summary, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_mean_with_shaded_std(ax, mean_curve, std_curve, "Baseline: mean ± std")
    ax.axhline(threshold, linestyle="--", alpha=0.7, label=f"Threshold={threshold:.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "baseline_curve.png"), dpi=150)
    plt.close(fig)

    return threshold

# -----------------------------
# Stage B: LR sweep
# -----------------------------
def stage_lr_sweep(threshold):
    outdir = os.path.join(ROOT, "lr_sweep")
    os.makedirs(outdir, exist_ok=True)

    rows = []
    all_means = {}
    all_stds = {}

    for lr in LR_SWEEP:
        label = f"lr_{lr}"
        metrics, mean_curve, std_curve = run_setting(
            label, BASELINE_CFG, lr=lr, episodes=EPISODES_MAIN, seeds=SEEDS_PER_SETTING, outdir=outdir
        )

        # Episodes to threshold (from mean curve)
        ett = episodes_to_threshold(mean_curve, threshold, window=ROLLING_WINDOW)
        metrics["episodes_to_threshold"] = float(ett if np.isfinite(ett) else np.inf)

        rows.append(metrics)
        all_means[label] = mean_curve
        all_stds[label] = std_curve

    # Save CSV
    import csv
    csv_path = os.path.join(outdir, "summary_lr.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "lr", "episodes", "auc_mavg",
            "final100_mean", "final100_std", "episodes_to_threshold", "per_seed_files"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plots: mean ± std, final box, AUC bars
    # 1) Overlay curves
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, mean_curve in all_means.items():
        ax.plot(moving_average(mean_curve, ROLLING_WINDOW), label=label)
    ax.axhline(threshold, linestyle="--", alpha=0.7, label="threshold")
    ax.set_title("LR sweep – rolling mean reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lr_curves.png"), dpi=150)
    plt.close(fig)

    # 2) Final 100-episode boxplot (per seed)
    per_setting_final_lists = []
    labels = []
    for r in rows:
        finals = []
        for fpath in r["per_seed_files"]:
            rr = np.load(fpath)
            finals.append(final_mean(rr, tail=min(100, len(rr))))
        per_setting_final_lists.append(finals)
        labels.append(r["label"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(per_setting_final_lists, labels=labels, showmeans=True)
    ax.set_title("LR sweep – final-100 reward (per seed)")
    ax.set_ylabel("Final-100 mean reward")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lr_final_boxplot.png"), dpi=150)
    plt.close(fig)

    # 3) AUC bars
    fig, ax = plt.subplots(figsize=(9, 5))
    aucs = [r["auc_mavg"] for r in rows]
    ax.bar(labels, aucs)
    ax.set_title("LR sweep – AUC of rolling mean")
    ax.set_ylabel("AUC")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lr_auc_bars.png"), dpi=150)
    plt.close(fig)

    # Determine best LR by AUC (tie-breaker: lower final100_std)
    rows_sorted = sorted(rows, key=lambda r: (-r["auc_mavg"], r["final100_std"]))
    best = rows_sorted[0]
    with open(os.path.join(outdir, "best_lr.json"), "w") as f:
        json.dump(best, f, indent=2)

    return best["lr"]

# -----------------------------
# Stage C: Local reward sensitivity (tornado)
# -----------------------------
def stage_reward_sensitivity(best_lr, threshold):
    outdir = os.path.join(ROOT, "reward_sensitivity")
    os.makedirs(outdir, exist_ok=True)

    rows = []

    # Baseline at best LR for comparator
    base_label = "baseline_at_bestlr"
    base_metrics, base_mean, _ = run_setting(
        base_label, BASELINE_CFG, lr=best_lr, episodes=EPISODES_MAIN, seeds=SEEDS_PER_SETTING, outdir=outdir
    )
    base_auc = base_metrics["auc_mavg"]

    # For each reward term: run x0.5 and x1.5
    for term in REWARD_TERMS_TO_SCALE:
        base_val = BASELINE_CFG[term]
        for factor in [0.5, 1.5]:
            cfg = dict(BASELINE_CFG)
            cfg[term] = base_val * factor
            label = f"{term}_x{factor}"
            metrics, mean_curve, std_curve = run_setting(
                label, cfg, lr=best_lr, episodes=EPISODES_MAIN, seeds=SEEDS_PER_SETTING, outdir=outdir
            )

            # Episodes-to-threshold for reporting
            ett = episodes_to_threshold(mean_curve, threshold, window=ROLLING_WINDOW)
            metrics["episodes_to_threshold"] = float(ett if np.isfinite(ett) else np.inf)

            rows.append({
                "term": term,
                "factor": factor,
                "base_value": base_val,
                **metrics
            })

    # Save CSV
    import csv
    csv_path = os.path.join(outdir, "summary_rewards.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "term", "factor", "base_value", "label", "lr", "episodes",
            "auc_mavg", "final100_mean", "final100_std", "episodes_to_threshold", "per_seed_files"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Tornado-style effect (ΔAUC vs baseline_at_bestlr)
    effects = {t: {"low": None, "high": None} for t in REWARD_TERMS_TO_SCALE}

    for r in rows:
        term = r["term"]
        factor = r["factor"]
        delta = r["auc_mavg"] - base_auc
        if factor < 1.0:
            effects[term]["low"] = delta
        else:
            effects[term]["high"] = delta

    # Sort by max absolute effect
    order = sorted(
        REWARD_TERMS_TO_SCALE,
        key=lambda t: -max(abs(effects[t]["low"] or 0), abs(effects[t]["high"] or 0))
    )

    # Plot tornado (horizontal bars: low and high)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(order))
    lows = [effects[t]["low"] if effects[t]["low"] is not None else 0 for t in order]
    highs = [effects[t]["high"] if effects[t]["high"] is not None else 0 for t in order]

    ax.barh(y_pos - 0.2, lows, height=0.4, label="×0.5 vs baseline")
    ax.barh(y_pos + 0.2, highs, height=0.4, label="×1.5 vs baseline")
    ax.axvline(0, linestyle="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order)
    ax.set_xlabel("ΔAUC (rolling) relative to baseline")
    ax.set_title("Local Reward Sensitivity (Tornado)")
    ax.grid(True, axis="x")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "tornado_auc.png"), dpi=150)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    # Stage A: Baseline to set threshold
    threshold = stage_baseline()

    # Stage B: LR sweep using baseline rewards
    best_lr = stage_lr_sweep(threshold)

    # Stage C: Local reward sensitivity at best LR
    stage_reward_sensitivity(best_lr, threshold)

    # Write top-level README-ish metadata
    meta = {
        "root": ROOT,
        "episodes_baseline": EPISODES_BASELINE,
        "episodes_main": EPISODES_MAIN,
        "seeds_per_setting": SEEDS_PER_SETTING,
        "rolling_window": ROLLING_WINDOW,
        "chosen_best_lr_from_sweep": best_lr,
        "note": "All runs used train_agent API unchanged; env reward terms patched at runtime.",
    }
    with open(os.path.join(ROOT, "experiment_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone. Results in:", ROOT)
    print("Files:\n- baseline/baseline_summary.json\n- lr_sweep/summary_lr.csv (plus plots)\n- reward_sensitivity/summary_rewards.csv (plus tornado plot)")

if __name__ == "__main__":
    main()
