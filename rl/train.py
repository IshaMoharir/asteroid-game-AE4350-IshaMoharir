import os
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from rl.env import AsteroidsEnv
from rl.dqn_agent import DQNAgent

def train_agent(run_id, episodes=500, max_steps=500):
    # Setup
    env = AsteroidsEnv(render_mode=False)
    state_dim = len(env.reset())
    action_dim = 6
    agent = DQNAgent(state_dim, action_dim)

    all_rewards = []
    alignment_rewards = []
    shooting_rewards = []
    best_avg_reward = float('-inf')
    moving_avg_window = 10
    episode_durations = []

    metric_sums = {
        "bullets_fired": 0,
        "hits_landed": 0,
        "idle_steps": 0,
        "ship_deaths": 0,
        "alignment_reward": 0,
        "shooting_reward": 0
    }

    global_start = time.time()

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        ep_start = time.time()
        action_counts = [0] * action_dim

        while not done and steps < max_steps:
            action = agent.act(state)
            action_counts[action] += 1
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
            steps += 1

        if ep % 10 == 0:
            agent.update_target()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        all_rewards.append(total_reward)
        alignment_rewards.append(info.get("alignment_reward", 0))
        shooting_rewards.append(info.get("shooting_reward", 0))

        for key in metric_sums:
            if key in info:
                metric_sums[key] += info[key]

        if len(all_rewards) >= moving_avg_window:
            moving_avg = sum(all_rewards[-moving_avg_window:]) / moving_avg_window
            if moving_avg > best_avg_reward:
                best_avg_reward = moving_avg
                torch.save(agent.model.state_dict(), f"models/best_model_run{run_id}.pth")

        # Time tracking
        ep_time = time.time() - ep_start
        episode_durations.append(ep_time)
        if len(episode_durations) > 10:
            episode_durations = episode_durations[-10:]
        avg_ep_time = sum(episode_durations) / len(episode_durations)
        remaining = (episodes - (ep + 1)) * avg_ep_time

        # Periodic print
        if (ep + 1) % 10 == 0 or ep == 0:
            avg_metrics = {k: v / 10 for k, v in metric_sums.items()}
            print(f"\n[Run {run_id}] Episode {ep + 1}/{episodes} | Reward = {total_reward:.2f} | Epsilon = {agent.epsilon:.3f}")
            print(f"  🔫 Bullets: {avg_metrics['bullets_fired']:.1f} | 🎯 Hits: {avg_metrics['hits_landed']:.1f} | "
                  f"🛑 Idle: {avg_metrics['idle_steps']:.1f} | 💥 Deaths: {avg_metrics['ship_deaths']:.1f}")
            print(f"  🎯 Alignment reward: {avg_metrics['alignment_reward']:.2f} | 🔫 Shooting reward: {avg_metrics['shooting_reward']:.2f}")
            print(f"  🎮 Action counts: {action_counts}")
            print(f"  ⏱️ ETA: {remaining:.1f}s (~{remaining / 60:.1f} min)")
            print("---------------------------------------------------------")
            for key in metric_sums:
                metric_sums[key] = 0

    # Save final model
    torch.save(agent.model.state_dict(), f"models/final_model_run{run_id}.pth")
    np.save(f"models/alignment_rewards_run{run_id}.npy", alignment_rewards)
    np.save(f"models/shooting_rewards_run{run_id}.npy", shooting_rewards)

    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards, label="Total reward per episode")
    if len(all_rewards) >= moving_avg_window:
        avg = np.convolve(all_rewards, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        plt.plot(avg, label=f"{moving_avg_window}-ep moving avg", color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Training Run {run_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"models/training_curve_run{run_id}.png")
    plt.close()

    total_time = time.time() - global_start
    print(f"\n✅ Run {run_id} complete | Best avg reward = {best_avg_reward:.2f} | Time: {total_time/60:.1f} min")

    return best_avg_reward, f"models/best_model_run{run_id}.pth", all_rewards

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    results = []
    all_rewards_all_runs = []

    for i in range(5):
        avg_reward, model_path, rewards = train_agent(run_id=i)
        results.append((avg_reward, model_path))
        all_rewards_all_runs.append(rewards)

    # Compute mean and std per episode index
    rewards_array = np.array(all_rewards_all_runs)
    mean_per_ep = np.mean(rewards_array, axis=0)
    std_per_ep = np.std(rewards_array, axis=0)

    np.save("models/mean_rewards.npy", mean_per_ep)
    np.save("models/std_rewards.npy", std_per_ep)

    best = max(results, key=lambda x: x[0])
    with open("models/best_model_path.txt", "w") as f:
        f.write(f"rl/{best[1]}")

    # Plot average curve
    plt.figure(figsize=(10, 5))
    plt.plot(mean_per_ep, label="Mean reward per episode")
    plt.fill_between(range(len(mean_per_ep)),
                     mean_per_ep - std_per_ep,
                     mean_per_ep + std_per_ep,
                     color="orange", alpha=0.3, label="±1 std dev")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Average Reward Across 10 Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/avg_std_across_runs.png")
    plt.show()
