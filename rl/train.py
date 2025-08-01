from rl.env import AsteroidsEnv
from rl.dqn_agent import DQNAgent
import torch
import matplotlib.pyplot as plt
import time
import numpy as np

# --- Settings ---
episodes = 300
MAX_STEPS = 300
print_interval = 10
moving_avg_window = 10
best_avg_reward = float('-inf')
all_rewards = []

# --- Setup ---
env = AsteroidsEnv(render_mode=False)
state_dim = len(env.reset())
action_dim = 6
agent = DQNAgent(state_dim, action_dim)

# --- Resume training if desired ---
try:
    agent.model.load_state_dict(torch.load("dqn_asteroids.pth"))
    agent.update_target()
    print(" Loaded previous model.")
except FileNotFoundError:
    print(" No previous model found. Starting fresh.")

# --- Timer ---
episode_durations = []
global_start = time.time()

# --- Training loop ---
for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    ep_start = time.time()
    action_counts = [0] * action_dim

    while not done and steps < MAX_STEPS:
        action = agent.act(state)
        action_counts[action] += 1
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.train_step()
        state = next_state
        total_reward += reward
        steps += 1

    agent.update_target()
    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
    all_rewards.append(total_reward)

    # Save best model by moving average
    if len(all_rewards) >= moving_avg_window:
        moving_avg = sum(all_rewards[-moving_avg_window:]) / moving_avg_window
        if moving_avg > best_avg_reward:
            best_avg_reward = moving_avg
            torch.save(agent.model.state_dict(), "best_model.pth")
            print(f"💾 Saved new best model (avg reward = {moving_avg:.2f})")

    # Episode timing
    ep_time = time.time() - ep_start
    episode_durations.append(ep_time)
    if len(episode_durations) > 10:
        episode_durations = episode_durations[-10:]
    avg_ep_time = sum(episode_durations) / len(episode_durations)
    remaining = (episodes - (ep + 1)) * avg_ep_time

    # Periodic print
    if (ep + 1) % print_interval == 0 or ep == 0:
        print(f"Episode {ep + 1}/{episodes} | Reward = {total_reward:.2f} | Epsilon = {agent.epsilon:.3f}")
        print(f"Action counts: {action_counts}")
        print(f"⏱️ ETA: {remaining:.1f}s (~{remaining/60:.1f} min)")
        print(f"---------------------------------------------------------")

# --- Save final model ---
torch.save(agent.model.state_dict(), "dqn_asteroids.pth")

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(all_rewards, label="Total reward per episode")

if len(all_rewards) >= moving_avg_window:
    avg = np.convolve(all_rewards, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
    plt.plot(avg, label=f"{moving_avg_window}-episode moving avg", color='orange')

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Reward Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_curve.png")
# plt.show()

# --- Done ---
total_time = time.time() - global_start
print(f"\n Training complete in {total_time:.1f} seconds (~{total_time/60:.1f} min)")
