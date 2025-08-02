import torch
import time
import pygame
from rl.dqn_agent import DQN
from rl.env import AsteroidsEnv

# --- Load best model path from file ---
with open("models/best_model_path.txt", "r") as f:
    MODEL_PATH = f.read().strip()

episodes = 5  # Number of episodes to watch

# --- Load environment ---
env = AsteroidsEnv(render_mode=True)
state_dim = len(env.reset())
action_dim = 6

# --- Load model ---
model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# --- Run episodes ---
start_time = time.time()
for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    print(f"️ Episode {ep + 1}/{episodes} running...")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

        time.sleep(0.01)

    print(f" Episode {ep + 1} complete | Reward = {total_reward:.2f}")
    time.sleep(0.5)

# --- Done ---
total_time = time.time() - start_time
print(f"\n All {episodes} episodes completed in {total_time:.1f} seconds (~{total_time/60:.1f} min)")
pygame.quit()
