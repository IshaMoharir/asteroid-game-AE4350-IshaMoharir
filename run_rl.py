import torch
import time
import pygame
from rl.dqn_agent import DQN
from rl.env import AsteroidsEnv

# Create the environment in render mode
env = AsteroidsEnv(render_mode=True)
state_dim = len(env.reset())
action_dim = 6

# Load the trained model
model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load("dqn_asteroids.pth", map_location=torch.device("cpu")))
model.eval()

episodes = 5  # Number of episodes to watch

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Handle quit events manually
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Convert state to tensor, get predicted action
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = model(state_tensor).argmax().item()

        # Step through environment
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

        # Optional: slow down for visibility
        time.sleep(0.01)

    print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}")
    time.sleep(1)

pygame.quit()
