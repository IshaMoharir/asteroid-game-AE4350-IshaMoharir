# AE4350 Bio-Inspired Intelligence Assignment – Asteroids RL Agent

**Author**: Isha Moharir  
**Course**: AE4350: Bio-Inspired Intelligence for Aerospace Applications  

## Assignment Overview

This assignment is part of the AE4350 course at TU Delft, which introduces various bio-inspired algorithms. The task is to apply one such algorithm to a self-proposed environment or problem. I chose to implement a reinforcement learning agent using Deep Q-Learning (DQN) in a simplified version of the Asteroids game.

The agent learns to navigate a 2D space environment, avoid collisions, and destroy asteroids using discrete control actions and a reward-based feedback system. The environment and agent were implemented from scratch using Python, Gym-style interfaces, and PyTorch.

## Project Description

- **Learning algorithm**: Deep Q-Network (DQN)
- **Task**: Play and survive in a custom Asteroids game
- **Goal**: Maximise cumulative reward by destroying asteroids and avoiding death
- **State space**: Ship position, velocity, and nearest asteroids
- **Action space**: 6 discrete actions (idle, thrust directions, shoot)
- **Reward structure**: Tuned incentives for aiming, shooting, hitting, and movement, with reduced penalties for exploration

## Repository Structure
## Repository Structure

```
asteroid-game-AE4350/
│
├── game/
│   ├── __init__.py
│   ├── asteroid.py
│   ├── bullet.py
│   ├── config.py
│   └── ship.py
│
├── rl/
│   ├── dqn_agent.py           # DQN agent class
│   ├── env.py                 # Asteroids environment
│   ├── train.py               # Training loop
│   ├── eval/                  # Evaluation plots
│   └── models/                # Saved models and logs
│       ├── best_model.pth
│       ├── training_curve.png
│
├── auto_train_and_eval.py     # Script to automate training + evaluation
├── run_rl.py                  # Script to run trained agent
├── main.py                    # Optional: single-entry runner
├── training_curve.png         # Duplicate of training plot
├── best_model.pth             # Output model (copied to root)
├── dqn_asteroids.pth          # Manually saved model
├── .gitignore
└── README.md                  # This file
```


## Notes

- Report will include sensitivity analysis and discussion of how reward changes impacted learning stability
- All learning parameters (e.g. reward shaping) are clearly documented in `env.py`
- Training results are visualised using moving average plots and episode rewards

## Author

Isha Moharir  
MSc Aerospace Engineering, TU Delft


