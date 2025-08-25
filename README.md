# AE4350 Bio-Inspired Intelligence Assignment - Asteroids RL Agent

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

```text
asteroid-game-AE4350/
│
├── game/                      # Game logic and entities
│   ├── asteroid.py
│   ├── bullet.py
│   ├── config.py
│   └── ship.py
│
├── rl/                        # RL logic
│   ├── dqn_agent.py           # DQN agent class and network
│   ├── env.py                 # Gym-style Asteroids environment
│   └── train.py               # Training loop
│
├── models/                    # Saved models and logs
│
├── results/                   # General results/plots
├── results_random/            # Random hyperparameter search results
├── results_sensitivity_v2/    # LR sweep + reward sensitivity results
│
├── best_model.pth             # Example saved model (root copy)
├── dqn_asteroids.pth          # Another saved checkpoint
├── training_curve.png         # Training curve example (root copy)
│
├── main.py                    # Playing the game yourself
├── run_rl.py                  # Run trained agent visually in pygame
├── sensitivity.py             # Random search for reward/lr configs
├── sensitivity_v2.py          # LR sweep + reward sensitivity (tornado plots)
│
├── .gitignore
└── README.md
```

## Software Structure and Modularity

The project is organised in a modular fashion to ensure reproducibility and extensibility:

- **Game logic**
  - `ship.py`, `asteroid.py`, `bullet.py`: define object behaviours and game physics  
  - `config.py`: centralises constants for physics, rendering, and reward shaping  

- **Reinforcement Learning**
  - `env.py`: implements the Gym-compatible Asteroids environment  
  - `dqn_agent.py`: defines the neural network, replay buffer, and training logic  
  - `train.py`: runs the main training loop, handles target updates and checkpointing  

- **Evaluation**
  - `run_rl.py`: evaluates trained agents interactively in the pygame environment  
  - Results (plots, statistics) are saved in `results/` and `rl/eval/`  

- **Hyperparameter and Sensitivity Analysis**
  - `sensitivity.py`: random search across reward parameters and learning rates, results saved in `results_random/`  
  - `sensitivity_v2.py`: structured LR sweep and reward sensitivity (tornado plots), results saved in `results_sensitivity_v2/`  

- **Models and Outputs**
  - `models/`: stores trained models (per-run checkpoints and best models)  
  - Reference artefacts (`best_model.pth`, `training_curve.png`) are included at the repo root for quick reference  

This modular design makes it easy to modify components independently, experiment with different learning algorithms, or adjust training configurations without breaking the overall pipeline.

## Notes

- Training results are logged and visualised with moving average reward curves (`train.py`).
- Evaluation of trained models is handled with `run_rl.py` (pygame visualisation) and additional plots/statistics saved in `rl/eval/`.
- Hyperparameter and reward sensitivity analyses are included:
  - `sensitivity.py`: runs a random search over learning rates and reward parameters, saving results in `results_random/`.
  - `sensitivity_v2.py`: performs a structured learning rate sweep and local reward sensitivity (tornado plots), saving results in `results_sensitivity_v2/`.
- All reward-shaping terms are clearly documented in `env.py` and patched dynamically during sensitivity experiments.
- Example outputs (`best_model.pth`, `training_curve.png`) are included at the root for quick reference; full logs and models are stored under `models/`.

## Author

Isha Moharir  
MSc Aerospace Engineering, TU Delft


