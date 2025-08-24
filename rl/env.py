import gymnasium as gym
import numpy as np
import random
import pygame
from game.ship import Ship
from game.asteroid import Asteroid
from game.bullet import Bullet
from game.config import *

# -----------------------------
# Asteroids Gym Environment
# -----------------------------
class AsteroidsEnv(gym.Env):
    """Custom Gym environment for the Asteroids game."""

    # -----------------------------
    # Initialisation
    # -----------------------------
    def __init__(self, render_mode=False, reward_per_hit=50.0):
        """
        Args:
            render_mode (bool): If True, render with pygame.
            reward_per_hit (float): Reward added per asteroid hit.
        """
        self.render_mode = render_mode
        self.action_history = []       # Recent actions for repetition detection
        self.history_window = 30       # Window size for action repetition metric
        self.episode_number = 0        # Track episodes for curriculum
        self.reward_per_hit = reward_per_hit  # Make asteroid reward tunable

        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Asteroids RL Agent")

        self.reset()

    # -----------------------------
    # Reset Episode
    # -----------------------------
    def reset(self):
        """
        Reset the environment to start a new episode.

        Returns:
            np.ndarray: Initial observation/state.
        """
        self.episode_number += 1

        # --- Curriculum: gradually increase number of asteroids ---
        max_episodes = 20000  # Total episodes for curriculum
        min_asteroids = 6     # Minimum number of asteroids
        max_asteroids = 18    # Maximum number of asteroids
        progress = min(1.0, self.episode_number / max_episodes)
        NUM_ASTEROIDS = int(min_asteroids + progress * (max_asteroids - min_asteroids))

        # World entities
        self.ship = Ship(WIDTH // 2, HEIGHT // 2)
        self.asteroids = [Asteroid(size=random.choice(["large", "medium", "small"])) for _ in range(NUM_ASTEROIDS)]
        self.bullets = []

        # Episode stats/flags
        self.score = 0
        self.done = False
        self.frame_count = 0
        self.bullets_fired = 0
        self.hits_landed = 0
        self.idle_steps = 0
        self.ship_deaths = 0
        self.edge_counter = 0
        self.step_counter = 0

        return self._get_state()

    # -----------------------------
    # Step Transition
    # -----------------------------
    def step(self, action):
        """
        Advance the environment by one step.

        Args:
            action (int): Discrete action index.
                0 = no-op, 1 = thrust up, 2 = thrust down,
                3 = thrust left, 4 = thrust right, 5 = shoot

        Returns:
            tuple: (state, reward, done, info)
        """
        if self.done:
            return self._get_state(), 0, True, {}

        def safe_normalize(v):
            """Return v.normalized() if length > 0, else zero vector."""
            return v.normalize() if v.length() > 0 else pygame.math.Vector2(0, 0)

        # --- Action Mapping ---
        thrust = pygame.math.Vector2(0, 0)
        shoot = False
        if action == 1:
            thrust.y -= 1
        elif action == 2:
            thrust.y += 1
        elif action == 3:
            thrust.x -= 1
        elif action == 4:
            thrust.x += 1
        elif action == 5:
            shoot = True

        # Apply thrust to ship
        self.ship.apply_thrust(thrust)

        # Track recent actions for repetition penalty
        self.action_history.append(action)
        if len(self.action_history) > self.history_window:
            self.action_history.pop(0)

        # --- Reward logic (includes shooting, hits, penalties, alignment) ---
        reward, alignment_reward, shooting_reward = self._reward(shoot, safe_normalize)

        # --- Game State Update ---
        self.ship.update()
        for a in self.asteroids:
            a.update()

        # Update score
        self.score += reward

        # Render if requested
        if self.render_mode:
            self._render()

        # Info dict for diagnostics/analytics
        return self._get_state(), reward, self.done, {
            "bullets_fired": self.bullets_fired,
            "hits_landed": self.hits_landed,
            "idle_steps": self.idle_steps,
            "ship_deaths": self.ship_deaths,
            "alignment_reward": alignment_reward,
            "shooting_reward": shooting_reward
        }

    # -----------------------------
    # Reward Function
    # -----------------------------
    def _reward(self, shoot, safe_normalize):
        """
        Compute reward and penalties for the current step.

        Includes:
        - Shooting and hit rewards
        - Idle vs movement incentives
        - Edge penalties and termination on prolonged edge-hovering
        - Center-distance penalty
        - Missed shot penalty if an asteroid is ahead
        - Alignment reward with closest asteroid
        - Small periodic survival bonus
        - Repetition penalty for action loops

        Args:
            shoot (bool): Whether the agent chose to shoot this step.
            safe_normalize (callable): Function to safely normalise a Vector2.

        Returns:
            tuple: (reward, alignment_reward, shooting_reward)
        """
        reward = 0
        alignment_reward = 0
        shooting_reward = 0

        # --- Shooting ---
        if shoot:
            if len(self.bullets) < 5:
                self.bullets.append(Bullet(self.ship.pos, self.ship.direction))
                self.bullets_fired += 1
                shooting_reward = 0.3
                reward += shooting_reward

        # --- Idle or movement ---
        if self.ship.vel.length() < 0.05:
            reward -= 0.01
            self.idle_steps += 1
        else:
            reward += 0.1

        # --- Bullet hits ---
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
                    reward += self.reward_per_hit
                    self.hits_landed += 1
                    break

        # --- Ship-asteroid collision (death) ---
        ship_rect = pygame.Rect(
            self.ship.pos.x - SHIP_RADIUS,
            self.ship.pos.y - SHIP_RADIUS,
            SHIP_RADIUS * 2,
            SHIP_RADIUS * 2
        )
        for a in self.asteroids:
            if ship_rect.colliderect(a.get_rect()):
                reward = -10.0
                self.done = True
                self.ship_deaths += 1
                return reward, alignment_reward, shooting_reward

        # --- Edge penalty ---
        norm_x = self.ship.pos.x / WIDTH
        norm_y = self.ship.pos.y / HEIGHT
        edge_margin = 0.1
        near_edge = (
            norm_x < edge_margin or norm_x > 1 - edge_margin or
            norm_y < edge_margin or norm_y > 1 - edge_margin
        )
        if near_edge:
            reward -= 0.05
            self.edge_counter += 1
        else:
            self.edge_counter = 0

        # Terminate if lingering near edge for too long
        if self.edge_counter > 120:
            reward -= 0.2
            self.done = True
            return reward, alignment_reward, shooting_reward

        # --- Center distance penalty (Manhattan distance from screen center) ---
        center_dist = abs(norm_x - 0.5) + abs(norm_y - 0.5)
        reward -= 0.1 * center_dist

        # --- Missed shot penalty (if asteroid ahead) ---
        if not shoot:
            ship_dir = safe_normalize(self.ship.direction)
            for asteroid in self.asteroids:
                to_asteroid = safe_normalize(asteroid.pos - self.ship.pos)
                angle = ship_dir.angle_to(to_asteroid)
                if abs(angle) < 15:
                    reward -= 0.05
                    break

        # --- Alignment reward with closest asteroid ---
        if self.asteroids:
            closest = min(self.asteroids, key=lambda a: self.ship.pos.distance_to(a.pos))
            to_asteroid = safe_normalize(closest.pos - self.ship.pos)
            ship_dir = safe_normalize(self.ship.direction)
            angle = ship_dir.angle_to(to_asteroid)
            if abs(angle) < 10:
                alignment_reward = 0.4
            elif abs(angle) < 25:
                alignment_reward = 0.2
            reward += alignment_reward

        # --- Small periodic survival bonus ---
        self.step_counter += 1
        if self.step_counter % 20 == 0:
            reward += 0.02

        # --- Detect repetitive action patterns ---
        if len(self.action_history) == self.history_window:
            most_common = max(set(self.action_history), key=self.action_history.count)
            freq = self.action_history.count(most_common)
            repetition_ratio = freq / self.history_window
            if repetition_ratio >= 0.85:
                reward -= 0.05

        return reward, alignment_reward, shooting_reward

    # -----------------------------
    # Observation Builder
    # -----------------------------
    def _get_state(self):
        """
        Construct the observation vector:
        - Ship position (normalised)
        - Ship velocity (scaled)
        - Up to 3 closest asteroids: position (normalised) and size index (scaled radius)
        - Zero-padding to fixed size if fewer than 3 asteroids are present
        """
        state = [
            self.ship.pos.x / WIDTH,
            self.ship.pos.y / HEIGHT,
            self.ship.vel.x / 10,
            self.ship.vel.y / 10
        ]

        # Include the three nearest asteroids
        asteroids = sorted(self.asteroids, key=lambda a: self.ship.pos.distance_to(a.pos))[:3]
        for a in asteroids:
            state += [
                a.pos.x / WIDTH,
                a.pos.y / HEIGHT,
                ASTEROID_SIZES[a.size]["radius"] / 40  # Normalise by max radius in config
            ]

        # Pad to constant length (4 + 3*3)
        while len(state) < 4 + 3 * 3:
            state += [0, 0, 0]

        return np.array(state, dtype=np.float32)

    # -----------------------------
    # Rendering
    # -----------------------------
    def _render(self):
        """
        Render the current game frame using pygame.
        """
        self.clock.tick(FPS)
        self.screen.fill((0, 0, 0))

        # Draw entities
        self.ship.draw(self.screen)
        for a in self.asteroids:
            a.draw(self.screen)
        for b in self.bullets:
            b.draw(self.screen)

        # HUD: score
        font = pygame.font.SysFont(None, 28)
        score_text = font.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
