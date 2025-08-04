import gymnasium as gym
import numpy as np
import random
import pygame
from game.ship import Ship
from game.asteroid import Asteroid
from game.bullet import Bullet
from game.config import *

class AsteroidsEnv(gym.Env):
    """Custom Gym environment for the Asteroids game."""
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.action_history = []
        self.history_window = 30
        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Asteroids RL Agent")

        self.reset()

    def reset(self):
        self.ship = Ship(WIDTH // 2, HEIGHT // 2)
        NUM_ASTEROIDS = 8  # 🔧 Tweak: number of starting asteroids
        self.asteroids = [Asteroid(size=random.choice(["large", "medium", "small"])) for _ in range(NUM_ASTEROIDS)]
        self.bullets = []
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

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        def safe_normalize(v):
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

        self.ship.apply_thrust(thrust)

        self.action_history.append(action)
        if len(self.action_history) > self.history_window:
            self.action_history.pop(0)

        # --- Call reward logic ---
        reward, alignment_reward, shooting_reward = self._reward(shoot, safe_normalize)

        # --- Game State Update ---
        self.ship.update()
        for a in self.asteroids:
            a.update()

        self.score += reward

        if self.render_mode:
            self._render()

        return self._get_state(), reward, self.done, {
            "bullets_fired": self.bullets_fired,
            "hits_landed": self.hits_landed,
            "idle_steps": self.idle_steps,
            "ship_deaths": self.ship_deaths,
            "alignment_reward": alignment_reward,
            "shooting_reward": shooting_reward
        }

    def _reward(self, shoot, safe_normalize):
        """Handles all reward and penalty logic."""
        reward = 0
        alignment_reward = 0
        shooting_reward = 0

        # --- Shooting ---
        if shoot:
            reward -= 0.005  # 🔧 Small penalty for firing
            if len(self.bullets) < 5:
                self.bullets.append(Bullet(self.ship.pos, self.ship.direction))
                self.bullets_fired += 1
                shooting_reward = 0.4  # 🔧 Reward for shooting
                reward += shooting_reward

        # --- Idle or movement ---
        if self.ship.vel.length() < 0.05:
            reward -= 0.03  # 🔧 Penalty for being idle
            self.idle_steps += 1
        else:
            reward += 0.06  # 🔧 Reward for movement

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
                    reward += 20.0  # 🔧 Big reward for destroying asteroid
                    self.hits_landed += 1
                    break

        # --- Ship-asteroid collision (death) ---
        ship_rect = pygame.Rect(self.ship.pos.x - SHIP_RADIUS, self.ship.pos.y - SHIP_RADIUS,
                                SHIP_RADIUS * 2, SHIP_RADIUS * 2)
        for a in self.asteroids:
            if ship_rect.colliderect(a.get_rect()):
                reward = -5.0  # 🔧 Death penalty
                self.done = True
                self.ship_deaths += 1
                return reward, alignment_reward, shooting_reward

        # --- Edge penalty ---
        norm_x = self.ship.pos.x / WIDTH
        norm_y = self.ship.pos.y / HEIGHT
        edge_margin = 0.1
        near_edge = norm_x < edge_margin or norm_x > 1 - edge_margin or \
                    norm_y < edge_margin or norm_y > 1 - edge_margin
        if near_edge:
            reward -= 0.05  # 🔧 Penalty for being near edge
            self.edge_counter += 1
        else:
            self.edge_counter = 0

        if self.edge_counter > 150:
            reward -= 0.2  # 🔧 Extra penalty if hugging edge too long
            self.done = True
            return reward, alignment_reward, shooting_reward

        # --- Center distance penalty ---
        center_dist = abs(norm_x - 0.5) + abs(norm_y - 0.5)
        reward -= 0.1 * center_dist  # 🔧 Discourage staying away from center

        # --- Missed shot penalty (if asteroid ahead) ---
        if not shoot:
            ship_dir = safe_normalize(self.ship.direction)
            for asteroid in self.asteroids:
                to_asteroid = safe_normalize(asteroid.pos - self.ship.pos)
                angle = ship_dir.angle_to(to_asteroid)
                if abs(angle) < 15:
                    reward -= 0.1  # 🔧 Penalty for not shooting when asteroid in front
                    break

        # --- Alignment reward with closest asteroid ---
        if self.asteroids:
            closest = min(self.asteroids, key=lambda a: self.ship.pos.distance_to(a.pos))
            to_asteroid = safe_normalize(closest.pos - self.ship.pos)
            ship_dir = safe_normalize(self.ship.direction)
            angle = ship_dir.angle_to(to_asteroid)
            if abs(angle) < 10:
                alignment_reward = 0.5  # 🔧 High reward for aiming precisely
            elif abs(angle) < 25:
                alignment_reward = 0.35  # 🔧 Smaller reward
            reward += alignment_reward

        self.step_counter += 1

        if self.step_counter % 20 == 0:  # Every 60 steps
            reward += 0.05  # 🔧 Small reward per timestep survived

        # --- Detect repetitive action patterns ---
        if len(self.action_history) == self.history_window:
            most_common = max(set(self.action_history), key=self.action_history.count)
            freq = self.action_history.count(most_common)
            repetition_ratio = freq / self.history_window
            if repetition_ratio >= 0.9:
                reward -= 1.0  # 🔧 Penalty for repetitive behaviour
                print(f"[INFO] Repetitive behaviour detected: {repetition_ratio*100:.1f}% → penalty applied")

        return reward, alignment_reward, shooting_reward

    def _get_state(self):
        state = [
            self.ship.pos.x / WIDTH,
            self.ship.pos.y / HEIGHT,
            self.ship.vel.x / 10,
            self.ship.vel.y / 10
        ]
        asteroids = sorted(self.asteroids, key=lambda a: self.ship.pos.distance_to(a.pos))[:3]
        for a in asteroids:
            state += [
                a.pos.x / WIDTH,
                a.pos.y / HEIGHT,
                ASTEROID_SIZES[a.size]["radius"] / 40  # 🔧 Normalize asteroid size
            ]
        while len(state) < 4 + 3 * 3:
            state += [0, 0, 0]  # Padding for missing asteroids
        return np.array(state, dtype=np.float32)

    def _render(self):
        self.clock.tick(FPS)
        self.screen.fill((0, 0, 0))
        self.ship.draw(self.screen)
        for a in self.asteroids:
            a.draw(self.screen)
        for b in self.bullets:
            b.draw(self.screen)
        font = pygame.font.SysFont(None, 28)
        score_text = font.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
