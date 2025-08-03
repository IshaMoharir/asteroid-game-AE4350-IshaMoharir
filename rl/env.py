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
        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Asteroids RL Agent")
        self.reset()

    def reset(self):
        self.ship = Ship(WIDTH // 2, HEIGHT // 2)
        NUM_ASTEROIDS = 8
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
        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        # Helper to safely normalize vectors
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

        # --- Rewards and Metrics ---
        reward = 0
        alignment_reward = 0
        shooting_reward = 0

        # --- Shooting logic ---
        if shoot:
            reward -= 0.01
            if len(self.bullets) < 5:
                self.bullets.append(Bullet(self.ship.pos, self.ship.direction))
                self.bullets_fired += 1
                shooting_reward = 0.2
                reward += shooting_reward

        # --- Idling penalty / movement reward ---
        if self.ship.vel.length() < 0.05:
            reward -= 0.1
            self.idle_steps += 1
        else:
            reward += 0.03

        # --- Bullet-asteroid collisions ---
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
                    reward += 10.0
                    self.hits_landed += 1
                    break

        # --- Ship-asteroid collisions ---
        ship_rect = pygame.Rect(self.ship.pos.x - SHIP_RADIUS, self.ship.pos.y - SHIP_RADIUS,
                                SHIP_RADIUS * 2, SHIP_RADIUS * 2)
        for a in self.asteroids:
            if ship_rect.colliderect(a.get_rect()):
                reward = -1.0
                self.done = True
                self.ship_deaths += 1
                break

        # --- Penalise being near edge ---
        edge_margin = 0.1
        norm_x = self.ship.pos.x / WIDTH
        norm_y = self.ship.pos.y / HEIGHT
        near_edge = norm_x < edge_margin or norm_x > 1 - edge_margin or \
                    norm_y < edge_margin or norm_y > 1 - edge_margin
        if near_edge:
            reward -= 0.15
            self.edge_counter += 1
        else:
            self.edge_counter = 0

        # --- Die if hugging edge too long ---
        if self.edge_counter > 150:
            reward -= 0.5
            self.done = True

        # --- Penalise distance from center ---
        center_dist = abs(norm_x - 0.5) + abs(norm_y - 0.5)
        reward -= 0.6 * center_dist

        # --- Penalise not shooting when asteroid in front ---
        if not shoot:
            ship_dir = safe_normalize(self.ship.direction)
            for asteroid in self.asteroids:
                to_asteroid = safe_normalize(asteroid.pos - self.ship.pos)
                angle = ship_dir.angle_to(to_asteroid)
                if abs(angle) < 15:
                    reward -= 0.4
                    break

        # --- Reward pointing at closest asteroid ---
        if self.asteroids:
            closest = min(self.asteroids, key=lambda a: self.ship.pos.distance_to(a.pos))
            to_asteroid = safe_normalize(closest.pos - self.ship.pos)
            ship_dir = safe_normalize(self.ship.direction)
            angle = ship_dir.angle_to(to_asteroid)
            if abs(angle) < 10:
                alignment_reward = 0.3
            elif abs(angle) < 25:
                alignment_reward = 0.2
            reward += alignment_reward

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
                ASTEROID_SIZES[a.size]["radius"] / 40
            ]
        while len(state) < 4 + 3 * 3:
            state += [0, 0, 0]
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
