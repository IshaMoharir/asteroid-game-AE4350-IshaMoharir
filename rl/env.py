import gymnasium as gym
import numpy as np
from game.ship import Ship
from game.asteroid import Asteroid
from game.bullet import Bullet
from game.config import *
import random
import pygame
import time

class AsteroidsEnv(gym.Env):
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
        self.asteroids = [Asteroid(size=random.choice(["large", "medium", "small"])) for _ in range(5)]
        self.bullets = []
        self.score = 0
        self.done = False
        self.frame_count = 0
        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        # Map action to control
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

        # Shooting (no cooldown here for simplicity)
        if shoot and len(self.bullets) < 5:
            self.bullets.append(Bullet(self.ship.pos, self.ship.direction))

        self.ship.update()
        for a in self.asteroids:
            a.update()

        reward = 0

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
                    reward += ASTEROID_POINTS[a.size] / 100  # small positive reward
                    break

        # Ship collision
        ship_rect = pygame.Rect(self.ship.pos.x - SHIP_RADIUS, self.ship.pos.y - SHIP_RADIUS, SHIP_RADIUS*2, SHIP_RADIUS*2)
        for a in self.asteroids:
            if ship_rect.colliderect(a.get_rect()):
                reward = -1
                self.done = True
                break

        # Small survival reward
        reward += 0.01
        self.frame_count += 1

        if self.render_mode:
            self._render()

        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        state = [
            self.ship.pos.x / WIDTH,
            self.ship.pos.y / HEIGHT,
            self.ship.vel.x / 10,
            self.ship.vel.y / 10
        ]

        # Add up to 3 nearest asteroid positions (normalised)
        asteroids = sorted(self.asteroids, key=lambda a: self.ship.pos.distance_to(a.pos))[:3]
        for a in asteroids:
            state += [
                a.pos.x / WIDTH,
                a.pos.y / HEIGHT,
                ASTEROID_SIZES[a.size]["radius"] / 40
            ]
        # Pad if fewer than 3
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
        # Draw score on screen
        font = pygame.font.SysFont(None, 28)
        score_text = font.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
