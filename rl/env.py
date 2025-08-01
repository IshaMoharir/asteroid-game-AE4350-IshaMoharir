import gymnasium as gym
import numpy as np
import random
import pygame
from game.ship import Ship
from game.asteroid import Asteroid
from game.bullet import Bullet
from game.config import *

class AsteroidsEnv(gym.Env):
    """Custom Gym environment for the Asteroids game.
    This environment simulates the Asteroids game where an agent controls a ship
    to avoid asteroids, shoot them, and maximize its score.
    The state space includes the ship's position, velocity, and nearby asteroids.
    The action space includes thrusting in different directions and shooting.
    The environment can be rendered using Pygame for visualization.
    """
    # ---------- Environment metadata ----------
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Asteroids RL Agent")
        self.reset()

    # ---------- Environment methods ----------
    def reset(self):
        self.ship = Ship(WIDTH // 2, HEIGHT // 2)
        self.asteroids = [Asteroid(size=random.choice(["large", "medium", "small"])) for _ in range(5)]
        self.bullets = []
        self.score = 0
        self.done = False
        self.frame_count = 0
        self.bullets_fired = 0
        self.hits_landed = 0
        self.idle_steps = 0
        self.ship_deaths = 0
        return self._get_state()

    # ---------- Action space ----------
    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        # Action mapping
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

        reward = 0

        # When shooting:
        if shoot:
            reward -= 0.01  # shooting penalty
            if len(self.bullets) < 5:
                self.bullets.append(Bullet(self.ship.pos, self.ship.direction))
                self.bullets_fired += 1


        # # Shooting
        # if shoot:
        #     reward -= 0.01  # small cost to discourage spam
        #     if len(self.bullets) < 5:
        #         self.bullets.append(Bullet(self.ship.pos, self.ship.direction))

        # Movement encouragement vs idling
        if self.ship.vel.length() < 0.05:
            reward -= 0.1  # Was idle
            self.idle_steps += 1
        else:
            reward += 0.03  # Small reward for being active

        #
        # # Ship movement
        # if self.ship.vel.length() > 0.01:
        #     reward += 0.01
        # else:
        #     reward -= 0.01

        # Bullet-asteroid collision
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
                    reward += 10.0  # Increased reward for hit
                    self.hits_landed += 1  # ✅ Log hit
                    break

        # Ship-asteroid collision
        ship_rect = pygame.Rect(self.ship.pos.x - SHIP_RADIUS, self.ship.pos.y - SHIP_RADIUS, SHIP_RADIUS * 2,
                                SHIP_RADIUS * 2)
        for a in self.asteroids:
            if ship_rect.colliderect(a.get_rect()):
                reward = -1.0
                self.done = True
                self.ship_deaths += 1
                break


        # Small survival bonus
        reward += 0.002

        # Update ship and asteroid positions
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
            "ship_deaths": self.ship_deaths
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

    # # ---------- Rendering ----------
    def _render(self):
        self.clock.tick(FPS)
        self.screen.fill((0, 0, 0))
        self.ship.draw(self.screen)
        for a in self.asteroids:
            a.draw(self.screen)
        for b in self.bullets:
            b.draw(self.screen)

        # Draw score
        font = pygame.font.SysFont(None, 28)
        score_text = font.render(f"Score: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))


        pygame.display.flip()

