import pygame
import random
from .config import *

class Asteroid:
    def __init__(self, size="large", pos=None):
        self.size = size
        self.radius = ASTEROID_SIZES[size]["radius"]
        speed = ASTEROID_SIZES[size]["speed"]

        if pos:
            self.pos = pygame.math.Vector2(pos)
        else:
            self.pos = pygame.math.Vector2(random.randint(0, WIDTH), random.randint(0, HEIGHT))

        angle = random.uniform(0, 360)
        self.vel = pygame.math.Vector2(speed, 0).rotate(angle)

    def update(self):
        self.pos += self.vel
        self.pos.x %= WIDTH
        self.pos.y %= HEIGHT

    def draw(self, screen):
        pygame.draw.circle(screen, (200, 200, 200), (int(self.pos.x), int(self.pos.y)), self.radius)

    def get_rect(self):
        return pygame.Rect(self.pos.x - self.radius, self.pos.y - self.radius, self.radius*2, self.radius*2)

    def split(self):
        """Returns smaller asteroids if possible"""
        if self.size == "large":
            return [Asteroid("medium", self.pos), Asteroid("medium", self.pos)]
        elif self.size == "medium":
            return [Asteroid("small", self.pos), Asteroid("small", self.pos)]
        else:
            return []
