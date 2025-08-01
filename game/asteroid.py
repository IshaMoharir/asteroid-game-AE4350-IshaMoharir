import pygame
import random
from .config import *

class Asteroid:
    def __init__(self):
        self.pos = pygame.math.Vector2(random.randint(0, WIDTH), random.randint(0, HEIGHT))
        angle = random.uniform(0, 360)
        self.vel = pygame.math.Vector2(ASTEROID_SPEED, 0).rotate(angle)

    def update(self):
        self.pos += self.vel
        self.pos.x %= WIDTH
        self.pos.y %= HEIGHT

    def draw(self, screen):
        pygame.draw.circle(screen, (200, 200, 200), (int(self.pos.x), int(self.pos.y)), ASTEROID_RADIUS)
