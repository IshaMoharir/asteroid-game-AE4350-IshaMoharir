import pygame
import math
from .config import *

class Ship:
    def __init__(self, x, y):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = 0

    def rotate(self, direction):
        self.angle += ROTATION_SPEED * direction

    def thrust(self):
        rad = math.radians(self.angle)
        force = pygame.math.Vector2(math.cos(rad), -math.sin(rad)) * SHIP_THRUST
        self.vel += force

    def update(self):
        self.pos += self.vel
        self.vel *= 0.99  # friction / damping
        self.pos.x %= WIDTH
        self.pos.y %= HEIGHT

    def draw(self, screen):
        rad = math.radians(self.angle)
        tip = self.pos + pygame.math.Vector2(math.cos(rad), -math.sin(rad)) * SHIP_RADIUS
        left = self.pos + pygame.math.Vector2(math.cos(rad + 2.5), -math.sin(rad + 2.5)) * SHIP_RADIUS
        right = self.pos + pygame.math.Vector2(math.cos(rad - 2.5), -math.sin(rad - 2.5)) * SHIP_RADIUS
        pygame.draw.polygon(screen, (255, 255, 255), [tip, left, right])
