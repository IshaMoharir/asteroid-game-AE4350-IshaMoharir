import pygame
from .config import *

class Ship:
    def __init__(self, x, y):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.acc = pygame.math.Vector2(0, 0)
        self.direction = pygame.math.Vector2(0, -1)  # Last direction of movement

    def apply_thrust(self, thrust_vector):
        if thrust_vector.length() > 0:
            thrust_vector = thrust_vector.normalize()
            self.direction = thrust_vector
            self.acc += thrust_vector * SHIP_THRUST

    def update(self):
        self.vel += self.acc
        self.pos += self.vel
        self.acc *= 0  # reset acceleration each frame

        # Apply friction
        self.vel *= 0.98

        # Screen wrap
        self.pos.x = max(SHIP_RADIUS, min(WIDTH - SHIP_RADIUS, self.pos.x))
        self.pos.y = max(SHIP_RADIUS, min(HEIGHT - SHIP_RADIUS, self.pos.y))

    def draw(self, screen):
        pygame.draw.polygon(screen, (255, 255, 255), self.get_triangle())

    def get_triangle(self):
        # Draw triangle pointing in direction of self.direction
        perp = self.direction.rotate(90)
        tip = self.pos + self.direction * SHIP_RADIUS
        left = self.pos + perp * (SHIP_RADIUS / 2)
        right = self.pos - perp * (SHIP_RADIUS / 2)
        return [tip, left, right]
