import pygame
import math
from .config import *

class Bullet:
    def __init__(self, pos, angle):
        rad = math.radians(angle)
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(math.cos(rad), -math.sin(rad)) * BULLET_SPEED

    def update(self):
        self.pos += self.vel

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (int(self.pos.x), int(self.pos.y)), BULLET_RADIUS)

    def off_screen(self):
        return self.pos.x < 0 or self.pos.x > WIDTH or self.pos.y < 0 or self.pos.y > HEIGHT
