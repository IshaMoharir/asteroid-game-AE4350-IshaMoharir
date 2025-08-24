import pygame
import random
from .config import *  # Import game configuration (ASTEROID_SIZES, WIDTH, HEIGHT)

# -----------------------------
# Asteroid Class
# -----------------------------
class Asteroid:
    """
    Asteroid entity for the game.
    Handles spawning, movement, rendering, collision bounding, and splitting.
    """

    # -----------------------------
    # Initialisation
    # -----------------------------
    def __init__(self, size="large", pos=None):
        """
        Create an asteroid with a given size and optional position.
        If no position is given, spawn at a random location.
        """
        self.size = size
        self.radius = ASTEROID_SIZES[size]["radius"]
        speed = ASTEROID_SIZES[size]["speed"]

        # Starting position: use given pos or random screen location
        if pos:
            self.pos = pygame.math.Vector2(pos)
        else:
            self.pos = pygame.math.Vector2(
                random.randint(0, WIDTH),
                random.randint(0, HEIGHT)
            )

        # Assign random velocity vector at the given speed
        angle = random.uniform(0, 360)
        self.vel = pygame.math.Vector2(speed, 0).rotate(angle)

    # -----------------------------
    # Update Position
    # -----------------------------
    def update(self):
        """
        Move asteroid according to velocity and wrap around screen edges.
        """
        self.pos += self.vel
        self.pos.x %= WIDTH   # Wrap horizontally
        self.pos.y %= HEIGHT  # Wrap vertically

    # -----------------------------
    # Drawing
    # -----------------------------
    def draw(self, screen):
        """
        Draw the asteroid as a grey circle.
        """
        pygame.draw.circle(
            screen,
            (200, 200, 200),  # Grey colour
            (int(self.pos.x), int(self.pos.y)),
            self.radius
        )

    # -----------------------------
    # Collision Rectangle
    # -----------------------------
    def get_rect(self):
        """
        Return axis-aligned bounding box for collision checks.
        """
        return pygame.Rect(
            self.pos.x - self.radius,
            self.pos.y - self.radius,
            self.radius * 2,
            self.radius * 2
        )

    # -----------------------------
    # Splitting
    # -----------------------------
    def split(self):
        """
        Split asteroid into smaller ones when destroyed.
        - Large -> 2 Medium
        - Medium -> 2 Small
        - Small -> No split
        """
        if self.size == "large":
            return [Asteroid("medium", self.pos), Asteroid("medium", self.pos)]
        elif self.size == "medium":
            return [Asteroid("small", self.pos), Asteroid("small", self.pos)]
        else:
            return []
