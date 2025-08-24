import pygame
import math
from .config import *  # Import constants (WIDTH, HEIGHT, BULLET_SPEED, BULLET_RADIUS)

# -----------------------------
# Bullet Class
# -----------------------------
class Bullet:
    """
    Bullet entity fired by the ship.
    Handles movement, drawing, and screen-bound checks.
    """

    # -----------------------------
    # Initialisation
    # -----------------------------
    def __init__(self, pos, direction):
        """
        Create a bullet at a given position moving in a given direction.

        Args:
            pos (tuple|list|Vector2): Starting position of the bullet.
            direction (Vector2): Direction vector, will be normalised and scaled to BULLET_SPEED.
        """
        self.pos = pygame.math.Vector2(pos)
        self.vel = direction.normalize() * BULLET_SPEED

    # -----------------------------
    # Update Position
    # -----------------------------
    def update(self):
        """
        Move bullet forward according to its velocity.
        """
        self.pos += self.vel

    # -----------------------------
    # Drawing
    # -----------------------------
    def draw(self, screen):
        """
        Draw the bullet as a red circle.
        """
        pygame.draw.circle(
            screen,
            (255, 0, 0),  # Red colour
            (int(self.pos.x), int(self.pos.y)),
            BULLET_RADIUS
        )

    # -----------------------------
    # Screen Bounds Check
    # -----------------------------
    def off_screen(self):
        """
        Check if the bullet has left the visible screen.

        Returns:
            bool: True if bullet is outside the screen, False otherwise.
        """
        return (
            self.pos.x < 0 or self.pos.x > WIDTH or
            self.pos.y < 0 or self.pos.y > HEIGHT
        )
