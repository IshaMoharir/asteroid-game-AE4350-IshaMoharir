import pygame
from .config import *  # Import constants (WIDTH, HEIGHT, SHIP_RADIUS, SHIP_THRUST)

# -----------------------------
# Ship Class
# -----------------------------
class Ship:
    """
    Player-controlled ship entity.
    Handles movement, thrust, drawing, and screen boundaries.
    """

    # -----------------------------
    # Initialisation
    # -----------------------------
    def __init__(self, x, y):
        """
        Create a ship at a given position.

        Args:
            x (int|float): Initial x-coordinate.
            y (int|float): Initial y-coordinate.
        """
        self.pos = pygame.math.Vector2(x, y)       # Position vector
        self.vel = pygame.math.Vector2(0, 0)       # Velocity vector
        self.acc = pygame.math.Vector2(0, 0)       # Acceleration vector
        self.direction = pygame.math.Vector2(0, -1)  # Facing upwards initially

    # -----------------------------
    # Apply Thrust
    # -----------------------------
    def apply_thrust(self, thrust_vector):
        """
        Apply thrust to the ship, adjusting acceleration and direction.

        Args:
            thrust_vector (Vector2): The thrust direction (normalised inside).
        """
        if thrust_vector.length() > 0:
            thrust_vector = thrust_vector.normalize()
            self.direction = thrust_vector         # Update facing direction
            self.acc += thrust_vector * SHIP_THRUST

    # -----------------------------
    # Update Movement
    # -----------------------------
    def update(self):
        """
        Update position and velocity each frame.
        Includes:
        - Applying acceleration to velocity
        - Applying friction
        - Keeping ship within screen bounds
        """
        self.vel += self.acc       # Apply acceleration
        self.pos += self.vel       # Update position
        self.acc *= 0              # Reset acceleration each frame

        # Apply friction (slows the ship down gradually)
        self.vel *= 0.98

        # Keep ship within screen bounds (clamped to edges)
        self.pos.x = max(SHIP_RADIUS, min(WIDTH - SHIP_RADIUS, self.pos.x))
        self.pos.y = max(SHIP_RADIUS, min(HEIGHT - SHIP_RADIUS, self.pos.y))

    # -----------------------------
    # Drawing
    # -----------------------------
    def draw(self, screen):
        """
        Draw the ship as a white triangle pointing in its direction.
        """
        pygame.draw.polygon(screen, (255, 255, 255), self.get_triangle())

    # -----------------------------
    # Ship Shape (Triangle)
    # -----------------------------
    def get_triangle(self):
        """
        Compute the ship's triangular shape (tip + left/right corners).

        Returns:
            list[Vector2]: Three vertices of the triangle.
        """
        # Perpendicular vector to direction
        perp = self.direction.rotate(90)

        # Compute triangle points
        tip = self.pos + self.direction * SHIP_RADIUS         # Tip points forward
        left = self.pos + perp * (SHIP_RADIUS / 2)            # Left wing
        right = self.pos - perp * (SHIP_RADIUS / 2)           # Right wing

        return [tip, left, right]
