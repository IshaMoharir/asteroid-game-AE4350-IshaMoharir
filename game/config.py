# -----------------------------
# Game Configuration Constants
# -----------------------------

# -----------------------------
# Screen & Frame Settings
# -----------------------------
WIDTH, HEIGHT = 800, 600   # Screen dimensions (pixels)
FPS = 60                   # Target frames per second

# -----------------------------
# Entity Dimensions
# -----------------------------
SHIP_RADIUS = 20           # Ship size (pixels)
ASTEROID_RADIUS = 30       # Default asteroid radius (if not using ASTEROID_SIZES below)
BULLET_RADIUS = 5          # Bullet size (pixels)

# -----------------------------
# Speeds & Movement
# -----------------------------
ASTEROID_SPEED = 2         # Default asteroid speed (if not using ASTEROID_SIZES below)
SHIP_THRUST = 0.15         # Acceleration per frame when thrusting
ROTATION_SPEED = 5         # Degrees of rotation per frame
BULLET_SPEED = 5           # Bullet travel speed (pixels per frame)

# -----------------------------
# Asteroid Size Variants
# -----------------------------
ASTEROID_SIZES = {
    "large":  {"radius": 40, "speed": 1},
    "medium": {"radius": 25, "speed": 1.5},
    "small":  {"radius": 15, "speed": 2}
}

# -----------------------------
# Asteroid Score Values
# -----------------------------
ASTEROID_POINTS = {
    "large":  20,   # Points for destroying a large asteroid
    "medium": 50,   # Points for destroying a medium asteroid
    "small":  100   # Points for destroying a small asteroid
}
