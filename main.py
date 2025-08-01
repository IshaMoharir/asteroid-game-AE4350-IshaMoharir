import pygame
import random
import time
from game.ship import Ship
from game.asteroid import Asteroid
from game.bullet import Bullet
from game.config import *

# Setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Asteroids RL")
clock = pygame.time.Clock()

# Game objects
ship = Ship(WIDTH // 2, HEIGHT // 2)
asteroids = [Asteroid(size=random.choice(["large", "medium", "small"])) for _ in range(6)]
bullets = []

# Scoring
score = 0
font = pygame.font.SysFont(None, 36)

# Shooting cooldown
last_shot_time = 0
SHOT_COOLDOWN = 0.3  # seconds

# Game loop
running = True
while running:
    clock.tick(FPS)
    screen.fill((0, 0, 0))  # Clear screen

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Controls
    keys = pygame.key.get_pressed()
    thrust = pygame.math.Vector2(0, 0)

    if keys[pygame.K_UP]:
        thrust.y -= 1
    if keys[pygame.K_DOWN]:
        thrust.y += 1
    if keys[pygame.K_LEFT]:
        thrust.x -= 1
    if keys[pygame.K_RIGHT]:
        thrust.x += 1

    ship.apply_thrust(thrust)

    # Shooting
    current_time = time.time()
    if keys[pygame.K_SPACE] and current_time - last_shot_time >= SHOT_COOLDOWN:
        bullets.append(Bullet(ship.pos, ship.direction))
        last_shot_time = current_time

    # Update ship and asteroids
    ship.update()
    for a in asteroids:
        a.update()

    # Update bullets + check collisions
    for b in bullets[:]:
        b.update()
        if b.off_screen():
            bullets.remove(b)
            continue

        for a in asteroids[:]:
            if a.get_rect().collidepoint(b.pos):
                bullets.remove(b)
                asteroids.remove(a)
                score += ASTEROID_POINTS[a.size]
                asteroids.extend(a.split())
                break

    # Draw all
    ship.draw(screen)
    for a in asteroids:
        a.draw(screen)
    for b in bullets:
        b.draw(screen)

    # Draw score
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    pygame.display.flip()

pygame.quit()
