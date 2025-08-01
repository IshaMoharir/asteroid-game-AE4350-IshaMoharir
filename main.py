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

# Game state
score = 0
lives = 3
respawning = False
respawn_start_time = 0
invincible_duration = 3  # seconds
game_over = False

# Fonts and cooldowns
font = pygame.font.SysFont(None, 36)
last_shot_time = 0
SHOT_COOLDOWN = 0.3  # seconds

# Game loop
running = True
while running:
    clock.tick(FPS)
    screen.fill((0, 0, 0))

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
    if not game_over:
        ship.update()
        for a in asteroids:
            a.update()

        # Bullet updates and collisions
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

        # Ship-asteroid collision
        if not respawning:
            ship_rect = pygame.Rect(ship.pos.x - SHIP_RADIUS, ship.pos.y - SHIP_RADIUS, SHIP_RADIUS*2, SHIP_RADIUS*2)
            for a in asteroids:
                if ship_rect.colliderect(a.get_rect()):
                    lives -= 1
                    if lives <= 0:
                        game_over = True
                    else:
                        respawning = True
                        respawn_start_time = time.time()
                    break

    # Draw
    if not game_over:
        # Blink during respawn
        if respawning:
            if time.time() - respawn_start_time >= invincible_duration:
                respawning = False
            elif int(time.time() * 10) % 2 == 0:
                ship.draw(screen)
        else:
            ship.draw(screen)

        for a in asteroids:
            a.draw(screen)
        for b in bullets:
            b.draw(screen)

        # Score and lives
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        lives_text = font.render(f"Lives: {lives}", True, (255, 255, 255))
        screen.blit(lives_text, (WIDTH - 120, 10))

        pygame.display.flip()

    else:
        # Game over screen
        game_over_text = font.render("GAME OVER", True, (255, 0, 0))
        score_text = font.render(f"Final Score: {score}", True, (255, 255, 255))
        restart_text = font.render("Press any key to play again", True, (200, 200, 200))

        screen.blit(game_over_text, (WIDTH // 2 - 100, HEIGHT // 2 - 50))
        screen.blit(score_text, (WIDTH // 2 - 100, HEIGHT // 2 - 10))
        screen.blit(restart_text, (WIDTH // 2 - 160, HEIGHT // 2 + 30))

        pygame.display.flip()

        # Wait for key to restart
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    # Reset game
                    ship = Ship(WIDTH // 2, HEIGHT // 2)
                    asteroids = [Asteroid(size=random.choice(["large", "medium", "small"])) for _ in range(6)]
                    bullets = []
                    score = 0
                    lives = 3
                    game_over = False
                    respawning = False
                    waiting = False
            clock.tick(15)

pygame.quit()
