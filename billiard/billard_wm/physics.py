"""Physique réelle du monde — simulation 2D des billes."""

from __future__ import annotations
import math
from billard_wm.config import (
    BALL_RADIUS, FRICTION, RESTITUTION, MAX_SPEED, MIN_SPEED,
    TABLE_X, TABLE_Y, TABLE_WIDTH, TABLE_HEIGHT, BAND_THICKNESS, GRAVITY_SCALE,
)


class Ball:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0

    def moving(self) -> bool:
        return math.hypot(self.vx, self.vy) > MIN_SPEED

    def copy(self) -> Ball:
        b = Ball(self.x, self.y)
        b.vx = self.vx
        b.vy = self.vy
        return b


def tilt_to_gravity(tilt_x_deg: float, tilt_y_deg: float) -> tuple[float, float]:
    """Convertit les angles d'inclinaison en composantes de gravité."""
    gx = math.sin(math.radians(tilt_x_deg)) * GRAVITY_SCALE
    gy = math.sin(math.radians(tilt_y_deg)) * GRAVITY_SCALE
    return gx, gy


def step(ball: Ball, gravity_x: float = 0.0, gravity_y: float = 0.0,
         friction: float = FRICTION,
         restitution: float = RESTITUTION) -> None:
    """Avance la bille d'un pas. Modifie ball in-place."""
    ball.vx += gravity_x
    ball.vy += gravity_y

    # Mise à jour position
    ball.x += ball.vx
    ball.y += ball.vy

    # Friction
    ball.vx *= friction
    ball.vy *= friction

    # Clamp vitesse
    speed = math.hypot(ball.vx, ball.vy)
    if speed > MAX_SPEED:
        ball.vx = ball.vx / speed * MAX_SPEED
        ball.vy = ball.vy / speed * MAX_SPEED
    elif speed < MIN_SPEED:
        ball.vx = 0.0
        ball.vy = 0.0

    # Rebonds sur les bandes
    left = TABLE_X + BAND_THICKNESS + BALL_RADIUS
    right = TABLE_X + TABLE_WIDTH - BAND_THICKNESS - BALL_RADIUS
    top = TABLE_Y + BAND_THICKNESS + BALL_RADIUS
    bottom = TABLE_Y + TABLE_HEIGHT - BAND_THICKNESS - BALL_RADIUS

    if ball.x < left:
        ball.x = left
        ball.vx = abs(ball.vx) * restitution
    elif ball.x > right:
        ball.x = right
        ball.vx = -abs(ball.vx) * restitution

    if ball.y < top:
        ball.y = top
        ball.vy = abs(ball.vy) * restitution
    elif ball.y > bottom:
        ball.y = bottom
        ball.vy = -abs(ball.vy) * restitution


def check_collision(a: Ball, b: Ball) -> bool:
    """Vérifie si deux billes se touchent."""
    return math.hypot(a.x - b.x, a.y - b.y) <= 2 * BALL_RADIUS


def simulate_trajectory(ball: Ball, gravity_x: float, gravity_y: float,
                        max_steps: int,
                        friction: float = FRICTION,
                        restitution: float = RESTITUTION,
                        target: Ball | None = None) -> list[tuple[float, float]]:
    """Simule une trajectoire complète et retourne la liste des positions."""
    b = ball.copy()
    points = [(b.x, b.y)]
    for _ in range(max_steps):
        step(b, gravity_x, gravity_y, friction, restitution)
        points.append((b.x, b.y))
        if not b.moving():
            break
        if target and check_collision(b, target):
            break
    return points
