"""World Model — modèle interne paramétrique avec adaptation."""

from __future__ import annotations
import colorsys
import math
from dataclasses import dataclass, field
from billard_wm.config import (
    WM_LEARNING_RATE, WM_PREDICTION_STEPS, FRICTION, RESTITUTION,
    TOP_K_COUNT,
)
from billard_wm.physics import Ball, simulate_trajectory, check_collision


@dataclass
class ShotCandidate:
    angle: float
    force: float
    trajectory: list[tuple[float, float]]
    min_dist: float
    color: tuple[int, int, int]


@dataclass
class PlanResult:
    candidates: list[ShotCandidate]
    top_k: list[int]       # indices des TOP_K meilleurs candidats
    best_idx: int
    best_angle: float
    best_force: float


class WorldModel:
    def __init__(self):
        # Paramètres estimés par le WM (commence par table plate)
        self.est_gx = 0.0  # gravité estimée sur x
        self.est_gy = 0.0  # gravité estimée sur y
        self.est_friction = FRICTION
        self.est_restitution = RESTITUTION
        self.shot_count = 0
        self.last_error = 0.0
        self.errors_history: list[float] = []

    def predict_trajectory(self, ball: Ball, target: Ball) -> list[tuple[float, float]]:
        """Prédit la trajectoire avec le modèle interne."""
        return simulate_trajectory(
            ball, self.est_gx, self.est_gy,
            max_steps=WM_PREDICTION_STEPS,
            friction=self.est_friction,
            restitution=self.est_restitution,
            target=target,
        )

    def plan_shot(self, white: Ball, target: Ball) -> tuple[float, float]:
        """Planifie un tir : retourne (angle, force)."""
        best_angle = 0.0
        best_force = 5.0
        best_dist = float('inf')

        dx = target.x - white.x
        dy = target.y - white.y
        direct_angle = math.atan2(dy, dx)

        # Compensation basée sur la gravité estimée
        gravity_compensation = math.atan2(-self.est_gy, -self.est_gx) if (abs(self.est_gx) + abs(self.est_gy)) > 0.001 else 0.0
        gravity_mag = math.hypot(self.est_gx, self.est_gy)
        center_angle = direct_angle + gravity_compensation * gravity_mag * 20

        for angle_offset in [i * 0.03 for i in range(-25, 26)]:
            angle = center_angle + angle_offset
            for force in [4, 6, 8, 10, 13]:
                test_ball = white.copy()
                test_ball.vx = math.cos(angle) * force
                test_ball.vy = math.sin(angle) * force
                traj = simulate_trajectory(
                    test_ball, self.est_gx, self.est_gy,
                    max_steps=WM_PREDICTION_STEPS,
                    friction=self.est_friction,
                    restitution=self.est_restitution,
                    target=target,
                )
                min_d = min(math.hypot(px - target.x, py - target.y) for px, py in traj)
                if min_d < best_dist:
                    best_dist = min_d
                    best_angle = angle
                    best_force = force

        return best_angle, best_force

    def plan_shot_detailed(self, white: Ball, target: Ball) -> PlanResult:
        """Comme plan_shot mais retourne toutes les 255 trajectoires + classement."""
        dx = target.x - white.x
        dy = target.y - white.y
        direct_angle = math.atan2(dy, dx)

        gravity_compensation = (
            math.atan2(-self.est_gy, -self.est_gx)
            if (abs(self.est_gx) + abs(self.est_gy)) > 0.001 else 0.0
        )
        gravity_mag = math.hypot(self.est_gx, self.est_gy)
        center_angle = direct_angle + gravity_compensation * gravity_mag * 20

        all_combos = []
        for angle_offset in [i * 0.03 for i in range(-25, 26)]:
            angle = center_angle + angle_offset
            for force in [4, 6, 8, 10, 13]:
                all_combos.append((angle, force))

        total = len(all_combos)
        candidates: list[ShotCandidate] = []

        for idx, (angle, force) in enumerate(all_combos):
            test_ball = white.copy()
            test_ball.vx = math.cos(angle) * force
            test_ball.vy = math.sin(angle) * force
            traj = simulate_trajectory(
                test_ball, self.est_gx, self.est_gy,
                max_steps=WM_PREDICTION_STEPS,
                friction=self.est_friction,
                restitution=self.est_restitution,
                target=target,
            )
            min_d = min(
                math.hypot(px - target.x, py - target.y) for px, py in traj
            )
            # Couleur arc-en-ciel
            hue = idx / total
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            color = (int(r * 255), int(g * 255), int(b * 255))

            candidates.append(ShotCandidate(
                angle=angle, force=force, trajectory=traj,
                min_dist=min_d, color=color,
            ))

        sorted_indices = sorted(
            range(len(candidates)), key=lambda i: candidates[i].min_dist
        )
        top_k = sorted_indices[:TOP_K_COUNT]
        best_idx = sorted_indices[0]

        return PlanResult(
            candidates=candidates,
            top_k=top_k,
            best_idx=best_idx,
            best_angle=candidates[best_idx].angle,
            best_force=candidates[best_idx].force,
        )

    def compute_update_preview(
        self, predicted: list[tuple[float, float]],
        actual: list[tuple[float, float]],
    ) -> tuple[float, float, float, float, float]:
        """Calcule l'aperçu de la mise à jour sans modifier le modèle.

        Returns (old_gx, old_gy, new_gx, new_gy, error).
        """
        old_gx, old_gy = self.est_gx, self.est_gy

        n = min(len(predicted), len(actual))
        if n < 2:
            return old_gx, old_gy, old_gx, old_gy, 0.0

        sample_indices = list(range(0, n, max(1, n // 20)))
        errors_x = []
        errors_y = []
        for i in sample_indices:
            errors_x.append(actual[i][0] - predicted[i][0])
            errors_y.append(actual[i][1] - predicted[i][1])

        avg_error_x = sum(errors_x) / len(errors_x)
        avg_error_y = sum(errors_y) / len(errors_y)
        error = math.sqrt(
            (sum(ex**2 for ex in errors_x) + sum(ey**2 for ey in errors_y))
            / len(errors_x)
        )

        lr = WM_LEARNING_RATE
        new_gx = old_gx + avg_error_x * lr * 0.002
        new_gy = old_gy + avg_error_y * lr * 0.002

        return old_gx, old_gy, new_gx, new_gy, error

    def apply_update(
        self, predicted: list[tuple[float, float]],
        actual: list[tuple[float, float]],
    ) -> None:
        """Applique la mise à jour calculée par compute_update_preview."""
        _, _, new_gx, new_gy, error = self.compute_update_preview(
            predicted, actual
        )
        self.shot_count += 1
        self.est_gx = new_gx
        self.est_gy = new_gy
        self.last_error = error
        self.errors_history.append(error)

    def update(self, predicted: list[tuple[float, float]],
               actual: list[tuple[float, float]]) -> None:
        """Recalibre le modèle interne à partir de l'erreur observée."""
        self.shot_count += 1

        n = min(len(predicted), len(actual))
        if n < 2:
            return

        sample_indices = list(range(0, n, max(1, n // 20)))
        errors_x = []
        errors_y = []
        for i in sample_indices:
            errors_x.append(actual[i][0] - predicted[i][0])
            errors_y.append(actual[i][1] - predicted[i][1])

        avg_error_x = sum(errors_x) / len(errors_x)
        avg_error_y = sum(errors_y) / len(errors_y)
        self.last_error = math.sqrt(
            (sum(ex**2 for ex in errors_x) + sum(ey**2 for ey in errors_y))
            / len(errors_x)
        )
        self.errors_history.append(self.last_error)

        lr = WM_LEARNING_RATE
        self.est_gx += avg_error_x * lr * 0.002
        self.est_gy += avg_error_y * lr * 0.002

    def reset(self):
        self.est_gx = 0.0
        self.est_gy = 0.0
        self.est_friction = FRICTION
        self.est_restitution = RESTITUTION
        self.shot_count = 0
        self.last_error = 0.0
        self.errors_history.clear()
