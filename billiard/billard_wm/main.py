"""Boucle principale — démo World Model billard."""

from __future__ import annotations
import math
import random
import pygame
from billard_wm.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT, FPS, MAX_TILT, GRAVITY_SCALE,
    TABLE_X, TABLE_Y, TABLE_WIDTH, TABLE_HEIGHT, BAND_THICKNESS, BALL_RADIUS,
    CANDIDATES_REVEAL_RATE,
    PHASE_DURATION_IMAGINING, PHASE_DURATION_PLANNING,
    PHASE_DURATION_PREDICTING, PHASE_DURATION_OBSERVING,
    PHASE_DURATION_LEARNING,
)
from billard_wm.physics import Ball, step, check_collision, tilt_to_gravity
from billard_wm.world_model import WorldModel, PlanResult
from billard_wm.renderer import Renderer


def random_target_pos() -> tuple[float, float]:
    """Position aléatoire dans la moitié droite de la table."""
    margin = BAND_THICKNESS + BALL_RADIUS + 20
    x = TABLE_X + TABLE_WIDTH // 2 + random.randint(margin, TABLE_WIDTH // 2 - margin)
    y = TABLE_Y + random.randint(margin, TABLE_HEIGHT - margin)
    return float(x), float(y)


def white_start_pos() -> tuple[float, float]:
    """Position de départ de la bille blanche (quart gauche)."""
    margin = BAND_THICKNESS + BALL_RADIUS + 20
    x = TABLE_X + margin + 50
    y = TABLE_Y + TABLE_HEIGHT // 2
    return float(x), float(y)


# Durées par état (ms) — pour les transitions automatiques
_PHASE_DURATIONS: dict[str, int] = {
    "imagining":  PHASE_DURATION_IMAGINING,
    "planning":   PHASE_DURATION_PLANNING,
    "predicting": PHASE_DURATION_PREDICTING,
    "observing":  PHASE_DURATION_OBSERVING,
    "learning":   PHASE_DURATION_LEARNING,
}


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("World Model — Billard Demo")
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.wm = WorldModel()
        self.wm_enabled = True

        self.tilt_x_deg = 0.0
        self.tilt_y_deg = 0.0
        self.white = Ball(*white_start_pos())
        tx, ty = random_target_pos()
        self.target = Ball(tx, ty)

        # Machine à états étendue
        # aiming → imagining → planning → predicting → shooting
        #       → observing → learning → (auto-reset) → aiming
        self.state = "aiming"
        self.phase_timer = 0          # ms écoulées dans la phase courante
        self.imagination_progress = 0  # nb de candidats révélés

        self.plan_result: PlanResult | None = None
        self.predicted_traj: list[tuple[float, float]] | None = None
        self.actual_traj: list[tuple[float, float]] = []
        self.update_preview: tuple[float, float, float, float, float] | None = None

        self.phase_text = "Phase 1 — Table plate"
        self.hit_target = False

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_shot(self) -> None:
        wx, wy = white_start_pos()
        self.white.x, self.white.y = wx, wy
        self.white.vx = self.white.vy = 0.0
        tx, ty = random_target_pos()
        self.target.x, self.target.y = tx, ty
        self.state = "aiming"
        self.phase_timer = 0
        self.imagination_progress = 0
        self.plan_result = None
        self.predicted_traj = None
        self.actual_traj = []
        self.update_preview = None
        self.hit_target = False

    def full_reset(self) -> None:
        self.tilt_x_deg = 0.0
        self.tilt_y_deg = 0.0
        self.wm.reset()
        self.reset_shot()
        self.phase_text = "Phase 1 — Table plate"

    # ------------------------------------------------------------------
    # Tir
    # ------------------------------------------------------------------
    def shoot(self) -> None:
        if self.state != "aiming":
            return

        if self.wm_enabled:
            # Pipeline pédagogique : imagining → planning → predicting → shooting
            self.plan_result = self.wm.plan_shot_detailed(self.white, self.target)
            self.state = "imagining"
            self.phase_timer = 0
            self.imagination_progress = 0
        else:
            # Pas de WM → tir direct
            angle, force = self.wm.plan_shot(self.white, self.target)
            self._fire_shot(angle, force)

    def _fire_shot(self, angle: float, force: float) -> None:
        """Déclenche le tir réel (transition vers shooting)."""
        # Trajectoire prédite
        pred_ball = self.white.copy()
        pred_ball.vx = math.cos(angle) * force
        pred_ball.vy = math.sin(angle) * force
        self.predicted_traj = self.wm.predict_trajectory(pred_ball, self.target)

        # Appliquer le tir
        self.white.vx = math.cos(angle) * force
        self.white.vy = math.sin(angle) * force
        self.actual_traj = [(self.white.x, self.white.y)]
        self.state = "shooting"
        self.phase_timer = 0

    # ------------------------------------------------------------------
    # Transitions entre phases
    # ------------------------------------------------------------------
    def _advance_phase(self) -> None:
        """Passe à la phase suivante dans le pipeline."""
        if self.state == "imagining":
            self.state = "planning"
            self.phase_timer = 0
        elif self.state == "planning":
            self.state = "predicting"
            self.phase_timer = 0
        elif self.state == "predicting":
            # Déclenche le tir
            if self.plan_result:
                self._fire_shot(
                    self.plan_result.best_angle,
                    self.plan_result.best_force,
                )
            else:
                self.reset_shot()
        elif self.state == "shooting":
            # Skip du tir en cours
            self.white.vx = 0.0
            self.white.vy = 0.0
            if self.wm_enabled:
                self.state = "observing"
                self.phase_timer = 0
            else:
                self.reset_shot()
        elif self.state == "observing":
            # Pré-calcul de la mise à jour
            if self.predicted_traj and self.actual_traj:
                self.update_preview = self.wm.compute_update_preview(
                    self.predicted_traj, self.actual_traj
                )
            self.state = "learning"
            self.phase_timer = 0
        elif self.state == "learning":
            # Appliquer la mise à jour et reset
            if self.predicted_traj and self.actual_traj:
                self.wm.apply_update(self.predicted_traj, self.actual_traj)
            self._update_phase_text()
            self.reset_shot()

    # ------------------------------------------------------------------
    # Physique (inchangée dans la logique)
    # ------------------------------------------------------------------
    def update_physics(self) -> None:
        if self.state != "shooting":
            return

        gx, gy = tilt_to_gravity(self.tilt_x_deg, self.tilt_y_deg)
        step(self.white, gx, gy)
        self.actual_traj.append((self.white.x, self.white.y))

        if check_collision(self.white, self.target):
            self.hit_target = True
            self.white.vx *= 0.3
            self.white.vy *= 0.3

        if not self.white.moving():
            if self.wm_enabled:
                self.state = "observing"
                self.phase_timer = 0
            else:
                self.state = "done"
                if self.predicted_traj:
                    self.wm.update(self.predicted_traj, self.actual_traj)
                self._update_phase_text()

    def _update_phase_text(self) -> None:
        tilt_mag = math.hypot(self.tilt_x_deg, self.tilt_y_deg)
        if tilt_mag < 0.5:
            self.phase_text = "Phase 1 — Table plate"
        elif self.wm.last_error < 20:
            self.phase_text = (
                f"Adapté ! (x:{self.tilt_x_deg:+.1f}° y:{self.tilt_y_deg:+.1f}°)"
            )
        elif self.wm.shot_count < 3:
            self.phase_text = (
                f"Incliné x:{self.tilt_x_deg:+.1f}° y:{self.tilt_y_deg:+.1f}°"
                " — Le WM découvre..."
            )
        else:
            self.phase_text = (
                f"Incliné x:{self.tilt_x_deg:+.1f}° y:{self.tilt_y_deg:+.1f}°"
                " — Adaptation..."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_phase_progress(self) -> float:
        """Progression normalisée [0, 1] dans la phase courante."""
        duration = _PHASE_DURATIONS.get(self.state, 1)
        return min(1.0, self.phase_timer / max(1, duration))

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------
    def run(self) -> None:
        running = True
        done_timer = 0

        while running:
            dt = self.clock.tick(FPS)

            # --- Événements ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        if self.state == "aiming":
                            self.shoot()
                        elif self.state in (
                            "imagining", "planning", "predicting",
                            "shooting", "observing", "learning",
                        ):
                            self._advance_phase()
                        elif self.state == "done":
                            self.reset_shot()

                    elif event.key == pygame.K_r:
                        self.full_reset()

                    elif event.key == pygame.K_w:
                        self.wm_enabled = not self.wm_enabled

            # --- Inclinaison continue ---
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.tilt_x_deg = max(-MAX_TILT, self.tilt_x_deg - 0.3)
            if keys[pygame.K_RIGHT]:
                self.tilt_x_deg = min(MAX_TILT, self.tilt_x_deg + 0.3)
            if keys[pygame.K_UP]:
                self.tilt_y_deg = max(-MAX_TILT, self.tilt_y_deg - 0.3)
            if keys[pygame.K_DOWN]:
                self.tilt_y_deg = min(MAX_TILT, self.tilt_y_deg + 0.3)

            # --- Mise à jour des phases temporisées ---
            if self.state == "imagining":
                self.phase_timer += dt
                self.imagination_progress = int(
                    self.phase_timer / max(1, CANDIDATES_REVEAL_RATE)
                )
                if self.phase_timer >= PHASE_DURATION_IMAGINING:
                    self._advance_phase()

            elif self.state == "planning":
                self.phase_timer += dt
                if self.phase_timer >= PHASE_DURATION_PLANNING:
                    self._advance_phase()

            elif self.state == "predicting":
                self.phase_timer += dt
                if self.phase_timer >= PHASE_DURATION_PREDICTING:
                    self._advance_phase()

            elif self.state == "shooting":
                self.update_physics()

            elif self.state == "observing":
                self.phase_timer += dt
                if self.phase_timer >= PHASE_DURATION_OBSERVING:
                    self._advance_phase()

            elif self.state == "learning":
                self.phase_timer += dt
                if self.phase_timer >= PHASE_DURATION_LEARNING:
                    self._advance_phase()

            elif self.state == "done":
                done_timer += dt
                if done_timer > 1200:
                    done_timer = 0
                    self.reset_shot()

            # --- Rendu ---
            self.renderer.draw_all(
                white=self.white,
                target=self.target,
                tilt_x_deg=self.tilt_x_deg,
                tilt_y_deg=self.tilt_y_deg,
                predicted_traj=self.predicted_traj,
                actual_traj=self.actual_traj,
                wm_error=self.wm.last_error,
                wm_shots=self.wm.shot_count,
                wm_est_gx=self.wm.est_gx,
                wm_est_gy=self.wm.est_gy,
                phase_text=self.phase_text,
                wm_enabled=self.wm_enabled,
                errors_history=self.wm.errors_history,
                state=self.state,
                plan_result=self.plan_result,
                revealed_count=self.imagination_progress,
                phase_progress=self._get_phase_progress(),
                update_preview=self.update_preview,
            )
            pygame.display.flip()

        pygame.quit()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
