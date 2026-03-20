"""Rendu Pygame — table, billes, trajectoires, UI, étapes pédagogiques."""

from __future__ import annotations
import math
import pygame
from billard_wm.config import (
    WINDOW_WIDTH, WINDOW_HEIGHT,
    TABLE_X, TABLE_Y, TABLE_WIDTH, TABLE_HEIGHT, BAND_THICKNESS,
    BALL_RADIUS, MAX_TILT,
    COLOR_BG, COLOR_TABLE, COLOR_BAND, COLOR_WHITE_BALL, COLOR_TARGET_BALL,
    COLOR_PREDICTED, COLOR_ACTUAL, COLOR_TEXT, COLOR_ERROR, COLOR_SUCCESS,
    COLOR_TILT_ARROW, COLOR_UI_BG,
    COLOR_IMAGINATION, COLOR_PLANNING, COLOR_PREDICTION,
    COLOR_ACTION, COLOR_OBSERVATION, COLOR_LEARNING,
)
from billard_wm.physics import Ball
from billard_wm.world_model import PlanResult


# Mapping état → (numéro, label, couleur)
STAGE_INFO: dict[str, tuple[int, str, tuple[int, int, int]]] = {
    "imagining":  (1, "IMAGINATION",    COLOR_IMAGINATION),
    "planning":   (2, "PLANIFICATION",  COLOR_PLANNING),
    "predicting": (3, "PRÉDICTION",     COLOR_PREDICTION),
    "shooting":   (4, "ACTION",         COLOR_ACTION),
    "observing":  (5, "OBSERVATION",    COLOR_OBSERVATION),
    "learning":   (6, "APPRENTISSAGE",  COLOR_LEARNING),
}


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont("Menlo", 18)
        self.font_big = pygame.font.SysFont("Menlo", 24, bold=True)
        self.font_small = pygame.font.SysFont("Menlo", 14)
        self.font_stage = pygame.font.SysFont("Menlo", 32, bold=True)
        # Surface translucide pour trajectoires
        self.overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------
    def draw_all(
        self,
        white: Ball,
        target: Ball,
        tilt_x_deg: float,
        tilt_y_deg: float,
        predicted_traj: list[tuple[float, float]] | None,
        actual_traj: list[tuple[float, float]],
        wm_error: float,
        wm_shots: int,
        wm_est_gx: float,
        wm_est_gy: float,
        phase_text: str,
        wm_enabled: bool,
        errors_history: list[float],
        *,
        state: str = "aiming",
        plan_result: PlanResult | None = None,
        revealed_count: int = 0,
        phase_progress: float = 0.0,
        update_preview: tuple[float, float, float, float, float] | None = None,
    ) -> None:
        self.screen.fill(COLOR_BG)
        self.overlay.fill((0, 0, 0, 0))

        self._draw_table(tilt_x_deg, tilt_y_deg)

        # --- Visualisations spécifiques à l'étape ---
        if state in ("imagining", "planning", "predicting") and plan_result:
            self._draw_candidate_trajectories(
                plan_result, revealed_count, state, phase_progress
            )
        elif state == "shooting":
            self._draw_trajectories(predicted_traj, actual_traj)
        elif state == "observing":
            self._draw_trajectories(predicted_traj, actual_traj)
            self._draw_observation_comparison(
                predicted_traj, actual_traj, phase_progress
            )
        elif state == "learning":
            self._draw_trajectories(predicted_traj, actual_traj)
            if update_preview:
                self._draw_learning_animation(
                    *update_preview, phase_progress, errors_history
                )
        elif state == "done":
            self._draw_trajectories(predicted_traj, actual_traj)

        self._draw_ball(white.x, white.y, COLOR_WHITE_BALL)
        self._draw_ball(target.x, target.y, COLOR_TARGET_BALL)
        self._draw_tilt_indicator(tilt_x_deg, tilt_y_deg)

        # Label d'étape (uniquement si WM actif et pas en aiming/done)
        if wm_enabled and state in STAGE_INFO:
            num, label, color = STAGE_INFO[state]
            self._draw_stage_label(label, num, color)

        self._draw_ui(
            wm_error, wm_shots, wm_est_gx, wm_est_gy,
            phase_text, wm_enabled, errors_history,
            state=state, phase_progress=phase_progress,
            update_preview=update_preview,
        )

        self.screen.blit(self.overlay, (0, 0))

    # ------------------------------------------------------------------
    # Étape label
    # ------------------------------------------------------------------
    def _draw_stage_label(
        self, name: str, number: int, color: tuple[int, int, int]
    ) -> None:
        label = f"{number}. {name}"
        txt = self.font_stage.render(label, True, color)
        x = TABLE_X + TABLE_WIDTH // 2 - txt.get_width() // 2
        y = TABLE_Y + BAND_THICKNESS + 10
        # Fond semi-transparent
        bg = pygame.Rect(x - 12, y - 4, txt.get_width() + 24, txt.get_height() + 8)
        pygame.draw.rect(self.overlay, (0, 0, 0, 170), bg, border_radius=8)
        self.screen.blit(txt, (x, y))

    # ------------------------------------------------------------------
    # Trajectoires candidates (imagining / planning / predicting)
    # ------------------------------------------------------------------
    def _draw_candidate_trajectories(
        self,
        plan_result: PlanResult,
        revealed_count: int,
        state: str,
        phase_progress: float,
    ) -> None:
        candidates = plan_result.candidates

        if state == "imagining":
            count = min(revealed_count, len(candidates))
            for i in range(count):
                c = candidates[i]
                self._draw_thin_trajectory(c.trajectory, c.color, alpha=100)

        elif state == "planning":
            top_set = set(plan_result.top_k)
            # Pulsation pour le top K
            pulse = int(80 + 80 * math.sin(phase_progress * math.pi * 6))
            for i, c in enumerate(candidates):
                if i in top_set:
                    self._draw_thin_trajectory(
                        c.trajectory, c.color, alpha=min(255, pulse + 100)
                    )
                    # Badge numéroté
                    if len(c.trajectory) > 15:
                        bx, by = c.trajectory[15]
                        rank = plan_result.top_k.index(i) + 1
                        badge = self.font_small.render(
                            f"#{rank}", True, (255, 255, 255)
                        )
                        # Fond du badge
                        br = pygame.Rect(
                            int(bx) + 4, int(by) - 12,
                            badge.get_width() + 6, badge.get_height() + 2,
                        )
                        pygame.draw.rect(
                            self.overlay, (0, 0, 0, 180), br, border_radius=4
                        )
                        self.screen.blit(badge, (int(bx) + 7, int(by) - 11))
                else:
                    self._draw_thin_trajectory(c.trajectory, c.color, alpha=18)

        elif state == "predicting":
            best = candidates[plan_result.best_idx]
            fade = min(1.0, phase_progress * 2.5)
            alpha = int(220 * fade)
            self._draw_thin_trajectory(
                best.trajectory, COLOR_PREDICTION, alpha=alpha
            )

    def _draw_thin_trajectory(
        self,
        traj: list[tuple[float, float]],
        color: tuple[int, int, int],
        alpha: int = 100,
    ) -> None:
        """Dessine une trajectoire sous forme de points espacés (~60 pts)."""
        step = max(1, len(traj) // 60)
        a = max(0, min(255, alpha))
        for i in range(0, len(traj), step):
            px, py = traj[i]
            pygame.draw.circle(
                self.overlay, (*color, a), (int(px), int(py)), 2
            )

    # ------------------------------------------------------------------
    # Observation : lignes d'erreur
    # ------------------------------------------------------------------
    def _draw_observation_comparison(
        self,
        predicted: list[tuple[float, float]] | None,
        actual: list[tuple[float, float]],
        phase_progress: float,
    ) -> None:
        if not predicted or not actual:
            return
        n = min(len(predicted), len(actual))
        if n < 2:
            return
        sample_step = max(1, n // 20)
        show_count = max(1, int(n * phase_progress))
        alpha = int(220 * min(1.0, phase_progress * 1.5))

        for i in range(0, min(show_count, n), sample_step):
            px, py = predicted[i]
            ax, ay = actual[i]
            pygame.draw.line(
                self.overlay, (255, 80, 80, alpha),
                (int(px), int(py)), (int(ax), int(ay)), 2,
            )
            pygame.draw.circle(
                self.overlay, (255, 80, 80, alpha), (int(px), int(py)), 3
            )
            pygame.draw.circle(
                self.overlay, (255, 200, 80, alpha), (int(ax), int(ay)), 3
            )

    # ------------------------------------------------------------------
    # Apprentissage : animation paramètres
    # ------------------------------------------------------------------
    def _draw_learning_animation(
        self,
        old_gx: float, old_gy: float,
        new_gx: float, new_gy: float,
        error: float,
        phase_progress: float,
        errors_history: list[float],
    ) -> None:
        panel_x = TABLE_X + 20
        panel_y = TABLE_Y + TABLE_HEIGHT // 2 - 50
        pw, ph = 310, 100

        # Fond du panel
        pygame.draw.rect(
            self.overlay, (0, 0, 0, 190),
            (panel_x, panel_y, pw, ph), border_radius=10,
        )

        # Titre
        title = self.font.render("Mise \u00e0 jour du mod\u00e8le", True, COLOR_LEARNING)
        self.screen.blit(title, (panel_x + 12, panel_y + 8))

        # Interpolation animée
        t = min(1.0, phase_progress)
        cur_gx = old_gx + (new_gx - old_gx) * t
        cur_gy = old_gy + (new_gy - old_gy) * t

        # Flèche animée pour gx
        gx_txt = f"gx: {old_gx:+.5f}  \u2192  {cur_gx:+.5f}"
        gy_txt = f"gy: {old_gy:+.5f}  \u2192  {cur_gy:+.5f}"
        err_txt = f"erreur: {error:.1f} px"

        t1 = self.font_small.render(gx_txt, True, (220, 220, 220))
        t2 = self.font_small.render(gy_txt, True, (220, 220, 220))
        t3 = self.font_small.render(err_txt, True, COLOR_ERROR)

        self.screen.blit(t1, (panel_x + 12, panel_y + 35))
        self.screen.blit(t2, (panel_x + 12, panel_y + 55))
        self.screen.blit(t3, (panel_x + 12, panel_y + 75))

    # ------------------------------------------------------------------
    # Table / billes / trajectoires (inchangés)
    # ------------------------------------------------------------------
    def _draw_table(self, tilt_x_deg: float, tilt_y_deg: float) -> None:
        inner = pygame.Rect(
            TABLE_X + BAND_THICKNESS, TABLE_Y + BAND_THICKNESS,
            TABLE_WIDTH - 2 * BAND_THICKNESS, TABLE_HEIGHT - 2 * BAND_THICKNESS
        )
        base_g = 100
        pygame.draw.rect(self.screen, (0, base_g, 50), inner)
        if abs(tilt_x_deg) > 0.3:
            for x_offset in range(inner.width):
                t = (x_offset / max(1, inner.width - 1) - 0.5) * tilt_x_deg * 4
                g = int(max(40, min(160, base_g + t)))
                pygame.draw.line(
                    self.screen, (0, g, max(20, 50 - abs(int(tilt_x_deg)))),
                    (inner.x + x_offset, inner.y),
                    (inner.x + x_offset, inner.y + inner.height),
                )
        if abs(tilt_y_deg) > 0.3:
            grad = pygame.Surface((inner.width, inner.height), pygame.SRCALPHA)
            for y_offset in range(inner.height):
                t = (y_offset / max(1, inner.height - 1) - 0.5) * tilt_y_deg * 4
                alpha = int(min(80, max(0, abs(t))))
                color = (0, 60, 0, alpha) if t < 0 else (0, 200, 0, alpha)
                pygame.draw.line(grad, color, (0, y_offset), (inner.width, y_offset))
            self.screen.blit(grad, (inner.x, inner.y))
        pygame.draw.rect(
            self.screen, COLOR_BAND,
            (TABLE_X, TABLE_Y, TABLE_WIDTH, TABLE_HEIGHT),
            BAND_THICKNESS, border_radius=4,
        )

    def _draw_ball(self, x: float, y: float, color: tuple) -> None:
        pygame.draw.circle(self.screen, color, (int(x), int(y)), BALL_RADIUS)
        pygame.draw.circle(
            self.screen, (255, 255, 255), (int(x) - 3, int(y) - 3), 3
        )

    def _draw_trajectories(
        self,
        predicted: list[tuple[float, float]] | None,
        actual: list[tuple[float, float]],
    ) -> None:
        if predicted and len(predicted) > 1:
            for i in range(1, len(predicted), 3):
                px, py = predicted[i]
                alpha = max(30, 120 - i // 4)
                pygame.draw.circle(
                    self.overlay, (*COLOR_PREDICTED[:3], alpha),
                    (int(px), int(py)), 2,
                )
        if len(actual) > 1:
            for i in range(1, len(actual)):
                alpha = max(40, 200 - i // 3)
                pygame.draw.line(
                    self.overlay, (*COLOR_ACTUAL[:3], alpha),
                    (int(actual[i - 1][0]), int(actual[i - 1][1])),
                    (int(actual[i][0]), int(actual[i][1])), 2,
                )

    def _draw_tilt_indicator(self, tilt_x_deg: float, tilt_y_deg: float) -> None:
        if abs(tilt_x_deg) < 0.3 and abs(tilt_y_deg) < 0.3:
            return
        cx = TABLE_X + TABLE_WIDTH // 2
        cy = TABLE_Y + 30
        dx = tilt_x_deg * 3
        dy = tilt_y_deg * 3
        end_x = cx + dx
        end_y = cy + dy
        pygame.draw.line(
            self.screen, COLOR_TILT_ARROW,
            (cx, cy), (int(end_x), int(end_y)), 3,
        )
        length = math.hypot(dx, dy)
        if length > 1:
            ux, uy = dx / length, dy / length
            tip_x, tip_y = int(end_x), int(end_y)
            pygame.draw.polygon(self.screen, COLOR_TILT_ARROW, [
                (tip_x, tip_y),
                (int(tip_x - ux * 8 + uy * 5), int(tip_y - uy * 8 - ux * 5)),
                (int(tip_x - ux * 8 - uy * 5), int(tip_y - uy * 8 + ux * 5)),
            ])
        txt = self.font_small.render(
            f"x:{tilt_x_deg:+.1f}\u00b0 y:{tilt_y_deg:+.1f}\u00b0",
            True, COLOR_TILT_ARROW,
        )
        self.screen.blit(txt, (cx - txt.get_width() // 2, cy + 15))

    # ------------------------------------------------------------------
    # UI basse + mini graphe
    # ------------------------------------------------------------------
    def _draw_ui(
        self,
        wm_error: float,
        wm_shots: int,
        wm_est_gx: float,
        wm_est_gy: float,
        phase_text: str,
        wm_enabled: bool,
        errors_history: list[float],
        *,
        state: str = "aiming",
        phase_progress: float = 0.0,
        update_preview: tuple[float, float, float, float, float] | None = None,
    ) -> None:
        ui_y = TABLE_Y + TABLE_HEIGHT + 10
        pygame.draw.rect(
            self.screen, COLOR_UI_BG,
            (TABLE_X, ui_y, TABLE_WIDTH, 55), border_radius=6,
        )

        color = (
            COLOR_SUCCESS
            if "adapt\u00e9" in phase_text.lower() or "plat" in phase_text.lower()
            else COLOR_TEXT
        )
        txt = self.font_big.render(phase_text, True, color)
        self.screen.blit(txt, (TABLE_X + 15, ui_y + 5))

        err_color = COLOR_ERROR if wm_error > 30 else COLOR_SUCCESS
        info = (
            f"Tirs: {wm_shots}   Erreur: {wm_error:.1f}px"
            f"   Grav. est. x:{wm_est_gx:.4f} y:{wm_est_gy:.4f}"
        )
        txt2 = self.font.render(info, True, err_color)
        self.screen.blit(txt2, (TABLE_X + 15, ui_y + 32))

        mode_txt = "WM: ON (adaptatif)" if wm_enabled else "WM: OFF (fig\u00e9)"
        mode_color = COLOR_SUCCESS if wm_enabled else COLOR_ERROR
        txt3 = self.font.render(mode_txt, True, mode_color)
        self.screen.blit(
            txt3, (TABLE_X + TABLE_WIDTH - txt3.get_width() - 15, ui_y + 5)
        )

        controls = "ESPACE: tir/skip  \u2190\u2192\u2191\u2193: incliner  W: toggle WM  R: reset"
        txt4 = self.font_small.render(controls, True, (140, 140, 140))
        self.screen.blit(
            txt4, (TABLE_X + TABLE_WIDTH - txt4.get_width() - 15, ui_y + 32)
        )

        # Mini graphe d'erreur
        preview_err = update_preview[4] if (
            state == "learning" and update_preview
        ) else None
        if len(errors_history) > 1 or preview_err is not None:
            self._draw_error_graph(
                errors_history, preview_err=preview_err,
                pulse=(state == "learning"), phase_progress=phase_progress,
            )

    def _draw_error_graph(
        self,
        errors: list[float],
        *,
        preview_err: float | None = None,
        pulse: bool = False,
        phase_progress: float = 0.0,
    ) -> None:
        gx = TABLE_X + TABLE_WIDTH - 180
        gy = TABLE_Y + BAND_THICKNESS + 5
        gw, gh = 170, 60
        pygame.draw.rect(
            self.overlay, (0, 0, 0, 100), (gx, gy, gw, gh), border_radius=4
        )
        txt = self.font_small.render(
            "Erreur de pr\u00e9diction", True, (180, 180, 180)
        )
        self.screen.blit(txt, (gx + 5, gy + 2))

        display_errors = list(errors)
        if preview_err is not None:
            display_errors.append(preview_err)

        if len(display_errors) < 2:
            return

        max_err = max(max(display_errors), 1)
        n = min(len(display_errors), 30)
        recent = display_errors[-n:]
        points = []
        for i, e in enumerate(recent):
            px = gx + 5 + int(i * (gw - 10) / max(1, n - 1))
            py = gy + gh - 5 - int((e / max_err) * (gh - 20))
            points.append((px, py))
        if len(points) > 1:
            pygame.draw.lines(
                self.overlay, (100, 180, 255, 200), False, points, 2
            )

        # Ping sur le dernier point pendant learning
        if pulse and len(points) >= 1:
            lx, ly = points[-1]
            ping_r = int(4 + 4 * abs(math.sin(phase_progress * math.pi * 4)))
            pygame.draw.circle(
                self.overlay, (80, 255, 120, 220), (lx, ly), ping_r
            )
