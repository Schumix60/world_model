# demo.py — Démo Pygame interactive — orchestre les 3 phases

import sys
import numpy as np
import pygame
from config import (
    DEMO_WINDOW_WIDTH, DEMO_WINDOW_HEIGHT, DEMO_FPS,
    DISPLAY_IMAGE_SIZE, ROOM_SIZE, NUM_OBJECTS,
    OBJECT_TYPES, OBJECT_COLORS, OBJECT_SIZE,
)
from scene import generate_random_scene, scene_to_position_vector, position_vector_to_objects
from renderer2d import render_scene_high_res, render_scene, get_occlusion_info
from renderer3d import InteractiveRenderer
from model import load_trained_model, predict


class Demo:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((DEMO_WINDOW_WIDTH, DEMO_WINDOW_HEIGHT))
        pygame.display.set_caption("Nano Marble — Reconstruction Spatiale 3D")
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("monospace", 28, bold=True)
        self.font_info = pygame.font.SysFont("monospace", 16)
        self.font_small = pygame.font.SysFont("monospace", 14)

        self.model = load_trained_model()
        self.phase = 1
        self._new_scene()

    def _new_scene(self):
        """Génère une nouvelle scène et ses rendus."""
        self.scene = generate_random_scene()
        self.image_hr = render_scene_high_res(self.scene)
        self.image_lr = render_scene(self.scene)  # 64×64 pour le modèle
        self.image_lr_float = self.image_lr.astype(np.float32) / 255.0

        # Prédiction
        self.pred_vector = predict(self.model, self.image_lr_float)
        self.gt_vector = scene_to_position_vector(self.scene)

        # Objets pour renderer3d
        self.pred_objects = position_vector_to_objects(self.pred_vector)
        self.gt_objects = position_vector_to_objects(self.gt_vector)

        # Occlusion
        self.visibility = get_occlusion_info(self.scene)

        # Surface Pygame de l'image 2D
        self.image_surface = pygame.surfarray.make_surface(
            np.transpose(self.image_hr, (1, 0, 2))
        )

    def _draw_text_centered(self, text, font, color, y):
        surf = font.render(text, True, color)
        x = (DEMO_WINDOW_WIDTH - surf.get_width()) // 2
        self.screen.blit(surf, (x, y))

    def _draw_text(self, text, font, color, pos):
        surf = font.render(text, True, color)
        self.screen.blit(surf, pos)

    def _fade_transition(self, duration_ms=300):
        """Court fondu noir entre les phases."""
        fade = pygame.Surface((DEMO_WINDOW_WIDTH, DEMO_WINDOW_HEIGHT))
        fade.fill((0, 0, 0))
        steps = max(1, duration_ms // 16)
        for i in range(steps):
            alpha = int(255 * i / steps)
            fade.set_alpha(alpha)
            self.screen.blit(fade, (0, 0))
            pygame.display.flip()
            pygame.time.wait(16)

    # ─────────────────────────── Phase 1 ───────────────────────────

    def _phase1(self):
        """Phase 1 : Ce que le modèle voit."""
        self.screen.fill((30, 30, 40))

        # Titre
        self._draw_text_centered(
            "PHASE 1 — Ce que le modèle voit",
            self.font_title, (255, 255, 255), 20
        )

        # Image au centre
        img_x = (DEMO_WINDOW_WIDTH - DISPLAY_IMAGE_SIZE) // 2
        img_y = (DEMO_WINDOW_HEIGHT - DISPLAY_IMAGE_SIZE) // 2 - 10
        self.screen.blit(self.image_surface, (img_x, img_y))

        # Cadre
        pygame.draw.rect(self.screen, (100, 100, 120),
                         (img_x - 2, img_y - 2,
                          DISPLAY_IMAGE_SIZE + 4, DISPLAY_IMAGE_SIZE + 4), 2)

        # Instructions
        self._draw_text_centered(
            "ESPACE → reconstruire  |  N → nouvelle scène  |  ESC → quitter",
            self.font_info, (180, 180, 200), DEMO_WINDOW_HEIGHT - 40
        )

        # Info résolution
        self._draw_text(
            f"Entrée réseau: 64×64 px  |  Affichage: {DISPLAY_IMAGE_SIZE}×{DISPLAY_IMAGE_SIZE} px",
            self.font_small, (120, 120, 140), (img_x, img_y + DISPLAY_IMAGE_SIZE + 10)
        )

    # ─────────────────────────── Phase 2 ───────────────────────────

    def _phase2(self):
        """Phase 2 : Ce que le modèle reconstruit (split-screen)."""
        self.screen.fill((30, 30, 40))

        # Titre
        self._draw_text_centered(
            "PHASE 2 — Reconstruction 3D",
            self.font_title, (255, 255, 255), 10
        )

        half_w = DEMO_WINDOW_WIDTH // 2

        # --- Gauche : image 2D ---
        img_size = min(DISPLAY_IMAGE_SIZE, half_w - 40)
        img_x = (half_w - img_size) // 2
        img_y = 60

        # Redimensionner si nécessaire
        if img_size != DISPLAY_IMAGE_SIZE:
            img_surf = pygame.transform.scale(self.image_surface, (img_size, img_size))
        else:
            img_surf = self.image_surface

        self.screen.blit(img_surf, (img_x, img_y))
        pygame.draw.rect(self.screen, (100, 100, 120),
                         (img_x - 2, img_y - 2, img_size + 4, img_size + 4), 2)

        self._draw_text("Image 2D (entrée)", self.font_small,
                        (180, 180, 200), (img_x, img_y + img_size + 8))

        # --- Droite : vue de dessus ---
        map_size = min(400, half_w - 40)
        map_x = half_w + (half_w - map_size) // 2
        map_y = 60
        self._draw_top_view(map_x, map_y, map_size)

        # --- Erreurs ---
        error_y = map_y + map_size + 30
        obj_names = ["Cube rouge", "Sphère bleue", "Cylindre vert"]
        total_err = 0.0

        for i in range(NUM_OBJECTS):
            pred = np.array(self.pred_objects[i]["position"])
            gt = np.array(self.gt_objects[i]["position"])
            err = np.linalg.norm(pred - gt)
            total_err += err

            vis_str = "visible" if self.visibility[i] else "OCCULTÉ"
            color = (180, 180, 200) if self.visibility[i] else (255, 200, 80)
            self._draw_text(
                f"{obj_names[i]:20s}  err: {err:.2f}u  [{vis_str}]",
                self.font_small, color, (half_w + 20, error_y + i * 22)
            )

        self._draw_text(
            f"Erreur totale: {total_err:.2f}u  |  Moyenne: {total_err/NUM_OBJECTS:.2f}u",
            self.font_info, (255, 255, 255),
            (half_w + 20, error_y + NUM_OBJECTS * 22 + 10)
        )

        # Instructions
        self._draw_text_centered(
            "ESPACE → explorer le monde  |  N → nouvelle scène  |  BACKSPACE → retour",
            self.font_info, (180, 180, 200), DEMO_WINDOW_HEIGHT - 40
        )

    def _draw_top_view(self, x, y, size):
        """Dessine la vue de dessus (plan x-z) de la pièce."""
        # Fond de la pièce
        pygame.draw.rect(self.screen, (60, 60, 70), (x, y, size, size))
        pygame.draw.rect(self.screen, (100, 100, 120), (x, y, size, size), 2)

        # Grille
        for i in range(1, 10):
            gx = x + int(i / ROOM_SIZE * size)
            gy = y + int(i / ROOM_SIZE * size)
            pygame.draw.line(self.screen, (70, 70, 80), (gx, y), (gx, y + size), 1)
            pygame.draw.line(self.screen, (70, 70, 80), (x, gy), (x + size, gy), 1)

        scale = size / ROOM_SIZE
        obj_names = ["Cube", "Sphère", "Cylindre"]

        # Ground truth (contours pointillés)
        for i, obj in enumerate(self.gt_objects):
            ox = x + int(obj["position"][0] * scale)
            oz = y + int(obj["position"][2] * scale)
            r = int(OBJECT_SIZE * scale * 0.8)
            color = obj["color"]
            pygame.draw.circle(self.screen, color, (ox, oz), r, 2)

        # Prédictions (formes pleines)
        for i, obj in enumerate(self.pred_objects):
            ox = x + int(obj["position"][0] * scale)
            oz = y + int(obj["position"][2] * scale)
            r = int(OBJECT_SIZE * scale * 0.6)
            color = obj["color"]

            if obj["type"] == "cube":
                pygame.draw.rect(self.screen, color,
                                 (ox - r, oz - r, r * 2, r * 2))
            elif obj["type"] == "sphere":
                pygame.draw.circle(self.screen, color, (ox, oz), r)
            elif obj["type"] == "cylinder":
                pygame.draw.rect(self.screen, color,
                                 (ox - int(r * 0.7), oz - r,
                                  int(r * 1.4), r * 2))

            # Flèche entre prédit et ground truth
            gt = self.gt_objects[i]
            gx = x + int(gt["position"][0] * scale)
            gz = y + int(gt["position"][2] * scale)
            pygame.draw.line(self.screen, (255, 255, 100),
                             (ox, oz), (gx, gz), 1)

            # Mise en évidence des objets occultés
            if not self.visibility[i]:
                self._draw_occluded_badge(ox, oz, r, obj["color"])

        # Légende
        self._draw_text("Vue de dessus (X-Z)", self.font_small,
                        (180, 180, 200), (x, y + size + 8))
        self._draw_text("Plein = prédit | Contour = réel | Jaune = erreur",
                        self.font_small, (140, 140, 160), (x, y + size + 24))

    def _draw_occluded_badge(self, cx, cy, radius, color):
        """Badge clignotant pour les objets occultés."""
        tick = pygame.time.get_ticks()
        if (tick // 400) % 2 == 0:
            # Étoile / indicateur
            r = radius + 8
            pygame.draw.circle(self.screen, (255, 200, 50), (cx, cy), r, 3)

            badge_text = "INVISIBLE"
            surf = self.font_small.render(badge_text, True, (255, 200, 50))
            self.screen.blit(surf, (cx - surf.get_width() // 2, cy - radius - 22))

    # ─────────────────────────── Phase 3 ───────────────────────────

    def _launch_phase3(self):
        """Lance l'exploration 3D interactive."""
        self._fade_transition(200)

        renderer = InteractiveRenderer(
            predicted_objects=self.pred_objects,
            ground_truth_objects=self.gt_objects,
        )
        result = renderer.run()

        # Restaurer l'écran de la démo
        self.screen = pygame.display.set_mode((DEMO_WINDOW_WIDTH, DEMO_WINDOW_HEIGHT))
        pygame.display.set_caption("Nano Marble — Reconstruction Spatiale 3D")

        if result == "quit":
            return False
        return True  # retour en phase 2

    # ─────────────────────────── Boucle principale ───────────────────────────

    def run(self):
        running = True
        while running:
            self.clock.tick(DEMO_FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_n:
                        self._new_scene()

                    elif event.key == pygame.K_SPACE:
                        if self.phase == 1:
                            self._fade_transition(200)
                            self.phase = 2
                        elif self.phase == 2:
                            if not self._launch_phase3():
                                running = False
                            else:
                                self.phase = 2

                    elif event.key == pygame.K_BACKSPACE:
                        if self.phase == 2:
                            self._fade_transition(200)
                            self.phase = 1

                    elif event.key == pygame.K_1:
                        self.phase = 1
                    elif event.key == pygame.K_2:
                        self.phase = 2
                    elif event.key == pygame.K_3:
                        if not self._launch_phase3():
                            running = False
                        else:
                            self.phase = 2

            if self.phase == 1:
                self._phase1()
            elif self.phase == 2:
                self._phase2()

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    demo = Demo()
    demo.run()
