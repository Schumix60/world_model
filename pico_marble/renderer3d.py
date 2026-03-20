# renderer3d.py — Rendu 3D interactif navigable (Pygame, caméra libre)
# NE dépend PAS de scene.py — reçoit des listes de dicts {type, color, position, size}

import math
import numpy as np
import pygame
from config import DEMO_WINDOW_WIDTH, DEMO_WINDOW_HEIGHT, DEMO_FPS, ROOM_SIZE, CAMERA_POSITION


def _look_at_matrix(eye, target, up=np.array([0.0, 1.0, 0.0])):
    """Matrice de vue lookAt."""
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        norm = np.linalg.norm(right)
    right = right / norm
    cam_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = cam_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(cam_up, eye)
    view[2, 3] = np.dot(forward, eye)
    return view


def _perspective_proj(fov_deg, aspect, near=0.1, far=100.0):
    """Matrice de projection perspective."""
    fov_rad = math.radians(fov_deg)
    f = 1.0 / math.tan(fov_rad / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def _project(point, view, proj, w, h):
    """Projette un point 3D → écran 2D. Retourne (sx, sy, depth) ou None."""
    p = np.array([*point, 1.0], dtype=np.float64)
    pv = view @ p
    depth = -pv[2]
    if depth <= 0.1:
        return None
    pp = proj @ pv
    if abs(pp[3]) < 1e-10:
        return None
    nx = pp[0] / pp[3]
    ny = pp[1] / pp[3]
    sx = (nx + 1.0) * 0.5 * w
    sy = (1.0 - ny) * 0.5 * h
    return (sx, sy, depth)


def _cube_vertices(cx, cy, cz, half):
    """8 sommets d'un cube centré en (cx, cy, cz)."""
    return [
        (cx - half, cy - half, cz - half),
        (cx + half, cy - half, cz - half),
        (cx + half, cy - half, cz + half),
        (cx - half, cy - half, cz + half),
        (cx - half, cy + half, cz - half),
        (cx + half, cy + half, cz - half),
        (cx + half, cy + half, cz + half),
        (cx - half, cy + half, cz + half),
    ]


# Faces d'un cube (indices des sommets) avec normales sortantes approx.
_CUBE_FACES = [
    (0, 1, 2, 3),  # bas
    (4, 5, 6, 7),  # haut
    (0, 1, 5, 4),  # avant
    (2, 3, 7, 6),  # arrière
    (0, 3, 7, 4),  # gauche
    (1, 2, 6, 5),  # droite
]


class InteractiveRenderer:
    def __init__(self, predicted_objects: list[dict],
                 ground_truth_objects: list[dict] = None):
        self.predicted = predicted_objects
        self.ground_truth = ground_truth_objects

        # Caméra — démarre à la position d'observation 2D
        self.cam_pos = np.array(CAMERA_POSITION, dtype=np.float64)
        # Calculer yaw/pitch initiaux pour regarder vers le centre de la pièce
        target = np.array([5.0, 2.0, 5.0])
        diff = target - self.cam_pos
        self.yaw = math.atan2(diff[0], diff[2])
        self.pitch = math.atan2(diff[1], math.sqrt(diff[0]**2 + diff[2]**2))

        self.move_speed = 5.0
        self.mouse_sensitivity = 0.003
        self.fov = 70

    def _get_forward_right(self):
        """Vecteurs forward et right depuis yaw/pitch."""
        fx = math.sin(self.yaw) * math.cos(self.pitch)
        fy = math.sin(self.pitch)
        fz = math.cos(self.yaw) * math.cos(self.pitch)
        forward = np.array([fx, fy, fz])
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        norm = np.linalg.norm(right)
        if norm > 1e-6:
            right = right / norm
        return forward, right

    def _build_view(self):
        forward, _ = self._get_forward_right()
        target = self.cam_pos + forward
        return _look_at_matrix(self.cam_pos, target)

    def _draw_room(self, surface, view, proj, w, h):
        """Dessine la pièce en wireframe."""
        s = ROOM_SIZE
        corners = [
            (0, 0, 0), (s, 0, 0), (s, 0, s), (0, 0, s),
            (0, s, 0), (s, s, 0), (s, s, s), (0, s, s),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        projected = [_project(c, view, proj, w, h) for c in corners]

        # Sol semi-transparent
        sol = [projected[i] for i in [0, 1, 2, 3]]
        if all(p is not None for p in sol):
            poly = [(int(p[0]), int(p[1])) for p in sol]
            sol_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.polygon(sol_surf, (200, 200, 195, 80), poly)
            surface.blit(sol_surf, (0, 0))

        # Arêtes
        for i, j in edges:
            p1, p2 = projected[i], projected[j]
            if p1 is not None and p2 is not None:
                pygame.draw.line(surface, (160, 160, 160),
                                 (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])), 1)

        # Grille au sol
        grid_color = (190, 190, 185)
        for g in range(1, int(s)):
            p1 = _project((g, 0, 0), view, proj, w, h)
            p2 = _project((g, 0, s), view, proj, w, h)
            if p1 and p2:
                pygame.draw.line(surface, grid_color,
                                 (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])), 1)
            p1 = _project((0, 0, g), view, proj, w, h)
            p2 = _project((s, 0, g), view, proj, w, h)
            if p1 and p2:
                pygame.draw.line(surface, grid_color,
                                 (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])), 1)

    def _draw_object(self, surface, obj, view, proj, w, h, ghost=False):
        """Dessine un objet 3D projeté."""
        pos = obj["position"]
        color = obj["color"]
        size = obj["size"]
        obj_type = obj["type"]

        p = _project(pos, view, proj, w, h)
        if p is None:
            return None  # pas visible

        sx, sy, depth = p
        # Taille apparente
        fov_rad = math.radians(self.fov)
        pix_size = (size / depth) * (h / (2.0 * math.tan(fov_rad / 2.0)))
        pix_size = max(pix_size, 2)

        if ghost:
            # Contour pointillé pour ground truth
            alpha_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            ghost_color = (*color, 80)

            if obj_type == "cube":
                half = size
                verts = _cube_vertices(*pos, half)
                # Dessiner les arêtes du cube
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ]
                proj_verts = [_project(v, view, proj, w, h) for v in verts]
                for i, j in edges:
                    p1, p2 = proj_verts[i], proj_verts[j]
                    if p1 and p2:
                        _draw_dashed_line(alpha_surf, (*color, 150),
                                          (int(p1[0]), int(p1[1])),
                                          (int(p2[0]), int(p2[1])), 2)
            elif obj_type == "sphere":
                pygame.draw.circle(alpha_surf, ghost_color,
                                   (int(sx), int(sy)), int(pix_size), 2)
            elif obj_type == "cylinder":
                rect = pygame.Rect(int(sx - pix_size * 0.7),
                                   int(sy - pix_size * 1.2),
                                   int(pix_size * 1.4),
                                   int(pix_size * 2.4))
                pygame.draw.rect(alpha_surf, ghost_color, rect, 2)
            surface.blit(alpha_surf, (0, 0))
            return depth
        else:
            if obj_type == "cube":
                half = size
                verts = _cube_vertices(*pos, half)
                proj_verts = [_project(v, view, proj, w, h) for v in verts]
                # Dessiner les faces triées par profondeur
                face_list = []
                for face_indices in _CUBE_FACES:
                    face_projs = [proj_verts[fi] for fi in face_indices]
                    if all(fp is not None for fp in face_projs):
                        avg_depth = sum(fp[2] for fp in face_projs) / 4
                        face_list.append((avg_depth, face_projs, color))
                face_list.sort(key=lambda x: -x[0])
                for _, fps, c in face_list:
                    poly = [(int(fp[0]), int(fp[1])) for fp in fps]
                    shade = tuple(max(0, min(255, ci)) for ci in c)
                    pygame.draw.polygon(surface, shade, poly)
                    pygame.draw.polygon(surface, tuple(max(0, ci - 40) for ci in c), poly, 2)

            elif obj_type == "sphere":
                pygame.draw.circle(surface, color, (int(sx), int(sy)), int(pix_size))
                pygame.draw.circle(surface, tuple(max(0, c - 40) for c in color),
                                   (int(sx), int(sy)), int(pix_size), 2)

            elif obj_type == "cylinder":
                rect = pygame.Rect(int(sx - pix_size * 0.7),
                                   int(sy - pix_size * 1.2),
                                   int(pix_size * 1.4),
                                   int(pix_size * 2.4))
                pygame.draw.rect(surface, color, rect)
                pygame.draw.ellipse(surface, color,
                                    pygame.Rect(rect.left, rect.top - int(pix_size * 0.3),
                                                rect.width, int(pix_size * 0.6)))
                pygame.draw.ellipse(surface, color,
                                    pygame.Rect(rect.left, rect.bottom - int(pix_size * 0.3),
                                                rect.width, int(pix_size * 0.6)))
                pygame.draw.rect(surface, tuple(max(0, c - 40) for c in color), rect, 2)

            return depth

    def _draw_objects_sorted(self, surface, view, proj, w, h):
        """Dessine tous les objets triés par profondeur."""
        draw_items = []

        for obj in self.predicted:
            p = _project(obj["position"], view, proj, w, h)
            if p:
                draw_items.append((p[2], obj, False))

        if self.ground_truth:
            for obj in self.ground_truth:
                p = _project(obj["position"], view, proj, w, h)
                if p:
                    draw_items.append((p[2], obj, True))

        # Trier du plus loin au plus proche
        draw_items.sort(key=lambda x: -x[0])

        for _, obj, ghost in draw_items:
            self._draw_object(surface, obj, view, proj, w, h, ghost=ghost)

    def _draw_hud(self, surface, font):
        """Dessine le HUD."""
        texts = [
            "PHASE 3 — Exploration libre",
            f"Caméra: ({self.cam_pos[0]:.1f}, {self.cam_pos[1]:.1f}, {self.cam_pos[2]:.1f})",
            "ZQSD/WASD = déplacer | Souris = regarder | Espace/Shift = haut/bas",
            "BACKSPACE = retour | ESC = quitter",
        ]
        y = 10
        for txt in texts:
            text_surf = font.render(txt, True, (255, 255, 255))
            bg = pygame.Surface((text_surf.get_width() + 10, text_surf.get_height() + 4),
                                pygame.SRCALPHA)
            bg.fill((0, 0, 0, 150))
            surface.blit(bg, (8, y - 2))
            surface.blit(text_surf, (12, y))
            y += text_surf.get_height() + 6

    def run(self) -> str:
        """Lance la boucle Pygame. Retourne 'back' ou 'quit'."""
        pygame.init()
        screen = pygame.display.set_mode((DEMO_WINDOW_WIDTH, DEMO_WINDOW_HEIGHT))
        pygame.display.set_caption("Nano Marble — Exploration 3D")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 16)

        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

        aspect = DEMO_WINDOW_WIDTH / DEMO_WINDOW_HEIGHT
        proj = _perspective_proj(self.fov, aspect)

        running = True
        result = "quit"

        while running:
            dt = clock.tick(DEMO_FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    result = "quit"
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        result = "quit"
                    elif event.key == pygame.K_BACKSPACE:
                        running = False
                        result = "back"
                elif event.type == pygame.MOUSEMOTION:
                    dx, dy = event.rel
                    self.yaw += dx * self.mouse_sensitivity
                    self.pitch -= dy * self.mouse_sensitivity
                    self.pitch = max(-math.pi / 2 + 0.1,
                                     min(math.pi / 2 - 0.1, self.pitch))

            # Déplacement clavier
            keys = pygame.key.get_pressed()
            forward, right = self._get_forward_right()
            move = np.zeros(3)

            if keys[pygame.K_w] or keys[pygame.K_z]:
                move += forward
            if keys[pygame.K_s]:
                move -= forward
            if keys[pygame.K_a] or keys[pygame.K_q]:
                move -= right
            if keys[pygame.K_d]:
                move += right
            if keys[pygame.K_SPACE]:
                move[1] += 1.0
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                move[1] -= 1.0

            norm = np.linalg.norm(move)
            if norm > 0:
                move = move / norm * self.move_speed * dt
                self.cam_pos += move

            # Rendu
            screen.fill((40, 40, 50))
            view = self._build_view()
            self._draw_room(screen, view, proj, DEMO_WINDOW_WIDTH, DEMO_WINDOW_HEIGHT)
            self._draw_objects_sorted(screen, view, proj,
                                      DEMO_WINDOW_WIDTH, DEMO_WINDOW_HEIGHT)
            self._draw_hud(screen, font)

            pygame.display.flip()

        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        return result


def _draw_dashed_line(surface, color, start, end, width=1, dash_len=8, gap_len=4):
    """Dessine une ligne pointillée."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1:
        return
    dx /= dist
    dy /= dist

    pos = 0
    drawing = True
    while pos < dist:
        seg_len = dash_len if drawing else gap_len
        seg_end = min(pos + seg_len, dist)
        if drawing:
            p1 = (int(start[0] + dx * pos), int(start[1] + dy * pos))
            p2 = (int(start[0] + dx * seg_end), int(start[1] + dy * seg_end))
            pygame.draw.line(surface, color, p1, p2, width)
        pos = seg_end
        drawing = not drawing
