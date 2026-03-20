# renderer2d.py — Projection perspective : scène 3D → image 2D (numpy array)

import numpy as np
from PIL import Image, ImageDraw
from scene import Scene
from config import (
    CAMERA_POSITION, CAMERA_TARGET, CAMERA_FOV, IMAGE_SIZE, DISPLAY_IMAGE_SIZE,
)


def _look_at(eye, target, up=np.array([0.0, 1.0, 0.0])):
    """Construit la matrice de vue (lookAt) 4×4."""
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    cam_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = cam_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(cam_up, eye)
    view[2, 3] = np.dot(forward, eye)

    return view


def _perspective_matrix(fov_deg, aspect, near=0.1, far=100.0):
    """Construit la matrice de projection perspective."""
    fov_rad = np.radians(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)

    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0

    return proj


def _project_point(point_3d, view_matrix, proj_matrix, img_size):
    """Projette un point 3D en coordonnées écran 2D.

    Retourne (screen_x, screen_y, depth) ou None si le point est derrière la caméra.
    """
    p = np.array([*point_3d, 1.0], dtype=np.float64)

    # Appliquer vue
    p_view = view_matrix @ p
    depth = -p_view[2]  # profondeur (positive = devant la caméra)

    if depth <= 0.1:
        return None

    # Appliquer projection
    p_proj = proj_matrix @ p_view

    # Division perspective
    if abs(p_proj[3]) < 1e-10:
        return None

    ndc_x = p_proj[0] / p_proj[3]
    ndc_y = p_proj[1] / p_proj[3]

    # NDC [-1,1] → screen [0, img_size]
    screen_x = (ndc_x + 1.0) * 0.5 * img_size
    screen_y = (1.0 - ndc_y) * 0.5 * img_size  # Y inversé

    return (screen_x, screen_y, depth)


def _apparent_size(obj_size, depth, fov_deg, img_size):
    """Calcule la taille apparente d'un objet en pixels."""
    fov_rad = np.radians(fov_deg)
    # Taille en pixels proportionnelle à la taille réelle / distance
    pixel_size = (obj_size / depth) * (img_size / (2.0 * np.tan(fov_rad / 2.0)))
    return max(pixel_size, 1.0)


def _draw_room_wireframe(draw, view_matrix, proj_matrix, img_size, room_size):
    """Dessine les arêtes de la pièce en perspective (wireframe)."""
    s = room_size
    # Les 8 coins de la pièce
    corners = [
        (0, 0, 0), (s, 0, 0), (s, 0, s), (0, 0, s),  # sol
        (0, s, 0), (s, s, 0), (s, s, s), (0, s, s),    # plafond
    ]

    # Arêtes : sol, plafond, et piliers verticaux
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # sol
        (4, 5), (5, 6), (6, 7), (7, 4),  # plafond
        (0, 4), (1, 5), (2, 6), (3, 7),  # piliers
    ]

    projected = []
    for c in corners:
        p = _project_point(c, view_matrix, proj_matrix, img_size)
        projected.append(p)

    line_color = (180, 180, 180)
    for i, j in edges:
        p1 = projected[i]
        p2 = projected[j]
        if p1 is not None and p2 is not None:
            draw.line(
                [(p1[0], p1[1]), (p2[0], p2[1])],
                fill=line_color, width=max(1, img_size // 64),
            )

    # Dessiner le sol en remplissage pâle
    sol_corners = [projected[i] for i in [0, 1, 2, 3]]
    if all(c is not None for c in sol_corners):
        poly = [(c[0], c[1]) for c in sol_corners]
        draw.polygon(poly, fill=(230, 230, 225))


def render_scene(scene: Scene, size: int = IMAGE_SIZE) -> np.ndarray:
    """Rend une scène 3D en image 2D numpy array (H, W, 3) uint8."""
    view_matrix = _look_at(CAMERA_POSITION, CAMERA_TARGET)
    proj_matrix = _perspective_matrix(CAMERA_FOV, 1.0)

    # Créer l'image avec fond gris clair
    img = Image.new("RGB", (size, size), (210, 210, 210))
    draw = ImageDraw.Draw(img)

    # Dessiner le sol et les murs
    _draw_room_wireframe(draw, view_matrix, proj_matrix, size, scene.room_size)

    # Projeter et trier les objets par profondeur (painter's algorithm)
    obj_projections = []
    for obj in scene.objects:
        proj = _project_point(obj.position, view_matrix, proj_matrix, size)
        if proj is not None:
            app_size = _apparent_size(obj.size, proj[2], CAMERA_FOV, size)
            obj_projections.append((obj, proj, app_size))

    # Trier du plus loin au plus proche (dessiner les plus lointains d'abord)
    obj_projections.sort(key=lambda x: -x[1][2])

    for obj, (sx, sy, depth), app_size in obj_projections:
        color = obj.color

        if obj.obj_type == "cube":
            half = app_size
            draw.rectangle(
                [sx - half, sy - half, sx + half, sy + half],
                fill=color,
                outline=tuple(max(0, c - 40) for c in color),
                width=max(1, size // 64),
            )

        elif obj.obj_type == "sphere":
            radius = app_size
            draw.ellipse(
                [sx - radius, sy - radius, sx + radius, sy + radius],
                fill=color,
                outline=tuple(max(0, c - 40) for c in color),
                width=max(1, size // 64),
            )

        elif obj.obj_type == "cylinder":
            # Rectangle vertical avec demi-cercles en haut et bas
            half_w = app_size * 0.7
            half_h = app_size * 1.2
            # Corps rectangulaire
            draw.rectangle(
                [sx - half_w, sy - half_h, sx + half_w, sy + half_h],
                fill=color,
            )
            # Demi-cercle haut
            draw.ellipse(
                [sx - half_w, sy - half_h - half_w * 0.5,
                 sx + half_w, sy - half_h + half_w * 0.5],
                fill=color,
            )
            # Demi-cercle bas
            draw.ellipse(
                [sx - half_w, sy + half_h - half_w * 0.5,
                 sx + half_w, sy + half_h + half_w * 0.5],
                fill=color,
            )
            # Contour
            outline_color = tuple(max(0, c - 40) for c in color)
            draw.ellipse(
                [sx - half_w, sy - half_h - half_w * 0.5,
                 sx + half_w, sy - half_h + half_w * 0.5],
                outline=outline_color, width=max(1, size // 64),
            )
            draw.line(
                [(sx - half_w, sy - half_h), (sx - half_w, sy + half_h)],
                fill=outline_color, width=max(1, size // 64),
            )
            draw.line(
                [(sx + half_w, sy - half_h), (sx + half_w, sy + half_h)],
                fill=outline_color, width=max(1, size // 64),
            )
            draw.ellipse(
                [sx - half_w, sy + half_h - half_w * 0.5,
                 sx + half_w, sy + half_h + half_w * 0.5],
                outline=outline_color, width=max(1, size // 64),
            )

    return np.array(img, dtype=np.uint8)


def render_scene_high_res(scene: Scene, size: int = DISPLAY_IMAGE_SIZE) -> np.ndarray:
    """Rendu haute résolution pour l'affichage dans la démo."""
    return render_scene(scene, size=size)


def get_occlusion_info(scene: Scene) -> list[bool]:
    """Détermine la visibilité de chaque objet.

    Retourne une liste de booléens : True = visible, False = occulté.
    """
    view_matrix = _look_at(CAMERA_POSITION, CAMERA_TARGET)
    proj_matrix = _perspective_matrix(CAMERA_FOV, 1.0)

    projections = []
    for obj in scene.objects:
        proj = _project_point(obj.position, view_matrix, proj_matrix, IMAGE_SIZE)
        if proj is not None:
            app_size = _apparent_size(obj.size, proj[2], CAMERA_FOV, IMAGE_SIZE)
            projections.append((proj[0], proj[1], proj[2], app_size))
        else:
            projections.append(None)

    visible = [True] * len(scene.objects)

    for i in range(len(scene.objects)):
        if projections[i] is None:
            visible[i] = False
            continue

        sx_i, sy_i, depth_i, size_i = projections[i]

        for j in range(len(scene.objects)):
            if i == j or projections[j] is None:
                continue

            sx_j, sy_j, depth_j, size_j = projections[j]

            # L'objet j est-il devant l'objet i ?
            if depth_j >= depth_i:
                continue

            # L'objet j recouvre-t-il entièrement l'objet i en 2D ?
            dist_2d = np.sqrt((sx_i - sx_j) ** 2 + (sy_i - sy_j) ** 2)
            if dist_2d + size_i <= size_j * 1.1:  # marge de 10%
                visible[i] = False
                break

    return visible
