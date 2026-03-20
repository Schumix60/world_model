# scene.py — Définition et génération de scènes 3D aléatoires

from dataclasses import dataclass
import numpy as np
from config import (
    ROOM_SIZE, NUM_OBJECTS, OBJECT_TYPES, OBJECT_COLORS,
    OBJECT_SIZE, OBJECT_MIN_POS, OBJECT_MAX_POS, MIN_OBJECT_DISTANCE,
)


@dataclass
class SceneObject:
    obj_type: str       # "cube", "sphere", "cylinder"
    color: tuple        # RGB tuple
    position: np.ndarray  # (x, y, z) — centre de l'objet
    size: float         # rayon ou demi-côté


@dataclass
class Scene:
    room_size: float
    objects: list  # list[SceneObject]


def generate_random_scene() -> Scene:
    """Génère une scène aléatoire avec NUM_OBJECTS objets bien séparés."""
    objects = []
    max_attempts = 1000

    for i in range(NUM_OBJECTS):
        obj_type = OBJECT_TYPES[i]
        color = OBJECT_COLORS[i]

        for attempt in range(max_attempts):
            x = np.random.uniform(OBJECT_MIN_POS, OBJECT_MAX_POS)
            z = np.random.uniform(OBJECT_MIN_POS, OBJECT_MAX_POS)
            y = OBJECT_SIZE  # objets posés au sol (y=0), centre à y=OBJECT_SIZE

            position = np.array([x, y, z], dtype=np.float32)

            # Vérifier la distance minimale avec les objets déjà placés
            valid = True
            for other in objects:
                dist = np.linalg.norm(position[[0, 2]] - other.position[[0, 2]])
                if dist < MIN_OBJECT_DISTANCE:
                    valid = False
                    break

            if valid:
                objects.append(SceneObject(
                    obj_type=obj_type,
                    color=color,
                    position=position,
                    size=OBJECT_SIZE,
                ))
                break
        else:
            raise RuntimeError(
                f"Impossible de placer l'objet {i} après {max_attempts} tentatives. "
                "Vérifier MIN_OBJECT_DISTANCE vs la zone disponible."
            )

    return Scene(room_size=ROOM_SIZE, objects=objects)


def scene_to_position_vector(scene: Scene) -> np.ndarray:
    """Convertit une scène en vecteur de positions normalisé [0,1].

    Retourne un array de taille NUM_OBJECTS * 3 (x,y,z pour chaque objet).
    """
    positions = []
    for obj in scene.objects:
        positions.extend(obj.position / ROOM_SIZE)
    return np.array(positions, dtype=np.float32)


def position_vector_to_objects(vector: np.ndarray) -> list[dict]:
    """Convertit un vecteur de positions normalisé en liste de dicts pour renderer3d.

    Dénormalise et reconstruit les dicts {type, color, position, size}.
    """
    vector = vector * ROOM_SIZE
    objects = []
    for i in range(NUM_OBJECTS):
        x = float(vector[i * 3])
        y = float(vector[i * 3 + 1])
        z = float(vector[i * 3 + 2])
        objects.append({
            "type": OBJECT_TYPES[i],
            "color": OBJECT_COLORS[i],
            "position": (x, y, z),
            "size": OBJECT_SIZE,
        })
    return objects
