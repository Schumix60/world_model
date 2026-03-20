# dataset.py — Génération et sauvegarde du dataset d'entraînement

import os
import numpy as np
from tqdm import tqdm
from scene import generate_random_scene, scene_to_position_vector
from renderer2d import render_scene
from config import IMAGE_SIZE, TRAIN_SIZE, VAL_SIZE, DATASET_DIR


def generate_dataset(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Génère un dataset de paires (image, vecteur de positions).

    Retourne:
        images: shape (n_samples, IMAGE_SIZE, IMAGE_SIZE, 3), float32, [0,1]
        positions: shape (n_samples, NUM_OBJECTS * 3), float32, [0,1]
    """
    images = np.zeros((n_samples, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    positions = np.zeros((n_samples, 9), dtype=np.float32)

    for i in tqdm(range(n_samples), desc="Génération du dataset"):
        scene = generate_random_scene()
        img = render_scene(scene)  # uint8
        pos = scene_to_position_vector(scene)

        images[i] = img.astype(np.float32) / 255.0
        positions[i] = pos

    return images, positions


def save_dataset(images: np.ndarray, positions: np.ndarray, split: str):
    """Sauvegarde un dataset sur disque."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    np.save(os.path.join(DATASET_DIR, f"{split}_images.npy"), images)
    np.save(os.path.join(DATASET_DIR, f"{split}_positions.npy"), positions)
    print(f"Dataset '{split}' sauvegardé: {images.shape[0]} samples")


def load_dataset(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Charge un dataset depuis le disque."""
    images = np.load(os.path.join(DATASET_DIR, f"{split}_images.npy"))
    positions = np.load(os.path.join(DATASET_DIR, f"{split}_positions.npy"))
    print(f"Dataset '{split}' chargé: {images.shape[0]} samples")
    return images, positions


if __name__ == "__main__":
    print(f"Génération du dataset d'entraînement ({TRAIN_SIZE} scènes)...")
    train_images, train_positions = generate_dataset(TRAIN_SIZE)
    save_dataset(train_images, train_positions, "train")

    print(f"\nGénération du dataset de validation ({VAL_SIZE} scènes)...")
    val_images, val_positions = generate_dataset(VAL_SIZE)
    save_dataset(val_images, val_positions, "val")

    print("\nTerminé.")
