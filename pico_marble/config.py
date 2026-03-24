# config.py — Hyperparamètres et constantes (source unique de vérité)

import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Scène ---
ROOM_SIZE = 10.0
NUM_OBJECTS = 3
OBJECT_TYPES = ["cube", "sphere", "cylinder"]
OBJECT_COLORS = [(220, 60, 60), (60, 100, 220), (60, 180, 80)]  # Rouge, Bleu, Vert
OBJECT_SIZE = 1.2
OBJECT_MIN_POS = 1.5
OBJECT_MAX_POS = 8.5
MIN_OBJECT_DISTANCE = 2.5

# --- Caméra d'observation (renderer2d) ---
CAMERA_POSITION = (-2.0, 12.0, -2.0)
CAMERA_TARGET = (5.0, 2.0, 5.0)
CAMERA_FOV = 60
IMAGE_SIZE = 64

# --- Dataset ---
TRAIN_SIZE = 50000
VAL_SIZE = 1000
DATASET_DIR = os.path.join(_BASE_DIR, "data")

# --- Réseau ---
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
EPOCHS = 50
MODEL_PATH = os.path.join(_BASE_DIR, "weights", "model.pth")

# --- Démo ---
DEMO_WINDOW_WIDTH = 1200
DEMO_WINDOW_HEIGHT = 700
DEMO_FPS = 60
DISPLAY_IMAGE_SIZE = 256
