# model.py — Architecture CNN (PyTorch)

import numpy as np
import torch
import torch.nn as nn
from config import IMAGE_SIZE, NUM_OBJECTS, MODEL_PATH


class SpatialReconstructionNet(nn.Module):
    """CNN qui prédit les positions 3D des objets à partir d'une image 2D.

    Input:  (batch, 3, IMAGE_SIZE, IMAGE_SIZE)
    Output: (batch, NUM_OBJECTS * 3)  — positions normalisées [0,1]
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 64 → 32

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 32 → 16

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 16 → 8

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 8 → 4
        )

        # 256 * 4 * 4 = 4096
        flat_size = 256 * (IMAGE_SIZE // 16) ** 2

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_OBJECTS * 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


def load_trained_model(path: str = MODEL_PATH) -> SpatialReconstructionNet:
    """Charge les poids et met le modèle en mode eval."""
    model = SpatialReconstructionNet()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: SpatialReconstructionNet, image_numpy: np.ndarray) -> np.ndarray:
    """Prédit les positions à partir d'une image numpy (H, W, 3) float32 [0,1].

    Retourne un array (NUM_OBJECTS * 3,) float32 [0,1].
    """
    # (H, W, 3) → (1, 3, H, W)
    tensor = torch.from_numpy(image_numpy).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        output = model(tensor)

    return output.squeeze(0).numpy()
