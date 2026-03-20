# train.py — Boucle d'entraînement + sauvegarde des poids

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import (
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, EPOCHS, MODEL_PATH, ROOM_SIZE,
    NUM_OBJECTS, TRAIN_SIZE, VAL_SIZE,
)
from model import SpatialReconstructionNet
from dataset import load_dataset, generate_dataset, save_dataset


def main():
    # Charger ou générer le dataset
    try:
        train_images, train_positions = load_dataset("train")
    except FileNotFoundError:
        print(f"Dataset train non trouvé. Génération ({TRAIN_SIZE} scènes)...")
        train_images, train_positions = generate_dataset(TRAIN_SIZE)
        save_dataset(train_images, train_positions, "train")

    try:
        val_images, val_positions = load_dataset("val")
    except FileNotFoundError:
        print(f"Dataset val non trouvé. Génération ({VAL_SIZE} scènes)...")
        val_images, val_positions = generate_dataset(VAL_SIZE)
        save_dataset(val_images, val_positions, "val")

    # Convertir en tenseurs PyTorch — (N, H, W, 3) → (N, 3, H, W)
    X_train = torch.from_numpy(train_images).permute(0, 3, 1, 2).float()
    y_train = torch.from_numpy(train_positions).float()
    X_val = torch.from_numpy(val_images).permute(0, 3, 1, 2).float()
    y_val = torch.from_numpy(val_positions).float()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    # Modèle, optimiseur, loss
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = SpatialReconstructionNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Entraînement
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best_val_loss = float("inf")
    patience = 15
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * X_batch.size(0)
            train_count += X_batch.size(0)

        train_loss = train_loss_sum / train_count

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss_sum += loss.item() * X_batch.size(0)
                val_count += X_batch.size(0)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        val_loss = val_loss_sum / val_count

        # Métriques interprétables
        preds = np.concatenate(all_preds) * ROOM_SIZE
        targets = np.concatenate(all_targets) * ROOM_SIZE

        # Erreur de position moyenne par objet (en unités de la pièce)
        errors = []
        for i in range(NUM_OBJECTS):
            pred_pos = preds[:, i*3:(i+1)*3]
            true_pos = targets[:, i*3:(i+1)*3]
            err = np.sqrt(np.sum((pred_pos - true_pos) ** 2, axis=1))
            errors.append(err)
        all_errors = np.concatenate(errors)
        mean_error = all_errors.mean()

        # % de scènes avec tous les objets à < 1.0 unité
        per_scene = np.stack(errors, axis=0)  # (NUM_OBJECTS, N)
        all_under_1 = np.all(per_scene < 1.0, axis=0).mean() * 100

        # LR scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Sauvegarde du meilleur modèle + early stopping
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            marker = " ★ saved"
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train loss: {train_loss:.6f} | "
            f"Val loss: {val_loss:.6f} | "
            f"Err moy: {mean_error:.2f}u | "
            f"<1u: {all_under_1:.1f}% | "
            f"lr: {current_lr:.1e}{marker}"
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping: pas d'amélioration depuis {patience} epochs.")
            break

    # Recharger le meilleur modèle pour les exemples
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    print("\n--- Exemples de prédictions vs ground truth (meilleur modèle) ---")
    model.eval()
    with torch.no_grad():
        sample_X = X_val[:5].to(device)
        sample_pred = model(sample_X).cpu().numpy() * ROOM_SIZE
        sample_true = y_val[:5].numpy() * ROOM_SIZE

    obj_names = ["Cube rouge", "Sphère bleue", "Cylindre vert"]
    for i in range(5):
        print(f"\nScène {i+1}:")
        for j in range(NUM_OBJECTS):
            p = sample_pred[i, j*3:(j+1)*3]
            t = sample_true[i, j*3:(j+1)*3]
            err = np.sqrt(np.sum((p - t) ** 2))
            print(f"  {obj_names[j]:20s} | "
                  f"prédit ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) | "
                  f"réel ({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) | "
                  f"err: {err:.2f}u")

    print(f"\nMeilleure val loss: {best_val_loss:.6f}")
    print(f"Poids sauvegardés: {MODEL_PATH}")


if __name__ == "__main__":
    main()
