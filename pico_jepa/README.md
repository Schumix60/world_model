# Pendulum-JEPA Demo

A real-time demo of a **Joint Embedding Predictive Architecture** (JEPA) learning the physics of a simple pendulum without supervision.

A CNN encoder compresses 84x84 grayscale frames into a **2D latent space**, and an MLP predictor learns to forecast the next embedding from the 3 previous ones. The latent space is visualized as a live scatter plot — after training, it reveals the pendulum's **phase portrait** (an ellipse mapping angle and velocity), discovered entirely from pixels.

## How it works

1. **Training** (press T) — the pendulum swings, the network trains in a background thread, and the scatter plot gradually organizes into an ellipse
2. **Interaction** (press Space + mouse drag) — move the pendulum manually and watch the latent point follow in real time
3. **Imagination** (press M) — the encoder is disconnected; the predictor runs autoregressively, generating a predicted trajectory (green) on the scatter plot. Press M again to reveal the real trajectory (red) alongside it

## Architecture

- **Encoder** (CNN): 3 stacked 84x84 frames → z in R² (2 numbers)
- **Predictor** (MLP): 3 consecutive embeddings (6D) → predicted next z (2D)
- **Target encoder**: EMA copy of the encoder (prevents collapse)
- **Variance regularization**: forces the 2 latent dimensions to stay informative

## Controls

| Key | Action |
|-----|--------|
| **T** | Toggle training ON/OFF |
| **Space** | Pause/resume physics |
| **Mouse** | Drag the pendulum mass |
| **M** | Imagination → reveal → exit |
| **R** | Reset everything (weights, buffer, scatter) |
| **1/2/3** | Switch to phase 1/2/3 |
| **Q/Esc** | Quit |

## Configuration

Edit `config.py` to tune physics, initial conditions, JEPA hyper-parameters, colors, and display layout.

## Run

```bash
cd pico_jepa
uv run python main.py
```
