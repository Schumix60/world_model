# Pico Marble — Spatial Reconstruction Demo

A CNN that reconstructs the **3D positions of objects in a room from a single 2D image**, including objects that are fully occluded.

Inspired by World Labs / Marble (Fei-Fei Li): given one perspective view (64x64 pixels), the network predicts the (x, y, z) coordinates of 3 colored objects (cube, sphere, cylinder) placed in a 10x10x10 room.

## How it works

1. **Dataset generation** — 50K random scenes are rendered from a fixed camera angle into 64x64 images
2. **Training** — a CNN learns to regress 9 values (3 positions x 3 coordinates) from pixels, using MSE loss
3. **Demo** — an interactive Pygame app shows the model's predictions vs ground truth

### The 3 phases

1. **Phase 1** — shows the 2D input image (what the model sees)
2. **Phase 2** — split-screen: 2D image + top-down view with predicted (filled) vs ground truth (outlined) positions, error metrics, and occlusion badges
3. **Phase 3** — free-roam 3D exploration with WASD + mouse camera controls

## Controls

### Demo (Phases 1 & 2)

| Key | Action |
|-----|--------|
| **Space** | Next phase |
| **Backspace** | Previous phase |
| **N** | New random scene |
| **1/2/3** | Jump to phase |
| **Esc** | Quit |

### 3D Explorer (Phase 3)

| Key | Action |
|-----|--------|
| **WASD** | Move camera |
| **Mouse** | Look around |
| **Shift/Ctrl** | Up/down |
| **Backspace** | Return to Phase 2 |
| **Esc** | Quit |

## Configuration

Edit `config.py` to tune room geometry, object properties, camera settings, training hyper-parameters, and display layout.

## Run

First train the model (takes a few minutes on CPU/MPS):

```bash
cd pico_marble
uv run python train.py
```

Then run the demo:

```bash
cd pico_marble
uv run python main.py
```

## Project structure

```
pico_marble/
  main.py            # Entry point (runs the demo)
  config.py          # All constants and hyper-parameters
  scene.py           # Scene definition and random generation
  renderer2d.py      # Perspective projection: scene → 64x64 image
  renderer3d.py      # Interactive 3D renderer (Pygame, no OpenGL)
  dataset.py         # Dataset generation and I/O
  model.py           # CNN architecture (SpatialReconstructionNet)
  train.py           # Training loop with early stopping
  demo.py            # 3-phase interactive Pygame demo
```
