# World Model Demos

A collection of pedagogical demos illustrating World Model concepts for a general audience. Each demo is a minimal, self-contained implementation that makes a key idea visually tangible.

A World Model is an internal representation of the environment that an agent uses to **simulate**, **predict**, and **plan** before acting. When the real world changes, the agent detects the mismatch between its predictions and reality, and updates its internal model — learning to adapt.

Each demo in this repository makes this process visually explicit so that a non-technical audience can see how an adaptive system "thinks".

## Demos

| Demo | Description |
|------|-------------|
| [billiard](billiard/) | A 2D billiard table where a World Model learns to compensate for table tilt by observing prediction errors and adapting its internal physics parameters |
| [pico_jepa](pico_jepa/) | A tiny implementation of Yann LeCun's **JEPA** (Joint Embedding Predictive Architecture, 2022) applied to a simple pendulum. A CNN compresses raw pixel frames into a 2D latent space, and a predictor learns the dynamics in that space. After training, the latent trajectory reveals the pendulum's **phase portrait** — proving that the network has discovered the underlying physics (angle and velocity) purely from images, without any supervision |
| [pico_marble](pico_marble/) | A minimal take on the core idea behind Fei-Fei Li's **World Labs / Marble** project: reconstructing a full 3D scene from a single 2D image. A CNN takes one 64x64 perspective view of a room containing 3 colored objects and predicts all 9 spatial coordinates (x, y, z for each object) — including objects that are **fully occluded** behind others. The network learns projective geometry and occlusion reasoning from data alone |

## Setup

```bash
uv sync
```

Then run any demo:

```bash
cd billiard && uv run python main.py
cd pico_jepa && uv run python main.py
cd pico_marble && uv run python train.py && uv run python main.py
```

## Requirements

- Python 3.11+
- Dependencies managed via `uv` (see `pyproject.toml`)
