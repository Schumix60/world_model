# Billiard World Model Demo

A Pygame demo showing a World Model that learns to play billiards on a tilting table.

An agent must send a white ball to hit a red target ball. It uses an internal physics model to **imagine** trajectories, **plan** the best shot, then **observe** the result and **correct** its parameters. Each step is animated so the audience can see the reasoning process.

## How it works

1. **Table flat** — the World Model predicts trajectories accurately from the start
2. **Tilt the table** (arrow keys) — gravity pulls the ball off course, predictions fail visibly
3. **Adaptation** — after a few shots, the World Model recalibrates and predictions converge again

### The 6-step pipeline

Each shot goes through 6 animated phases:

1. **Imagination** (purple) — 255 candidate trajectories appear as a rainbow fan
2. **Planning** (orange) — top 5 candidates pulse and are ranked
3. **Prediction** (blue) — the best trajectory is selected
4. **Action** (yellow) — the ball is launched, real trajectory drawn alongside the prediction
5. **Observation** (white) — red error lines show the gap between predicted and actual
6. **Learning** (green) — internal parameters are updated, error graph records progress

## Controls

| Key | Action |
|-----|--------|
| **Space** | Fire / skip to next phase |
| **Arrow keys** | Tilt the table (X and Y axes, up to 15 degrees) |
| **W** | Toggle World Model ON/OFF |
| **R** | Full reset |
| **Esc** | Quit |

## Run

```bash
cd billiard
uv run python main.py
```

## Project structure

```
billiard/
  main.py              # Entry point
  billard_wm/
    config.py          # Constants (physics, colors, durations)
    physics.py         # Real physics: Ball, step(), collisions
    world_model.py     # Internal model: planning, prediction, learning
    renderer.py        # Pygame rendering: table, trajectories, UI
    main.py            # Game loop and state machine
```
