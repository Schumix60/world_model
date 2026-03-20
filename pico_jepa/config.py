"""Constants and tunable parameters for Pendulum-JEPA."""

import numpy as np

# ===========================================================================
# Window
# ===========================================================================
WIDTH, HEIGHT = 900, 500
LEFT_W = 350
FPS = 60
DT = 1.0 / FPS
FRAME_SZ = 84

# ===========================================================================
# Physics
# ===========================================================================
G = 9.81
L = 1.0
DAMP = 0.001
IMPULSE_EVERY = 500        # frames between impulses during training
IMPULSE_STRENGTH = 1.0     # multiplier on velocity kick (lower = gentler)
IMPULSE_RANGE = 0.8        # max angle target for impulses (rad, ~45 deg)
OMEGA_MAX = 6.0            # clamp angular velocity (prevents wild spins)

# Initial conditions
INIT_THETA = 0.8           # initial angle (rad, ~45 deg)
INIT_OMEGA = 0.0           # initial angular velocity

# ===========================================================================
# Colors
# ===========================================================================
BG = (26, 26, 46)
GRID_COL = (40, 40, 70)
TEXT_COL = (200, 200, 200)
PEND_COL = (220, 220, 220)
PIVOT_COL = (150, 150, 150)
TRAIL_OLD = np.array([0, 100, 180], dtype=np.float64)   # blue (oldest)
TRAIL_NEW = np.array([0, 220, 255], dtype=np.float64)   # cyan (newest)
COL_CUR = (255, 60, 60)       # red  — current real point
COL_PRED = (60, 255, 60)      # green — predicted point

# ===========================================================================
# JEPA hyper-parameters
# ===========================================================================
ZDIM = 2
NSTACK = 3
BATCH = 64
BUF_CAP = 10000
LR = 3e-4
TAU = 0.996
WVAR = 0.5
LOSS_STOP = 0.025          # stop training below this threshold (rolling avg)
LOSS_RESUME = 0.08         # resume training if loss drifts back above this
TRAIL_LEN = 200

# ===========================================================================
# Display geometry
# ===========================================================================
PIV_DISP = (LEFT_W // 2, 80)
ARM_DISP = 150
ARM_SMALL = 34
