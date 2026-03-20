"""Pendulum-JEPA: Interactive demo of a Joint Embedding Predictive Architecture
learning the physics of a simple pendulum without supervision.

Controls:
    T       - Toggle training ON/OFF
    Space   - Pause/resume physics
    Mouse   - Drag the pendulum mass
    M       - Toggle imagination mode (hide -> reveal -> exit)
    R       - Reset everything
    1/2/3   - Switch to phase 1/2/3
    Q/Esc   - Quit
"""

import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import threading
import time
from collections import deque

from config import (
    WIDTH, HEIGHT, LEFT_W, FPS, DT, FRAME_SZ,
    G, L, DAMP, IMPULSE_EVERY, IMPULSE_STRENGTH, IMPULSE_RANGE, OMEGA_MAX,
    INIT_THETA, INIT_OMEGA,
    BG, GRID_COL, TEXT_COL, PEND_COL, PIVOT_COL,
    TRAIL_OLD, TRAIL_NEW, COL_CUR, COL_PRED,
    ZDIM, NSTACK, BATCH, BUF_CAP, LR, TAU, WVAR, LOSS_STOP, LOSS_RESUME,
    TRAIL_LEN,
    PIV_DISP, ARM_DISP, ARM_SMALL,
)

# Limit PyTorch intra-op threads to reduce GIL contention with training thread
torch.set_num_threads(2)


# ===========================================================================
# Neural Networks
# ===========================================================================
class Encoder(nn.Module):
    """CNN encoder: 3 stacked 84x84 grayscale frames -> z in R^2."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NSTACK, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
        )
        # 84 -> 42 -> 21 -> 10, so flatten = 64*10*10 = 6400
        self.fc = nn.Sequential(
            nn.Linear(64 * 10 * 10, 128), nn.ReLU(),
            nn.Linear(128, ZDIM),
        )

    def forward(self, x):
        h = self.conv(x)
        return self.fc(h.reshape(h.size(0), -1))


class Predictor(nn.Module):
    """MLP predictor: 3 consecutive embeddings (6D) -> predicted next z (2D)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NSTACK * ZDIM, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, ZDIM),
        )

    def forward(self, x):
        return self.net(x)


# ===========================================================================
# Replay Buffer  (thread-safe: main appends, trainer reads)
# ===========================================================================
class ReplayBuffer:
    """Stores frame stacks. z values are re-encoded during training."""
    def __init__(self):
        self.stacks = []   # list of (3, 84, 84) uint8 numpy arrays

    def add(self, stack):
        self.stacks.append(stack)

    def can_train(self, bs):
        return len(self.stacks) >= bs + 4

    def sample(self, bs):
        n = len(self.stacks)
        js = np.random.randint(0, n - 3, size=bs)
        s0, s1, s2, s3 = [], [], [], []
        for j in js:
            s0.append(self.stacks[j])
            s1.append(self.stacks[j + 1])
            s2.append(self.stacks[j + 2])
            s3.append(self.stacks[j + 3])

        def _t(lst):
            return torch.from_numpy(np.array(lst)).float() / 255.0

        return _t(s0), _t(s1), _t(s2), _t(s3)


# ===========================================================================
# Pendulum
# ===========================================================================
class Pendulum:
    def __init__(self, theta=INIT_THETA, omega=INIT_OMEGA):
        self.theta = theta
        self.omega = omega
        self.steps = 0

    def step(self):
        """Semi-implicit Euler integration."""
        acc = -(G / L) * math.sin(self.theta) - DAMP * self.omega
        self.omega += acc * DT
        self.omega = max(-OMEGA_MAX, min(OMEGA_MAX, self.omega))
        self.theta += self.omega * DT
        self.steps += 1

    def impulse(self):
        """Apply a gentle impulse toward a random angle."""
        target = np.random.uniform(-IMPULSE_RANGE, IMPULSE_RANGE)
        self.omega += (target - self.theta) * IMPULSE_STRENGTH

    def mass_pos(self, pivot, length):
        """Pixel position of the mass given pivot and arm length in pixels."""
        x = pivot[0] + length * math.sin(self.theta)
        y = pivot[1] + length * math.cos(self.theta)
        return int(x), int(y)

    def render_frame(self, surf):
        """Draw on a pre-allocated FRAME_SZ x FRAME_SZ surface for the network."""
        surf.fill((0, 0, 0))
        pv = (FRAME_SZ // 2, 8)
        mx, my = self.mass_pos(pv, ARM_SMALL)
        pygame.draw.line(surf, (255, 255, 255), pv, (mx, my), 1)
        pygame.draw.circle(surf, (255, 255, 255), (mx, my), 5)


# ===========================================================================
# Shared state container (main thread <-> training thread)
# ===========================================================================
class Shared:
    """Mutable container so both threads see the same objects."""
    def __init__(self):
        self.encoder = None
        self.predictor = None
        self.target_enc = None
        self.optimizer = None
        self.buf = ReplayBuffer()
        self.losses = deque(maxlen=100)
        self.training = False
        self.converged = False          # auto-pause when loss is low enough
        self.running = True


# ===========================================================================
# Background training loop
# ===========================================================================
def train_loop(shared, lock):
    """Runs in a daemon thread. Re-encodes all z values each step
    so there is never any staleness in the predictor input.
    Auto-pauses when loss drops below LOSS_STOP; resumes above LOSS_RESUME."""
    ema_loss = 1.0  # exponential moving average for stop/resume decisions
    train_steps = 0

    while shared.running:
        # --- grab current references under lock ---
        with lock:
            if not shared.training:
                active = False
            else:
                active = True
                enc = shared.encoder
                pred = shared.predictor
                tgt_enc = shared.target_enc
                opt = shared.optimizer
                buf = shared.buf
                loss_q = shared.losses
                converged = shared.converged

        if not active or not buf.can_train(BATCH):
            time.sleep(0.02)
            continue

        # If converged, only probe periodically to check if loss drifted up
        if converged:
            time.sleep(0.2)
            # Probe: compute loss without updating weights
            s0, s1, s2, s3 = buf.sample(BATCH)
            with torch.no_grad():
                z_prev = enc(torch.cat([s0, s1]))
                z0, z1 = z_prev[:BATCH], z_prev[BATCH:]
                z_target = tgt_enc(s3)
                z_t = enc(s2)
                z_pred = pred(torch.cat([z0, z1, z_t], dim=1))
                probe_loss = F.mse_loss(z_pred, z_target).item()
            ema_loss = 0.5 * ema_loss + 0.5 * probe_loss
            if ema_loss > LOSS_RESUME:
                shared.converged = False
            continue

        # --- sample & build tensors (outside lock) ---
        s0, s1, s2, s3 = buf.sample(BATCH)

        # Re-encode z_{t-2}, z_{t-1} with current encoder (no grad)
        with torch.no_grad():
            z_prev = enc(torch.cat([s0, s1]))       # (2B, 2)
            z0, z1 = z_prev[:BATCH], z_prev[BATCH:]
            z_target = tgt_enc(s3)                   # (B, 2)

        # Encode z_t with gradients (drives encoder learning)
        z_t = enc(s2)                                # (B, 2)

        # Predict z_{t+1}
        z_pred = pred(torch.cat([z0, z1, z_t], dim=1))  # (B, 2)

        # Loss = prediction + variance regularization
        loss_pred = F.mse_loss(z_pred, z_target)
        loss_var = torch.mean(F.relu(1.0 - z_t.std(dim=0)))
        loss = loss_pred + WVAR * loss_var

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(pred.parameters()), max_norm=1.0
        )
        opt.step()

        # EMA update target encoder
        with torch.no_grad():
            for p_e, p_t in zip(enc.parameters(), tgt_enc.parameters()):
                p_t.data.mul_(TAU).add_(p_e.data, alpha=1 - TAU)

        lp = loss_pred.item()
        loss_q.append(lp)  # display prediction loss only (meaningful)

        # Update EMA and check convergence on prediction loss
        ema_loss = 0.9 * ema_loss + 0.1 * lp
        train_steps += 1

        # Decay LR every 20 steps (gentle annealing toward finer convergence)
        if train_steps % 20 == 0:
            for pg in opt.param_groups:
                pg['lr'] = max(pg['lr'] * 0.85, 1e-5)

        if ema_loss < LOSS_STOP:
            shared.converged = True


# ===========================================================================
# Main
# ===========================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pendulum-JEPA — World Model Demo")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 26)
    font_lg = pygame.font.Font(None, 38)
    small_surf = pygame.Surface((FRAME_SZ, FRAME_SZ))

    # --- Shared state & lock ---
    shared = Shared()
    lock = threading.Lock()

    def init_models():
        shared.encoder = Encoder()
        shared.predictor = Predictor()
        shared.target_enc = copy.deepcopy(shared.encoder)
        for p in shared.target_enc.parameters():
            p.requires_grad = False
        shared.optimizer = torch.optim.Adam(
            list(shared.encoder.parameters())
            + list(shared.predictor.parameters()),
            lr=LR,
        )
        shared.buf = ReplayBuffer()
        shared.losses = deque(maxlen=100)

    init_models()

    # Start background training thread
    trainer = threading.Thread(
        target=train_loop, args=(shared, lock), daemon=True
    )
    trainer.start()

    # --- Local state (main thread only) ---
    pend = Pendulum()
    fstack = deque(maxlen=NSTACK)
    trail = deque(maxlen=TRAIL_LEN)

    training = False
    paused = False
    imagining = False
    revealed = False
    dragging = False
    phase = 1
    tick = 0

    imag_zs = []
    imag_trail = deque(maxlen=TRAIL_LEN)

    zlo = np.array([-2.0, -2.0], dtype=np.float64)
    zhi = np.array([2.0, 2.0], dtype=np.float64)

    running = True
    while running:
        # ==============================================================
        # Events
        # ==============================================================
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif k == pygame.K_t:
                    training = not training

                elif k == pygame.K_SPACE:
                    paused = not paused

                elif k == pygame.K_m:
                    if not imagining:
                        imagining, revealed = True, False
                        imag_zs = (
                            [z.clone() for z in list(trail)[-3:]]
                            if len(trail) >= 3
                            else []
                        )
                        imag_trail.clear()
                    elif not revealed:
                        revealed = True
                    else:
                        imagining, revealed = False, False

                elif k == pygame.K_r:
                    with lock:
                        shared.training = False
                        shared.converged = False
                        init_models()
                    pend = Pendulum()
                    fstack.clear()
                    trail.clear()
                    training = paused = imagining = revealed = dragging = False
                    phase, tick = 1, 0
                    imag_zs = []
                    imag_trail.clear()
                    zlo[:], zhi[:] = -2.0, 2.0

                elif k == pygame.K_1:
                    phase = 1
                    imagining = revealed = False
                    paused = False
                    training = True

                elif k == pygame.K_2:
                    phase = 2
                    imagining = revealed = False

                elif k == pygame.K_3:
                    phase = 3
                    imagining, revealed = True, False
                    imag_zs = (
                        [z.clone() for z in list(trail)[-3:]]
                        if len(trail) >= 3
                        else []
                    )
                    imag_trail.clear()

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if not (imagining and not revealed):
                    mpos = pend.mass_pos(PIV_DISP, ARM_DISP)
                    if math.hypot(ev.pos[0] - mpos[0], ev.pos[1] - mpos[1]) < 30:
                        dragging = True

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False

        # Propagate training flag to shared state
        shared.training = training

        # ==============================================================
        # Mouse drag
        # ==============================================================
        if dragging:
            mx, my = pygame.mouse.get_pos()
            pend.theta = math.atan2(mx - PIV_DISP[0], my - PIV_DISP[1])
            pend.omega = 0.0

        # ==============================================================
        # Physics
        # ==============================================================
        if not paused and not dragging:
            pend.step()
            if training and pend.steps > 0 and pend.steps % IMPULSE_EVERY == 0:
                pend.impulse()

        # ==============================================================
        # Render small frame for the network
        # ==============================================================
        pend.render_frame(small_surf)
        gray = pygame.surfarray.array3d(small_surf)[:, :, 0].astype(np.uint8)
        fstack.append(gray)

        # ==============================================================
        # Encode current state  (lightweight — single sample, no grad)
        # ==============================================================
        encoder = shared.encoder          # local ref (safe even during training)
        predictor = shared.predictor

        cur_z = None
        if len(fstack) == NSTACK:
            if not (imagining and not revealed):
                stk = np.stack(list(fstack), axis=0)
                with torch.no_grad():
                    cur_z = encoder(
                        torch.from_numpy(stk).float().unsqueeze(0) / 255.0
                    ).squeeze(0)
                if not imagining:
                    trail.append(cur_z.clone())
                    shared.buf.add(stk)

        # ==============================================================
        # Imagination (autoregressive prediction — lightweight)
        # ==============================================================
        imag_pt = None
        if imagining and len(imag_zs) >= 3:
            with torch.no_grad():
                z_next = predictor(
                    torch.cat(imag_zs[-3:]).unsqueeze(0)
                ).squeeze(0)
            imag_zs.append(z_next.clone())
            if len(imag_zs) > 500:
                imag_zs = imag_zs[-500:]
            imag_pt = z_next.numpy()
            imag_trail.append(z_next.clone())

        # ==============================================================
        # Rescale scatter axes (every ~1 second)
        # ==============================================================
        if tick % 60 == 0:
            pts = list(trail) + (list(imag_trail) if imagining else [])
            if len(pts) > 1:
                arr = torch.stack(pts).numpy()
                lo = arr.min(axis=0) - 0.5
                hi = arr.max(axis=0) + 0.5
                zlo = zlo * 0.8 + lo * 0.2
                zhi = zhi * 0.8 + hi * 0.2
                for d in range(2):
                    if zhi[d] - zlo[d] < 1.0:
                        mid = (zhi[d] + zlo[d]) / 2
                        zlo[d], zhi[d] = mid - 0.5, mid + 0.5

        # ==============================================================
        # RENDERING
        # ==============================================================
        screen.fill(BG)

        # --- Left panel: pendulum ---
        if imagining and not revealed:
            txt = font_lg.render("Le reseau imagine...", True, (100, 200, 255))
            screen.blit(
                txt, (LEFT_W // 2 - txt.get_width() // 2, HEIGHT // 2 - 20)
            )
        else:
            mp = pend.mass_pos(PIV_DISP, ARM_DISP)
            pygame.draw.line(screen, PEND_COL, PIV_DISP, mp, 2)
            pygame.draw.circle(screen, PEND_COL, mp, 10)
            pygame.draw.circle(screen, PIVOT_COL, PIV_DISP, 4)

        # Status bar
        yb = HEIGHT - 85
        losses = shared.losses
        avg_loss = float(np.mean(losses)) if losses else 0.0
        screen.blit(
            font.render(f"Loss: {avg_loss:.4f}", True, TEXT_COL), (10, yb)
        )
        screen.blit(
            font.render(f"Phase: {phase}", True, TEXT_COL), (10, yb + 25)
        )

        if imagining:
            status, status_col = "IMAGINATION", (100, 200, 255)
        elif paused:
            status, status_col = "PAUSE", (255, 200, 100)
        elif training and shared.converged:
            status, status_col = "CONVERGE", (80, 220, 180)
        elif training:
            status, status_col = "ENTRAINEMENT", (100, 255, 100)
        else:
            status, status_col = "OBSERVATION", TEXT_COL
        screen.blit(font.render(status, True, status_col), (10, yb + 50))

        # --- Right panel: scatter plot ---
        rx = LEFT_W + 25
        ry = 25
        rw = WIDTH - LEFT_W - 50
        rh = HEIGHT - 50

        for i in range(5):
            gx = rx + rw * i // 4
            gy = ry + rh * i // 4
            pygame.draw.line(screen, GRID_COL, (gx, ry), (gx, ry + rh))
            pygame.draw.line(screen, GRID_COL, (rx, gy), (rx + rw, gy))
        pygame.draw.rect(screen, GRID_COL, (rx, ry, rw, rh), 1)

        screen.blit(
            font.render("z1", True, TEXT_COL), (rx + rw // 2 - 8, ry + rh + 5)
        )
        screen.blit(
            font.render("z2", True, TEXT_COL), (rx - 22, ry + rh // 2 - 8)
        )

        def z_to_screen(z):
            fx = (z[0] - zlo[0]) / max(zhi[0] - zlo[0], 1e-8)
            fy = (z[1] - zlo[1]) / max(zhi[1] - zlo[1], 1e-8)
            return (
                int(np.clip(rx + fx * rw, rx, rx + rw)),
                int(np.clip(ry + rh - fy * rh, ry, ry + rh)),
            )

        # Historical trail (blue -> cyan gradient)
        trail_list = list(trail)
        n_trail = max(len(trail_list) - 1, 1)
        for i, z in enumerate(trail_list):
            alpha = i / n_trail
            color = tuple(
                (TRAIL_OLD + (TRAIL_NEW - TRAIL_OLD) * alpha).astype(int)
            )
            radius = max(2, int(2 + 2 * alpha))
            pygame.draw.circle(screen, color, z_to_screen(z.numpy()), radius)

        # Current real point (red)
        if cur_z is not None and not (imagining and not revealed):
            pygame.draw.circle(screen, COL_CUR, z_to_screen(cur_z.numpy()), 6)

        # Imagination trail + predicted point (green)
        if imagining:
            itl = list(imag_trail)
            n_imag = max(len(itl) - 1, 1)
            for i, z in enumerate(itl):
                a = i / n_imag
                col = (60 + int(140 * a), 255, 60 + int(140 * a))
                pygame.draw.circle(
                    screen, col, z_to_screen(z.numpy()), max(2, int(2 + 2 * a))
                )
            if imag_pt is not None:
                pygame.draw.circle(screen, COL_PRED, z_to_screen(imag_pt), 6)

            if revealed and cur_z is not None:
                pygame.draw.circle(
                    screen, COL_CUR, z_to_screen(cur_z.numpy()), 6
                )
                lx, ly = rx + 12, ry + rh - 35
                pygame.draw.circle(screen, COL_PRED, (lx, ly), 5)
                screen.blit(
                    font.render("Predit", True, COL_PRED), (lx + 14, ly - 7)
                )
                pygame.draw.circle(screen, COL_CUR, (lx, ly + 20), 5)
                screen.blit(
                    font.render("Reel", True, COL_CUR), (lx + 14, ly + 13)
                )

        # Panel separator
        pygame.draw.line(screen, GRID_COL, (LEFT_W, 0), (LEFT_W, HEIGHT))

        pygame.display.flip()
        tick += 1
        clock.tick(FPS)

    shared.running = False
    pygame.quit()


if __name__ == "__main__":
    main()
