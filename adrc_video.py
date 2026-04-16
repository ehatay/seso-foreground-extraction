#!/usr/bin/env python3
"""
Per-pixel Saturated Extended State Observer (SESO) for video
foreground extraction, compared against a linear ESO (LESO) and the
classical Gaussian Mixture Model (MOG2) baseline. Companion code for

    "Extended-Memory Video Foreground Extraction via a Saturated
     Extended State Observer"

Per-pixel plant (first-order ADRC formulation):

    y_dot(t) = f(t)              first-order plant
    y(k)     = x_1(k) + v(k)     measurement with sensor noise

With extended state x_2 := f (total disturbance), the state is
[x_1, x_2] = [y, f] and the ESO estimates both from y alone:

    zhat_1  -> y    (denoised measurement)
    zhat_2  -> f    (total disturbance: illumination drift, shadows,
                     foreground arrivals -- everything that makes the
                     pixel change)

Bandwidth-parameterized linear gains for a first-order plant:
    L1 = 2 * wo
    L2 = wo**2

Foreground mask from the innovation (not from zhat_2 -- see paper
Section II.C for the bandwidth-separation argument):
    mask(i, j, k) = | y(i, j, k) - zhat_1(i, j, k) | > threshold

Three detectors are compared:

    LESO  linear ESO        zhat_1 <- zhat_1 + zhat_2 + L1 * e
                            zhat_2 <- zhat_2 + L2 * e

    SESO  saturated ESO     e_sat  = sat(e, umax)
                            zhat_1 <- zhat_1 + zhat_2 + L1 * e_sat
                            zhat_2 <- zhat_2 + L2 * e_sat

    MOG2  Gaussian Mixture Model (classical baseline from OpenCV;
          Zivkovic 2004). Per-pixel mixture fit to recent history;
          pixels that don't fit the mixture are flagged as foreground.

All three methods share the same compute tier (classical, CPU,
milliseconds per frame) but have distinct failure modes on the
stationary-foreground absorption problem that the paper addresses.

Usage
-----
    python adrc_video.py                                 # defaults
    python adrc_video.py --wo 0.05 --umax 0.005          # paper tuned params
    python adrc_video.py --mog2-history 80               # retune MOG2
    python adrc_video.py --out-dir results/             # save figures
    python adrc_video.py --save-anim results/anim.gif   # GIF animation
    python adrc_video.py --benchmark                    # numbers only
    python adrc_video.py --no-mog2                      # skip MOG2

Dependencies:
    Required : numpy, matplotlib
    Optional : opencv-python  (for the MOG2 baseline)
               Pillow         (for GIF animation; bundled with matplotlib)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# Synthetic video generator
# ----------------------------------------------------------------------------

def make_synthetic_video(H=120, W=160, n_frames=200,
                         disk_start=25, intruder_start=50, seed=0):
    """
    Synthesize a video with known ground truth and three phenomena:

      1. slow illumination drift on a spatial-gradient background
      2. a moving disk appearing at `disk_start` (basic foreground
         detection test; delayed so observers can converge on a
         clean background first)
      3. a stationary intruder appearing at `intruder_start` and staying
         (tests all three methods' absorption behavior)

    Returns
    -------
    frames : (n_frames, H, W) float32 in [0, 1]
    gt     : (n_frames, H, W) bool, True where foreground is present
    meta   : dict with timing / geometry info
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    bg_spatial = 0.35 + 0.12 * (xx / max(W - 1, 1))

    ir0, ir1 = int(H * 0.50), int(H * 0.72)
    ic0, ic1 = int(W * 0.62), int(W * 0.82)
    intruder_intensity = 0.72

    disk_radius = 8
    disk_intensity = 0.92

    frames = np.zeros((n_frames, H, W), dtype=np.float32)
    gt = np.zeros((n_frames, H, W), dtype=bool)

    for k in range(n_frames):
        drift = 0.04 * np.sin(2 * np.pi * k / 300.0)
        bg = bg_spatial + drift

        frame = bg.copy()
        mask = np.zeros_like(frame, dtype=bool)

        if k >= disk_start:
            cx = int(W * 0.28 + W * 0.22 * np.cos(2 * np.pi * k / 37.0))
            cy = int(H * 0.42 + H * 0.18 * np.sin(2 * np.pi * k / 53.0))
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 < disk_radius ** 2
            frame[disk] = disk_intensity
            mask = mask | disk

        if k >= intruder_start:
            frame[ir0:ir1, ic0:ic1] = intruder_intensity
            mask[ir0:ir1, ic0:ic1] = True

        frame = frame + rng.standard_normal(frame.shape).astype(np.float32) * 0.012
        frame = np.clip(frame, 0.0, 1.0)

        frames[k] = frame
        gt[k] = mask

    meta = {
        "disk_start": disk_start,
        "intruder_start": intruder_start,
        "intruder_rect": (ir0, ir1, ic0, ic1),
        "H": H, "W": W, "n_frames": n_frames,
    }
    return frames, gt, meta


# ----------------------------------------------------------------------------
# Observers (2-state ESO for a 1st-order plant)
# ----------------------------------------------------------------------------

def run_leso(frames, wo=0.10):
    """Linear ESO per pixel. Returns (z1_hist, z2_hist)."""
    n = frames.shape[0]
    z1 = frames[0].astype(np.float32).copy()
    z2 = np.zeros_like(z1)
    L1, L2 = 2.0 * wo, wo ** 2

    z1_hist = np.empty_like(frames)
    z2_hist = np.empty_like(frames)
    z1_hist[0] = z1
    z2_hist[0] = z2

    for k in range(1, n):
        y = frames[k]
        e = y - z1
        z1 = z1 + z2 + L1 * e
        z2 = z2 + L2 * e
        z1_hist[k] = z1
        z2_hist[k] = z2
    return z1_hist, z2_hist


def run_seso(frames, wo=0.10, umax=0.015):
    """Saturated-innovation ESO per pixel. Returns (z1_hist, z2_hist)."""
    n = frames.shape[0]
    z1 = frames[0].astype(np.float32).copy()
    z2 = np.zeros_like(z1)
    L1, L2 = 2.0 * wo, wo ** 2

    z1_hist = np.empty_like(frames)
    z2_hist = np.empty_like(frames)
    z1_hist[0] = z1
    z2_hist[0] = z2

    for k in range(1, n):
        y = frames[k]
        e = y - z1
        e_sat = np.clip(e, -umax, umax)
        z1 = z1 + z2 + L1 * e_sat
        z2 = z2 + L2 * e_sat
        z1_hist[k] = z1
        z2_hist[k] = z2
    return z1_hist, z2_hist


# ----------------------------------------------------------------------------
# MOG2 baseline (OpenCV)
# ----------------------------------------------------------------------------

def run_mog2(frames, history=100, var_threshold=16.0):
    """
    Classical MOG2 baseline using OpenCV's
    cv2.createBackgroundSubtractorMOG2.

    Per-pixel Gaussian-mixture model fit to the last `history` frames.
    Pixels that don't fit the mixture (Mahalanobis distance
    > sqrt(var_threshold)) are flagged as foreground.

    Parameters
    ----------
    history       : sliding-window length in frames (default 100). MOG2
                    maintains its model on the last `history` frames.
                    Smaller -> adapts faster, absorbs stationary intruder
                    faster. Larger -> slower adaptation.
    var_threshold : Mahalanobis-distance threshold for the fit test
                    (default 16.0 = 4 sigma).

    Returns
    -------
    (masks, bg_estimates) both (n, H, W)
        masks        : bool, True where foreground detected
        bg_estimates : float32 in [0, 1], MOG2's internal background
                       image at each frame

    Returns (None, None) if opencv-python is not installed.
    """
    try:
        import cv2
    except ImportError:
        return None, None

    n, H, W = frames.shape
    mog = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=False,
    )

    masks = np.zeros((n, H, W), dtype=bool)
    bg_est = np.zeros((n, H, W), dtype=np.float32)

    for k in range(n):
        frame_u8 = (frames[k] * 255).astype(np.uint8)
        fg_mask = mog.apply(frame_u8)
        masks[k] = fg_mask > 127

        bg_u8 = mog.getBackgroundImage()
        if bg_u8 is not None:
            bg_est[k] = bg_u8.astype(np.float32) / 255.0
        else:
            # Before the model is built, getBackgroundImage returns None.
            bg_est[k] = frames[k]

    return masks, bg_est


# ----------------------------------------------------------------------------
# Masks and metrics
# ----------------------------------------------------------------------------

def to_mask(frames, z1_hist, threshold):
    return np.abs(frames - z1_hist) > threshold


def compute_f1(pred, gt):
    n = pred.shape[0]
    f1 = np.zeros(n, dtype=np.float64)
    for k in range(n):
        p = pred[k]; g = gt[k]
        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, ~g).sum()
        fn = np.logical_and(~p, g).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1[k] = 2 * prec * rec / (prec + rec + 1e-9)
    return f1


def compute_iou(pred, gt):
    n = pred.shape[0]
    iou = np.zeros(n, dtype=np.float64)
    for k in range(n):
        p = pred[k]; g = gt[k]
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        iou[k] = inter / (union + 1e-9)
    return iou


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def _color_mask(pred, gt):
    """TP green, FP red, FN blue, TN black."""
    img = np.zeros((*pred.shape, 3))
    img[np.logical_and(pred,  gt)] = [0.20, 0.80, 0.30]
    img[np.logical_and(pred, ~gt)] = [0.90, 0.25, 0.20]
    img[np.logical_and(~pred, gt)] = [0.20, 0.40, 0.95]
    return img


def fig_contact_sheet(frames, z1_L, z1_S, gt, sample_k, threshold, title):
    """Observer-internals contact sheet: LESO vs SESO across sample frames."""
    ns = len(sample_k)
    fig, ax = plt.subplots(7, ns, figsize=(2.2 * ns + 0.6, 13.2))

    inn_L = np.abs(frames - z1_L)
    inn_S = np.abs(frames - z1_S)
    vmax_e = max(inn_L.max(), inn_S.max()) + 1e-9

    row_labels = [
        "frame  $y(k)$",
        r"$\hat{z}_1$  LESO",
        r"$\hat{z}_1$  SESO",
        r"$|e|$  LESO",
        r"$|e|$  SESO",
        "mask LESO\n(TP / FP / FN)",
        "mask SESO\n(TP / FP / FN)",
    ]

    for j, k in enumerate(sample_k):
        ax[0, j].imshow(frames[k], cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(f"$k = {k}$", fontsize=11)
        ax[1, j].imshow(z1_L[k], cmap="gray", vmin=0, vmax=1)
        ax[2, j].imshow(z1_S[k], cmap="gray", vmin=0, vmax=1)
        ax[3, j].imshow(inn_L[k], cmap="inferno", vmin=0, vmax=vmax_e)
        ax[4, j].imshow(inn_S[k], cmap="inferno", vmin=0, vmax=vmax_e)
        mL = inn_L[k] > threshold
        mS = inn_S[k] > threshold
        ax[5, j].imshow(_color_mask(mL, gt[k]))
        ax[6, j].imshow(_color_mask(mS, gt[k]))

    for i, lab in enumerate(row_labels):
        ax[i, 0].set_ylabel(lab, fontsize=10)
    for a in ax.flatten():
        a.set_xticks([]); a.set_yticks([])

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def fig_method_comparison(frames, mask_L, mask_S, mask_M, bg_M, gt, sample_k,
                          title):
    """
    Three-method comparison. Rows: frame | bg estimates | masks.
    Shows LESO, SESO, and MOG2 side by side.
    """
    ns = len(sample_k)
    fig, ax = plt.subplots(4, ns, figsize=(2.2 * ns + 0.6, 7.8))

    for j, k in enumerate(sample_k):
        ax[0, j].imshow(frames[k], cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(f"$k = {k}$", fontsize=11)
        ax[1, j].imshow(_color_mask(mask_L[k], gt[k]))
        ax[2, j].imshow(_color_mask(mask_S[k], gt[k]))
        ax[3, j].imshow(_color_mask(mask_M[k], gt[k]))

    row_labels = [
        "frame  $y(k)$",
        "mask LESO\n(TP / FP / FN)",
        "mask SESO\n(TP / FP / FN)",
        "mask MOG2\n(TP / FP / FN)",
    ]
    for i, lab in enumerate(row_labels):
        ax[i, 0].set_ylabel(lab, fontsize=10)
    for a in ax.flatten():
        a.set_xticks([]); a.set_yticks([])

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def fig_pixel_trace(frames, z1_L, z2_L, z1_S, z2_S, pixel_rc, intruder_start,
                     threshold, title):
    """Three-panel pixel trace: y + zhat_1 | e = y - zhat_1 | zhat_2."""
    r, c = pixel_rc
    n = frames.shape[0]
    k = np.arange(n)

    y  = frames[:, r, c]
    lz1 = z1_L[:, r, c]
    sz1 = z1_S[:, r, c]
    lz2 = z2_L[:, r, c]
    sz2 = z2_S[:, r, c]
    eL = y - lz1
    eS = y - sz1

    fig, (a, b, c_ax) = plt.subplots(3, 1, figsize=(10.5, 8.4), sharex=True)

    a.plot(k, y,   color="black",    lw=1.6, label=r"$y(k)$  (measurement)")
    a.plot(k, lz1, color="tab:blue", lw=1.4, label=r"$\hat{z}_1$  LESO")
    a.plot(k, sz1, color="tab:red",  lw=1.4, label=r"$\hat{z}_1$  SESO")
    a.axvline(intruder_start, color="gray", ls="--", lw=1, alpha=0.7,
              label=f"intruder arrives ($k = {intruder_start}$)")
    a.set_ylabel("intensity")
    a.set_title(f"{title}   |   pixel at (row = {r}, col = {c})", fontsize=11)
    a.legend(loc="best", fontsize=9); a.grid(alpha=0.3)

    b.plot(k, np.abs(eL), color="tab:blue", lw=1.4, label=r"$|e|$  LESO")
    b.plot(k, np.abs(eS), color="tab:red",  lw=1.4, label=r"$|e|$  SESO")
    b.axhline(threshold, color="black", ls=":", lw=1, alpha=0.7,
              label=f"detection threshold = {threshold}")
    b.axvline(intruder_start, color="gray", ls="--", lw=1, alpha=0.7)
    b.set_ylabel(r"$|e| = |y - \hat{z}_1|$")
    b.legend(loc="best", fontsize=9); b.grid(alpha=0.3)

    c_ax.plot(k, lz2, color="tab:blue", lw=1.4, label=r"$\hat{z}_2$  LESO")
    c_ax.plot(k, sz2, color="tab:red",  lw=1.4, label=r"$\hat{z}_2$  SESO")
    c_ax.axhline(0, color="k", lw=0.5, alpha=0.6)
    c_ax.axvline(intruder_start, color="gray", ls="--", lw=1, alpha=0.7)
    c_ax.set_xlabel("frame  $k$")
    c_ax.set_ylabel(r"$\hat{z}_2$  (total disturbance)")
    c_ax.legend(loc="best", fontsize=9); c_ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def fig_f1_over_time(f1_L, f1_S, f1_M, intruder_start, title, disk_start=None):
    """F1 score vs frame for all three methods."""
    k = np.arange(len(f1_L))
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(k, f1_L, color="tab:blue",  lw=1.6, label="LESO")
    ax.plot(k, f1_S, color="tab:red",   lw=1.6, label="SESO  (saturated)")
    if f1_M is not None:
        ax.plot(k, f1_M, color="tab:green", lw=1.6, label="MOG2  (baseline)")
    if disk_start is not None:
        ax.axvline(disk_start, color="gray", ls=":", lw=1, alpha=0.6,
                   label=f"disk appears  ($k = {disk_start}$)")
    ax.axvline(intruder_start, color="gray", ls="--", lw=1, alpha=0.7,
               label=f"intruder arrives  ($k = {intruder_start}$)")
    ax.set_xlabel("frame  $k$")
    ax.set_ylabel("F1 score")
    ax.set_title(f"Detection quality over time   |   {title}", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def save_animation(frames, z1_L, z1_S, threshold, path, fps=15):
    """Save 2x3 animated LESO vs SESO comparison as GIF."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = frames.shape[0]
    inn_L = np.abs(frames - z1_L)
    inn_S = np.abs(frames - z1_S)
    emax  = max(inn_L.max(), inn_S.max()) + 1e-9

    fig, ax = plt.subplots(2, 3, figsize=(11, 6.5))
    im_y  = ax[0, 0].imshow(frames[0], cmap="gray",    vmin=0, vmax=1)
    im_l1 = ax[0, 1].imshow(z1_L[0],   cmap="gray",    vmin=0, vmax=1)
    im_s1 = ax[0, 2].imshow(z1_S[0],   cmap="gray",    vmin=0, vmax=1)
    im_le = ax[1, 1].imshow(inn_L[0],  cmap="inferno", vmin=0, vmax=emax)
    im_se = ax[1, 2].imshow(inn_S[0],  cmap="inferno", vmin=0, vmax=emax)
    ax[1, 0].axis("off")

    ax[0, 0].set_title("frame  $y(k)$")
    ax[0, 1].set_title(r"$\hat{z}_1$  LESO")
    ax[0, 2].set_title(r"$\hat{z}_1$  SESO")
    ax[1, 1].set_title(r"$|e|$  LESO")
    ax[1, 2].set_title(r"$|e|$  SESO")
    for a in ax.flatten():
        a.set_xticks([]); a.set_yticks([])

    ktxt = ax[1, 0].text(0.5, 0.5, "$k = 0$", ha="center", va="center",
                         fontsize=16, transform=ax[1, 0].transAxes)

    def update(k):
        im_y.set_data(frames[k])
        im_l1.set_data(z1_L[k])
        im_s1.set_data(z1_S[k])
        im_le.set_data(inn_L[k])
        im_se.set_data(inn_S[k])
        ktxt.set_text(f"$k = {k}$")
        return im_y, im_l1, im_s1, im_le, im_se, ktxt

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--height",     type=int,   default=120)
    ap.add_argument("--width",      type=int,   default=160)
    ap.add_argument("--frames",     type=int,   default=200)
    ap.add_argument("--disk-start",     type=int, default=25,
                    help="frame index at which the moving disk appears "
                         "(default 25, giving observers time to converge "
                         "on the clean background first).")
    ap.add_argument("--intruder-start", type=int, default=50)
    ap.add_argument("--seed",       type=int,   default=0)

    ap.add_argument("--wo",         type=float, default=0.10,
                    help="ESO bandwidth in 1/frame (default 0.10).")
    ap.add_argument("--umax",       type=float, default=0.015,
                    help="saturation limit on innovation (default 0.015).")
    ap.add_argument("--threshold",  type=float, default=0.08,
                    help="foreground threshold on |y - z1| (default 0.08).")

    ap.add_argument("--mog2-history", type=int, default=100,
                    help="MOG2 sliding-window length in frames (default 100).")
    ap.add_argument("--mog2-var-threshold", type=float, default=16.0,
                    help="MOG2 Mahalanobis threshold (default 16.0).")
    ap.add_argument("--no-mog2",    action="store_true",
                    help="skip MOG2 baseline even if OpenCV is available.")

    ap.add_argument("--out-dir",    type=str,   default=None,
                    help="directory to save figures (PNG).")
    ap.add_argument("--save-anim",  type=str,   default=None,
                    help="path for GIF animation (e.g. results/anim.gif).")
    ap.add_argument("--no-show",    action="store_true")
    ap.add_argument("--benchmark",  action="store_true",
                    help="timing and metrics only, no figures.")
    args = ap.parse_args()

    # --- 1. Synthesize video ---
    print("\nGenerating synthetic video ...")
    frames, gt, meta = make_synthetic_video(
        H=args.height, W=args.width, n_frames=args.frames,
        disk_start=args.disk_start, intruder_start=args.intruder_start,
        seed=args.seed,
    )
    print(f"  {args.frames} frames at {args.height}x{args.width}, "
          f"disk appears at k = {args.disk_start}, "
          f"intruder arrives at k = {args.intruder_start}")

    # --- 2. Run methods ---
    t0 = time.perf_counter()
    z1_L, z2_L = run_leso(frames, wo=args.wo)
    t_leso = time.perf_counter() - t0

    t0 = time.perf_counter()
    z1_S, z2_S = run_seso(frames, wo=args.wo, umax=args.umax)
    t_seso = time.perf_counter() - t0

    mask_M, bg_M, t_mog2 = None, None, None
    if not args.no_mog2:
        t0 = time.perf_counter()
        mask_M, bg_M = run_mog2(frames, history=args.mog2_history,
                                var_threshold=args.mog2_var_threshold)
        t_mog2 = time.perf_counter() - t0
        if mask_M is None:
            print("  NOTE: opencv-python not installed; MOG2 baseline skipped.")
            print("        Install with  pip install opencv-python")

    mog2_ok = mask_M is not None

    # --- 3. Metrics ---
    mL = to_mask(frames, z1_L, args.threshold)
    mS = to_mask(frames, z1_S, args.threshold)
    f1_L  = compute_f1(mL, gt)
    f1_S  = compute_f1(mS, gt)
    iou_L = compute_iou(mL, gt)
    iou_S = compute_iou(mS, gt)

    if mog2_ok:
        f1_M  = compute_f1(mask_M, gt)
        iou_M = compute_iou(mask_M, gt)
    else:
        f1_M = None; iou_M = None

    ks = args.intruder_start
    pre  = slice(1, ks)
    post = slice(ks + 5, None)

    print(f"\nObserver parameters:")
    print(f"  wo        = {args.wo}   (bandwidth in 1/frame)")
    print(f"  umax      = {args.umax}  (saturation limit, SESO)")
    print(f"  threshold = {args.threshold}  (on |y - z1|)")
    if mog2_ok:
        print(f"  MOG2      : history = {args.mog2_history}, "
              f"varThreshold = {args.mog2_var_threshold}")

    print(f"\nTiming  ({args.frames} frames, {args.height}x{args.width})")
    print(f"  LESO : {t_leso * 1000:6.1f} ms   "
          f"({args.frames / t_leso:6.0f} frames/s)")
    print(f"  SESO : {t_seso * 1000:6.1f} ms   "
          f"({args.frames / t_seso:6.0f} frames/s)")
    if mog2_ok:
        print(f"  MOG2 : {t_mog2 * 1000:6.1f} ms   "
              f"({args.frames / t_mog2:6.0f} frames/s)")

    print(f"\nDetection quality  (mean F1 per regime)")
    if mog2_ok:
        print(f"  regime            LESO     SESO     MOG2")
        print(f"  pre-intruder    : {f1_L[pre].mean():6.3f}   "
              f"{f1_S[pre].mean():6.3f}   {f1_M[pre].mean():6.3f}")
        print(f"  post-intruder   : {f1_L[post].mean():6.3f}   "
              f"{f1_S[post].mean():6.3f}   {f1_M[post].mean():6.3f}")
        print(f"  whole video     : {f1_L[1:].mean():6.3f}   "
              f"{f1_S[1:].mean():6.3f}   {f1_M[1:].mean():6.3f}")
    else:
        print(f"  regime            LESO     SESO")
        print(f"  pre-intruder    : {f1_L[pre].mean():6.3f}   {f1_S[pre].mean():6.3f}")
        print(f"  post-intruder   : {f1_L[post].mean():6.3f}   {f1_S[post].mean():6.3f}")
        print(f"  whole video     : {f1_L[1:].mean():6.3f}   {f1_S[1:].mean():6.3f}")
    print()

    if args.benchmark:
        return

    # --- 4. Figures ---
    title = (f"ADRC video   |   wo = {args.wo},  umax = {args.umax},  "
             f"thr = {args.threshold}")

    # Sample frames chosen to showcase the three regimes:
    #   k_pre     : moving disk only (before intruder arrives)
    #   k_arrival : intruder fresh, all methods detect it
    #   k_mid     : ~20 frames post-arrival; SESO still holds, MOG2 dropping,
    #               LESO already lost the intruder
    #   k_late    : ~70 frames post-arrival; all methods have converged to
    #               their long-term (absorbed) regime
    #   k_end     : end of video
    k_pre     = max(ks - 10, 1)
    k_arrival = ks + 3
    k_mid     = ks + 20
    k_late    = min(ks + 70, args.frames - 1)
    k_end     = args.frames - 1
    sample_k  = [k_pre, k_arrival, k_mid, k_late, k_end]

    fig_sheet = fig_contact_sheet(frames, z1_L, z1_S, gt, sample_k,
                                  args.threshold, title)

    ir0, ir1, ic0, ic1 = meta["intruder_rect"]
    pixel_rc = ((ir0 + ir1) // 2, (ic0 + ic1) // 2)
    fig_trace = fig_pixel_trace(frames, z1_L, z2_L, z1_S, z2_S,
                                 pixel_rc, ks, args.threshold, title)

    fig_f1 = fig_f1_over_time(f1_L, f1_S, f1_M, ks, title,
                              disk_start=args.disk_start)

    fig_compare = None
    if mog2_ok:
        fig_compare = fig_method_comparison(frames, mL, mS, mask_M, bg_M, gt,
                                            sample_k, title)

    # --- 5. Save / show ---
    if args.out_dir:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        saves = [
            ("contact_sheet.png",      fig_sheet),
            ("pixel_trace.png",        fig_trace),
            ("f1_over_time.png",       fig_f1),
        ]
        if fig_compare is not None:
            saves.append(("method_comparison.png", fig_compare))
        for name, f in saves:
            p = out / name
            f.savefig(p, dpi=120, bbox_inches="tight")
            print(f"saved: {p}")

    if args.save_anim:
        p = Path(args.save_anim)
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nsaving animation to {p} ... (~30 s)")
        save_animation(frames, z1_L, z1_S, args.threshold, str(p), fps=15)
        print(f"saved: {p}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()