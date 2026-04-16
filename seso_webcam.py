#!/usr/bin/env python3
"""
Real-time SESO webcam demo.

Interactive demonstration of the Saturated Extended State Observer
(SESO) for per-pixel background estimation and foreground detection.
Companion code for the paper

    "Extended-Memory Video Foreground Extraction via a Saturated
     Extended State Observer"

Four live panels arranged in a 2x2 grid:

    +----------------+----------------+
    |   camera       |  background    |
    |   (raw feed)   |  (SESO z1)     |
    +----------------+----------------+
    | innovation |e| | foreground     |
    | heatmap        | mask overlay   |
    +----------------+----------------+

The demo showcases the extended-memory property: stationary foreground
stays flagged for much longer than a linear background model would
allow. To see it:

    1. Start the script and LOOK AWAY from the camera for ~3 seconds.
       This lets the observer converge on the empty background.
    2. Walk into frame. You light up red in the mask panel.
    3. Stand still. Keep standing still. At the default parameters
       SESO will keep detecting you for a long time (a linear
       observer would have absorbed you into the background by now).
    4. Drag the umax trackbar to the right (try 0.050). Watch yourself
       disappear from the mask within a second or two --- that is the
       stationary-foreground absorption problem SESO is designed to
       mitigate.
    5. Drag it back to 0.005. Reset with 'r'. Try again.

    Bonus experiment: place an object on your desk. Walk away.
    SESO will flag the object as foreground for minutes, because it
    appeared suddenly and hasn't moved since.

Controls:
    Trackbars    : adjust wo, umax, threshold live
    'r' key      : reset observer state (re-learn background)
    's' key      : save a snapshot of the current display
    'q' or ESC   : quit

Usage:
    python seso_webcam.py
    python seso_webcam.py --camera 1        # for second camera
    python seso_webcam.py --wo 0.05 --umax 0.005    # paper defaults
    python seso_webcam.py --no-mirror       # disable horizontal flip

Dependencies: opencv-python, numpy
"""

import argparse
import time
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit(
        "ERROR: opencv-python is required.\n"
        "Install it with:  pip install opencv-python"
    )


def label_panel(img, text, color=(255, 255, 255)):
    """Draw a small text label in the top-left corner of an image."""
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (5, 5), (15 + tw, 15 + th + 8), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 15 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--camera", type=int, default=0,
                    help="camera index (default 0). Try 1, 2 if you have "
                         "multiple cameras.")
    ap.add_argument("--width", type=int, default=640,
                    help="requested capture width (default 640).")
    ap.add_argument("--height", type=int, default=480,
                    help="requested capture height (default 480).")
    ap.add_argument("--no-mirror", action="store_true",
                    help="don't horizontally flip the video "
                         "(mirror mode is on by default, matches how people "
                         "expect webcams to look).")
    ap.add_argument("--wo", type=float, default=0.05,
                    help="initial observer bandwidth (default 0.05).")
    ap.add_argument("--umax", type=float, default=0.005,
                    help="initial saturation limit (default 0.005).")
    ap.add_argument("--threshold", type=float, default=0.08,
                    help="initial foreground threshold (default 0.08).")
    args = ap.parse_args()

    # --- Open webcam ---------------------------------------------------------
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {args.camera}.")
        print("  Try --camera 1 or --camera 2 if you have multiple cameras.")
        print("  On Linux, check `ls /dev/video*` and permissions.")
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        print("ERROR: Camera opened but failed to read the first frame.")
        cap.release()
        return

    mirror = not args.no_mirror
    if mirror:
        frame = cv2.flip(frame, 1)

    H, W = frame.shape[:2]
    print(f"\nWebcam opened: {W} x {H}")
    print(__doc__)
    print("Tip: look away from the camera for 3 seconds at startup so the "
          "observer can converge on the empty background.")
    print()

    # --- Initialize observer state ------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    z1 = gray.copy()
    z2 = np.zeros_like(z1)

    # --- Window and trackbars -----------------------------------------------
    win = "SESO webcam demo   |   q=quit   r=reset   s=save"
    cv2.namedWindow(win)
    # Trackbars work with integers, so we scale the parameters.
    cv2.createTrackbar("wo  x1000",   win, int(args.wo * 1000),
                       200, lambda x: None)
    cv2.createTrackbar("umax x10000", win, int(args.umax * 10000),
                       500, lambda x: None)
    cv2.createTrackbar("thr x1000",   win, int(args.threshold * 1000),
                       300, lambda x: None)

    # --- FPS tracking (exponential moving average) --------------------------
    fps_ewma = 30.0
    last_t = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # Some webcams occasionally drop a frame; just try again.
            continue

        if mirror:
            frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Read current parameter values from trackbars.
        wo   = max(cv2.getTrackbarPos("wo  x1000",   win) / 1000.0,  0.001)
        umax = max(cv2.getTrackbarPos("umax x10000", win) / 10000.0, 0.0001)
        thr  = cv2.getTrackbarPos("thr x1000", win) / 1000.0
        L1 = 2.0 * wo
        L2 = wo * wo

        # ---- SESO update (one frame) ---------------------------------------
        e = gray - z1
        e_sat = np.clip(e, -umax, umax)
        z1 = z1 + z2 + L1 * e_sat
        z2 = z2 + L2 * e_sat

        innovation = np.abs(gray - z1)
        mask = innovation > thr

        # ---- Build the four display panels ---------------------------------
        # 1. Raw camera feed
        p_raw = frame

        # 2. Background estimate z1
        p_bg = cv2.cvtColor(
            np.clip(z1 * 255.0, 0, 255).astype(np.uint8),
            cv2.COLOR_GRAY2BGR,
        )

        # 3. Innovation magnitude as a heatmap.
        # Multiply by 5 to make subtle innovations visible; saturates at 0.2
        # (which is a large |e| for webcam data).
        inn_u8 = np.clip(innovation * 255.0 * 5.0, 0, 255).astype(np.uint8)
        p_inn = cv2.applyColorMap(inn_u8, cv2.COLORMAP_INFERNO)

        # 4. Foreground mask overlaid on raw frame.
        # Where mask is True, blend 50% red into the raw pixels.
        p_mask = frame.copy()
        if mask.any():
            blend = frame.astype(np.float32)
            blend[mask] = 0.5 * blend[mask] + 0.5 * np.array([0, 0, 220])
            p_mask = np.clip(blend, 0, 255).astype(np.uint8)

        # Labels
        p_raw  = label_panel(p_raw,  "camera")
        p_bg   = label_panel(p_bg,   "background  z1")
        p_inn  = label_panel(p_inn,  "innovation  |e|")
        p_mask = label_panel(p_mask, "foreground  mask",
                             color=(100, 100, 255))

        # 2x2 grid
        top = np.hstack([p_raw, p_bg])
        bot = np.hstack([p_inn, p_mask])
        grid = np.vstack([top, bot])

        # ---- FPS and status bar --------------------------------------------
        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps_ewma = 0.9 * fps_ewma + 0.1 * (1.0 / dt)

        status_h = 42
        status = np.zeros((status_h, grid.shape[1], 3), dtype=np.uint8)
        txt = (f"wo = {wo:.3f}    "
               f"umax = {umax:.4f}    "
               f"thr = {thr:.3f}    |    "
               f"fps = {fps_ewma:5.0f}    |    "
               f"mean |e| = {innovation.mean():.3f}    "
               f"mask density = {mask.mean() * 100:5.1f}%")
        cv2.putText(status, txt, (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210),
                    1, cv2.LINE_AA)

        display = np.vstack([grid, status])
        cv2.imshow(win, display)

        # ---- Keyboard ------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # 27 = ESC
            break
        elif key == ord('r'):
            z1 = gray.copy()
            z2 = np.zeros_like(z1)
            print("observer state reset")
        elif key == ord('s'):
            fname = f"seso_snap_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            print(f"saved snapshot: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()