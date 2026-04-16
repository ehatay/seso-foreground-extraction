#!/usr/bin/env python3
"""
Prints a structured text report covering everything needed to analyze
the results honestly:

    1. Experiment configuration
    2. Multi-seed headline metrics (mean +/- std)
    3. Time-to-absorption analysis
    4. Per-phenomenon F1 (moving disk vs stationary intruder separately)
    5. Parameter sensitivity sweeps (SESO over wo x umax, MOG2 over history)
    6. Single-pixel observer dynamics (numeric dump at key frames)
    7. Precision / recall breakdown
    8. Timing and FLOP comparison
    9. Overall summary and rankings

Usage:
    python adrc_analyze.py                       # full analysis (~30 s)
    python adrc_analyze.py --quick               # skip parameter sweeps
    python adrc_analyze.py --out report.txt      # save report to file
    python adrc_analyze.py --seeds 10            # more seeds (slower)

Requires adrc_video.py in the same directory.
Dependencies: numpy. Optional: opencv-python (for the MOG2 section).
"""

import argparse
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import the algorithms from the main script
# ---------------------------------------------------------------------------
try:
    from adrc_video import (
        make_synthetic_video,
        run_leso, run_seso, run_mog2,
        to_mask, compute_f1, compute_iou,
    )
except ImportError as exc:
    print("ERROR: could not import from adrc_video.py. "
          "Make sure it is in the same directory.")
    print("       ImportError:", exc)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Output helpers (everything goes through OUT so we can capture to file)
# ---------------------------------------------------------------------------

OUT = sys.stdout

def out(*args, **kw):
    print(*args, file=OUT, **kw)

def section(n, title):
    out()
    out("=" * 78)
    out(f" [{n}]  {title}")
    out("=" * 78)

def subsection(title):
    out()
    out("-- " + title + " " + "-" * max(0, 75 - len(title)))

def fmt(x, width=8, prec=3):
    """Format a float for a table cell."""
    if x is None:
        return " " * (width - 1) + "-"
    if not np.isfinite(x):
        return " " * (width - 3) + "inf"
    return f"{x:>{width}.{prec}f}"

def mean_std(arr):
    """Return (mean, std) or (arr[0], 0.0) if single value."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


# ---------------------------------------------------------------------------
# Extra metrics
# ---------------------------------------------------------------------------

def compute_precision_recall(pred, gt):
    """Per-frame precision and recall."""
    n = pred.shape[0]
    p = np.zeros(n)
    r = np.zeros(n)
    for k in range(n):
        a = pred[k]; g = gt[k]
        tp = np.logical_and(a, g).sum()
        fp = np.logical_and(a, ~g).sum()
        fn = np.logical_and(~a, g).sum()
        p[k] = tp / (tp + fp + 1e-9)
        r[k] = tp / (tp + fn + 1e-9)
    return p, r


def time_to_f1_drop(f1_series, start_idx, threshold):
    """
    Frames after start_idx until F1 drops below `threshold`. Returns None
    if never drops. If F1 at start is already below threshold returns 0.
    """
    for k in range(start_idx, len(f1_series)):
        if f1_series[k] < threshold:
            return k - start_idx
    return None


def phenomenon_split_f1(pred, gt, meta):
    """
    Compute F1 separately on two spatial regions:
      disk region     = full frame MINUS the intruder rectangle
      intruder region = only within the intruder rectangle

    Returns (f1_disk, f1_intruder), each of length n_frames.
    """
    ir0, ir1, ic0, ic1 = meta["intruder_rect"]
    H, W = pred.shape[1:]
    in_intruder = np.zeros((H, W), dtype=bool)
    in_intruder[ir0:ir1, ic0:ic1] = True
    in_disk_region = ~in_intruder

    n = pred.shape[0]
    f1_disk     = np.zeros(n)
    f1_intruder = np.zeros(n)

    for k in range(n):
        p = pred[k]; g = gt[k]

        # Disk region
        pd = p & in_disk_region; gd = g & in_disk_region
        tp = (pd & gd).sum()
        fp = (pd & ~gd).sum()
        fn = (~pd & gd & in_disk_region).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1_disk[k] = 2 * prec * rec / (prec + rec + 1e-9)

        # Intruder region
        pi = p & in_intruder; gi = g & in_intruder
        tp = (pi & gi).sum()
        fp = (pi & ~gi).sum()
        fn = (~pi & gi & in_intruder).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1_intruder[k] = 2 * prec * rec / (prec + rec + 1e-9)

    return f1_disk, f1_intruder


# ---------------------------------------------------------------------------
# Experiment wrappers
# ---------------------------------------------------------------------------

def run_one_seed(seed, wo, umax, threshold,
                 mog2_history, mog2_var, H, W, n_frames, intruder_start,
                 want_mog2=True):
    """Run all three methods on one seed. Returns a dict of arrays."""
    frames, gt, meta = make_synthetic_video(
        H=H, W=W, n_frames=n_frames,
        intruder_start=intruder_start, seed=seed,
    )

    t0 = time.perf_counter()
    z1_L, z2_L = run_leso(frames, wo=wo)
    t_leso = time.perf_counter() - t0

    t0 = time.perf_counter()
    z1_S, z2_S = run_seso(frames, wo=wo, umax=umax)
    t_seso = time.perf_counter() - t0

    mask_M, bg_M, t_mog2 = None, None, None
    if want_mog2:
        t0 = time.perf_counter()
        mask_M, bg_M = run_mog2(frames, history=mog2_history,
                                var_threshold=mog2_var)
        t_mog2 = time.perf_counter() - t0

    mL = to_mask(frames, z1_L, threshold)
    mS = to_mask(frames, z1_S, threshold)

    return dict(
        frames=frames, gt=gt, meta=meta,
        z1_L=z1_L, z2_L=z2_L, mask_L=mL, t_L=t_leso,
        z1_S=z1_S, z2_S=z2_S, mask_S=mS, t_S=t_seso,
        mask_M=mask_M, bg_M=bg_M, t_M=t_mog2,
    )


def summarize_one(run, pre_slice, post_slice):
    """Compute headline metrics for one run."""
    out_rows = {}
    methods = [("LESO", run["mask_L"]),
               ("SESO", run["mask_S"])]
    if run["mask_M"] is not None:
        methods.append(("MOG2", run["mask_M"]))

    for name, pred in methods:
        gt = run["gt"]
        f1  = compute_f1(pred, gt)
        iou = compute_iou(pred, gt)
        prec, rec = compute_precision_recall(pred, gt)
        f1_disk, f1_intr = phenomenon_split_f1(pred, gt, run["meta"])
        out_rows[name] = dict(
            f1=f1, iou=iou, prec=prec, rec=rec,
            f1_disk=f1_disk, f1_intr=f1_intr,
            f1_pre=f1[pre_slice].mean(),
            f1_post=f1[post_slice].mean(),
            f1_whole=f1[1:].mean(),
            iou_pre=iou[pre_slice].mean(),
            iou_post=iou[post_slice].mean(),
            iou_whole=iou[1:].mean(),
            prec_post=prec[post_slice].mean(),
            rec_post=rec[post_slice].mean(),
            f1_disk_post=f1_disk[post_slice].mean(),
            f1_intr_post=f1_intr[post_slice].mean(),
        )
    return out_rows


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def report_config(args, ks, n_seeds, quick):
    section(1, "EXPERIMENT CONFIGURATION")
    out(f"  video               : {args.height} x {args.width}, "
        f"{args.frames} frames")
    out(f"  intruder arrives at : k = {ks}")
    out(f"  observer bandwidth  : wo    = {args.wo}")
    out(f"  saturation limit    : umax  = {args.umax}")
    out(f"  detection threshold : thr   = {args.threshold}  (on |y - z1|)")
    out(f"  MOG2 history        : {args.mog2_history} frames")
    out(f"  MOG2 varThreshold   : {args.mog2_var}")
    out(f"  seeds used          : {n_seeds}")
    out(f"  mode                : {'quick (no sweeps)' if quick else 'full'}")


def report_multiseed(results_per_seed, pre_slice, post_slice):
    section(2, "HEADLINE METRICS  (mean +/- std across seeds)")

    methods = ["LESO", "SESO"]
    # Include MOG2 only if present in first seed
    if "MOG2" in results_per_seed[0]:
        methods.append("MOG2")

    def collect(method, key):
        return np.array([r[method][key] for r in results_per_seed])

    subsection("F1 score")
    header = "  regime          " + "   ".join(
        [f"{m:>15}" for m in methods])
    out(header)
    for regime, key in [("pre-intruder", "f1_pre"),
                        ("post-intruder", "f1_post"),
                        ("whole video", "f1_whole")]:
        row = f"  {regime:<14}:  "
        for m in methods:
            mu, sd = mean_std(collect(m, key))
            row += f"{mu:6.3f} +/- {sd:5.3f}   "
        out(row.rstrip())

    subsection("IoU score")
    out(header)
    for regime, key in [("pre-intruder", "iou_pre"),
                        ("post-intruder", "iou_post"),
                        ("whole video", "iou_whole")]:
        row = f"  {regime:<14}:  "
        for m in methods:
            mu, sd = mean_std(collect(m, key))
            row += f"{mu:6.3f} +/- {sd:5.3f}   "
        out(row.rstrip())

    subsection("Post-intruder precision and recall")
    out(header)
    for regime, key in [("precision", "prec_post"),
                        ("recall", "rec_post")]:
        row = f"  {regime:<14}:  "
        for m in methods:
            mu, sd = mean_std(collect(m, key))
            row += f"{mu:6.3f} +/- {sd:5.3f}   "
        out(row.rstrip())


def report_time_to_absorption(results_per_seed, ks):
    section(3, "TIME-TO-ABSORPTION  (frames after intruder arrival)")
    out("  Frames after k = %d until F1 drops below threshold.\n" % ks)
    out("  'never' means F1 remained above threshold for the rest of the video.\n")

    methods = ["LESO", "SESO"]
    if "MOG2" in results_per_seed[0]:
        methods.append("MOG2")

    thresholds = [0.90, 0.75, 0.50, 0.30]

    header = "  F1 drops below   " + "   ".join(f"{m:>15}" for m in methods)
    out(header)

    for thr in thresholds:
        row = f"  {thr:.2f}            :  "
        for m in methods:
            tvals = []
            for r in results_per_seed:
                t = time_to_f1_drop(r[m]["f1"], ks, thr)
                tvals.append(t if t is not None else np.nan)
            tvals = np.array(tvals, dtype=np.float64)
            finite = tvals[~np.isnan(tvals)]
            if finite.size == 0:
                row += f"{'never':>15}   "
            elif finite.size < len(tvals):
                mu, sd = mean_std(finite)
                row += f"{mu:5.1f} +/- {sd:4.1f} ({len(finite)}/{len(tvals)})   "
            else:
                mu, sd = mean_std(finite)
                row += f"{mu:5.1f} +/- {sd:4.1f}         "
        out(row.rstrip())


def report_phenomenon_split(results_per_seed, post_slice):
    section(4, "PER-PHENOMENON F1  (post-intruder regime, mean +/- std)")
    out("  The mask is split spatially into two regions and F1 computed")
    out("  on each separately:")
    out("    disk region     = full frame EXCLUDING the intruder rectangle")
    out("    intruder region = only WITHIN the intruder rectangle")
    out("")

    methods = ["LESO", "SESO"]
    if "MOG2" in results_per_seed[0]:
        methods.append("MOG2")

    def collect(method, key):
        return np.array([r[method][key] for r in results_per_seed])

    header = "  region          " + "   ".join(f"{m:>15}" for m in methods)
    out(header)
    for region, key in [("moving disk", "f1_disk_post"),
                        ("stat. intruder", "f1_intr_post")]:
        row = f"  {region:<14}:  "
        for m in methods:
            mu, sd = mean_std(collect(m, key))
            row += f"{mu:6.3f} +/- {sd:5.3f}   "
        out(row.rstrip())


def report_seso_sensitivity(args, ks, pre_slice, post_slice, seed=0):
    section(5, "SESO PARAMETER SENSITIVITY  (post-intruder F1)")
    out(f"  Single-seed sweep (seed={seed}). Values are mean F1 over the")
    out(f"  post-intruder regime (frames {ks + 5} .. {args.frames - 1}).\n")

    wos   = [0.05, 0.08, 0.10, 0.15, 0.20]
    umaxs = [0.005, 0.010, 0.015, 0.020, 0.030, 0.050]

    frames, gt, meta = make_synthetic_video(
        H=args.height, W=args.width, n_frames=args.frames,
        intruder_start=args.intruder_start, seed=seed,
    )

    header = "  wo \\ umax       " + "".join(
        f"{u:>9.3f}" for u in umaxs)
    out(header)
    for wo in wos:
        row = f"  wo = {wo:5.2f}      : "
        for um in umaxs:
            z1, _ = run_seso(frames, wo=wo, umax=um)
            m = to_mask(frames, z1, args.threshold)
            f1 = compute_f1(m, gt)
            row += f"{f1[post_slice].mean():9.3f}"
        out(row)
    out("")
    out("  ^ Each cell is post-intruder F1 for that (wo, umax) pair.")
    out("  Look for where the landscape peaks -- is the default choice near optimum?")


def report_mog2_sensitivity(args, ks, pre_slice, post_slice, seed=0):
    section(6, "MOG2 SENSITIVITY  (post-intruder F1)")
    out(f"  Single-seed sweep (seed={seed}) over MOG2 history length.\n")

    frames, gt, meta = make_synthetic_video(
        H=args.height, W=args.width, n_frames=args.frames,
        intruder_start=args.intruder_start, seed=seed,
    )

    histories = [30, 50, 75, 100, 150, 200, 300]
    var_thrs  = [8.0, 16.0, 25.0, 50.0]

    header = "  hist. \\ varThr " + "".join(
        f"{v:>9.1f}" for v in var_thrs)
    out(header)
    for hist in histories:
        row = f"  history = {hist:4d}  : "
        for vt in var_thrs:
            res = run_mog2(frames, history=hist, var_threshold=vt)
            if res[0] is None:
                row += "        -"
                continue
            mask_M, _ = res
            f1 = compute_f1(mask_M, gt)
            row += f"{f1[post_slice].mean():9.3f}"
        out(row)
    out("")
    out("  ^ Shows whether our chosen MOG2 hyperparameters are well-tuned")
    out("  for the given video. If MOG2 can do much better with different")
    out("  params, the comparison in the paper needs qualifying.")


def report_pixel_trace(run, meta, ks):
    section(7, "SINGLE-PIXEL OBSERVER DYNAMICS  (seed=0)")

    ir0, ir1, ic0, ic1 = meta["intruder_rect"]
    r_intr = (ir0 + ir1) // 2
    c_intr = (ic0 + ic1) // 2
    out(f"  Pixel at (row={r_intr}, col={c_intr}) -- inside stationary intruder.")
    out("")

    frames = run["frames"]
    z1_L = run["z1_L"]; z1_S = run["z1_S"]
    bg_M = run["bg_M"]

    out("    k   |   y    |  z1_L   |e_L|  |  z1_S   |e_S|  |" +
        ("  bg_M   |e_M|  |" if bg_M is not None else ""))
    out("  " + "-" * (54 + (16 if bg_M is not None else 0)))

    key_ks = [0, 20, 40, 45, 49, 50, 52, 55, 60, 65, 70, 75, 80, 90, 100,
              120, 150, 175, 199]
    key_ks = [k for k in key_ks if k < run["frames"].shape[0]]

    for k in key_ks:
        y = frames[k, r_intr, c_intr]
        l1 = z1_L[k, r_intr, c_intr]
        s1 = z1_S[k, r_intr, c_intr]
        eL = abs(y - l1)
        eS = abs(y - s1)
        marker = " <<< intruder arrives" if k == ks else ""
        row = f"  {k:>4}  | {y:6.3f} | {l1:6.3f}  {eL:6.3f} | {s1:6.3f}  {eS:6.3f} |"
        if bg_M is not None:
            m1 = bg_M[k, r_intr, c_intr]
            eM = abs(y - m1)
            row += f" {m1:6.3f}  {eM:6.3f} |"
        out(row + marker)

    out("")
    out("  Pixel inside the moving-disk-trail region (samples a pixel the")
    out("  disk passes over):")
    out("")

    # Find a pixel on the disk path that is NOT in the intruder rectangle
    # The disk center oscillates around (x,y) ~ (W*0.28, H*0.42); pick
    # something on that path.
    r_disk = int(run["meta"]["H"] * 0.42)
    c_disk = int(run["meta"]["W"] * 0.50)
    out(f"  Pixel at (row={r_disk}, col={c_disk}) -- along moving-disk path.")
    out("")
    out("    k   |   y    |  z1_L   |e_L|  |  z1_S   |e_S|  |" +
        ("  bg_M   |e_M|  |" if bg_M is not None else ""))
    out("  " + "-" * (54 + (16 if bg_M is not None else 0)))
    for k in [0, 20, 40, 50, 60, 80, 100, 120, 150, 199]:
        if k >= run["frames"].shape[0]:
            continue
        y = frames[k, r_disk, c_disk]
        l1 = z1_L[k, r_disk, c_disk]
        s1 = z1_S[k, r_disk, c_disk]
        eL = abs(y - l1)
        eS = abs(y - s1)
        row = f"  {k:>4}  | {y:6.3f} | {l1:6.3f}  {eL:6.3f} | {s1:6.3f}  {eS:6.3f} |"
        if bg_M is not None:
            m1 = bg_M[k, r_disk, c_disk]
            eM = abs(y - m1)
            row += f" {m1:6.3f}  {eM:6.3f} |"
        out(row)


def report_timing(results_per_seed, args):
    section(8, "TIMING  (per-video mean +/- std across seeds)")

    methods = [("LESO", "t_L"), ("SESO", "t_S")]
    if results_per_seed[0]["t_M"] is not None:
        methods.append(("MOG2", "t_M"))

    px_per_frame = args.height * args.width

    out(f"  Video size: {args.frames} frames x {args.height} x {args.width}"
        f" = {args.frames * px_per_frame:,} pixel-updates")
    out("")
    out(f"  {'method':<6}  {'total ms':>10}  {'ms/frame':>10}  "
        f"{'frames/s':>10}  {'relative':>10}")
    out("  " + "-" * 62)

    ts = {m: np.array([r[k] for r in results_per_seed]) for m, k in methods}
    t_leso_mean = ts["LESO"].mean()

    for name, key in methods:
        arr_ms = ts[name] * 1000.0
        mu_ms, sd_ms = mean_std(arr_ms)
        ms_per_frame = mu_ms / args.frames
        fps = args.frames / (mu_ms / 1000.0)
        rel = arr_ms.mean() / (t_leso_mean * 1000.0)
        out(f"  {name:<6}  {mu_ms:>8.1f}   {ms_per_frame:>10.3f}  "
            f"{fps:>10.0f}  {rel:>9.2f}x")


def report_flops(args):
    section(9, "APPROXIMATE FLOP COSTS  (per frame)")
    H = args.height; W = args.width
    px = H * W

    leso_flops = 6 * px     # e, z1 += z2 + L1*e, z2 += L2*e -> ~6 ops/pixel
    seso_flops = 8 * px     # +2 for saturation (compare + clip)
    mog2_flops = 50 * px    # rough: per-pixel Gaussian updates, K=5 clusters

    out(f"  Image size: {H} x {W} = {px:,} pixels")
    out("")
    out(f"  {'method':<6}  {'FLOPs/frame':>15}  {'vs LESO':>10}")
    out("  " + "-" * 40)
    out(f"  LESO    {leso_flops:>15,}   {1.0:>9.1f}x  (reference)")
    out(f"  SESO    {seso_flops:>15,}   "
        f"{seso_flops / leso_flops:>9.1f}x  (+ saturation)")
    out(f"  MOG2    {mog2_flops:>15,}   "
        f"{mog2_flops / leso_flops:>9.1f}x  (rough estimate)")
    out("")
    out("  MOG2 estimate assumes 5 Gaussian clusters per pixel with")
    out("  Mahalanobis evaluation and parameter updates. The true number")
    out("  is likely between 30-80 FLOPs/pixel depending on cluster")
    out("  membership; wall-clock timing above is the more reliable signal.")


def report_summary(results_per_seed):
    section(10, "SUMMARY")
    methods = ["LESO", "SESO"]
    if "MOG2" in results_per_seed[0]:
        methods.append("MOG2")

    def mean_of(method, key):
        return np.mean([r[method][key] for r in results_per_seed])

    best = {}
    for key, label in [("f1_pre",  "pre-intruder F1"),
                       ("f1_post", "post-intruder F1"),
                       ("f1_whole", "whole-video F1"),
                       ("f1_intr_post", "stationary intruder F1 (post)"),
                       ("f1_disk_post", "moving disk F1 (post)")]:
        scores = {m: mean_of(m, key) for m in methods}
        winner = max(scores, key=scores.get)
        best[key] = (winner, scores)
        rest = ", ".join(f"{m}={v:.3f}" for m, v in scores.items())
        out(f"  {label:<32}: {winner} wins   ({rest})")

    out("")
    out("  Use the above to decide which claims the paper can defend.")
    out("  The story that the paper should tell is whichever one is")
    out("  consistent across all diagnostics above.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global OUT

    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--width",  type=int, default=160)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--intruder-start", type=int, default=50)
    ap.add_argument("--wo",        type=float, default=0.10)
    ap.add_argument("--umax",      type=float, default=0.015)
    ap.add_argument("--threshold", type=float, default=0.08)
    ap.add_argument("--mog2-history", type=int,   default=100)
    ap.add_argument("--mog2-var",     type=float, default=16.0)
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of seeds for multi-seed report (default 5).")
    ap.add_argument("--quick", action="store_true",
                    help="skip parameter sensitivity sweeps.")
    ap.add_argument("--out", type=str, default=None,
                    help="save the report to this file in addition to stdout.")
    args = ap.parse_args()

    # If --out, tee to file.
    if args.out:
        buf = StringIO()
        class Tee:
            def write(self, s): sys.__stdout__.write(s); buf.write(s)
            def flush(self):    sys.__stdout__.flush()
        OUT = Tee()

    ks = args.intruder_start
    pre  = slice(1, ks)
    post = slice(ks + 5, None)

    # Header
    out("#" * 78)
    out("#  ADRC VIDEO BACKGROUND SUBTRACTION - DIAGNOSTIC REPORT")
    out("#  Generated by adrc_analyze.py")
    out("#" * 78)

    report_config(args, ks, args.seeds, args.quick)

    # Multi-seed runs
    out("\n  Running %d seeds ..." % args.seeds)
    results_per_seed = []
    for seed in range(args.seeds):
        run = run_one_seed(
            seed, args.wo, args.umax, args.threshold,
            args.mog2_history, args.mog2_var,
            args.height, args.width, args.frames, args.intruder_start,
            want_mog2=True,
        )
        summary = summarize_one(run, pre, post)
        results_per_seed.append(summary)
        # Keep seed 0 for the single-pixel trace
        if seed == 0:
            run_seed0 = run

    report_multiseed(results_per_seed, pre, post)
    report_time_to_absorption(results_per_seed, ks)
    report_phenomenon_split(results_per_seed, post)

    if not args.quick:
        report_seso_sensitivity(args, ks, pre, post, seed=0)
        if run_seed0["mask_M"] is not None:
            report_mog2_sensitivity(args, ks, pre, post, seed=0)

    report_pixel_trace(run_seed0, run_seed0["meta"], ks)

    # Timing: collect the per-seed timings we recorded
    # need to stash them back -- re-run to collect properly
    per_seed_timings = []
    out("\n  (Re-running for timing stability ...)")
    for seed in range(args.seeds):
        run = run_one_seed(
            seed, args.wo, args.umax, args.threshold,
            args.mog2_history, args.mog2_var,
            args.height, args.width, args.frames, args.intruder_start,
            want_mog2=True,
        )
        per_seed_timings.append(dict(
            t_L=run["t_L"], t_S=run["t_S"], t_M=run["t_M"]
        ))
    report_timing(per_seed_timings, args)
    report_flops(args)
    report_summary(results_per_seed)

    out("\n" + "#" * 78)
    out("#  END OF REPORT")
    out("#" * 78 + "\n")

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(buf.getvalue())
        print(f"\n[report also saved to {args.out}]", file=sys.__stdout__)


if __name__ == "__main__":
    main()