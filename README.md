# SESO: Extended-Memory Video Foreground Extraction

Companion code for the paper

> **Extended-Memory Video Foreground Extraction via a Saturated Extended State Observer**
> Emre Hatay, Cleveland State University — Center for Advanced Control Technologies (CACT)

This repo contains:

- `adrc_video.py` — the main benchmark script (generates a synthetic test video, runs LESO / SESO / MOG2, produces the paper's figures)
- `adrc_analyze.py` — a diagnostic report generator (multi-seed metrics, parameter sweeps, timing, per-phenomenon analysis)
- `seso_webcam.py` — an interactive live demo that runs SESO on your webcam
- `main.tex` — the paper source (IEEEtran, compiles standalone)

## What is this?

Adaptive background models used for video foreground extraction — running averages, linear observers, Gaussian mixture models (MOG2), even deep networks — share a common failure mode: **foreground objects that come to rest are gradually absorbed into the background estimate and lost.** An abandoned bag becomes part of the wall. A parked vehicle becomes part of the street. A stationary intruder disappears from the mask within a handful of frames.

This is the *stationary-foreground absorption problem*. It has been studied for decades (see Cuevas, Yáñez & García, *CVIU* 2016 for a survey). Every adaptive method inherits a tradeoff: update fast enough to track illumination drift, but slow enough not to absorb stationary foreground. Those two requirements conflict directly.

SESO is a control-theoretic answer: a **Saturated Extended State Observer** running independently at every pixel, with a saturation element on the observer's innovation signal. The saturation bounds the rate at which the background estimate can drift, regardless of how large the true measurement change is. Stationary foreground is therefore retained for much longer than any linear method would allow.

### Key results on the synthetic benchmark

| Method | F1 post-intruder | T₅₀ | Moving disk F1 | Stationary intruder F1 | FPS |
|---|---|---|---|---|---|
| **SESO** | **0.847** | **124** | **0.976** | **0.789** | 46,800 |
| MOG2 | 0.340 | 10 | 0.997 | 0.035 | 2,600 |
| LESO | 0.257 | 9 | 0.511 | 0.027 | 58,900 |

*T₅₀ is frames after intruder arrival before F1 drops below 0.5. SESO is ~13× slower to lose a stationary object than MOG2, while matching MOG2 on moving-object detection and running ~18× faster.*

## Installation

```bash
git clone https://github.com/ehatay/seso-foreground-extraction.git
cd seso-foreground-extraction
pip install -r requirements.txt
```

Python 3.8 or later. The only hard dependencies are `numpy` and `matplotlib`. `opencv-python` is required for the MOG2 baseline and the webcam demo but is not needed for the SESO/LESO code itself.

## Quick start

### 1. Reproduce the paper's figures

```bash
python adrc_video.py --wo 0.05 --umax 0.005 --threshold 0.08 --out-dir figures
```

This generates the synthetic benchmark video and produces four PNGs in `figures/`:

- `pixel_trace.png` — observer states for a single pixel inside the stationary intruder (Figure 2)
- `f1_over_time.png` — per-frame F1 for all three methods (Figure 3)
- `contact_sheet.png` — observer-internals at five representative frames (Figure 4)
- `method_comparison.png` — foreground masks for all three methods (Figure 5)

Run it without arguments to use default parameters (not the paper's tuned values):

```bash
python adrc_video.py
```

Common flags:

| Flag | Default | Purpose |
|---|---|---|
| `--wo` | 0.10 | ESO bandwidth ω₀ in 1/frame (0.05 for paper) |
| `--umax` | 0.015 | Saturation limit for SESO (0.005 for paper) |
| `--threshold` | 0.08 | Detection threshold θ on \|y − ẑ₁\| |
| `--disk-start` | 25 | Frame at which the moving disk appears |
| `--intruder-start` | 50 | Frame at which the stationary intruder appears |
| `--out-dir` | *(none)* | Directory for saved PNGs |
| `--no-show` | — | Don't open matplotlib windows |
| `--no-mog2` | — | Skip MOG2 baseline even if OpenCV is available |
| `--benchmark` | — | Print timing only, no figures |

### 2. Reproduce the diagnostic report

```bash
python adrc_analyze.py --wo 0.05 --umax 0.005 --threshold 0.08
```

Produces a structured 10-section report covering multi-seed metrics, time-to-absorption, per-phenomenon F1 split, parameter sweeps (SESO over ω₀ × u_max, MOG2 over history × variance threshold), pixel-level dynamics, timing, and approximate FLOP counts. Add `--quick` to skip the parameter sweeps.

### 3. Run the live webcam demo

```bash
python seso_webcam.py
```

Opens a window showing your webcam feed with SESO running in real time. Four panels: raw camera, background estimate ẑ₁, innovation heatmap, foreground mask. Interactive trackbars let you adjust ω₀, u_max, and threshold live.

**To see the extended-memory property**, follow the demo protocol from the paper:

1. Start the script and **look away from the camera for ~3 seconds.** This lets the observer converge on the empty scene.
2. Walk into frame. You light up red in the mask panel.
3. **Stand perfectly still.** SESO keeps detecting you for a long time — a plain linear model would have absorbed you into the background by now.
4. Drag the u_max trackbar to the right (try 0.050). Watch yourself disappear from the mask within a second or two. That is the stationary-foreground absorption problem, made visible.
5. Drag u_max back to 0.005. Press `r` to reset. Try again.

**Bonus demo:** place an object on your desk and walk away. SESO will flag the object as foreground for minutes, because it appeared suddenly and hasn't moved since.

Controls: `q`/ESC to quit, `r` to reset observer state, `s` to save a snapshot, trackbars to adjust parameters.

## Reproducing the paper exactly

The paper's numbers come from the following single command:

```bash
python adrc_video.py --wo 0.05 --umax 0.005 --threshold 0.08 \
                      --disk-start 25 --intruder-start 50 \
                      --out-dir figures --no-show
```

This produces all four figures at the tuned parameters described in Section IV. The script is deterministic (uses a fixed seed by default), so results are reproducible frame-for-frame.

For multi-seed statistics (as reported in some of the tables):

```bash
python adrc_analyze.py --wo 0.05 --umax 0.005 --threshold 0.08 --seeds 5
```

## Algorithm

All three methods operate per-pixel. For each pixel at location (i, j), the measured intensity y(k) is treated as a scalar time series driven by a total disturbance f:

```
y_dot(t) = f(t),    y(k) = x_1(k) + v(k)
```

With extended state x_2 := f, the ESO estimates both state and total disturbance from y alone:

```python
# SESO update (one frame, vectorized across all pixels)
e = y - z1                       # innovation
e_sat = clip(e, -umax, umax)     # saturation (the only nonlinearity)
z1 = z1 + z2 + L1 * e_sat        # background estimate update
z2 = z2 +      L2 * e_sat        # total disturbance estimate update
mask = abs(e) > threshold        # foreground detection
```

where `L1 = 2*wo` and `L2 = wo**2` (bandwidth parameterization, Gao 2003). Setting `umax = infinity` recovers the plain linear ESO (LESO).

Detection reads off the *innovation* e, not the extended state ẑ₂. The paper (Section II.C) explains why: the ESO naturally separates slow and fast components of f — slow components (illumination drift) get absorbed into ẑ₂ while fast components (foreground arrivals) stay visible in e. Thresholding e gives detection of fast disturbances specifically, which is what we want for foreground.

## Repository layout

```
seso-foreground-extraction/
├── README.md              this file
├── requirements.txt       Python dependencies
├── main.tex               paper source (IEEEtran)
├── adrc_video.py          main benchmark script
├── adrc_analyze.py        diagnostic report generator
├── seso_webcam.py         interactive live demo
└── figures/               generated on first run
    ├── pixel_trace.png
    ├── f1_over_time.png
    ├── contact_sheet.png
    └── method_comparison.png
```

## Limitations

- All reported results are on synthetic video with known ground truth. Real-world imagery brings shadows, auto-exposure changes, and inter-pixel correlations that the per-pixel independent observer does not model.
- SESO delays absorption but does not prevent it altogether. A truly persistent intruder will eventually be absorbed; a complete solution would require an outer logic that freezes a pixel's background estimate once persistently classified as foreground. This hybrid switched observer is a natural next step.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{hatay2026seso,
  author       = {Emre Hatay},
  title        = {Extended-Memory Video Foreground Extraction via a
                  Saturated Extended State Observer},
  year         = {2026},
  institution  = {Cleveland State University, Center for Advanced
                  Control Technologies},
  howpublished = {\url{https://github.com/ehatay/seso-foreground-extraction}}
}
```

## Acknowledgements

Developed at the Center for Advanced Control Technologies (CACT) under the supervision of Dr. Zhiqiang Gao, Cleveland State University.

## License

Released under the MIT License. See `LICENSE` for details.