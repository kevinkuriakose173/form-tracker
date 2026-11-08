# Form Tracker

A Python prototype for reviewing bench-press footage with body landmarks and a bar-path overlay. MediaPipe estimates body landmarks, OpenCV locates the visible bar end through template matching, and FFmpeg exports an annotated replay.

Custom model workflow: see [TRAINING.md](TRAINING.md) for frame extraction, local annotation, YOLO keypoint fine-tuning, and session-based evaluation. The template matcher remains the default; custom bar-end weights must be trained before enabling model-based inference.

## Setup

Tested with Python 3.12 on macOS Apple Silicon.

```sh
git clone https://github.com/kevinkuriakose173/form-tracker.git
cd form-tracker
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p models form_videos
curl -fL 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task' -o models/pose_landmarker_heavy.task
```

Place the source clip at `form_videos/BarbellBenchPress.mp4`. Videos, model weights, generated results, and the Python environment are not included in the repository.

## Run

```sh
source .venv/bin/activate
python track_bench.py
```

The script reads the saved video and writes to `results/latest/`. It does not open an interactive window. Repeated runs replace output files in that directory.

On macOS, open the exported replay with:

```sh
open results/latest/bench_tracking.mp4
```

Choose a separate output directory for comparisons:

```sh
python track_bench.py --out results/experiment_02
```

Use `python track_bench.py --help` for input, model, output, and start-frame options. The default paths are resolved relative to the script.

## How it works

1. **Decode frames.** OpenCV reads the clip beginning at frame 300, about ten seconds into the reference video, and resizes frames to 540 pixels wide.
2. **Estimate body landmarks.** MediaPipe Pose Heavy predicts the shoulder, elbow, wrist, hip, knee, ankle, heel, and foot-index landmarks. The overlay shows the near-side/right landmarks when visibility and presence scores are at least 0.5 and coordinates are within the frame. Both sides' predictions are retained in the exported data.
3. **Locate the bar end.** Two manually selected crops from frames 300 and 436 represent the silver bar end. OpenCV compares grayscale patches at multiple scales in a wrist-guided search region. The wrist is not substituted for the bar location.
4. **Draw the trail.** Consecutive accepted positions are connected. Missing matches leave gaps instead of interpolated positions. No position smoothing is applied.
5. **Export results.** FFmpeg encodes an annotated MP4, and the script saves per-frame measurements separately.

The current matching threshold is `0.6`, defined by `BAR_MATCH_THRESHOLD` in `track_bench.py`. It controls acceptance and is recorded in the run summary. The similarity score is not a calibrated probability of correctness. Lower values can increase coverage and false matches.

## Files

```text
track_bench.py                    Processing entry point
requirements.txt                 Python dependencies
models/pose_landmarker_heavy.task Pretrained model, downloaded separately
form_videos/BarbellBenchPress.mp4  Source clip, supplied separately
results/latest/                   Generated output
```

Each run produces:

- `bench_tracking.mp4`: annotated H.264 replay without audio.
- `tracking.jsonl`: one JSON object per processed frame, including source timestamp, landmark positions and scores, and accepted bar position or null.
- `summary.json`: configuration, processing time, and tracking coverage.
- `preview_*.jpg`: sampled annotated frames.

## Current scope

The prototype is tuned to one 1080 x 1920 bench-press recording at approximately 29.97 FPS. Reference frames, crop coordinates, search region, visible side, and panel layout are specific to that clip. A different input path alone does not make bar tracking generalize.

The second reference comes from later in the same clip, so this is an offline analysis. A future reference-selection interface would remove the need to edit crop coordinates in code.

Positions are pixels in the resized image, not calibrated physical measurements. Perspective, camera movement, blur, and occlusion affect the trace. The bar point follows the visible end, not the midpoint of the complete barbell. Model visibility and matching coverage are not ground-truth accuracy. The prototype does not estimate force or provide form or injury scores.

## Next steps

- Evaluate positional error and missing detections against manually annotated frames.
- Add interactive reference selection and correction.
- Test additional clips and camera positions.
- Add repetition segmentation after tracking accuracy is established.
