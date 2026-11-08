# Train a bar-end keypoint model

The custom model predicts one landmark: the center of the visible near-side bar-end face. MediaPipe continues to estimate the lifter's body. The existing template tracker remains the default until custom weights are supplied.

## 1. Install

```sh
cd "$HOME/Desktop/Form Tracker"
source .venv/bin/activate
python -m pip install -r requirements-training.txt
```

Training dependencies have already been installed on the current computer. The extra requirements include Ultralytics, PyTorch, and PyYAML. No model has been trained on your recordings yet.

## 2. Extract training frames

The initial clip has already been sampled into 29 unlabeled frames, starting at frame 300 and sampling every 15 frames:

```sh
python barbell_workflow.py extract form_videos/BarbellBenchPress.mp4 \
  --session bench_day_01 --split train --start-frame 300 --every 15
```

Repeated extraction skips existing frame identities. Images are saved under `datasets/barbell/raw/`, and `manifest.json` records the recording session, video hash, frame number, timestamp, dimensions, and assigned split.

All clips from the same recording session must use the same session name and split. Renaming or re-encoding a clip does not make it a genuinely held-out recording. The code rejects the same session or identical video content appearing across splits, but cannot identify every re-encoded duplicate automatically.

The extractor preserves aspect ratio and limits width to 540 pixels by default. For the initial pixel-error comparison, record all evaluation videos in the same orientation/aspect ratio and extract them at the same dimensions. Wider images can be selected using `--width`; rebuild the dataset and reevaluate when changing resolution.

## 3. Label the frames

```sh
python barbell_workflow.py label --session bench_day_01
```

This opens a native OpenCV window on your computer. For each image:

1. Press **B**, then draw a tight box around the visible near-side sleeve/end. Include the circular end face. Confirm the box with **Enter** or **Space**.
2. Click the center of that circular face. Keep the same physical landmark across every image.
3. Press **S** to save and advance.

Other controls:

- **U**: unlabelable/ambiguous or occluded; exclude the frame from training and evaluation.
- **N**: verified negative; no visible target bar end exists in the frame. Do not use this merely because labeling is difficult.
- **Q**: quit; saved annotations remain on disk.

Labels are separate JSON files under `datasets/barbell/annotations/`. Resume the same command to continue; saved annotations are skipped. Use `--review` to revisit and replace labels. Skipped frames can also be revisited with `--review`.

Unlabeled images are never silently treated as negative examples. This first version trains and evaluates visible points; fully occluded points are excluded rather than guessed.

## 4. Add separate validation and test sessions

Use additional recordings, not random neighboring frames from the training video. Example commands below assume you supply the indicated files:

```sh
python barbell_workflow.py extract form_videos/bench_validation.mp4 \
  --session bench_day_02 --split val --every 15
python barbell_workflow.py label --session bench_day_02

python barbell_workflow.py extract form_videos/bench_test.mp4 \
  --session bench_day_03 --split test --every 15
python barbell_workflow.py label --session bench_day_03
```

Validation is used to compare settings and select checkpoints. Reserve the test recordings for final assessment. At least training and validation sets with visible labels are required to build a dataset version. A separate labeled test set is required for test evaluation.

## 5. Build a dataset version

```sh
python barbell_workflow.py build --name v1
```

This writes YOLO images, labels, a dataset YAML, and an annotation snapshot to `datasets/barbell/versions/v1/`. The model configuration has one class (`bar_end`) and one keypoint with visibility (`kpt_shape: [1, 3]`). Label coordinates are normalized to image width and height.

Version names cannot overwrite previous versions. After changing labels, use a new name such as `v2`. The validator checks image hashes and labels against the saved snapshot before training or evaluation. This is local experiment versioning, not automatic Git versioning of the private footage.

## 6. Fine-tune

```sh
python barbell_workflow.py train \
  --data datasets/barbell/versions/v1/data.yaml \
  --device mps --epochs 50 --batch 4 --name bench_v1
```

The default base is the small pretrained `yolo26n-pose.pt` model. It downloads to `models/` on first training use. The training library adapts the pose output head to the custom one-keypoint layout. This is transfer learning, not a newly trained network from scratch.

Epochs, batch size, and resolution are initial experiment settings, not promises of model quality. The default `--device auto` selects CUDA, then Apple MPS, then CPU when available. Use `--device cpu` if an operation is unsupported on MPS; reduce `--batch` if memory is limiting.

Training writes checkpoints and `dataset_provenance.json` under `runs/barbell/<run>/`. Read the actual output path from the command because the training library may increment an existing run name. Save the best checkpoint for deployment. Successful training alone does not establish generalization or measurement accuracy.

## 7. Evaluate localization error and detection coverage

```sh
python barbell_workflow.py evaluate \
  --data datasets/barbell/versions/v1/data.yaml \
  --weights runs/barbell/bench_v1/weights/best.pt \
  --split test --tolerance-px 5 --device mps \
  --out results/bench_v1_test.json
```

Five pixels is an example reporting tolerance at the extracted image resolution, not a validated physical tolerance. Choose and document it before inspecting test results.

The report includes:

- **Detection coverage:** accepted predictions divided by labeled visible frames.
- **Median and p95 localization error:** Euclidean pixel error on accepted predictions; missing predictions do not become zero errors.
- **Fraction within tolerance:** correctly localized predictions divided by all labeled visible frames. Missing predictions count as failures.
- **False-positive rate:** accepted predictions on verified negative frames, reported separately.
- **Per-session results:** avoid hiding a difficult session behind an aggregate score.

Thresholds are explicit. Set `--confidence` and `--keypoint-confidence` on validation data, then keep them fixed for the test run. Scores are not calibrated correctness probabilities. Confidence, image resolution, model hash, and evaluation split are recorded.

The evaluation command requires the training provenance and matching dataset snapshot. It does not label a training result as a held-out result. It selects the highest-confidence eligible target prediction without using the annotated center to choose a candidate.

## 8. Use trained weights in the existing replay

```sh
python track_bench.py \
  --bar-weights runs/barbell/bench_v1/weights/best.pt \
  --device mps --out results/bench_v1_replay
```

This runs MediaPipe for the body and YOLO for the bar. No template fallback is used. Human-pose weights with 17 keypoints are rejected: the adapter requires one `bar_end` keypoint and the expected class name.

The replay still uses the original bench-video display layout. Supporting different exercises, resolutions, and layouts is separate from training a model.

## Checks

```sh
python -m unittest discover -s tests -v
```

Tests cover split leakage, normalized labels, excluded versus negative frames, immutable snapshot checks, and missing-detection denominators. Synthetic training checks validate software compatibility only; they are not evidence of barbell recognition performance.
