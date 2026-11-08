"""Extract, annotate, prepare, train, and evaluate a single bar-end keypoint model."""
import argparse
import hashlib
import json
import math
import re
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / ".ultralytics"))
DEFAULT_DATASET = PROJECT / "datasets" / "barbell"
SPLITS = ("train", "val", "test")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_sessions(rows):
    """A recording session or identical video may occur in only one split."""
    sessions, videos = {}, {}
    ids = set()
    for row in rows:
        if row["split"] not in SPLITS:
            raise ValueError("Invalid split")
        if row["id"] in ids:
            raise ValueError("Duplicate frame id")
        ids.add(row["id"])
        for mapping, key in [(sessions, row["session"]), (videos, row["video_sha256"])]:
            previous = mapping.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError("Session/video leakage: use separate recordings and sessions for each split")


def extract(args):
    root = args.dataset.resolve()
    manifest = root / "manifest.json"
    rows = read_json(manifest) if manifest.exists() else []
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.session):
        raise ValueError("Session name may contain letters, digits, underscores, and hyphens")
    if args.every < 1 or args.width < 64 or args.start_frame < 0:
        raise ValueError("Use every >= 1, width >= 64, and start-frame >= 0")
    video_hash = file_hash(args.video)
    probe = {"id": "probe", "session": args.session, "video_sha256": video_hash, "split": args.split}
    check_sessions(rows + [probe])
    cap = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise ValueError("Cannot decode video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or args.start_frame >= total:
        cap.release()
        raise ValueError("Invalid FPS or start frame outside clip")
    existing = {row["id"] for row in rows}
    image_dir = root / "raw" / args.session
    image_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        for frame in range(args.start_frame, min(args.end_frame or total, total), args.every):
            identity = f"{args.session}_{video_hash[:12]}_{frame:07d}"
            if identity in existing:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Failed to decode frame {frame}")
            h, w = image.shape[:2]
            new_width = min(args.width, w)
            image = cv2.resize(image, (new_width, round(h * new_width / w)))
            relative = Path("raw") / args.session / f"{identity}.jpg"
            if not cv2.imwrite(str(root / relative), image):
                raise ValueError("Failed to write frame")
            rows.append({"id": identity, "session": args.session, "split": args.split,
                         "video_sha256": video_hash, "frame": frame, "timestamp_s": frame / fps,
                         "width": image.shape[1], "height": image.shape[0], "image": relative.as_posix()})
            count += 1
    finally:
        cap.release()
    check_sessions(rows)
    write_json(manifest, rows)
    print(f"Extracted {count} frames into {image_dir}")


def validate_annotation(row, annotation):
    state = annotation.get("state")
    if state in ("negative", "skip"):
        return
    if state != "visible":
        raise ValueError("Annotation state must be visible, negative, or skip")
    x1, y1, x2, y2 = annotation["bbox"]
    x, y = annotation["keypoint"]
    if not all(math.isfinite(v) for v in [x1, y1, x2, y2, x, y]):
        raise ValueError("Annotation coordinates must be finite")
    if not (0 <= x1 < x2 <= row["width"] and 0 <= y1 < y2 <= row["height"]):
        raise ValueError("Box must have positive area and lie inside the image")
    if not (x1 <= x < x2 and y1 <= y < y2):
        raise ValueError("Keypoint must be inside the box")


def yolo_label(row, annotation):
    validate_annotation(row, annotation)
    if annotation["state"] == "skip":
        return None
    if annotation["state"] == "negative":
        return ""
    w, h = row["width"], row["height"]
    x1, y1, x2, y2 = annotation["bbox"]
    x, y = annotation["keypoint"]
    return (f"0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} "
            f"{(y2-y1)/h:.8f} {x/w:.8f} {y/h:.8f} 2\n")


def annotate(args):
    root = args.dataset.resolve()
    rows = read_json(root / "manifest.json")
    selected = [row for row in rows if args.session is None or row["session"] == args.session]
    if not selected:
        raise ValueError("No extracted frames match this session")
    window = "Bar end annotation"
    print("Draw box around visible near-side sleeve/end. ENTER confirms, C cancels.")
    print("Then click the center of the circular end face and press S to save.")
    print("N = verified target absent; U = ambiguous/occluded, exclude; Q = quit.")
    print("Previously saved annotations are skipped unless --review is used.")
    try:
        for row in selected:
            dest = root / "annotations" / f"{row['id']}.json"
            if dest.exists() and not args.review:
                continue
            raw = cv2.imread(str(root / row["image"]))
            if raw is None:
                raise ValueError(f"Missing frame {row['image']}")
            # Fixed display scale: mouse coordinates are converted back to stored image pixels.
            scale = min(1.0, 900 / raw.shape[0], 1200 / raw.shape[1])
            display = cv2.resize(raw, (round(raw.shape[1]*scale), round(raw.shape[0]*scale)))
            point = []
            box = None
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            def click(event, x, y, flags, parameter):
                if event == cv2.EVENT_LBUTTONDOWN:
                    point[:] = [(x / scale, y / scale)]
            cv2.setMouseCallback(window, click)
            while True:
                preview = display.copy()
                if box:
                    x1,y1,x2,y2 = box
                    cv2.rectangle(preview, (round(x1*scale),round(y1*scale)),
                                  (round(x2*scale),round(y2*scale)), (0,200,255), 2)
                if point:
                    cv2.circle(preview, tuple(round(v*scale) for v in point[0]), 4, (0,255,0), -1)
                title = f"{row['frame']} | B box | click center | S save | N absent | U skip | Q quit"
                cv2.putText(preview,title,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.37,(0,255,255),1,cv2.LINE_AA)
                cv2.imshow(window, preview)
                key = cv2.waitKey(30) & 0xff
                if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return
                if key == ord("b"):
                    x,y,w,h = cv2.selectROI("Draw target box", display, showCrosshair=True, fromCenter=False)
                    cv2.destroyWindow("Draw target box")
                    if w > 0 and h > 0:
                        box = [x/scale,y/scale,(x+w)/scale,(y+h)/scale]
                elif key in (ord("n"), ord("u")):
                    annotation = {"state": "negative" if key == ord("n") else "skip"}
                    write_json(dest, annotation)
                    break
                elif key == ord("s") and box and point:
                    annotation = {"state":"visible", "bbox":box, "keypoint":list(point[0])}
                    try:
                        validate_annotation(row, annotation)
                    except ValueError as exc:
                        print(exc)
                        continue
                    write_json(dest,annotation)
                    break
    finally:
        cv2.destroyAllWindows()


def build(args):
    import yaml
    root = args.dataset.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise ValueError("Version name may contain letters, digits, underscores, and hyphens")
    out = root / "versions" / args.name
    if out.exists():
        raise ValueError("Dataset version already exists; choose a new --name")
    all_rows = read_json(root / "manifest.json")
    check_sessions(all_rows)
    prepared = []
    for row in all_rows:
        annotation = root / "annotations" / f"{row['id']}.json"
        if not annotation.exists():
            continue  # Unlabeled is NOT a negative training example.
        value = read_json(annotation)
        line = yolo_label(row,value)
        if line is not None:
            prepared.append((row,value,line))
    for split in ("train","val"):
        if not any(row["split"] == split and value["state"] == "visible" for row,value,line in prepared):
            raise ValueError(f"Need visible annotations in {split}, from a separate session")
    # Resolve all inputs before creating the version.
    for row,value,line in prepared:
        if not (root/row["image"]).is_file():
            raise ValueError(f"Missing {row['image']}")
    out.mkdir(parents=True)
    snapshot = []
    for row,value,line in prepared:
        image = out / "images" / row["split"] / f"{row['id']}.jpg"
        label = out / "labels" / row["split"] / f"{row['id']}.txt"
        image.parent.mkdir(parents=True,exist_ok=True)
        label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/row["image"], image)
        label.write_text(line)
        snapshot.append({**row,"image":image.relative_to(out).as_posix(),
                         "image_sha256":file_hash(image),"annotation":value})
    configuration = {"path":str(out),"train":"images/train","val":"images/val",
                     "names":{0:"bar_end"},"kpt_shape":[1,3],"flip_idx":[0]}
    if any(row["split"] == "test" for row,_,_ in prepared):
        configuration["test"]="images/test"
    (out/"data.yaml").write_text(yaml.safe_dump(configuration,sort_keys=False))
    write_json(out/"manifest.json",snapshot)
    print(f"Prepared {len(snapshot)} labeled frames: {out/'data.yaml'}")


def validate_version(data):
    import yaml
    data = data.resolve()
    config = yaml.safe_load(data.read_text())
    root = Path(config["path"])
    rows = read_json(root/"manifest.json")
    check_sessions(rows)
    if config.get("kpt_shape") != [1,3] or config.get("names") != {0:"bar_end"}:
        raise ValueError("Expected a one-keypoint bar_end dataset")
    for row in rows:
        path = root/row["image"]
        if file_hash(path) != row["image_sha256"]:
            raise ValueError("Prepared dataset image changed; build a new version")
        label = root/"labels"/row["split"]/(row["id"]+".txt")
        if label.read_text() != yolo_label(row,row["annotation"]):
            raise ValueError("Prepared labels changed; build a new version")
    return root, rows


def train(args):
    from ultralytics import YOLO
    import torch
    root, rows = validate_version(args.data)
    for split in ("train","val"):
        if not any(r["split"]==split and r["annotation"]["state"]=="visible" for r in rows):
            raise ValueError(f"Need visible labels in {split}")
    if args.epochs < 1 or args.batch < 1:
        raise ValueError("Epochs and batch size must be positive")
    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device}; adapting pretrained pose weights to one bar-end keypoint")
    # YOLO's trainer adapts the keypoint head to kpt_shape from data.yaml.
    model = YOLO(args.base_model)
    if model.task != "pose":
        raise ValueError("Base model must be a pose model")
    result = model.train(data=str(args.data.resolve()), epochs=args.epochs, batch=args.batch,
                        imgsz=args.imgsz, device=device, workers=0, seed=42,
                        project=str(args.runs.resolve()), name=args.name, exist_ok=False,
                        amp=False if device=="mps" else True)
    save_dir=Path(model.trainer.save_dir)
    write_json(save_dir/"dataset_provenance.json",{
        "manifest_sha256":file_hash(root/"manifest.json"),
        "dataset":str(args.data.resolve()),"base_model":args.base_model,
        "training_sessions":sorted({r["session"] for r in rows if r["split"]=="train"}),
        "validation_sessions":sorted({r["session"] for r in rows if r["split"]=="val"})})
    print(f"Weights: {save_dir/'weights/best.pt'}")


def summarize_metrics(records, tolerance_px):
    if tolerance_px <= 0:
        raise ValueError("Tolerance must be positive")
    visible = [r for r in records if r["state"] == "visible"]
    detected = [r for r in visible if r["prediction"] is not None]
    errors = [math.dist(r["target"],r["prediction"]) for r in detected]
    negatives = [r for r in records if r["state"] == "negative"]
    return {"visible_frames":len(visible),"accepted_predictions_on_visible_frames":len(detected),
            "detection_coverage":len(detected)/len(visible) if visible else None,
            "median_localization_error_px_on_accepted":float(np.median(errors)) if errors else None,
            "p95_localization_error_px_on_accepted":float(np.percentile(errors,95)) if errors else None,
            "tolerance_px":tolerance_px,
            "fraction_visible_frames_within_tolerance":sum(e<=tolerance_px for e in errors)/len(visible) if visible else None,
            "negative_frames":len(negatives),
            "false_positive_rate_on_negative_frames":sum(r["prediction"] is not None for r in negatives)/len(negatives) if negatives else None}


def evaluate(args):
    from barbell_detector import BarbellDetector
    root, rows = validate_version(args.data)
    chosen = [r for r in rows if r["split"] == args.split]
    if not chosen:
        raise ValueError(f"No labeled {args.split} frames in this version")
    provenance_path = args.weights.resolve().parent.parent/"dataset_provenance.json"
    if not provenance_path.exists():
        raise ValueError("Missing training provenance beside weights; use this workflow's train command")
    provenance=read_json(provenance_path)
    if provenance["manifest_sha256"] != file_hash(root/"manifest.json"):
        raise ValueError("Weights and dataset version do not match")
    if args.split=="test" and {r["session"] for r in chosen} & set(provenance["training_sessions"]+provenance["validation_sessions"]):
        raise ValueError("Test session was used for training or validation")
    detector=BarbellDetector(args.weights,args.confidence,args.keypoint_confidence,args.device)
    records=[]
    for row in chosen:
        frame=cv2.imread(str(root/row["image"]))
        point,score=detector.locate(frame)
        annotation=row["annotation"]
        records.append({"id":row["id"],"session":row["session"],"state":annotation["state"],
                        "target":annotation.get("keypoint"),"prediction":point,"score":score,
                        "image_width":row["width"],"image_height":row["height"]})
    dimensions={(r["image_width"],r["image_height"]) for r in records}
    if len(dimensions)!=1:
        raise ValueError("Pixel-error evaluation requires a common resolution; extract all clips with matching dimensions")
    report={"split":args.split,"image_resolution":list(next(iter(dimensions))),
            "weights_sha256":file_hash(args.weights),"confidence":args.confidence,
            "keypoint_confidence":args.keypoint_confidence,
            "overall":summarize_metrics(records,args.tolerance_px),
            "sessions":{s:summarize_metrics([r for r in records if r["session"]==s],args.tolerance_px)
                        for s in sorted({r["session"] for r in records})},
            "notes":["Error statistics condition on accepted detections; missing predictions still fail tolerance metric.",
                     "Confidence is not a calibrated correctness probability.",
                     "Validation guides model selection; reserve test sessions for final evaluation."]}
    write_json(args.out,report)
    write_json(args.out.with_name(args.out.stem+"_predictions.json"),records)
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("extract",help="Sample frames; explicitly assign each recording session to a split")
    p.add_argument("video",type=Path);p.add_argument("--dataset",type=Path,default=DEFAULT_DATASET)
    p.add_argument("--session",required=True);p.add_argument("--split",choices=SPLITS,required=True)
    p.add_argument("--every",type=int,default=15);p.add_argument("--width",type=int,default=540)
    p.add_argument("--start-frame",type=int,default=0);p.add_argument("--end-frame",type=int)
    p.set_defaults(function=extract)
    p=commands.add_parser("label",help="Local OpenCV window: B box, click point, S save, U skip, N absent, Q quit")
    p.add_argument("--dataset",type=Path,default=DEFAULT_DATASET);p.add_argument("--session")
    p.add_argument("--review",action="store_true");p.set_defaults(function=annotate)
    p=commands.add_parser("build",help="Create an immutable YOLO dataset version from saved annotations")
    p.add_argument("--dataset",type=Path,default=DEFAULT_DATASET);p.add_argument("--name",required=True)
    p.set_defaults(function=build)
    p=commands.add_parser("train",help="Fine-tune one bar-end keypoint from pretrained pose weights")
    p.add_argument("--data",type=Path,required=True);p.add_argument("--base-model",default=str(PROJECT/"models"/"yolo26n-pose.pt"))
    p.add_argument("--epochs",type=int,default=50);p.add_argument("--batch",type=int,default=4)
    p.add_argument("--imgsz",type=int,default=640);p.add_argument("--device",default="auto")
    p.add_argument("--runs",type=Path,default=PROJECT/"runs"/"barbell");p.add_argument("--name",default="bar_end")
    p.set_defaults(function=train)
    p=commands.add_parser("evaluate",help="Evaluate on held-out labeled frames; missing predictions count against coverage")
    p.add_argument("--data",type=Path,required=True);p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--split",choices=["val","test"],default="test")
    p.add_argument("--confidence",type=float,default=.25);p.add_argument("--keypoint-confidence",type=float,default=.5)
    p.add_argument("--tolerance-px",type=float,required=True);p.add_argument("--device",default=None)
    p.add_argument("--out",type=Path,default=PROJECT/"results"/"evaluation.json")
    p.set_defaults(function=evaluate)
    args=parser.parse_args()
    try:
        args.function(args)
    except (ValueError,FileNotFoundError,KeyError) as exc:
        parser.exit(2,f"Error: {exc}\n")


if __name__=="__main__":
    main()
