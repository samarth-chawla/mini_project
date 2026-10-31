r"""Traffic violations + number-plate detection/OCR for tr2.mp4.

Run: venv\Scripts\python.exe traffic_signal_detection_anpr.py
Camera regions are measured from tr2.mp4; its monitoring line is estimated.
See README_ANPR.md for setup, output fields, and command-line options.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen

import cv2
import numpy as np

from anpr_processing import (PlateReader, analyze_camera, crosses_boundary,
                             draw_camera, ground_point_visible, move_plate_box, plate_owner,
                             recover_vehicle_detections, set_plate_box, update_consensus)

BASE = Path(__file__).resolve().parent
os.environ["YOLO_CONFIG_DIR"] = str(BASE / ".ultralytics_config")

TARGET_LABELS = {"bicycle", "car", "motorcycle", "bus", "truck"}
PLATE_URL = (
    "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/"
    "83c98fbe7412fe8b3950adb5637cfd08b0f04809/best.pt"
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=BASE / "tr2.mp4")
    parser.add_argument("--vehicle-model", type=Path, default=BASE / "yolov8m.pt")
    parser.add_argument("--plate-model", type=Path, default=BASE / "license_plate_detector.pt")
    parser.add_argument("--vehicle-conf", type=float, default=0.65)
    parser.add_argument("--recovery-conf", type=float, default=0.25,
                        help="Lower threshold for tracking partly occluded road cars")
    parser.add_argument("--plate-conf", type=float, default=0.3)
    parser.add_argument("--ocr-conf", type=float, default=0.85)
    parser.add_argument("--ocr-every", type=int, default=8, help="OCR interval per vehicle, in frames")
    parser.add_argument("--plate-format", choices=("hk", "general"), default="hk",
                        help="Expected plate layout; hk requires letter prefix + digits for this clip")
    parser.add_argument("--plate-observations", type=int, default=3,
                        help="Matching OCR observations required before publishing a plate number")
    parser.add_argument("--imgsz", type=int, default=960, help="Vehicle inference image size")
    parser.add_argument("--camera-config", type=Path, default=BASE / "tr2_camera.json")
    parser.add_argument("--recalibrate", action="store_true", help="Re-analyze the video and replace camera settings")
    parser.add_argument("--calibrate-only", action="store_true", help="Save camera JSON and preview, then exit")
    parser.add_argument("--start-frame", type=int, default=0, help="Start at a particular frame for inspection")
    parser.add_argument("--brightness-threshold", type=float, default=128)
    parser.add_argument("--device", default="cpu", help="cpu or a CUDA device, e.g. 0")
    parser.add_argument("--headless", action="store_true", help="Process without an OpenCV window")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the whole video")
    parser.add_argument("--output-video", type=Path, help="Optional annotated MP4")
    parser.add_argument("--violations-file", type=Path, default=BASE / "violations.json")
    parser.add_argument("--plates-file", type=Path, default=BASE / "plate_detections.json")
    parser.add_argument("--evidence-dir", type=Path, default=BASE.parent / "frontend/public/violations")
    parser.add_argument("--no-download", action="store_true", help="Require models already on disk")
    args = parser.parse_args()
    if args.ocr_every < 1 or args.max_frames < 0 or args.start_frame < 0:
        parser.error("--ocr-every must be positive; frame limits cannot be negative")
    if args.plate_observations < 2 or args.imgsz < 320:
        parser.error("--plate-observations must be at least 2 and --imgsz at least 320")
    for name in ("vehicle_conf", "recovery_conf", "plate_conf", "ocr_conf"):
        if not 0 < getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.output_video and args.output_video.resolve() == args.video.resolve():
        parser.error("--output-video must differ from the input video")
    if args.plates_file.resolve() == args.violations_file.resolve():
        parser.error("--plates-file and --violations-file must be separate files")
    return args


def write_json(path, data):
    """Replace atomically so the dashboard never reads half-written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_records(path):
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}; existing records were left untouched")
    return data


def ensure_plate_model(path, no_download):
    if path.is_file():
        return
    if no_download:
        raise FileNotFoundError(f"Missing plate model: {path}. Run once without --no-download.")
    print("Downloading the dedicated number-plate detector...", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        with urlopen(PLATE_URL, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary.stat().st_size < 100_000:
            raise RuntimeError("Downloaded plate model is unexpectedly small")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def is_region_light(image, polygon, brightness_threshold=128, color=None):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    if color is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.mean(gray, mask=mask)[0] > brightness_threshold
    # A bright red lamp is dark in grayscale (~76), so the old mean >128
    # test never sees the red icon in tr2.mp4. Check bright, saturated color.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    if color == "red":
        hue_match = (hue < 12) | (hue > 168)
    elif color == "yellow":
        hue_match = (hue >= 15) & (hue <= 35)
    else:
        hue_match = (hue > 35) & (hue < 90)
    illuminated = hue_match & (saturation > 80) & (value > brightness_threshold) & (mask > 0)
    return np.count_nonzero(illuminated) / max(1, np.count_nonzero(mask)) >= 0.15


def draw_text(frame, text, position, color=(255, 255, 255)):
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = max(0, min(position[0], frame.shape[1] - width - 5))
    y = max(18, min(position[1], frame.shape[0] - baseline - 3))
    cv2.rectangle(frame, (x, y-height-4), (x+width+4, y+baseline+2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x+2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


@dataclass
class Track:
    id: int
    box: tuple
    last_seen: int
    last_ocr: int = -10000
    plate_text: str = ""
    ocr_confidence: float = 0.0
    plate_confidence: float = 0.0
    plate_image: str | None = None
    plate_box: tuple | None = None
    plate_points: object = field(default=None, repr=False)
    plate_gray: object = field(default=None, repr=False)
    plate_candidate: str = ""
    plate_quality: float = 0.0
    votes: dict = field(default_factory=dict)
    logged: bool = False
    previous_point: tuple | None = None
    line_armed: bool = False
    label: str = ""
    hits: int = 0
    high_confidence_seen: bool = False


class Tracker:
    """One-to-one overlap/motion matching with stale-track removal."""

    def __init__(self, max_age):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age

    def update(self, detections, frame_number):
        self.tracks = {i: t for i, t in self.tracks.items()
                       if frame_number - t.last_seen <= self.max_age}
        candidates = []
        for index, (box, label, _) in enumerate(detections):
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            for tid, track in self.tracks.items():
                old = track.box
                if track.label and track.label != label:
                    continue
                distance = math.hypot(cx-(old[0]+old[2])/2, cy-(old[1]+old[3])/2)
                width = max(0, min(box[2], old[2])-max(box[0], old[0]))
                height = max(0, min(box[3], old[3])-max(box[1], old[1]))
                area = max(1, (box[2]-box[0])*(box[3]-box[1]))
                old_area = max(1, (old[2]-old[0])*(old[3]-old[1]))
                iou = width*height / max(1, area+old_area-width*height)
                limit = max(25, 0.30 * math.hypot(old[2]-old[0], old[3]-old[1]))
                if iou >= 0.2 and distance < limit and 0.45 < area/old_area < 2.2:
                    candidates.append((1-iou+distance/(limit*4), index, tid))
        assigned, used = {}, set()
        for _, index, tid in sorted(candidates):
            if index not in assigned and tid not in used:
                assigned[index] = self.tracks[tid]
                used.add(tid)
        result = []
        for index, (box, label, _) in enumerate(detections):
            track = assigned.get(index)
            if track is None:
                self.next_id += 1
                track = Track(self.next_id, box, frame_number)
                self.tracks[track.id] = track
            track.hits = track.hits+1 if frame_number-track.last_seen <= 3 else 1
            track.box, track.last_seen = box, frame_number
            track.label = label
            result.append(track)
        return result


def plate_fields(track):
    return {"plate_number": track.plate_text or None,
            "plate_confidence": round(track.plate_confidence, 4),
            "ocr_confidence": round(track.ocr_confidence, 4),
            "plate_image": track.plate_image,
            "plate_candidate": track.plate_candidate or None,
            "plate_observations": track.votes.get(track.plate_text or track.plate_candidate, (0,))[0],
            "plate_status": "confirmed" if track.plate_text else ("pending" if track.plate_candidate else "unreadable")}


def violation_evidence(raw, camera, track, video_seconds):
    """Each stored event highlights only its own vehicle, with matching identity."""
    evidence = raw.copy()
    draw_camera(evidence, camera, crossed=True)
    x1, y1, x2, y2 = map(int, track.box)
    cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 0, 255), 3)
    draw_text(evidence, f"VIOLATION - ID:{track.id} {track.label}", (x1, y1-12), (0, 0, 255))
    draw_text(evidence, f"Vehicle ID:{track.id} | {track.label} | Video: {video_seconds:.2f}s", (10, 28))
    draw_text(evidence, f"Plate: {track.plate_text or 'PENDING / UNREADABLE'}", (10, 52), (0, 255, 255))
    if track.plate_box:
        a, b, c, d = map(int, track.plate_box)
        cv2.rectangle(evidence, (a, b), (c, d), (0, 255, 255), 2)
    return evidence


def main():
    args = arguments()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    camera = analyze_camera(args.video, args.camera_config, force=args.recalibrate)
    if args.calibrate_only:
        print(f"Camera settings: {args.camera_config}\nPreview: {args.camera_config.with_suffix('.jpg')}")
        return 0
    red_light = np.asarray(camera["red_light"], np.int32)
    green_light = np.asarray(camera["green_light"], np.int32)
    yellow_light = np.asarray(camera["yellow_light"], np.int32)
    if not args.vehicle_model.is_file():
        raise FileNotFoundError(f"Vehicle model not found: {args.vehicle_model}")
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(2)
    try:
        reader = PlateReader(BASE / ".plate_ocr", args.no_download)
    except ImportError as exc:
        raise RuntimeError("OCR dependencies are missing. Follow README_ANPR.md setup instructions.") from exc

    ensure_plate_model(args.plate_model, args.no_download)
    vehicle_model, plate_model = YOLO(str(args.vehicle_model)), YOLO(str(args.plate_model))
    if not any("plate" in str(name).lower() for name in plate_model.names.values()):
        raise ValueError("--plate-model must be a dedicated license-plate detector, not the COCO vehicle model")
    records = read_records(args.violations_file)
    plate_records = read_records(args.plates_file)
    for path, data in ((args.violations_file, records), (args.plates_file, plate_records)):
        if not path.exists():
            write_json(path, data)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cap = cv2.VideoCapture(str(args.video))
    writer = None
    frame_number, processed, violation_count, plate_count = args.start_frame, 0, 0, 0
    logged_entries, plate_entries, evidence_cache = {}, {}, {}
    recent_violations = []
    start = time.monotonic()
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        tracker = Tracker(max_age=max(5, int(fps*0.5)))
        if args.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        if args.output_video:
            args.output_video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1100, 700))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot write output video: {args.output_video}")
        print(f"Processing {args.video.name}; ESC exits. Camera settings: {args.camera_config.name}", flush=True)
        print(camera.get("line_note", "Configured monitoring boundary"), flush=True)
        while not args.max_frames or processed < args.max_frames:
            success, native = cap.read()
            if not success:
                break
            frame_number += 1
            processed += 1
            raw = cv2.resize(native, (1100, 700))
            gray_frame = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            frame = raw.copy()
            video_seconds = (frame_number-1) / fps
            red_on = is_region_light(raw, red_light, args.brightness_threshold, "red")
            green_on = is_region_light(raw, green_light, args.brightness_threshold, "green")
            yellow_on = is_region_light(raw, yellow_light, args.brightness_threshold, "yellow")
            signal = "RED" if red_on else ("YELLOW" if yellow_on else ("GREEN" if green_on else "OFF"))
            draw_camera(frame, camera)
            prediction = vehicle_model.predict(raw, conf=min(args.vehicle_conf, args.recovery_conf), device=args.device,
                                               imgsz=args.imgsz, iou=0.5, verbose=False)[0]
            detections = []
            for box in prediction.boxes:
                label = vehicle_model.names[int(box.cls.item())]
                if label in TARGET_LABELS:
                    detections.append((tuple(map(int, box.xyxy[0].tolist())), label, float(box.conf.item())))
            detections = recover_vehicle_detections(detections, args.vehicle_conf, args.recovery_conf,
                                                    camera["road_polygon"])
            tracks = tracker.update(detections, frame_number)
            current_ids = {t.id for t in tracks}
            vehicles = [(t.id, t.box, t.label) for t in tracks]
            # Retain short occlusion context to avoid donating an obscured car's
            # plate to a large background bus when the car misses one detection.
            for recent in tracker.tracks.values():
                if recent.id not in current_ids and frame_number-recent.last_seen <= 3:
                    a, b, c, d = recent.box
                    vehicles.append((recent.id, (a-10, b-10, c+10, d+10), recent.label))
            pending = []
            recent_violations = [v for v in recent_violations if video_seconds-v[2] < 10]
            for (box, label, confidence), track in zip(detections, tracks):
                x1, y1, x2, y2 = box
                cx = (x1+x2)//2
                move_plate_box(track, gray_frame)
                if track.plate_box and plate_owner(track.plate_box, vehicles) != track.id:
                    track.plate_box = None
                if ground_point_visible(track.id, vehicles):
                    crossed = crosses_boundary(track, (cx, y2), camera["monitoring_line"], camera["road_polygon"])
                else:
                    track.previous_point, track.line_armed = None, False
                    crossed = False
                track.high_confidence_seen |= confidence >= args.vehicle_conf
                reliable = track.high_confidence_seen or track.hits >= 3
                violated = red_on and crossed and reliable
                violation_visible = violated or track.logged
                interval = args.ocr_every if track.plate_text else min(3, args.ocr_every)
                if frame_number-track.last_ocr >= interval:
                    track.last_ocr = frame_number
                    track.plate_box = None
                    # Detect plates on the native-resolution vehicle crop, before drawing overlays.
                    sx, sy = native.shape[1]/1100, native.shape[0]/700
                    nx1, ny1 = max(0, int(x1*sx)), max(0, int(y1*sy))
                    nx2, ny2 = min(native.shape[1], int(x2*sx)), min(native.shape[0], int(y2*sy))
                    vehicle_crop = native[ny1:ny2, nx1:nx2]
                    if vehicle_crop.size:
                        plates = plate_model.predict(vehicle_crop, conf=args.plate_conf, device=args.device,
                                                     imgsz=640, verbose=False)[0]
                        owned_plates = []
                        for candidate in plates.boxes:
                            a, b, c, d = candidate.xyxy[0].tolist()
                            owner = plate_owner(((a+nx1)/sx, (b+ny1)/sy, (c+nx1)/sx, (d+ny1)/sy), vehicles)
                            if owner == track.id:
                                owned_plates.append(candidate)
                        if owned_plates:
                            best = max(owned_plates, key=lambda b: float(b.conf.item()))
                            px1, py1, px2, py2 = map(int, best.xyxy[0].tolist())
                            px1, py1 = max(0, px1-2), max(0, py1-2)
                            px2, py2 = min(vehicle_crop.shape[1], px2+2), min(vehicle_crop.shape[0], py2+2)
                            plate_crop = vehicle_crop[py1:py2, px1:px2]
                            if plate_crop.shape[0] >= 6 and plate_crop.shape[1] >= 12:
                                set_plate_box(track, gray_frame,
                                              ((px1+nx1)/sx, (py1+ny1)/sy,
                                               (px2+nx1)/sx, (py2+ny1)/sy))
                                text, ocr_conf = reader.read(plate_crop, args.ocr_conf, args.plate_format)
                                old_text = track.plate_text
                                update_consensus(track, text, ocr_conf, args.plate_observations)
                                gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                                quality = float(cv2.Laplacian(gray_plate, cv2.CV_64F).var()) * plate_crop.shape[0]*plate_crop.shape[1]
                                if (track.plate_image is None or
                                    (text and text == (track.plate_text or track.plate_candidate) and
                                     (old_text != track.plate_text or quality > track.plate_quality))):
                                    filename = f"plate_{session}_{track.id}.png"
                                    if not cv2.imwrite(str(args.evidence_dir / filename), plate_crop):
                                        raise OSError("Could not save plate evidence")
                                    track.plate_quality = quality
                                    track.plate_confidence = float(best.conf.item())
                                    track.plate_image = f"/violations/{filename}"
                                if track.id not in plate_entries:
                                    plate_entries[track.id] = {"session": session, "id": track.id,
                                                              "source_video": args.video.name,
                                                              "vehicle_type": label}
                                    plate_records.append(plate_entries[track.id])
                                    plate_count += 1
                                plate_entries[track.id].update(plate_fields(track), video_seconds=round(video_seconds, 3),
                                                               vehicle_box=list(track.box))
                                write_json(args.plates_file, plate_records)
                                if track.id in logged_entries:
                                    logged_entries[track.id].update(plate_fields(track))
                                    if track.plate_text and old_text != track.plate_text:
                                        saved_raw, saved_track, saved_seconds, saved_path = evidence_cache[track.id]
                                        saved_track.plate_text = track.plate_text
                                        evidence = violation_evidence(saved_raw, camera, saved_track, saved_seconds)
                                        if not cv2.imwrite(str(saved_path), evidence):
                                            raise OSError("Could not update violation plate annotation")
                                        logged_entries[track.id]["plate_read_video_seconds"] = round(video_seconds, 3)
                                    write_json(args.violations_file, records)
                                if track.plate_text and old_text != track.plate_text:
                                    print(f"Frame {frame_number}, vehicle {track.id}: confirmed plate {track.plate_text}", flush=True)
                if not reliable:
                    continue
                color = (0, 0, 255) if violation_visible else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_text(frame, f"ID:{track.id} {label} ({confidence:.1%})", (x1, y1-10))
                if track.plate_image:
                    status = track.plate_text or ("READING..." if track.plate_candidate else "UNREADABLE")
                    draw_text(frame, f"Plate: {status}", (x1, y2+20), (0, 255, 255))
                if track.plate_box:
                    a, b, c, d = map(int, track.plate_box)
                    cv2.rectangle(frame, (a, b), (c, d), (0, 255, 255), 2)
                if violation_visible:
                    draw_text(frame, "VIOLATION!", (x1, y1-32), color)
                if violated:
                    draw_camera(frame, camera, crossed=True)
                    duplicate = any((track.plate_text and track.plate_text == plate) or
                                    (video_seconds-vt < 1 and abs(cx-vx) < 35 and abs(y2-vy) < 35)
                                    for vx, vy, vt, plate in recent_violations)
                    if not track.logged and not duplicate:
                        pending.append((track, label, confidence))
                        recent_violations.append((cx, y2, video_seconds, track.plate_text))
            signal_color = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255),
                            "GREEN": (0, 255, 0), "OFF": (180, 180, 180)}[signal]
            draw_text(frame, f"Signal: {signal}", (10, 28), signal_color)
            draw_text(frame, f"Violations: {violation_count+len(pending)} | Plates: {plate_count}", (10, 52))
            # Save after drawing every box and the status overlays.
            for track, label, confidence in pending:
                filename = f"violation_{session}_{track.id}_{frame_number}.png"
                evidence = violation_evidence(raw, camera, track, video_seconds)
                if not cv2.imwrite(str(args.evidence_dir / filename), evidence):
                    raise OSError("Could not save violation evidence")
                now = datetime.now()
                entry = {"id": track.id, "time": now.strftime("%H:%M:%S"), "timestamp": now.isoformat(),
                         "image": f"/violations/{filename}", "session": session, "source_video": args.video.name,
                         "video_seconds": round(video_seconds, 3), "vehicle_type": label,
                         "vehicle_box": list(track.box), "evidence_frame": frame_number,
                         "boundary_kind": camera["line_kind"], "monitoring_line": camera["monitoring_line"],
                         "vehicle_confidence": round(confidence, 4), **plate_fields(track)}
                records.append(entry)
                logged_entries[track.id] = entry
                event_track = Track(track.id, track.box, frame_number, label=label,
                                    plate_text=track.plate_text, plate_box=track.plate_box)
                evidence_cache[track.id] = (raw.copy(), event_track, video_seconds, args.evidence_dir / filename)
                track.logged = True
                violation_count += 1
                write_json(args.violations_file, records)
                print(f"Violation: vehicle {track.id}, plate {track.plate_text or 'UNREADABLE'}", flush=True)
            if writer:
                writer.write(frame)
            if not args.headless:
                cv2.imshow("Traffic Violations + Number Plates", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if frame_number % 50 == 0:
                print(f"Processed {frame_number} frames", flush=True)
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"Finished: {processed} frames, {violation_count} violations, {plate_count} vehicles with plates "
          f"in {time.monotonic()-start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Stopped.")
        sys.exit(130)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
