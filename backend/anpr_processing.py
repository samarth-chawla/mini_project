"""Plate-specific OCR and video camera calibration used by the ANPR script."""

import json
from pathlib import Path
import re

import cv2
import numpy as np


class PlateReader:
    """Recognize the whole plate, retaining two-row layouts instead of dropping a row."""

    def __init__(self, model_dir, no_download=False):
        from fast_plate_ocr import LicensePlateRecognizer
        import onnxruntime as ort

        model_dir = Path(model_dir)
        model = model_dir / "cct_s_v2_global.onnx"
        config = model_dir / "cct_s_v2_global_plate_config.yaml"
        if not model.is_file() or not config.is_file():
            if no_download:
                raise FileNotFoundError("Plate OCR model is missing; run once without --no-download")
            from fast_plate_ocr.inference.hub import download_model
            model, config = download_model("cct-s-v2-global-model", save_directory=model_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.reader = LicensePlateRecognizer(
            onnx_model_path=model, plate_config_path=config,
            device="cpu", sess_options=options)

    def read(self, crop, minimum_confidence=0.85, plate_format="hk"):
        if crop.shape[0] < 10 or crop.shape[1] < 20:
            return "", 0.0
        prediction = self.reader.run(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                     return_confidence=True)[0]
        text = re.sub(r"[^A-Z0-9]", "", prediction.plate.upper())
        # This clip has letter prefixes and digit suffixes. Do not promote
        # '4540' to a complete plate when its 'KT' row was missed.
        pattern = r"[A-Z]{1,2}[0-9]{1,4}" if plate_format == "hk" else r"(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]{4,10}"
        if not re.fullmatch(pattern, text):
            return "", 0.0
        probabilities = np.asarray(prediction.char_probs)[:len(text)]
        confidence = float(np.mean(probabilities))
        if confidence < minimum_confidence or float(np.min(probabilities)) < 0.6:
            return "", 0.0
        return text, confidence


def update_consensus(track, text, confidence, required=3):
    """Count independent video observations, never preprocessing variants as extra votes."""
    if not text:
        return
    count, weight, peak = track.votes.get(text, (0, 0.0, 0.0))
    track.votes[text] = (count+1, weight+confidence, max(peak, confidence))
    winner = max(track.votes, key=lambda value: (track.votes[value][0], track.votes[value][1]))
    track.plate_candidate = winner
    votes, weight, _ = track.votes[winner]
    total = sum(v[0] for v in track.votes.values())
    if votes >= required and votes/total >= 0.67:
        track.plate_text = winner
        track.ocr_confidence = weight/votes


def plate_owner(plate_box, vehicles):
    """A plate belongs to the smallest containing vehicle, never every overlapping crop.

    vehicles contains (track_id, box, label). The lower-body check excludes
    background plates/signs above a vehicle's front/rear panel.
    """
    px1, py1, px2, py2 = plate_box
    area = max(1, (px2-px1)*(py2-py1))
    cx, cy = (px1+px2)/2, (py1+py2)/2
    candidates = []
    for track_id, (x1, y1, x2, y2), _ in vehicles:
        intersection = max(0, min(px2, x2)-max(px1, x1))*max(0, min(py2, y2)-max(py1, y1))
        if intersection/area >= 0.8 and x1 <= cx <= x2 and y1+0.3*(y2-y1) <= cy <= y2:
            candidates.append(((x2-x1)*(y2-y1), track_id))
    return min(candidates)[1] if candidates else None


def ground_point_visible(track_id, vehicles):
    """Do not use the bottom of a bus box that includes a foreground car."""
    current = next(box for tid, box, _ in vehicles if tid == track_id)
    x1, y1, x2, y2 = current
    cx, area = (x1+x2)/2, max(1, (x2-x1)*(y2-y1))
    for tid, (ox1, oy1, ox2, oy2), _ in vehicles:
        if tid == track_id:
            continue
        other_area = max(1, (ox2-ox1)*(oy2-oy1))
        if (other_area < area*0.8 and ox1 <= cx <= ox2 and
                oy1 > y1+0.2*(y2-y1) and oy1 < y2 and oy2 >= y2-25):
            return False
    return True


def recover_vehicle_detections(candidates, confidence, recovery_confidence, road_polygon):
    """Keep weak, partly occluded road cars for temporal tracking, not immediate alerts."""
    result = []
    for box, label, score in candidates:
        x1, y1, x2, y2 = box
        if score >= confidence:
            result.append((box, label, score))
        elif (label == "car" and score >= recovery_confidence and y1 >= 245 and
              x2-x1 >= 50 and y2-y1 >= 60 and
              cv2.pointPolygonTest(np.asarray(road_polygon, np.int32),
                                       ((x1+x2)/2, float(y2)), False) >= 0):
            result.append((box, label, score))
    return result


def set_plate_box(track, gray, box):
    """Seed optical flow on actual plate pixels, rather than the whole car box."""
    track.plate_box = tuple(map(float, box))
    x1, y1, x2, y2 = np.rint(box).astype(int)
    mask = np.zeros_like(gray)
    mask[max(0, y1):min(gray.shape[0], y2), max(0, x1):min(gray.shape[1], x2)] = 255
    track.plate_points = cv2.goodFeaturesToTrack(gray, maxCorners=30, qualityLevel=0.02,
                                                minDistance=2, mask=mask, blockSize=3)
    track.plate_gray = gray


def move_plate_box(track, gray):
    """Forward/backward LK flow; suppress uncertain boxes instead of guessing."""
    if track.plate_box is None or track.plate_gray is None:
        return
    points = track.plate_points
    if points is None or len(points) < 4:
        track.plate_box = None
        return
    current, status, _ = cv2.calcOpticalFlowPyrLK(
        track.plate_gray, gray, points, None, winSize=(15, 15), maxLevel=2)
    if current is None:
        track.plate_box = None
        return
    backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, track.plate_gray, current, None, winSize=(15, 15), maxLevel=2)
    if backward is None:
        track.plate_box = None
        return
    valid = ((status.ravel() > 0) & (reverse_status.ravel() > 0) &
             (np.linalg.norm(backward-points, axis=2).ravel() < 1.5))
    if np.count_nonzero(valid) < 4:
        track.plate_box = None
        return
    before, after = points[valid].reshape(-1, 2), current[valid].reshape(-1, 2)
    transform, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC,
                                                   ransacReprojThreshold=2)
    if transform is None or np.count_nonzero(inliers) < 4:
        track.plate_box = None
        return
    scale = np.linalg.norm(transform[:, 0])
    if not 0.85 < scale < 1.2:
        track.plate_box = None
        return
    x1, y1, x2, y2 = track.plate_box
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    moved = cv2.transform(corners[None], transform)[0]
    box = (*moved.min(axis=0), *moved.max(axis=0))
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    vx1, vy1, vx2, vy2 = track.box
    if not (vx1-5 <= cx <= vx2+5 and vy1-5 <= cy <= vy2+5):
        track.plate_box = None
        return
    set_plate_box(track, gray, box)


def line_side(point, line):
    a, b = np.asarray(line, dtype=float)
    direction = b-a
    return float(np.cross(direction, np.asarray(point)-a) / max(1.0, np.linalg.norm(direction)))


def crosses_boundary(track, point, line, road_polygon, margin=3):
    """Arm above the line and trigger on an actual downward crossing, once per passage."""
    side = line_side(point, line)
    previous = track.previous_point
    track.previous_point = point
    if side < -margin:
        track.line_armed = True
    if not track.line_armed or previous is None or side < margin:
        return False
    previous_side = line_side(previous, line)
    track.line_armed = False
    if previous_side >= side:
        return False
    fraction = -previous_side / (side-previous_side) if previous_side != side else 0.0
    # The last sample can be inside the hysteresis band; allow its segment
    # endpoint rather than extrapolating backwards beyond that sample.
    fraction = float(np.clip(fraction, 0, 1))
    intersection = np.asarray(previous) + fraction*(np.asarray(point)-previous)
    a, b = np.asarray(line, dtype=float)
    projection = float(np.dot(intersection-a, b-a) / max(1, np.dot(b-a, b-a)))
    return (0 <= projection <= 1 and
            cv2.pointPolygonTest(np.asarray(road_polygon, np.int32), tuple(map(float, intersection)), False) >= 0)


def _largest_lamp_box(mask, fallback):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    candidates = [s for s in stats[1:count] if 150 <= s[4] <= 5000 and 0.5 < s[2]/s[3] < 1.6]
    if not candidates:
        return fallback
    x, y, w, h, _ = max(candidates, key=lambda s: s[4])
    return [[int(x-4), int(y-4)], [int(x+w+4), int(y-4)],
            [int(x+w+4), int(y+h+4)], [int(x-4), int(y+h+4)]]


def _road_span(polygon, y):
    intersections = []
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        if min(a[1], b[1]) <= y < max(a[1], b[1]):
            intersections.append(a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]))
    if len(intersections) < 2:
        raise ValueError("Road polygon does not intersect the proposed monitoring line")
    return [[round(min(intersections)), y], [round(max(intersections)), y]]


def _add_yellow_box(data):
    # The middle lamp lies halfway between the measured top/bottom lamps.
    if "yellow_light" not in data:
        data["yellow_light"] = np.rint((np.asarray(data["red_light"])+
                                        np.asarray(data["green_light"]))/2).astype(int).tolist()
    return data


def analyze_camera(video, config_path, force=False):
    """Find the graphic's lamps and look for a transverse white road marking.

    When no reliable marking exists, explicitly return an estimated virtual
    boundary, not a purported physical stop line. Coordinates use 1100 x 700.
    """
    config_path, video = Path(config_path), Path(video)
    if config_path.exists() and not force:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if data.get("source_video") != video.name or data.get("source_bytes") != video.stat().st_size:
            raise ValueError("Camera configuration belongs to another video; use --recalibrate")
        if data.get("frame_size") != [1100, 700]:
            raise ValueError("Camera configuration frame_size must be [1100, 700]")
        for key, count in (("red_light", 4), ("green_light", 4), ("road_polygon", 4), ("monitoring_line", 2)):
            points = np.asarray(data.get(key), dtype=float)
            if points.shape != (count, 2) or not np.isfinite(points).all():
                raise ValueError(f"Invalid {key} coordinates in camera configuration")
            if np.any(points < 0) or np.any(points[:, 0] >= 1100) or np.any(points[:, 1] >= 700):
                raise ValueError(f"{key} coordinates must be within the 1100 x 700 frame")
        if data["monitoring_line"][0][0] >= data["monitoring_line"][1][0]:
            raise ValueError("monitoring_line must run left to right")
        return _add_yellow_box(data)
    cap = cv2.VideoCapture(str(video))
    frames = []
    red_mask, green_mask = np.zeros((700, 1100), bool), np.zeros((700, 1100), bool)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in np.linspace(0, max(0, total-1), 24).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (1100, 700))
            frames.append(frame)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            region = np.zeros_like(red_mask)
            region[:230, 950:] = True
            red_mask |= ((h < 12) | (h > 168)) & (s > 100) & (v > 150) & region
            green_mask |= (h > 35) & (h < 90) & (s > 100) & (v > 150) & region
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Cannot read video for camera calibration: {video}")
    background = np.median(np.stack(frames), axis=0).astype(np.uint8)
    # Road area traced from this clip's perspective, excluding the pavement.
    road = [[420, 350], [1055, 350], [1095, 695], [0, 695]]
    road_mask = np.zeros((700, 1100), np.uint8)
    cv2.fillPoly(road_mask, [np.asarray(road, np.int32)], 255)
    hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 170) & (road_mask > 0)).astype(np.uint8)*255
    white[:470] = 0  # Ignore the watermark, signs, and vehicle roofs.
    lines = cv2.HoughLinesP(white, 1, np.pi/180, threshold=70,
                            minLineLength=200, maxLineGap=12)
    candidates = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2-x1) > 250 and abs(y2-y1) < 0.08*abs(x2-x1):
                candidates.append((int(x1), int(y1), int(x2), int(y2)))
    # A white transverse candidate alone cannot establish a legal stop line.
    # Store it for inspection, and retain an honest virtual boundary by default.
    data = {
        "source_video": video.name, "source_bytes": video.stat().st_size,
        "frame_size": [1100, 700],
        "red_light": _largest_lamp_box(red_mask, [[1015, 30], [1060, 30], [1060, 80], [1015, 80]]),
        "green_light": _largest_lamp_box(green_mask, [[1015, 142], [1060, 142], [1060, 192], [1015, 192]]),
        "road_polygon": road, "monitoring_line": _road_span(road, 602),
        "line_kind": "estimated_virtual",
        "line_note": "No verified physical stop line visible; estimated foreground road boundary.",
        "transverse_marking_candidates": candidates,
    }
    _add_yellow_box(data)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    preview = frames[0].copy()
    draw_camera(preview, data)
    cv2.imwrite(str(config_path.with_suffix(".jpg")), preview)
    return data


def draw_camera(frame, camera, crossed=False):
    for name, color in (("red_light", (0, 0, 255)), ("yellow_light", (0, 255, 255)),
                        ("green_light", (0, 255, 0))):
        cv2.polylines(frame, [np.asarray(camera[name], np.int32)], True, color, 1)
    line = camera["monitoring_line"]
    color = (0, 0, 255) if crossed else (255, 180, 0)
    # Dashed virtual line distinguishes the inferred boundary from road paint.
    a, b = np.asarray(line[0], float), np.asarray(line[1], float)
    for fraction in np.arange(0, 1, 0.05):
        p = tuple((a+(b-a)*fraction).astype(int))
        q = tuple((a+(b-a)*min(1, fraction+0.03)).astype(int))
        cv2.line(frame, p, q, color, 2)
    x, y = line[0]
    label = "Estimated monitoring line" if camera["line_kind"] == "estimated_virtual" else "Configured stop line"
    cv2.putText(frame, label, (max(5, x), max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
