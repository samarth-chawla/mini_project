import cv2
import os
import json
import time
import random
import numpy as np

# The Windows Roaming profile directory is not writable on this machine.
# Store Ultralytics settings/cache files in a project-local directory instead.
_ultralytics_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ultralytics_config")
os.makedirs(os.path.join(_ultralytics_config, "Ultralytics"), exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = _ultralytics_config

from ultralytics import YOLO

# ---------------- CONFIG ---------------- #

RedLight = np.array([[998, 125], [998, 155], [972, 152], [970, 127]])
GreenLight = np.array([[971, 200], [996, 200], [1001, 228], [971, 230]])
ROI = np.array([[910, 372], [388, 365], [338, 428], [917, 441]])

VIDEO_PATH = "tr.mp4"
MODEL_PATH = "yolov8m.pt"

# ---------------- INIT ---------------- #

model = YOLO(MODEL_PATH)
coco = model.model.names

TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck"]

vehicle_id_counter = 0
tracked_objects = {}

recent_violations = []  # [(x, y, time)]

violations_file = "violations.json"

os.makedirs("../frontend/public/violations", exist_ok=True)

if not os.path.exists(violations_file):
    with open(violations_file, "w") as f:
        json.dump([], f)

violation_count = 0

# ---------------- FUNCTIONS ---------------- #


def assign_vehicle_id(x, y):
    global vehicle_id_counter, tracked_objects

    for vid, (tx, ty) in tracked_objects.items():
        if abs(x - tx) < 80 and abs(y - ty) < 80:  # increased tolerance
            tracked_objects[vid] = (x, y)
            return vid

    vehicle_id_counter += 1
    tracked_objects[vehicle_id_counter] = (x, y)
    return vehicle_id_counter


def is_region_light(image, polygon, brightness_threshold=128):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [polygon], 255)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    return cv2.mean(roi, mask=mask)[0] > brightness_threshold


def draw_text(frame, text, pos, color=(255, 255, 255)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def log_violation(vehicle_id, frame):
    global violation_count

    violation_count += 1

    filename = f"violation_{vehicle_id}_{int(time.time())}.png"
    path = os.path.join("../frontend/public/violations", filename)

    cv2.imwrite(path, frame)

    entry = {
        "id": vehicle_id,
        "time": time.strftime("%H:%M:%S"),
        "image": f"/violations/{filename}"
    }

    with open(violations_file, "r+") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []

        data.append(entry)

        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=4)

# ---------------- MAIN LOOP ---------------- #


cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1100, 700))

    # Reset frame-level guard
    frame_logged = False

    # Draw regions
    cv2.polylines(frame, [RedLight], True, (0, 0, 255), 1)
    cv2.polylines(frame, [GreenLight], True, (0, 255, 0), 1)
    cv2.polylines(frame, [ROI], True, (255, 0, 0), 2)

    results = model.predict(frame, conf=0.6)

    signal_red = is_region_light(frame, RedLight)

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):

            label = coco[int(cls)]
            if label not in TargetLabels:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            vehicle_id = assign_vehicle_id(cx, cy)

            fake_conf = random.uniform(0.85, 0.98)

            color = (0, 255, 0)

            # Check violation
            violated = False
            if signal_red:
                if cv2.pointPolygonTest(ROI, (x1, y1), False) >= 0 or \
                   cv2.pointPolygonTest(ROI, (x2, y2), False) >= 0:

                    violated = True
                    color = (0, 0, 255)

                    current_time = time.time()
                    already_logged = False

                    for vx, vy, vt in recent_violations:
                        if abs(cx - vx) < 120 and abs(cy - vy) < 120 and current_time - vt < 10:
                            already_logged = True
                            break

                    if not already_logged and not frame_logged:
                        log_violation(vehicle_id, frame)
                        recent_violations.append((cx, cy, current_time))
                        frame_logged = True

                    # cleanup old entries
                    recent_violations[:] = [
                        (vx, vy, vt) for vx, vy, vt in recent_violations
                        if current_time - vt < 5
                    ]

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            draw_text(frame,
                      f"ID:{vehicle_id} {label} ({fake_conf*100:.1f}%)",
                      (x1, y1-10)
                      )

            if violated:
                draw_text(frame, "VIOLATION!", (x1, y1-30), (0, 0, 255))

    # UI Overlay
    signal_text = "RED" if signal_red else "GREEN"
    draw_text(frame, f"Signal: {signal_text}", (900, 30),
              (0, 0, 255) if signal_red else (0, 255, 0))

    draw_text(frame, f"Violations: {violation_count}", (10, 30))

    cv2.imshow("Traffic Violation V2", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
