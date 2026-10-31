import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_PATH = "tr2.mp4"          # CCTV or test video
MODEL_PATH = "yolov8m.pt"       # Model (auto-downloads if missing)
CONF_THRESHOLD = 0.75
# ----------------------------------------

# 🧠 Automatically download YOLOv8 if not found
if not os.path.exists(MODEL_PATH):
    print(f"🔽 Model '{MODEL_PATH}' not found — downloading automatically...")
model = YOLO(MODEL_PATH)  # This triggers download if file is missing

# COCO class names from YOLO model
coco = model.model.names
TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]

# ---------------- COORDINATES ----------------
# Adjusted regions for given frame (1100x700 resolution)
RedLight = np.array([[1035, 30], [1075, 30], [1075, 70], [1035, 70]])
GreenLight = np.array([[1035, 150], [1075, 150], [1075, 190], [1035, 190]])
ROI_line = [(220, 420), (880, 420)]  # single horizontal ROI line
# ---------------------------------------------

def is_region_light(image, polygon, brightness_threshold=128):
    """Check if a region (polygon) is bright enough to consider the light ON."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [np.array(polygon)], 255)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    mean_brightness = cv2.mean(roi, mask=mask)[0]
    return mean_brightness > brightness_threshold

def draw_text_with_background(frame, text, position, font, scale,
                              text_color, background_color, border_color,
                              thickness=2, padding=5):
    """Draw text with background and border on the frame."""
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x - padding, y - th - padding),
                  (x + tw + padding, y + base + padding),
                  background_color, cv2.FILLED)
    cv2.rectangle(frame, (x - padding, y - th - padding),
                  (x + tw + padding, y + base + padding),
                  border_color, thickness)
    cv2.putText(frame, text, (x, y), font, scale, text_color,
                thickness, lineType=cv2.LINE_AA)

# ---------------- MAIN PROCESS ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error: Could not open video '{VIDEO_PATH}'")
    exit()

print("✅ YOLOv8 Model loaded successfully. Starting detection...")
frame_count = 0
roi_y = ROI_line[0][1]  # y-coordinate for ROI line

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("📹 Video ended or cannot read frames.")
        break

    frame = cv2.resize(frame, (1100, 700))
    frame_count += 1

    # Draw reference regions
    cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1)
    cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
    cv2.line(frame, ROI_line[0], ROI_line[1], (255, 0, 0), 2)
    cv2.circle(frame, ROI_line[0], 4, (255, 0, 0), -1)
    cv2.circle(frame, ROI_line[1], 4, (255, 0, 0), -1)

    # Run YOLO prediction
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls

        for box, conf, cls in zip(boxes, confs, classes):
            label = coco[int(cls)]
            if label not in TargetLabels:
                continue

            x1, y1, x2, y2 = map(int, box)

            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 255, 0], 2)
            draw_text_with_background(
                frame,
                f"{label.capitalize()} ({conf*100:.1f}%)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (255, 255, 255),
                (0, 0, 0),
                (0, 0, 255)
            )

            # Check red light ON
            red_on = is_region_light(frame, RedLight)
            green_on = is_region_light(frame, GreenLight)

            # Draw status on screen
            color_status = "🟥 RED" if red_on else ("🟩 GREEN" if green_on else "⬛ OFF")
            draw_text_with_background(
                frame,
                f"Signal: {color_status}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                (0, 0, 0),
                (0, 255, 0) if green_on else (0, 0, 255)
            )

            # Check violation (crossing ROI when red)
            if red_on and y2 >= roi_y:
                draw_text_with_background(
                    frame,
                    f"🚨 {label.capitalize()} violated red signal!",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.8,
                    (255, 255, 255),
                    (0, 0, 0),
                    (0, 0, 255)
                )
                cv2.line(frame, ROI_line[0], ROI_line[1], (0, 0, 255), 4)
                cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 3)

                # Save violation frame
                timestamp = int(time.time())
                filename = f"violation_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[{frame_count}] 🚗 Violation saved → {filename}")

    cv2.imshow("Traffic Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Process finished.")
