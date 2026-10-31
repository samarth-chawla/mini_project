# Traffic violations with number-plate recognition

Run from the project root in PowerShell:

```powershell
.\backend\venv\Scripts\python.exe .\backend\traffic_signal_detection_anpr.py
```

The original two scripts remain unchanged. The ANPR script defaults to
`backend/tr2.mp4`. Paths resolve relative to the script. Press **Esc** in the
video window to stop.

## Plate recognition fixes

The reader uses the plate-specific `cct-s-v2-global-model` from FastPlateOCR
instead of general-purpose EasyOCR. It accepts the entire native-resolution
plate crop, including two-line plates. Regression crops from this video read
`KW527` and `KT4540`, correcting the previous `KH527` and `4540` results. These
registration strings are test expectations, not hardcoded recognition rules.

For this clip, `--plate-format hk` requires a one/two-letter prefix and one to
four digits. Digits alone are rejected as incomplete. `--plate-format general`
permits other letter-and-digit layouts of four to ten characters.

Three agreeing observations and at least a two-thirds majority are required
before a plate is published. Pending reads show **READING...**, and unreadable
plates keep `plate_number: null`. Before confirmation, recognition runs every
three frames; after confirmation it uses `--ocr-every` (eight by default).
An image variant does not count as an independent observation.

Mean OCR confidence must be at least 0.85 and each character at least 0.6.
Bounding-box overlap, position, size, and vehicle class are used to match tracks,
reducing cached registrations carrying over to different vehicles. Blur,
occlusion, and tiny plates can still cause errors. "Confirmed" means agreement
across video frames, not independent verification; inspect the saved crop.

Partly occluded cars are retained at `--recovery-conf` (0.25 by default) to
preserve their path before they become fully visible. A weak track needs three
observations before it can raise an alert. Plate ownership is resolved across
overlapping vehicles: a small foreground car's plate cannot also belong to the
bus crop behind it. Short occlusion context covers missed detections. A vehicle
whose lower edge is hidden by a smaller foreground vehicle cannot raise a
crossing alert from that obscured edge.

OCR weights are cached in `backend/.plate_ocr/`. The YOLO plate detector is
`backend/license_plate_detector.pt`; the vehicle model is `backend/yolov8m.pt`.
Missing plate/OCR models download on first use. `--no-download` requires cached
models. Plate OCR uses CPU ONNX Runtime; `--device 0` selects CUDA for the YOLO
detectors when supported.

## Camera calibration and crossing rule

`backend/tr2_camera.json` stores coordinates in a 1100 x 700 display frame.
Red/green boxes are measured from the lamp colors across sampled video frames.
The middle yellow lamp is positioned between them and reported as YELLOW.
`backend/tr2_camera.jpg` previews the measured boxes and boundary.

The clip contains a traffic-light graphic and **no verified painted stop line**.
Analysis of the road background did not find a reliable white transverse marking.
The boundary is explicitly labeled **Estimated monitoring line**, approximately
`(113, 602)` to `(1084, 602)` across the foreground road. This is a virtual
boundary, not a claimed physical stop line. The road polygon is traced for this
clip's perspective; this is not a general automatic road-segmentation model.

A vehicle's bottom-center ground point must move from above the line to below
it during red, inside the road polygon and the finite line segment. Vehicles
already below the boundary when first detected or when the signal turns red
are not automatically violations. Crossing during green consumes that crossing,
so a later red light does not retroactively flag it. One violation is logged per
track, with short-term position/plate suppression for duplicate tracks.

To re-measure the lamps and regenerate the default boundary and preview:

```powershell
.\backend\venv\Scripts\python.exe .\backend\traffic_signal_detection_anpr.py --calibrate-only --recalibrate
```

Edit `red_light`, `green_light`, `road_polygon`, and `monitoring_line` in the
camera JSON to refine a boundary. Endpoints must run left to right. Set
`line_kind` to `configured` and update `line_note` when defining a specific stop
line. A normal run preserves edits; `--recalibrate` replaces them. Another video
needs its own `--camera-config` and a reviewed road polygon.

## Outputs and inspection

- `backend/violations.json`: existing records are preserved. New records include
  `id`, `time`, `image`, `timestamp`, `video_seconds`, `source_video`, `session`,
  vehicle type/confidence, plate number/confidence/crop, `plate_candidate`,
  `plate_observations`, `plate_status`, `boundary_kind`, and `monitoring_line`.
- `backend/plate_detections.json`: best observation per tracked vehicle, including
  vehicles without violations. Pending candidates remain separate from the
  published `plate_number`.
- `frontend/public/violations/`: plate crops and annotated violation images.
  Each event image highlights only its own vehicle and matching ID/type. Later
  successful OCR updates both the metadata and that original event's plate
  annotation. Records include `vehicle_box` and `evidence_frame` for inspection.
- `--output-video`: optional annotated MP4. Yellow plate boxes follow plate-image
  features between detection frames with forward/backward optical-flow checks;
  uncertain boxes are hidden. Labels stay inside the video boundaries.

The web dashboard still displays ID, time, and violation screenshots; it does not
render the extra plate metadata. Old records remain unchanged. Only one process
should write to a given JSON output file.

Example headless inspection starting at a chosen frame:

```powershell
.\backend\venv\Scripts\python.exe .\backend\traffic_signal_detection_anpr.py --headless --start-frame 240 --max-frames 100 --output-video .\backend\anpr_output.mp4
```

`--help` lists paths, thresholds, observation counts, and inference size. Vehicle
inference size is 960 by default; `--imgsz 640` is faster with less detail.

## Dependencies and tests

The project virtual environment has been updated. To recreate the extra OCR
dependencies in the existing backend environment:

```powershell
.\backend\venv\Scripts\python.exe -m pip install --no-deps -r .\backend\requirements-anpr.txt
```

`--no-deps` preserves GUI-enabled OpenCV and NumPy. The environment must already
provide the backend requirements, PyTorch, PyYAML, Pillow, protobuf, packaging,
sympy, and tqdm. EasyOCR is no longer required by this script.

Run regression checks, including OCR against the two saved video crops:

```powershell
.\backend\venv\Scripts\python.exe -m unittest discover -s .\backend\tests -v
```

Sources: [FastPlateOCR](https://github.com/ankandrew/fast-plate-ocr),
[plate detector](https://huggingface.co/Koushim/yolov8-license-plate-detection).
