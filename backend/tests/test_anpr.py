"""Regressions for lost plate rows, incorrect carry-over, and false crossings."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from anpr_processing import (PlateReader, crosses_boundary, move_plate_box,
                             ground_point_visible, plate_owner, recover_vehicle_detections,
                             set_plate_box, update_consensus)
from traffic_signal_detection_anpr import Track, Tracker, is_region_light, violation_evidence


class RecognitionRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (BACKEND / ".plate_ocr/cct_s_v2_global.onnx").exists():
            raise unittest.SkipTest("Run ANPR once to cache the OCR model")
        cls.reader = PlateReader(BACKEND / ".plate_ocr", no_download=True)

    def test_previous_kw_kh_confusion(self):
        crop = cv2.imread(str(Path(__file__).parent / "fixtures/kw527.png"))
        text, confidence = self.reader.read(crop)
        self.assertEqual(text, "KW527")
        self.assertGreaterEqual(confidence, 0.85)

    def test_keep_kt_prefix_above_digits(self):
        crop = cv2.imread(str(Path(__file__).parent / "fixtures/kt4540.png"))
        self.assertEqual(self.reader.read(crop)[0], "KT4540")

    def test_digits_alone_are_not_complete_plate(self):
        prediction = SimpleNamespace(plate="4540", char_probs=np.array([0.99]*4))
        with patch.object(self.reader.reader, "run", return_value=[prediction]):
            self.assertEqual(self.reader.read(np.zeros((25, 50, 3), np.uint8)), ("", 0.0))

    def test_tiny_plate_is_not_guessed(self):
        self.assertEqual(self.reader.read(np.zeros((5, 15, 3), np.uint8)), ("", 0.0))


class TemporalRegression(unittest.TestCase):
    def test_saved_evidence_marks_only_the_event_vehicle(self):
        raw = np.zeros((700, 1100, 3), np.uint8)
        camera = json.loads((BACKEND / "tr2_camera.json").read_text())
        car = Track(67, (440, 378, 629, 614), 381, label="car", plate_text="NN773")
        evidence = violation_evidence(raw, camera, car, 14.432)
        np.testing.assert_array_equal(evidence[500, 440], [0, 0, 255])
        # The old bus outline began at x=313; it must not appear in the event.
        np.testing.assert_array_equal(evidence[400, 313], [0, 0, 0])
        self.assertFalse(np.any(raw))

    def test_car_plate_inside_bus_crop_belongs_to_car(self):
        vehicles = [(66, (313, 195, 630, 614), "bus"), (67, (440, 378, 629, 614), "car")]
        self.assertEqual(plate_owner((500, 555, 560, 580), vehicles), 67)
        self.assertEqual(plate_owner((350, 470, 400, 495), vehicles), 66)

    def test_bus_lower_edge_obscured_by_car_is_not_used(self):
        vehicles = [(66, (313, 195, 630, 614), "bus"), (67, (440, 378, 629, 614), "car")]
        self.assertFalse(ground_point_visible(66, vehicles))
        self.assertTrue(ground_point_visible(67, vehicles))

    def test_occluded_car_keeps_crossing_history_when_it_reappears(self):
        tracker = Tracker(max_age=13)
        road = [[420, 350], [1055, 350], [1095, 695], [0, 695]]
        line = [[113, 602], [1084, 602]]
        first = ((532, 375, 640, 589), "car", 0.511)
        second = ((526, 380, 636, 592), "car", 0.287)
        third = ((440, 378, 629, 614), "car", 0.653)
        track_id = None
        for frame, candidate in [(378, first), (379, second), (381, third)]:
            recovered = recover_vehicle_detections([candidate], .65, .25, road)
            self.assertEqual(len(recovered), 1)
            track = tracker.update(recovered, frame)[0]
            track_id = track.id if track_id is None else track_id
            self.assertEqual(track.id, track_id)
            box = track.box
            crossing = crosses_boundary(track, ((box[0]+box[2])/2, box[3]), line, road)
            self.assertEqual(crossing, frame == 381)
        self.assertEqual(track.hits, 3)

    def test_plate_box_follows_its_pixels_when_car_box_changes(self):
        before = np.zeros((140, 200), np.uint8)
        cv2.rectangle(before, (50, 40), (120, 80), 200, -1)
        cv2.putText(before, "KT4540", (52, 64), cv2.FONT_HERSHEY_SIMPLEX, .4, 0, 1)
        track = Track(1, (0, 0, 180, 130), 1)
        set_plate_box(track, before, (48, 38, 122, 82))
        after = cv2.warpAffine(before, np.float32([[1, 0, 8], [0, 1, 6]]), (200, 140))
        track.box = (5, 5, 195, 139)
        move_plate_box(track, after)
        self.assertIsNotNone(track.plate_box)
        np.testing.assert_allclose(track.plate_box, (56, 44, 130, 88), atol=2)

    def test_featureless_plate_box_is_hidden(self):
        image = np.zeros((140, 200), np.uint8)
        track = Track(1, (0, 0, 180, 130), 1)
        set_plate_box(track, image, (48, 38, 122, 82))
        move_plate_box(track, image)
        self.assertIsNone(track.plate_box)

    def test_one_bad_read_is_not_published(self):
        track = Track(1, (0, 0, 100, 100), 1)
        update_consensus(track, "KH527", 0.99)
        self.assertEqual(track.plate_text, "")
        for _ in range(3):
            update_consensus(track, "KW527", 0.98)
        self.assertEqual(track.plate_text, "KW527")

    def test_new_car_cannot_inherit_nonoverlapping_plate(self):
        tracker = Tracker(max_age=10)
        old = tracker.update([((0, 20, 100, 120), "car", 0.9)], 1)[0]
        old.plate_text = "KW527"
        new = tracker.update([((90, 20, 190, 120), "car", 0.9)], 2)[0]
        self.assertNotEqual(old.id, new.id)
        self.assertEqual(new.plate_text, "")

    def test_nearby_vehicle_keeps_id(self):
        tracker = Tracker(max_age=10)
        old = tracker.update([((0, 20, 100, 120), "car", 0.9)], 1)[0]
        new = tracker.update([((5, 22, 105, 122), "car", 0.9)], 2)[0]
        self.assertEqual(old.id, new.id)

    def test_already_beyond_line_is_not_crossing(self):
        track = Track(1, (0, 0, 100, 100), 1)
        line = [[0, 60], [100, 60]]
        road = [[0, 0], [100, 0], [100, 100], [0, 100]]
        self.assertFalse(crosses_boundary(track, (50, 80), line, road))
        self.assertFalse(crosses_boundary(track, (50, 90), line, road))

    def test_true_crossing_triggers_once(self):
        track = Track(1, (0, 0, 100, 100), 1)
        line = [[0, 60], [100, 60]]
        road = [[0, 0], [100, 0], [100, 100], [0, 100]]
        self.assertFalse(crosses_boundary(track, (50, 50), line, road))
        self.assertTrue(crosses_boundary(track, (50, 70), line, road))
        self.assertFalse(crosses_boundary(track, (50, 80), line, road))

    def test_outside_line_segment_is_not_crossing(self):
        track = Track(1, (0, 0, 100, 100), 1)
        line = [[20, 60], [80, 60]]
        road = [[0, 0], [100, 0], [100, 100], [0, 100]]
        crosses_boundary(track, (90, 50), line, road)
        self.assertFalse(crosses_boundary(track, (90, 70), line, road))


class SignalRegression(unittest.TestCase):
    def test_yellow_is_detected_instead_of_off(self):
        camera = json.loads((BACKEND / "tr2_camera.json").read_text())
        cap = cv2.VideoCapture(str(BACKEND / "tr2.mp4"))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 268)
            ok, frame = cap.read()
            self.assertTrue(ok)
            frame = cv2.resize(frame, (1100, 700))
            self.assertTrue(is_region_light(frame, np.array(camera["yellow_light"]), color="yellow"))
            self.assertFalse(is_region_light(frame, np.array(camera["red_light"]), color="red"))
            self.assertFalse(is_region_light(frame, np.array(camera["green_light"]), color="green"))
        finally:
            cap.release()

    def test_measured_boxes_follow_actual_lamps(self):
        camera = json.loads((BACKEND / "tr2_camera.json").read_text())
        cap = cv2.VideoCapture(str(BACKEND / "tr2.mp4"))
        try:
            for index, expected_red in [(0, False), (300, True), (600, True)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                self.assertTrue(ok)
                frame = cv2.resize(frame, (1100, 700))
                red = is_region_light(frame, np.array(camera["red_light"]), color="red")
                green = is_region_light(frame, np.array(camera["green_light"]), color="green")
                self.assertEqual((red, green), (expected_red, not expected_red))
        finally:
            cap.release()


if __name__ == "__main__":
    unittest.main()
