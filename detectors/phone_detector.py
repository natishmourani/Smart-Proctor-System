import cv2
from ultralytics import YOLO
from typing import List, Tuple

class PhoneDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.65):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def _calculate_iou(self, boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea)

    def detect_phones(self, frame) -> Tuple[bool, List[Tuple[int, int, int, int, float]]]:
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h

        results = self.model(frame, conf=0.30, verbose=False)

        candidate_phones = []
        conflicting_objects = []

        CONFLICTING_CLASSES = {
            "remote", "book", "handbag", "suitcase",
            "laptop", "keyboard", "mouse", "tablet",   # ← NEW: added these
            "wallet", "calculator"
        }

        for result in results:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf     = float(box.conf[0])
                xyxy     = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                bbox = (x1, y1, x2 - x1, y2 - y1)

                if cls_name == "cell phone":
                    candidate_phones.append((bbox, conf))
                elif cls_name in CONFLICTING_CLASSES:
                    conflicting_objects.append((bbox, cls_name, conf))

        valid_phones = []

        for (bbox, conf) in candidate_phones:
            x, y, w, h = bbox

            # ── 1. Confidence ──────────────────────────────────────────
            if conf < self.confidence_threshold:
                continue

            if w == 0 or h == 0:
                continue

            # ── 2. Aspect ratio ────────────────────────────────────────
            # A phone held vertically:  h/w is roughly 1.4 – 2.6
            # A phone held horizontally: w/h is roughly 1.4 – 2.6
            # ID cards: h/w ≈ 0.63  (landscape) → caught by this filter
            # Fan remotes: h/w > 2.8 → caught by this filter
            aspect = h / w  # > 1 means taller than wide

            is_vertical   = 1.40 <= aspect <= 2.60   # upright phone
            is_horizontal = 0.38 <= aspect <= 0.71   # landscape phone

            if not (is_vertical or is_horizontal):
                continue   # ← ID cards, remotes, square objects all rejected here

            # ── 3. Size sanity check ───────────────────────────────────
            area = w * h
            min_area = frame_area * 0.005   # must be at least 0.5% of frame
            max_area = frame_area * 0.25    # must be under 25% of frame

            if not (min_area <= area <= max_area):
                continue   # ← tiny blips and huge false-positive fills rejected

            # ── 4. Conflicting class overlap ───────────────────────────
            rejected = False
            for (c_bbox, c_name, c_conf) in conflicting_objects:
                iou = self._calculate_iou(bbox, c_bbox)
                if iou > 0.30:
                    # Always reject if the overlapping object is a remote
                    if c_name == "remote":
                        rejected = True
                        break
                    # Reject if the conflicting class is more confident
                    if c_conf > conf:
                        rejected = True
                        break

            if rejected:
                continue

            valid_phones.append((x, y, w, h, conf))

        return len(valid_phones) > 0, valid_phones