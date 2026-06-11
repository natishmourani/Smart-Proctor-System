import cv2
from ultralytics import YOLO
from typing import List, Tuple

class PhoneDetector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.60):
        """
        Initializes the PhoneDetector.
        Loads the pre-trained YOLOv8 model (downloads it if not already present).
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        
    def _calculate_iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """
        Calculates the Intersection over Union (IoU) of two bounding boxes in (x, y, w, h) format.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
            
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
        
    def detect_phones(self, frame) -> Tuple[bool, List[Tuple[int, int, int, int, float]]]:
        """
        Detects cell phones in the frame using the YOLOv8 model.
        Filters out false positives (like remotes or wallets) using:
          1. Aspect ratio checks (remotes are too long, wallets are too square).
          2. Class overlap checks (if YOLOv8 double-detects it as a remote, book, etc.).
        """
        # Run inference with a lower internal threshold to catch conflicting classes (e.g. remote, book, bag)
        results = self.model(frame, conf=0.30, verbose=False)
        
        candidate_phones = []
        conflicting_objects = []
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                # Get coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                w = x2 - x1
                h = y2 - y1
                bbox = (x1, y1, w, h)
                
                if class_name == "cell phone":
                    candidate_phones.append((bbox, conf))
                elif class_name in ["remote", "book", "handbag", "suitcase"]:
                    conflicting_objects.append((bbox, class_name, conf))
                    
        valid_phones = []
        
        for (bbox, conf) in candidate_phones:
            # 1. Check confidence threshold
            if conf < self.confidence_threshold:
                continue
                
            x, y, w, h = bbox
            if w == 0 or h == 0:
                continue
                
            # 2. Check Aspect Ratio (height / width)
            aspect_ratio = h / w
            
            # Remotes are typically very narrow and long (aspect ratio > 2.8 or < 0.35 when rotated)
            if aspect_ratio > 2.8 or aspect_ratio < 0.35:
                continue
                
            # Wallets and cups are typically close to square (aspect ratio between 0.75 and 1.30)
            if 0.75 <= aspect_ratio <= 1.30:
                continue
                
            # 3. Check for overlapping conflicting classes
            is_conflicted = False
            for (c_bbox, c_name, c_conf) in conflicting_objects:
                iou = self._calculate_iou(bbox, c_bbox)
                # If there's an overlapping class and it's a remote, or has higher confidence, skip
                if iou > 0.35:
                    if c_name == "remote":
                        is_conflicted = True
                        break
                    elif c_conf > conf:
                        is_conflicted = True
                        break
                        
            if is_conflicted:
                continue
                
            # Passed all filters!
            valid_phones.append((x, y, w, h, conf))
            
        phone_detected = len(valid_phones) > 0
        return phone_detected, valid_phones

