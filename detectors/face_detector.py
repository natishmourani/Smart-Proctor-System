import cv2
import mediapipe as mp
from typing import List, Tuple

class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        """
        Initializes the FaceDetector using MediaPipe's Face Detection.
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, # 0 for short-range faces (within 2 meters), perfect for webcams
            min_detection_confidence=min_detection_confidence
        )
        
    def detect_faces(self, frame) -> Tuple[int, List[Tuple[int, int, int, int]]]:
        """
        Detects faces in the given frame.
        Returns:
            - count: The number of detected faces.
            - boxes: A list of bounding boxes (x, y, w, h) for each detected face.
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        count = 0
        boxes = []
        
        if results.detections:
            count = len(results.detections)
            h, w, _ = frame.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                # Convert relative bounding box coordinates to pixels
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Clip box to frame boundaries
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                boxes.append((x, y, width, height))
                
        return count, boxes
