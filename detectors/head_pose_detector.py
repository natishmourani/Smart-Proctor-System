import cv2
import numpy as np
import mediapipe as mp
import math
from typing import Tuple, Optional

class HeadPoseDetector:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initializes the HeadPoseDetector using MediaPipe's Face Mesh.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 3D model points of standard face features (in world coordinates)
        # Nose tip, Chin, Left eye outer corner, Right eye outer corner, Left mouth corner, Right mouth corner
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Right eye outer corner (from subject's perspective, appears on left of image)
            (225.0, 170.0, -135.0),      # Left eye outer corner (from subject's perspective, appears on right of image)
            (-150.0, -150.0, -125.0),    # Right mouth corner
            (150.0, -150.0, -125.0)      # Left mouth corner
        ], dtype=np.float32)

    def estimate_pose(self, frame) -> Tuple[str, Optional[Tuple[float, float, float]]]:
        """
        Estimates the head pose (pitch, yaw, roll) and determines looking direction.
        Returns:
            - direction: "Forward", "Left", "Right", "Down", or "Unknown"
            - angles: Tuple of (pitch, yaw, roll) in degrees, or None if no face is detected.
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return "Unknown", None

        # Process the first detected face (assumed to be the candidate)
        face_landmarks = results.multi_face_landmarks[0]
        
        # MediaPipe landmark indices corresponding to 3D model points:
        # Nose tip: 1
        # Chin: 152
        # Right eye outer corner: 33 (subject's right eye, image left)
        # Left eye outer corner: 263 (subject's left eye, image right)
        # Right mouth corner: 61
        # Left mouth corner: 291
        
        indices = [1, 152, 33, 263, 61, 291]
        image_points = []
        
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            image_points.append((lm.x * w, lm.y * h))
            
        image_points = np.array(image_points, dtype=np.float32)

        # Camera internals (approximate focal length and optical center)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return "Unknown", None

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rotation_vector)

        # Calculate Euler angles
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(rmat[2, 1], rmat[2, 2])
            y = math.atan2(-rmat[2, 0], sy)
            z = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            x = math.atan2(-rmat[1, 2], rmat[1, 1])
            y = math.atan2(-rmat[2, 0], sy)
            z = 0

        # Convert to degrees
        pitch = math.degrees(x) # Rotation around X-axis (Looking up/down)
        yaw = math.degrees(y)   # Rotation around Y-axis (Looking left/right)
        roll = math.degrees(z)  # Rotation around Z-axis (Head tilt)

        # Classify the looking direction
        # Thresholds:
        # Looking Left/Right: Yaw threshold (typically |yaw| > 15)
        # Looking Down: Pitch threshold (typically pitch < -10 or pitch > 10 depending on camera orientation)
        # Standard solvePnP coordinate system:
        # Positive Yaw -> Looking Left
        # Negative Yaw -> Looking Right
        # Negative Pitch -> Looking Down (usually, face pointing downwards causes nose to move down relative to chin/eyes)
        
        direction = "Forward"
        
        if yaw > 15:
            direction = "Left"
        elif yaw < -15:
            direction = "Right"
        elif pitch < -12:
            direction = "Down"

        return direction, (pitch, yaw, roll)
