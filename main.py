import cv2
import time
import os
import sys
from datetime import datetime

# Add the project root directory to python path to support modular imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.face_detector import FaceDetector
from detectors.head_pose_detector import HeadPoseDetector
from detectors.phone_detector import PhoneDetector
from utils.logger import ViolationLogger
from utils.screenshot import ScreenshotManager

def main():
    # Initialize components
    print("[INFO] Initializing detectors...")
    face_detector = FaceDetector(min_detection_confidence=0.5)
    head_pose_detector = HeadPoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    phone_detector = PhoneDetector(model_name="yolov8n.pt", confidence_threshold=0.60)
    
    logger = ViolationLogger()
    screenshot_manager = ScreenshotManager()
    
    # Webcam capture setup
    # Try index 0, fallback to other standard indices if needed
    print("[INFO] Accessing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Trying fallback camera index...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[CRITICAL] Webcam access failed. Please connect a webcam and try again.")
            sys.exit(1)
            
    # Set webcam resolution to standard HD for balanced speed and accuracy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # State tracking variables
    no_face_start_time = None
    looking_away_start_time = None
    
    # Cooldown timers (to prevent logging violations and saving screenshots on every single frame)
    # Holds the last timestamp a particular violation type was recorded
    last_log_time = {
        "No Face Detected": 0.0,
        "Looking Away": 0.0,
        "Multiple Persons": 0.0,
        "Phone Detected": 0.0
    }
    COOLDOWN_PERIOD = 5.0 # seconds between logs/screenshots of the same violation type
    
    # Window setup
    window_name = "Smart Exam Proctoring System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("[INFO] Proctoring system started. Press 'q' inside the window to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame from webcam. Retrying...")
            time.sleep(0.1)
            continue
            
        # Create a clean copy of the frame for screenshot purposes (without the dashboard/HUD overlays)
        raw_frame = frame.copy()
        current_time = time.time()
        
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Lists and dicts to store findings for the current frame
        active_warnings = []
        status = "Monitoring"
        looking_direction = "N/A"
        
        # 1. RUN DETECTORS
        # Face Detection
        face_count, face_boxes = face_detector.detect_faces(frame)
        
        # Phone Detection
        phone_detected, phone_boxes = phone_detector.detect_phones(frame)
        
        # 2. EVALUATE LOGIC BASED ON FACE COUNT
        if face_count == 1:
            # Candidate is detected
            status = "Monitoring"
            no_face_start_time = None # Reset no face timer
            
            # Head Pose Estimation
            direction, angles = head_pose_detector.estimate_pose(frame)
            looking_direction = direction
            
            # Check if looking away (Left, Right, or Down)
            if direction in ["Left", "Right", "Down"]:
                if looking_away_start_time is None:
                    looking_away_start_time = current_time
                
                # Calculate duration of looking away
                duration = current_time - looking_away_start_time
                if duration > 2.0:
                    status = "Warning"
                    active_warnings.append("Warning: Candidate Looking Away")
                    
                    # Log violation with cooldown
                    if current_time - last_log_time["Looking Away"] > COOLDOWN_PERIOD:
                        logger.log_violation("Looking Away", f"Looking {direction} for {duration:.1f}s")
                        screenshot_manager.save_screenshot(raw_frame, "Looking Away")
                        last_log_time["Looking Away"] = current_time
                        print(f"[VIOLATION] Candidate Looking Away ({direction}) - Logged & Screenshotted")
            else:
                # Candidate is looking forward
                looking_away_start_time = None
                
            # Draw Face Bounding Box (Green for single candidate)
            x, y, fw, fh = face_boxes[0]
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(frame, "Candidate", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
        elif face_count == 0:
            # No candidate detected
            status = "Warning"
            looking_direction = "N/A"
            looking_away_start_time = None # Reset looking away timer
            
            if no_face_start_time is None:
                no_face_start_time = current_time
                
            duration = current_time - no_face_start_time
            if duration > 3.0:
                active_warnings.append("Warning: No Face Detected")
                
                # Log violation with cooldown
                if current_time - last_log_time["No Face Detected"] > COOLDOWN_PERIOD:
                    logger.log_violation("No Face Detected", f"No face present for {duration:.1f}s")
                    screenshot_manager.save_screenshot(raw_frame, "No Face Detected")
                    last_log_time["No Face Detected"] = current_time
                    print("[VIOLATION] No Face Detected - Logged & Screenshotted")
            else:
                active_warnings.append("Warning: Detecting Face...")
                
        else:
            # Multiple persons detected (face_count > 1)
            status = "Warning"
            looking_direction = "N/A"
            no_face_start_time = None
            looking_away_start_time = None
            
            active_warnings.append("Warning: Multiple Persons Detected")
            
            # Log violation with cooldown
            if current_time - last_log_time["Multiple Persons"] > COOLDOWN_PERIOD:
                logger.log_violation("Multiple Persons", f"{face_count} faces detected in frame")
                screenshot_manager.save_screenshot(raw_frame, "Multiple Persons")
                last_log_time["Multiple Persons"] = current_time
                print(f"[VIOLATION] Multiple Persons ({face_count} detected) - Logged & Screenshotted")
                
            # Draw Face Bounding Boxes (Red for multiple faces)
            for i, (x, y, fw, fh) in enumerate(face_boxes):
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 0, 255), 2)
                cv2.putText(frame, f"Person {i+1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
        # 3. EVALUATE PHONE DETECTION
        if phone_detected:
            status = "Warning"
            active_warnings.append("Warning: Mobile Phone Detected")
            
            # Log violation with cooldown
            if current_time - last_log_time["Phone Detected"] > COOLDOWN_PERIOD:
                logger.log_violation("Phone Detected", f"Mobile phone detected with {phone_boxes[0][4]:.2f} confidence")
                screenshot_manager.save_screenshot(raw_frame, "Phone Detected")
                last_log_time["Phone Detected"] = current_time
                print("[VIOLATION] Mobile Phone Detected - Logged & Screenshotted")
                
            # Draw Bounding Boxes around phones (Red)
            for (px, py, pw, ph, conf) in phone_boxes:
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
                cv2.putText(frame, f"Phone {conf:.2f}", (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                
        # 4. DRAW DASHBOARD OVERLAY
        overlay = frame.copy()
        
        # Transparent background panel for dashboard
        cv2.rectangle(overlay, (15, 15), (320, 320), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (15, 15), (320, 320), (80, 80, 80), 2)
        
        # Render text metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Header
        cv2.putText(frame, "PROCTORING DASHBOARD", (25, 45), font, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (25, 55), (310, 55), (120, 120, 120), 1)
        
        # Status
        status_color = (0, 255, 0) # Green for monitoring
        if status == "Warning":
            status_color = (0, 0, 255) # Red for warning
        cv2.putText(frame, f"Status: {status}", (25, 85), font, 0.6, status_color, 2, cv2.LINE_AA)
        
        # Metrics
        cv2.putText(frame, f"Faces Counted: {face_count}", (25, 115), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        dir_color = (255, 255, 255)
        if looking_direction in ["Left", "Right", "Down"]:
            dir_color = (0, 165, 255) # Orange warnings
        cv2.putText(frame, f"Looking Dir: {looking_direction}", (25, 145), font, 0.55, dir_color, 1, cv2.LINE_AA)
        
        phone_status_text = "Detected" if phone_detected else "Not Detected"
        phone_status_color = (0, 0, 255) if phone_detected else (0, 255, 0)
        cv2.putText(frame, f"Phone Status: {phone_status_text}", (25, 175), font, 0.55, phone_status_color, 1, cv2.LINE_AA)
        
        # Violation summary header
        cv2.line(frame, (25, 190), (310, 190), (120, 120, 120), 1)
        cv2.putText(frame, "TOTAL VIOLATIONS", (25, 215), font, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Fetch current counts from logger
        counts = logger.get_counts()
        cv2.putText(frame, f"Looking Away: {counts['Looking Away']}", (25, 245), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Multiple Persons: {counts['Multiple Persons']}", (25, 275), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Phone Detected: {counts['Phone Detected']}", (25, 305), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Render Active Warnings at the top center of the screen
        warning_y = 45
        for warning in active_warnings:
            # Dynamic text centering
            (tw, th), _ = cv2.getTextSize(warning, font, 0.75, 2)
            wx = int((w - tw) / 2)
            
            # Semi-transparent background banner for warning
            overlay_w = frame.copy()
            cv2.rectangle(overlay_w, (wx - 15, warning_y - 25), (wx + tw + 15, warning_y + 10), (0, 0, 150), -1)
            cv2.addWeighted(overlay_w, 0.7, frame, 0.3, 0, frame)
            
            # Draw warning text and red border
            cv2.rectangle(frame, (wx - 15, warning_y - 25), (wx + tw + 15, warning_y + 10), (0, 0, 255), 2)
            cv2.putText(frame, warning, (wx, warning_y), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            warning_y += 50
            
        # Show image in window
        cv2.imshow(window_name, frame)
        
        # Stop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    print("[INFO] Releasing webcam and closing windows...")
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Proctoring system closed. Violation logs written to violations.csv.")

if __name__ == "__main__":
    main()
