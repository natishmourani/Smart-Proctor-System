import os
import cv2
from datetime import datetime

class ScreenshotManager:
    def __init__(self, folder_name="screenshots"):
        """
        Initializes the ScreenshotManager.
        Creates the screenshots directory if it does not exist.
        """
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.folder_path = os.path.join(self.project_root, folder_name)
        
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            
    def save_screenshot(self, frame, violation_type: str) -> str:
        """
        Saves a copy of the current frame as a JPEG image.
        Filename format: violation_YYYYMMDD_HHMMSS.jpg
        In case of duplicate timestamps, appends a microsecond suffix to prevent overwriting.
        Returns the path of the saved screenshot.
        """
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{timestamp}.jpg"
        filepath = os.path.join(self.folder_path, filename)
        
        # Avoid collision if multiple screenshots are taken at the same second
        if os.path.exists(filepath):
            timestamp_ms = now.strftime("%Y%m%d_%H%M%S_%f")[:19] # up to deciseconds/centiseconds
            filename = f"violation_{timestamp_ms}.jpg"
            filepath = os.path.join(self.folder_path, filename)
            
        # Write image
        cv2.imwrite(filepath, frame)
        return filepath
