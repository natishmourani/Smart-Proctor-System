import os
import csv
from datetime import datetime

class ViolationLogger:
    def __init__(self, filename="violations.csv"):
        """
        Initializes the ViolationLogger.
        Creates the CSV file with headers if it does not exist.
        """
        # Get the directory of the current file and determine the project root
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.filepath = os.path.join(self.project_root, filename)
        
        # Dictionary to keep track of violation counts in the current session
        self.counts = {
            "Looking Away": 0,
            "Multiple Persons": 0,
            "Phone Detected": 0
        }
        
        # Create CSV and write header if it doesn't exist
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Violation Type", "Additional Details"])
                
    def log_violation(self, violation_type: str, details: str):
        """
        Logs a violation to the CSV file and updates the session counts.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write to CSV
        with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, violation_type, details])
            
        # Update session count if it's one of the monitored types
        if violation_type in self.counts:
            self.counts[violation_type] += 1
            
    def get_counts(self):
        """
        Returns the dictionary of violation counts.
        """
        return self.counts
