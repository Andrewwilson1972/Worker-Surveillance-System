
import cv2
import torch
import numpy as np
import ollama
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from fpdf import FPDF
from datetime import datetime
import mediapipe as mp

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Initialize DeepSORT Tracker
deep_sort = DeepSort(max_age=50)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Dummy names to assign to worker IDs
dummy_names = ["Andrew", "Pradeep", "Neha", "Rahul", "Ayesha", "John", "Kiran", "Divya", "Alex", "Sneha"]
track_id_to_name = {}

# Open video file or webcam
cap = cv2.VideoCapture(r"C:\Users\andre\Downloads\Example of Hi-Definition Video Surveillance of a Factory Floor - by CCTVDOC.COM.mp4")

detection_log = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    detections = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2, conf, cls = box.xyxy[0].tolist() + box.conf.tolist() + box.cls.tolist()
            class_name = yolo_model.names[int(cls)]
            if class_name == "person":
                detections.append([[x1, y1, x2, y2], conf, int(cls)])
                detection_log.append(f"{timestamp}: Detected worker at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) with confidence {conf:.2f}")

    # DeepSORT tracking
    tracked_objects = deep_sort.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_id = int(track.track_id)

        if track_id not in track_id_to_name:
            track_id_to_name[track_id] = dummy_names[track_id % len(dummy_names)]
        name = track_id_to_name[track_id]

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id} - {name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        detection_log.append(f"{timestamp}: {name} (ID {track_id}) tracked at ({x1}, {y1}, {x2}, {y2})")

        # Safe crop
        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)

        if x2 > x1 and y2 > y1:
            person_roi = frame[y1:y2, x1:x2]

            if person_roi.size != 0:
                rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                pose_result = pose.process(rgb_roi)

                if pose_result.pose_landmarks:
                    for lm in pose_result.pose_landmarks.landmark:
                        cx = int(x1 + lm.x * (x2 - x1))
                        cy = int(y1 + lm.y * (y2 - y1))
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    detection_log.append(f"{timestamp}: Pose detected for {name} (ID {track_id})")
            else:
                print(f"⚠️ Skipped pose estimation for ID {track_id} due to empty ROI")

    # Show frame
    cv2.imshow("Worker Surveillance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === LLaMA 3 Report Generation ===
def generate_worker_report(worker_data):
    prompt = f"""
    Generate a detailed worker activity report based on the following observations:
    {worker_data}
    Include total workers detected, tracking details, timestamps, pose estimation insights, and safety recommendations.
    """
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# Generate and save report
worker_data = "\n".join(detection_log)
report = generate_worker_report(worker_data)

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(200, 10, "Worker Activity Report", ln=True, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, report)

pdf.output("worker_activity_report.pdf")
print("✅ PDF report saved as worker_activity_report.pdf")
