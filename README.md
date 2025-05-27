
# 🧠 Worker Surveillance System

A real-time industrial worker surveillance system using computer vision and generative AI to detect PPE compliance, track workers, estimate poses, and generate natural language safety reports.

---

## 🚀 Features

- ✅ **YOLOv8**: Detects workers and PPE items like gloves, goggles, and shoes
- 🧍 **DeepSORT**: Tracks individuals with persistent IDs across frames
- 🧘 **MediaPipe Pose**: Performs real-time human pose estimation
- 🧾 **LLaMA 3 (via Ollama)**: Generates human-like safety reports
- 📄 **FPDF**: Saves a detailed PDF report for compliance records

---

## 🎥 Input

- Video file or live webcam feed

## 📄 Output

- Real-time annotated video stream
- `worker_activity_report.pdf`: detailed safety report with timestamps and insights

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Required packages:
- `ultralytics`
- `deep_sort_realtime`
- `mediapipe`
- `fpdf`
- `torch`
- `ollama`
- `opencv-python`

---

## 🛠️ Usage

```bash
python worker_surveillance.py
```

To quit the video stream, press `q`.

Make sure:
- Your video file path is correct.
- Ollama is installed and running with the LLaMA 3 model loaded locally.

---

## 🧾 Sample Report

The system generates a professional PDF report like:

```
Worker Activity Report
----------------------
- Total workers detected: 6
- Rahul (ID 2) tracked at (120, 340, 200, 420)
- Pose detected for Sneha (ID 3)
- ...

Safety Recommendation:
Ensure gloves and goggles are worn in all designated zones.
```

---

## 📍 Credits

- YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
- DeepSORT: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)
- MediaPipe: [Google](https://github.com/google/mediapipe)
- LLaMA 3: [Ollama](https://ollama.com)
