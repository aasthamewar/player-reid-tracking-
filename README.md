# 🎯 Player Re-Identification and Tracking from a Single Video Feed

This project simulates a real-time player tracking system from a sports video using a custom YOLOv11 model. Each player is detected, assigned a unique ID, and re-identified when they leave and re-enter the frame — all in a single continuous simulation.

---

## 🔍 Project Objective

- Detect and track multiple players in a video
- Assign consistent IDs to players across frames
- Re-identify players when they leave and reappear
- Simulate live frame-by-frame tracking (no batch processing)
- Save results as an annotated video and tracking CSV

---

## 🧠 Key Features

- 🟢 **YOLOv11 Detection** – detects players in real-time
- 🟢 **Kalman Filter Tracking** – predicts player position smoothly
- 🟢 **IoU Matching + Hungarian Logic** – associates detections to existing tracks
- 🟢 **Re-Identification** – matches returning players using appearance-based color histograms
- 🟢 **Video & CSV Logging** – saves tracking results for post-analysis

---

## 📁 Project Structure

````
player_reid_project/
├── input/
│ └── 15sec_input_720p.mp4 # Input sports video
├── models/
│ └── yolov11.pt # Pretrained YOLOv11 model
├── output/
│ └── annotated_output.mp4 # Tracked video with IDs
├── logs/
│ └── track_log.csv # Frame-by-frame tracking log
├── src/
│ ├── main.py # Main pipeline
│ ├── detector.py # YOLOv11 detection code
│ ├── tracker.py # Kalman tracking + re-ID
│ ├── reid.py # Appearance-based re-ID utils
````


---

## 🛠 Installation

> 💡 Recommended: Use Python 3.10 or above

```bash
# Create environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install opencv-python torch torchvision torchaudio ultralytics numpy filterpy scikit-image scipy
```


🚀 Run the Program
```
python src/main.py
```
You will see:

->A live window showing detected & tracked players

->ESC to quit

->Output video saved to /output

->CSV saved to /logs


📦 Model Info

->Custom YOLOv11 trained on sports datasets

->Model must be placed in models/yolov11.pt

->If not available, contact the author

🧩 Techniques Used

->Detection: YOLOv11 via ultralytics 

->Tracking: Kalman Filter with motion estimation

->Re-ID: Color histogram (HSV) with histogram correlation

->Matching: IoU threshold + histogram similarity fallback

### Demo Video

## 🎥 Demo Video
[Watch the demo on Google Drive](https://drive.google.com/file/d/1hIN8OWxxLm8CdcvOn386ssTc3oYg7UZo/view?usp=sharing)


