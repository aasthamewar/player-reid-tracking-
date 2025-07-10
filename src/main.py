from tracker import Tracker
import csv
import os
import cv2
from detector import Detector

VIDEO_PATH = "input/15sec_input_720p.mp4"
MODEL_PATH = "models/yolov11.pt"

cap = cv2.VideoCapture(VIDEO_PATH)
detector = Detector(MODEL_PATH)
tracker = Tracker()

OUTPUT_VIDEO = "output/annotated_output.mp4"
LOG_PATH = "logs/track_log.csv"

# Set up video writer (after getting first frame)
out = None
frame_no = 0

# Create CSV
os.makedirs("logs", exist_ok=True)
log_file = open(LOG_PATH, 'w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(["frame", "id", "x", "y", "w", "h"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and track
    detections = detector.detect(frame)
    tracker.update(detections, frame)


    # Initialize writer if not yet
    if out is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30, (w, h))

    # Draw and log
    for track in tracker.tracks:
        x, y, w, h = track.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track.id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        csv_writer.writerow([frame_no, track.id, x, y, w, h])

    # Show and write
    cv2.imshow("Detection", frame)
    out.write(frame)
    frame_no += 1

    if cv2.waitKey(1) == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()
out.release()
