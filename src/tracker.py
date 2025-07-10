import numpy as np
from filterpy.kalman import KalmanFilter
from reid import get_color_histogram, compare_hist  # Add this

class Track:
    count = 0

    def __init__(self, bbox):
        self.id = Track.count
        Track.count += 1
        self.kf = self.create_kf(bbox)
        self.bbox = bbox
        self.hits = 0
        self.no_losses = 0
        self.hist = None  # 🔹 Added for re-ID appearance feature

    def create_kf(self, bbox):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        kf.R[2:, 2:] *= 10.
        kf.P *= 10.
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01
        kf.x[:4] = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]])
        return kf

    def predict(self):
        self.kf.predict()
        x, y, w, h = self.kf.x[:4].reshape(4)
        self.bbox = [int(x), int(y), int(w), int(h)]

    def update(self, bbox):
        self.kf.update(np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]]))
        self.hits += 1
        self.no_losses = 0

class Tracker:
    def __init__(self, iou_threshold=0.3):
        self.tracks = []
        self.iou_threshold = iou_threshold

    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area != 0 else 0

    def update(self, detections, frame):
        for track in self.tracks:
            track.predict()

        assigned_tracks = set()
        assigned_detections = set()

        # First pass: match using IoU
        for i, det in enumerate(detections):
            best_iou = 0
            best_track = None
            for track in self.tracks:
                if track.id in assigned_tracks:
                    continue
                iou_score = self.iou(track.bbox, det[:4])
                if iou_score > best_iou and iou_score > self.iou_threshold:
                    best_iou = iou_score
                    best_track = track

            if best_track:
                best_track.update(det[:4])
                x, y, w, h = det[:4]
                crop = frame[y:y+h, x:x+w]
                if crop.size != 0:
                    best_track.hist = get_color_histogram(crop)
                assigned_tracks.add(best_track.id)
                assigned_detections.add(i)

        # Second pass: match remaining detections using appearance
        for i, det in enumerate(detections):
            if i in assigned_detections:
                continue

            x, y, w, h = det[:4]
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            new_hist = get_color_histogram(crop)

            best_score = 0
            best_track = None
            for track in self.tracks:
                if track.id in assigned_tracks or track.hist is None:
                    continue
                score = compare_hist(new_hist, track.hist)
                if score > best_score:
                    best_score = score
                    best_track = track

            if best_score > 0.85:
                best_track.update(det[:4])
                best_track.hist = new_hist
                assigned_tracks.add(best_track.id)
            else:
                new_track = Track(det[:4])
                new_track.hist = new_hist
                self.tracks.append(new_track)
