# detect_and_score.py
import os
import cv2
import csv
import time
import argparse
import numpy as np
import torch
from deepface import DeepFace
from collections import OrderedDict
from math import sqrt

# ---------------------------
# Simple centroid tracker
# ---------------------------
class CentroidTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = OrderedDict()  # id -> centroid
        self.max_distance = max_distance

    def update(self, rects):
        # rects: list of (x1,y1,x2,y2)
        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for c in input_centroids:
                self.objects[self.next_id] = c
                self.next_id += 1
            return list(self.objects.keys())

        # match existing
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # build distance matrix
        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = sqrt((oc[0] - ic[0])**2 + (oc[1] - ic[1])**2)

        # greedy match
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()
        new_objects = OrderedDict(self.objects)

        for r, c in zip(rows, cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            obj_id = object_ids[r]
            new_objects[obj_id] = input_centroids[c]
            assigned_rows.add(r)
            assigned_cols.add(c)

        # unmatched input -> new ids
        for i, ic in enumerate(input_centroids):
            if i not in assigned_cols:
                new_objects[self.next_id] = ic
                self.next_id += 1

        self.objects = new_objects
        return list(self.objects.keys())

    def get_centroid(self, obj_id):
        return self.objects.get(obj_id, None)

# ---------------------------
# Engagement scoring maps
# ---------------------------
ACTION_WEIGHTS = {
    # adjust these if your custom YOLOv5 uses different class names
    'focused': 20,
    'writing': 10,
    'raising_hand': 15,
    'hand_raise': 15,
    'mobile': -40,
    'phone': -40,
    'sleeping': -50,
    'distracted': -20,
    'idle': -10,
    'reading': 5,
    'talking': -5
}
EMOTION_WEIGHTS = {
    'happy': 10,
    'neutral': 0,
    'sad': -20,
    'angry': -15,
    'surprise': 5,
    'disgust': -10,
    'fear': -15
}

# ---------------------------
# Helpers
# ---------------------------
def clamp_score(s):
    return max(0, min(100, int(round(s))))

# ---------------------------
# Main pipeline
# ---------------------------
def main(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    # load YOLOv5 custom model
    print("Loading YOLOv5 model:", args.weights)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights, force_reload=False)
    # model.conf = 0.4  # confidence threshold
    names = model.names  # class id -> name

    # OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # video I/O
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: " + args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    # tracker and logs
    tracker = CentroidTracker(max_distance=80)
    fieldnames = ['frame','timestamp','id','bbox','action','emotion','engagement_score']
    csvfile = open(args.csv, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    frame_idx = 0
    start_time = time.time()

    print("Processing video... (this may take a while on first run because DeepFace downloads models)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        timestamp = frame_idx / fps

        # run yolov5 inference
        results = model(frame)  # returns ultralytics Results
        detections = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,class]

        # collect person/action boxes (if model trained on actions it returns action classes)
        rects = []
        det_infos = []  # parallel list with (x1,y1,x2,y2,cls_name,conf)
        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            cls = int(cls)
            cls_name = names.get(cls, str(cls))
            # if this model was trained on action classes, it might not use 'person' class.
            # We assume model returns action classes (like 'writing', 'sleeping', etc.)
            rects.append((x1,y1,x2,y2))
            det_infos.append((x1,y1,x2,y2,cls_name,float(conf)))

        # update tracker to assign ids
        ids = tracker.update(rects)

        # prepare mapping from centroid->(bbox,cls)
        # match current centroids order to det_infos by distance
        centroid_list = [tracker.get_centroid(i) for i in ids]
        bbox_assignments = {}  # id -> det_info
        for i, cid in enumerate(ids):
            centroid = centroid_list[i]
            # find nearest det_info
            best_j = None
            best_dist = 1e9
            for j, (x1,y1,x2,y2,cls_name,conf) in enumerate(det_infos):
                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)
                d = sqrt((centroid[0]-cx)**2 + (centroid[1]-cy)**2)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j is not None:
                bbox_assignments[cid] = det_infos[best_j]

        # For each tracked id, analyze face and compute engagement
        per_frame_scores = []
        for tid in ids:
            if tid not in bbox_assignments:
                continue
            x1,y1,x2,y2,action_name,conf = bbox_assignments[tid]
            # crop person area and search for face inside it
            px1 = max(0, x1); py1 = max(0, y1); px2 = min(w-1, x2); py2 = min(h-1, y2)
            person_roi = frame[py1:py2, px1:px2].copy()
            emotion_label = 'neutral'
            # detect faces inside person box
            try:
                gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
            except Exception as e:
                faces = []

            if len(faces) > 0:
                # choose largest face
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                (fx,fy,fw,fh) = faces[0]
                face_img = person_roi[fy:fy+fh, fx:fx+fw]
                # DeepFace analyze (may download models first time)
                try:
                    analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    dominant = analysis.get('dominant_emotion', None)
                    if dominant:
                        emotion_label = dominant.lower()
                except Exception as e:
                    # fallback if DeepFace fails
                    emotion_label = 'neutral'
            else:
                # no face detected -> keep neutral
                emotion_label = 'neutral'

            # compute engagement score
            base = 50
            wa = ACTION_WEIGHTS.get(action_name.lower(), 0)
            we = EMOTION_WEIGHTS.get(emotion_label.lower(), 0)
            score = clamp_score(base + wa + we)

            # draw annotations
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
            label = f"ID:{tid} {action_name} {emotion_label} S:{score}"
            cv2.putText(frame, label, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # append
            per_frame_scores.append(score)

            # write CSV row
            writer.writerow({
                'frame': frame_idx,
                'timestamp': round(timestamp, 3),
                'id': tid,
                'bbox': f"{x1},{y1},{x2},{y2}",
                'action': action_name,
                'emotion': emotion_label,
                'engagement_score': score
            })

        # aggregate class-level engagement
        class_engagement = int(round(np.mean(per_frame_scores))) if per_frame_scores else 0
        cv2.putText(frame, f"Class Engagement: {class_engagement}%", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        out.write(frame)

        # optional fast display
        if args.show:
            cv2.imshow("Annotated", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    csvfile.close()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print("Done. Frames:", frame_idx, "Elapsed:", round(elapsed,2), "s")
    print("Output video:", args.output)
    print("CSV log:", args.csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/classroomvideo.mp4", help="input video path")
    parser.add_argument("--weights", type=str, default="models/yolov5_action_best.pt", help="yolov5 weights")
    parser.add_argument("--output", type=str, default="results/video_annotated.mp4", help="output annotated video")
    parser.add_argument("--csv", type=str, default="results/video_log.csv", help="output csv log")
    parser.add_argument("--show", action="store_true", help="show frames while processing")
    args = parser.parse_args()
    main(args)
