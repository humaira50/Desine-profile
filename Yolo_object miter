# pip install ultralytics opencv-python cvzone numpy
import cv2
import math
import numpy as np
import cvzone
from ultralytics import YOLO

# ---------------------- Config ----------------------
VIDEO_SOURCE =  "video/human.mp4"
MODEL_PATH = "../Yolo-Weights/yolov8n.pt"  # n/s/m/l আপনার পছন্দ
CONF_THRES = 0.4
IOU_THRES = 0.45
TARGET_CLASSES = None  # None => সব ক্লাস; উদাহরণ: {'person','car'}
PIXELS_PER_METER = None  # উদাহরণ: 50 মানে 50px == 1m; না দিলে পিক্সেলেই দেখাবে
MAX_MATCH_DIST = 80      # পিক্সেলে; ট্র্যাক/ডিটেকশন ম্যাচের সর্বোচ্চ দূরত্ব
MAX_DISAPPEARED = 30     # কত ফ্রেম মিস হলে ট্র্যাক ড্রপ করবে
DRAW_TRAIL_LEN = 30      # ট্রেইলে সর্বোচ্চ কত পয়েন্ট দেখাবো

# COCO class names (80)
CLASS_NAMES = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
               "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat"]


# ------------------- Simple Centroid Tracker -------------------
class CentroidTracker:
    def __init__(self, max_match_dist=80, max_disappeared=30, trail_len=30):
        self.next_id = 1
        self.objects = {}         # id -> (cx, cy)
        self.totals = {}          # id -> cumulative distance (pixels)
        self.disappeared = {}     # id -> frames disappeared
        self.trails = {}          # id -> list of (cx,cy)
        self.max_match_dist = max_match_dist
        self.max_disappeared = max_disappeared
        self.trail_len = trail_len

    def _dist(self, a, b):
        ax, ay = a; bx, by = b
        return math.hypot(ax - bx, ay - by)

    def update(self, detections):
        """
        detections: list of (cx, cy, box, cls_name, conf)
        returns: dict id -> (cx, cy, box, cls_name, conf, total_dist_px)
        """
        det_points = [(cx, cy) for (cx, cy, *_rest) in detections]

        if len(self.objects) == 0:
            # Register all detections
            for (cx, cy, *_rest) in detections:
                oid = self.next_id; self.next_id += 1
                self.objects[oid] = (cx, cy)
                self.totals[oid] = 0.0
                self.disappeared[oid] = 0
                self.trails[oid] = [(cx, cy)]
        else:
            # Try to match existing objects to detections (greedy nearest neighbor)
            object_ids = list(self.objects.keys())
            object_pts = [self.objects[oid] for oid in object_ids]

            if len(det_points) == 0:
                # No detections: everyone disappears
                for oid in object_ids:
                    self.disappeared[oid] += 1
                self._purge()
            else:
                # Build distance matrix
                D = np.zeros((len(object_pts), len(det_points)), dtype=np.float32)
                for i, op in enumerate(object_pts):
                    for j, dp in enumerate(det_points):
                        D[i, j] = self._dist(op, dp)

                # Greedy: repeatedly pick smallest pair under threshold
                unmatched_objs = set(range(len(object_pts)))
                unmatched_dets = set(range(len(det_points)))
                matches = []

                while True:
                    if len(unmatched_objs) == 0 or len(unmatched_dets) == 0:
                        break
                    # find global min among remaining
                    sub = D[np.ix_(list(unmatched_objs), list(unmatched_dets))]
                    i_min, j_min = np.unravel_index(np.argmin(sub), sub.shape)
                    obj_index = list(unmatched_objs)[i_min]
                    det_index = list(unmatched_dets)[j_min]
                    dist = D[obj_index, det_index]
                    if dist > self.max_match_dist:
                        break
                    matches.append((obj_index, det_index))
                    unmatched_objs.remove(obj_index)
                    unmatched_dets.remove(det_index)

                # Update matched
                for obj_index, det_index in matches:
                    oid = object_ids[obj_index]
                    (cx, cy) = det_points[det_index]
                    # distance increment
                    prev = self.objects[oid]
                    self.totals[oid] += self._dist(prev, (cx, cy))
                    self.objects[oid] = (cx, cy)
                    self.disappeared[oid] = 0
                    self.trails[oid].append((cx, cy))
                    if len(self.trails[oid]) > self.trail_len:
                        self.trails[oid] = self.trails[oid][-self.trail_len:]

                # Unmatched objects disappear
                for obj_index in unmatched_objs:
                    oid = object_ids[obj_index]
                    self.disappeared[oid] += 1

                # New detections become new objects
                for det_index in unmatched_dets:
                    (cx, cy) = det_points[det_index]
                    oid = self.next_id; self.next_id += 1
                    self.objects[oid] = (cx, cy)
                    self.totals[oid] = 0.0
                    self.disappeared[oid] = 0
                    self.trails[oid] = [(cx, cy)]

                self._purge()

        # Build return dictionary (attach detection info by nearest)
        result = {}
        # Map each object to nearest detection (to get box/cls/conf for drawing)
        for oid, (ox, oy) in self.objects.items():
            # find nearest detection for visual info
            nearest_idx, nearest_dist = None, float('inf')
            for i, (cx, cy, *_rest) in enumerate(detections):
                d = self._dist((ox, oy), (cx, cy))
                if d < nearest_dist:
                    nearest_dist, nearest_idx = d, i
            if nearest_idx is not None:
                cx, cy, box, cls_name, conf = detections[nearest_idx]
                result[oid] = (ox, oy, box, cls_name, conf, self.totals[oid])
        return result

    def _purge(self):
        to_delete = [oid for oid, cnt in self.disappeared.items() if cnt > self.max_disappeared]
        for oid in to_delete:
            for d in (self.objects, self.totals, self.disappeared, self.trails):
                d.pop(oid, None)


# ---------------------- Main ----------------------
def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if isinstance(VIDEO_SOURCE, int):
        cap.set(3, 1280)
        cap.set(4, 720)

    model = YOLO(MODEL_PATH)

    tracker = CentroidTracker(
        max_match_dist=MAX_MATCH_DIST,
        max_disappeared=MAX_DISAPPEARED,
        trail_len=DRAW_TRAIL_LEN
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run YOLO
        results = model.predict(source=frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
        detections = []

        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2
                conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
                cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
                cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else "obj"

                # filter by target class if provided
                if TARGET_CLASSES is not None and cls_name not in TARGET_CLASSES:
                    continue

                detections.append((cx, cy, (x1, y1, x2, y2), cls_name, conf))

        # Update tracker & get per-ID info
        tracked = tracker.update(detections)

        # Draw
        for oid, (ox, oy, (x1, y1, x2, y2), cls_name, conf, total_px) in tracked.items():
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h), l=10)
            # distance text
            if PIXELS_PER_METER:
                dist_m = total_px / float(PIXELS_PER_METER)
                dist_text = f"{dist_m:.2f} m"
            else:
                dist_text = f"{total_px:.1f} px"

            label = f"ID {oid} | {cls_name} {conf:.2f} | moved {dist_text}"
            cvzone.putTextRect(frame, label, (max(0, x1), max(30, y1 - 10)),
                               scale=0.7, thickness=1, offset=3)

        # Optional: draw trails
        for oid, pts in tracker.trails.items():
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), 2)
            # draw center
            cx, cy = tracker.objects[oid]
            cv2.circle(frame, (int(cx), int(cy)), 3, (255, 255, 255), -1)

        cv2.imshow("YOLOv8 Movement Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
