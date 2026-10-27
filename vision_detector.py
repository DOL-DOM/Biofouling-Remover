import cv2
import numpy as np


def _bbox_gap(a, b):
    """두 bbox(x1, y1, x2, y2) 사이의 최소 간격(px). 겹치면 0."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def cluster_detections(detections, cluster_distance=60, min_cluster_size=1):
    """
    개별 따개비 검출 결과를 밀집 영역(구역, Cluster/Area) 단위로 묶는다.
    - bbox 간 간격이 cluster_distance(px) 이하이면 같은 구역으로 연결 (Union-Find)
    - 묶인 뒤 구역 bbox끼리 겹치면 다시 병합
    반환: [{"id", "bbox", "center", "count", "area", "members"}...]
    """
    n = len(detections)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_gap(detections[i]["bbox"], detections[j]["bbox"]) <= cluster_distance:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(detections[i])

    # 구역 bbox 생성
    zones = []
    for members in groups.values():
        x1 = min(m["bbox"][0] for m in members)
        y1 = min(m["bbox"][1] for m in members)
        x2 = max(m["bbox"][2] for m in members)
        y2 = max(m["bbox"][3] for m in members)
        zones.append({"bbox": [x1, y1, x2, y2], "members": list(members)})

    # 구역 bbox가 서로 겹치면 하나로 병합 (안정될 때까지 반복)
    merged = True
    while merged:
        merged = False
        for i in range(len(zones)):
            for j in range(i + 1, len(zones)):
                if _bbox_gap(zones[i]["bbox"], zones[j]["bbox"]) == 0:
                    a, b = zones[i], zones[j]
                    a["bbox"] = [min(a["bbox"][0], b["bbox"][0]), min(a["bbox"][1], b["bbox"][1]),
                                 max(a["bbox"][2], b["bbox"][2]), max(a["bbox"][3], b["bbox"][3])]
                    a["members"].extend(b["members"])
                    zones.pop(j)
                    merged = True
                    break
            if merged:
                break

    clusters = []
    for z in zones:
        if len(z["members"]) < min_cluster_size:
            continue
        x1, y1, x2, y2 = z["bbox"]
        clusters.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
            "count": len(z["members"]),
            "area": int((x2 - x1) * (y2 - y1)),
            "members": z["members"],
        })

    # 면적 큰 순서로 ID 부여
    clusters.sort(key=lambda c: c["area"], reverse=True)
    for idx, c in enumerate(clusters, start=1):
        c["id"] = idx
    return clusters


def draw_detections(frame, detections, clusters):
    """개별 바운딩 박스(얇은 초록) + 구역(두꺼운 주황) 오버레이."""
    vis = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 220, 80), 1)

    overlay = vis.copy()
    for c in clusters:
        x1, y1, x2, y2 = c["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 140, 255), -1)
    cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)

    for c in clusters:
        x1, y1, x2, y2 = c["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 140, 255), 2)
        label = f"Zone {c['id']} ({c['count']})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (0, 140, 255), -1)
        cv2.putText(vis, label, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(vis, f"Barnacle zones: {len(clusters)}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return vis


class BarnacleDetector:
    def __init__(self, model_path="best.pt", conf_threshold=0.25,
                 cluster_distance=60, min_cluster_size=1):
        from ultralytics import YOLO  # 시뮬레이션 모드에서는 불필요하므로 지연 import
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.min_confidence_limit = 0.35
        self.cluster_distance = cluster_distance
        self.min_cluster_size = min_cluster_size

    def infer_frame(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=640,
            verbose=False
        )

        detections = []
        low_confidence_flag = False
        total_dirty_area = 0

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                area = int((x2 - x1) * (y2 - y1))
                total_dirty_area += area
                detections.append({
                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "area": area
                })
                if conf < self.min_confidence_limit:
                    low_confidence_flag = True

        status = "SUCCESS"
        if low_confidence_flag or (detections and np.mean([d["confidence"] for d in detections]) < self.min_confidence_limit):
            status = "LOW_CONFIDENCE"

        return detections, total_dirty_area, status

    def detect_zones(self, frame):
        """프레임 → (개별 검출, 구역 목록, 상태)."""
        detections, _, status = self.infer_frame(frame)
        clusters = cluster_detections(detections, self.cluster_distance, self.min_cluster_size)
        return detections, clusters, status
