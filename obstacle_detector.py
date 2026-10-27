"""
전방 카메라 기반 장애물 인식 (추가 센서 / 재학습 없이 동작)

원리: 화면 대부분은 선체 표면이라는 가정하에
  1) 화면 전체의 '중앙값 색'을 선체 바탕색으로 추정
  2) 바탕색과 크게 다른 픽셀 중 따개비 박스 영역은 제외
  3) 일정 크기 이상 뭉친 덩어리 = 장애물 후보
  4) 그중 로봇 진행 경로 바로 앞(화면 하단 중앙 '위험 구역')에 걸친 것 = "장애물 발견"

한계: 카메라 1대로는 거리를 정확히 알 수 없고, 그림자·반사·탁도에 따라 오검출 가능.
      → 웹 화면의 빨간 박스를 보며 --obs-thresh 를 조절하세요. (클수록 둔감)
      → 가장 정확한 방법은 YOLO 모델에 'obstacle' 클래스를 추가 학습하는 것입니다.
"""
import cv2
import numpy as np


class CameraObstacleDetector:
    def __init__(self, diff_thresh=25, min_area_ratio=0.004, barnacle_pad=8,
                 danger_x=(0.22, 0.78), danger_y=0.55, danger_overlap=0.04):
        self.diff_thresh = diff_thresh
        self.min_area_ratio = min_area_ratio
        self.barnacle_pad = barnacle_pad
        self.danger_x = danger_x          # 위험 구역 가로 범위 (화면 비율) ≈ 로봇 폭
        self.danger_y = danger_y          # 위험 구역 시작 높이 (이 아래 = 가까움)
        self.danger_overlap = danger_overlap  # 위험 구역 면적 중 이 비율 이상 덮이면 '장애물 발견'

    def danger_rect(self, w, h):
        return int(self.danger_x[0] * w), int(self.danger_y * h), int(self.danger_x[1] * w), h

    def detect(self, frame, barnacle_detections=()):
        """반환: (전체 장애물 박스 목록, 위험 구역 내 장애물 여부, 장애물 중심 x(위험구역 내, 없으면 None))"""
        h, w = frame.shape[:2]
        lab = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        bg = np.median(lab.reshape(-1, 3), axis=0)
        mask = (np.linalg.norm(lab - bg, axis=2) > self.diff_thresh).astype(np.uint8) * 255

        p = self.barnacle_pad
        for d in barnacle_detections:
            x1, y1, x2, y2 = d["bbox"]
            mask[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p] = 0

        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        boxes, clean = [], np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= self.min_area_ratio * w * h:
                cv2.drawContours(clean, [c], -1, 255, -1)
                boxes.append(cv2.boundingRect(c))

        dx1, dy1, dx2, dy2 = self.danger_rect(w, h)
        zone = clean[dy1:dy2, dx1:dx2]
        blocked = zone.size and zone.mean() / 255 >= self.danger_overlap
        obs_cx = None
        if blocked:
            xs = np.nonzero(zone)[1]
            obs_cx = int(dx1 + xs.mean())
        return boxes, bool(blocked), obs_cx

    def draw(self, vis, boxes, blocked):
        h, w = vis.shape[:2]
        dx1, dy1, dx2, dy2 = self.danger_rect(w, h)
        cv2.rectangle(vis, (dx1, dy1), (dx2, dy2 - 1), (40, 40, 230) if blocked else (160, 160, 160), 1)
        for x, y, bw, bh in boxes:
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (40, 40, 230), 2)
            cv2.putText(vis, "OBSTACLE", (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 230), 2)
        return vis
