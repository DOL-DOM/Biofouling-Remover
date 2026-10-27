"""
하드웨어 없이 전체 동작을 테스트하는 가상 선체 + 전방 카메라 시뮬레이터.
- 위에서 본 가상 선체(따개비 군집, 장애물)를 만들고
- 로봇 앞쪽 카메라가 보는 영역을 잘라 640x480 '카메라 화면'으로 렌더링
- 로봇 몸체가 지나간 따개비는 확률적으로 제거 (청소 효율 < 100% → 반복 청소 확인)
- 장애물/선체 가장자리에 부딪히면 이동이 막히고 '[Sim] 충돌'을 기록 (회피 성능 확인용)
- 따개비 검출은 YOLO 대신 가상 정답 사용 (합성 이미지는 실제 모델로 검출 불가)
  장애물 인식은 실제와 같은 obstacle_detector 를 사용
"""
import math
import random

import cv2
import numpy as np

HULL = (120, 80, 50)          # 선체 바탕색 (BGR)
OBST = (55, 55, 65)           # 장애물/선체 밖 색


class SimulatedHull:
    def __init__(self, seed=3, clean_efficiency=0.85, world=(1400, 1000),
                 view_w=420, view_near=70, view_depth=330, step_px=35, body=110):
        self.rng = random.Random(seed)
        self.W, self.H = world
        self.view_w, self.near, self.depth = view_w, view_near, view_depth
        self.step_px, self.body = step_px, body
        self.clean_efficiency = clean_efficiency
        self.out_w, self.out_h = 640, 480

        self.x, self.y, self.th = 110.0, 900.0, -90.0   # 시작: 왼쪽 아래 모서리, 위쪽(북) 방향
        self.trail = [(self.x, self.y)]
        self.collisions = 0

        self.obstacles = [(380, 420, 470, 520), (900, 250, 1000, 330), (640, 760, 760, 830)]
        self.barnacles = []
        for cx, cy, n, sp in [(700, 250, 9, 35), (300, 230, 7, 30), (1080, 560, 10, 40),
                              (420, 760, 6, 28), (1150, 180, 5, 22)]:
            for _ in range(n):
                self.barnacles.append([self.rng.gauss(cx, sp), self.rng.gauss(cy, sp), self.rng.randint(7, 11)])
        self._world = None
        noise = np.random.default_rng(seed).integers(0, 14, (self.H, self.W, 1), dtype=np.uint8)
        self._bg = cv2.add(np.full((self.H, self.W, 3), HULL, np.uint8), np.repeat(noise, 3, axis=2))
        for x1, y1, x2, y2 in self.obstacles:
            cv2.rectangle(self._bg, (x1, y1), (x2, y2), OBST, -1)

    # ---------- 좌표 ----------
    def _axes(self):
        t = math.radians(self.th)
        return (math.cos(t), math.sin(t)), (-math.sin(t), math.cos(t))  # forward, right

    def _world_img(self):
        if self._world is None:
            img = self._bg.copy()
            for x, y, r in self.barnacles:
                cv2.circle(img, (int(x), int(y)), r, (200, 210, 220), -1)
                cv2.circle(img, (int(x), int(y)), max(2, r // 3), (60, 70, 80), -1)
            self._world = img
        return self._world

    # ---------- 카메라 ----------
    def read(self):
        (fx, fy), (rx, ry) = self._axes()
        u = (np.arange(self.out_w) / self.out_w - 0.5) * self.view_w
        f = self.near + (1 - np.arange(self.out_h) / self.out_h) * self.depth
        U, F = np.meshgrid(u, f)
        mx = (self.x + F * fx + U * rx).astype(np.float32)
        my = (self.y + F * fy + U * ry).astype(np.float32)
        frame = cv2.remap(self._world_img(), mx, my, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=OBST)
        return True, frame

    def detect(self, frame=None):
        (fx, fy), (rx, ry) = self._axes()
        s = self.out_w / self.view_w
        dets = []
        for bx, by, r in self.barnacles:
            dx, dy = bx - self.x, by - self.y
            f, lat = dx * fx + dy * fy, dx * rx + dy * ry
            v = (1 - (f - self.near) / self.depth) * self.out_h
            u = (lat / self.view_w + 0.5) * self.out_w
            if 0 <= u < self.out_w and 0 <= v < self.out_h:
                rr = int(r * s)
                dets.append({"center": (int(u), int(v)),
                             "bbox": (int(u) - rr, int(v) - rr, int(u) + rr, int(v) + rr),
                             "confidence": round(self.rng.uniform(0.55, 0.95), 2), "area": 4 * rr * rr})
        return dets, sum(d["area"] for d in dets), "SUCCESS"

    # ---------- 이동 ----------
    def _blocked(self, x, y):
        h = self.body / 2
        if x - h < 0 or y - h < 0 or x + h > self.W or y + h > self.H:
            return True
        return any(x + h > a and x - h < c and y + h > b and y - h < d for a, b, c, d in self.obstacles)

    def move(self, steps):
        (fx, fy), _ = self._axes()
        sgn = 1 if steps > 0 else -1
        for _ in range(abs(steps)):
            nx, ny = self.x + sgn * self.step_px * fx, self.y + sgn * self.step_px * fy
            if self._blocked(nx, ny):
                self.collisions += 1
                print(f"[Sim] 충돌! 장애물/가장자리에 막힘 (누적 {self.collisions}회)")
                return False
            self.x, self.y = nx, ny
            self.trail.append((self.x, self.y))
            self._clean_under_body()
        return True

    def turn(self, deg):
        self.th = (self.th + deg) % 360

    def _clean_under_body(self):
        h = self.body / 2
        keep = []
        for b in self.barnacles:
            under = abs(b[0] - self.x) <= h and abs(b[1] - self.y) <= h
            if under and self.rng.random() < self.clean_efficiency:
                continue
            keep.append(b)
        if len(keep) != len(self.barnacles):
            self.barnacles, self._world = keep, None

    # ---------- 위에서 본 지도 (웹 화면 오른쪽) ----------
    def render_map(self, height=480):
        img = self._world_img().copy()
        pts = np.array(self.trail, np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, (90, 200, 90), 3)
        (fx, fy), (rx, ry) = self._axes()
        corners = []
        for f, u in [(self.near, -.5), (self.near, .5), (self.near + self.depth, .5), (self.near + self.depth, -.5)]:
            corners.append((int(self.x + f * fx + u * self.view_w * rx), int(self.y + f * fy + u * self.view_w * ry)))
        cv2.polylines(img, [np.array(corners, np.int32)], True, (0, 220, 255), 3)
        h = self.body / 2
        body = [(self.x + a * h * fx + b * h * rx, self.y + a * h * fy + b * h * ry) for a, b in [(1, 1), (1, -1), (-1, -1), (-1, 1)]]
        cv2.fillPoly(img, [np.array(body, np.int32)], (255, 190, 0))
        cv2.line(img, (int(self.x), int(self.y)), (int(self.x + h * 1.4 * fx), int(self.y + h * 1.4 * fy)), (255, 255, 255), 4)
        cv2.putText(img, f"[SIM] top view  remaining barnacles: {len(self.barnacles)}  collisions: {self.collisions}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        return cv2.resize(img, (int(self.W * height / self.H), height))

    def release(self):
        pass
