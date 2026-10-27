"""
추측 항법(Dead Reckoning) + 왕복(지그재그) 커버리지 계획

엔코더/위치센서가 없으므로 로봇에게 내린 명령(전진 n스텝, 회전 n도)을 누적해 대략적인 위치를 추정한다.
좌표계: 시작 위치 = (0, 0), 시작 방향 = 0°,  x = 시작 방향 앞쪽(스텝), y = 오른쪽(스텝),  우회전 = +각도
※ 시간이 지날수록 오차가 누적되므로 정확한 지도가 아니라 '어느 줄을 청소 중인지' 정도의 참고용.
"""
import math


class DeadReckoning:
    def __init__(self):
        self.x = self.y = 0.0
        self.th = 0.0

    def on_motion(self, kind, value):
        if kind == "turn":
            self.th = (self.th + value) % 360
        elif kind == "move":
            t = math.radians(self.th)
            self.x += value * math.cos(t)
            self.y += value * math.sin(t)

    def heading_error(self, target_deg):
        """target 방향을 보려면 돌아야 할 각도 (-180~180, +는 우회전)."""
        return (target_deg - self.th + 180) % 360 - 180


class LanePlanner:
    """
    시작 위치(작업 구역의 왼쪽 아래 모서리)에서 앞쪽으로 lane_length 스텝 길이의 줄을
    오른쪽으로 lane_spacing 간격씩 lanes 개 왕복하며 훑는다.
    한 번(스윕) 다 훑으면 반대 방향으로 다시 훑는다.
    """

    def __init__(self, lane_length=20, lane_spacing=9, lanes=4):
        self.L, self.spacing, self.lanes = lane_length, lane_spacing, lanes
        self.sweep = 0
        self.order = list(range(lanes))
        self.idx = 0
        self.heading = 0.0
        self.lane_y = 0.0

    @property
    def lane(self):
        return self.order[self.idx]

    def start_lane(self, pose):
        self.lane_y = self.lane * self.spacing
        self.heading = 0.0 if pose.x < self.L / 2 else 180.0

    def lane_finished(self, pose):
        return pose.x >= self.L if self.heading == 0.0 else pose.x <= 0

    def next_lane(self, pose):
        """다음 줄로. 한 스윕이 끝났으면 True."""
        self.idx += 1
        done = False
        if self.idx >= len(self.order):
            self.sweep += 1
            self.order.reverse()
            self.idx = 1 if len(self.order) > 1 else 0   # 방금 훑은 마지막 줄은 건너뜀
            done = True
        self.start_lane(pose)
        return done

    def progress(self):
        return f"스윕 {self.sweep + 1} · 줄 {self.idx + 1}/{len(self.order)}"
