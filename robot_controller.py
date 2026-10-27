import threading
import time
from collections import deque


class SubseaCleaningRobot:
    """
    좌/우 궤도(차동 구동) 로봇.
    hardware : hardware.Hardware (실기)  / None
    motion_sim : 시뮬레이션 객체 (forward/turn 시 가상 위치 갱신) / None
    """

    def __init__(self, hardware=None, motion_sim=None):
        self.hw = hardware
        self.sim = motion_sim
        self.motion_listeners = []   # f(kind, value): 추측 항법 등
        self.state = "BOOTING"
        self.action = "-"
        self.led_level = 50
        self.camera_fps = 30
        self.brush_rpm = 0
        self.magnetic_hold = False
        self.network_connected = False
        self.telemetry_buffer = deque(maxlen=1000)
        self._stop_lock = threading.Lock()
        self.stopped = False

    # ---------------- 부팅 / 준비 ----------------
    def boot_and_connect(self, retries=3):
        print("[Robot] Power ON: 라즈베리파이 부팅 완료")
        backoff = 1
        for i in range(retries):
            print(f"[Robot] 무선 링크 연결 시도 중... (시도 {i+1})")
            if i == 0:
                print("[Robot] Warning: 신호 약함 / 연결 실패")
                time.sleep(backoff)
                backoff *= 2
            else:
                self.network_connected = True
                print("[Robot] Link OK: 관제 시스템 통신 연결 성공 (IP 할당)")
                return True
        return False

    def init_sensors(self, turbidity_high=True):
        print("[Robot] MissionStart: 센서 및 조명 구동 시작")
        if turbidity_high:
            self.boost_led()
        self.state = "SENSORS_READY"
        return True

    def boost_led(self):
        print("[Robot] 고탁도/어두움 감지 -> LED 밝기 증강 및 노출/FPS 자동 보정")
        self.led_level = 90
        self.camera_fps = 15

    def approach_and_attach(self, lidar_distance, slip_detected=False):
        print(f"[Robot] 선체 거리 측정값: {lidar_distance}cm")
        if slip_detected:
            print("[Robot] Warning: 슬립/이탈 감지! 속도 감속 및 자력 재확보 시도")
        self.magnetic_hold = True
        print("[Robot] ApproachReady: 선체 밀착 완료, 청소 준비 상태 도달")
        self.state = "READY"

    # ---------------- 주행 (차동 구동) ----------------
    def _go(self, l_dir, r_dir, sec, speed):
        if self.hw:
            self.hw.run(l_dir, r_dir, sec, speed, lambda: self.stopped)

    def forward(self, steps=1):
        if self.stopped:
            return False
        self.action = "FORWARD"
        if self.hw:
            self._go(1, 1, steps * self.hw.cfg["step_sec"], self.hw.cfg["forward_speed"])
        ok = self.sim.move(steps) if self.sim else True
        self._notify("move", steps)
        return ok and not self.stopped

    def _notify(self, kind, value):
        for f in self.motion_listeners:
            f(kind, value)

    def backward(self, steps=1):
        if self.stopped:
            return False
        self.action = "BACKWARD"
        if self.hw:
            self._go(-1, -1, steps * self.hw.cfg["step_sec"], self.hw.cfg["forward_speed"])
        ok = self.sim.move(-steps) if self.sim else True
        self._notify("move", -steps)
        return ok and not self.stopped

    def turn(self, deg):
        """deg > 0 : 우회전,  deg < 0 : 좌회전 (제자리 회전 - 양쪽 궤도 반대 방향)"""
        if self.stopped or abs(deg) < 1:
            return not self.stopped
        self.action = f"TURN {'R' if deg > 0 else 'L'} {abs(deg):.0f}°"
        if self.hw:
            cfg = self.hw.cfg
            s = 1 if deg > 0 else -1
            self._go(s, -s, abs(deg) / 90.0 * cfg["turn_90_sec"], cfg["turn_speed"])
        if self.sim:
            self.sim.turn(deg)
        self._notify("turn", deg)
        return not self.stopped

    def set_brush(self, rpm=800):
        if self.stopped:
            return
        self.brush_rpm = rpm
        if self.hw and self.hw.brush:
            self.hw.set_brush_duty(rpm / self.hw.cfg["brush_max_rpm"])
            print(f"[Actuator] 브러시: {rpm} RPM")

    # ---------------- 정지 ----------------
    def emergency_stop(self, reason="사용자 요청"):
        """즉시 하드웨어 안전 정지. 어느 스레드에서 호출해도 안전(1회만 실행)."""
        with self._stop_lock:
            if self.stopped:
                return
            self.stopped = True
            if self.hw:
                self.hw.stop_all()
            self.brush_rpm = 0
            self.state, self.action = "E-STOP", "STOP"
            print(f"[Robot] !!! 하드웨어 안전 정지 실행 ({reason}) - 궤도/브러시 모터 OFF, 자석 흡착 유지")

    def safe_shutdown(self):
        with self._stop_lock:
            if self.stopped:
                return
            self.stopped = True
            if self.hw:
                self.hw.stop_all()
            self.brush_rpm = 0
            self.state, self.action = "STOPPED", "STOP"
            print("[Robot] 궤도/브러시 모터 정지 완료 (안전 정지 상태)")

    # ---------------- 텔레메트리 ----------------
    def log_telemetry(self, zones, passes):
        payload = {"time": time.time(), "state": self.state, "action": self.action,
                   "zones": zones, "passes": passes, "led": self.led_level}
        if self.network_connected:
            return f"[Telemetry UpLink] {payload}"
        self.telemetry_buffer.append(payload)
        return f"[Buffer Saved] 로컬 큐 저장 (대기 {len(self.telemetry_buffer)}건)"
