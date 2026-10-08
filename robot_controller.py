import time
from collections import deque

class SubseaCleaningRobot:
    def __init__(self):
        self.state = "BOOTING"
        self.led_level = 50
        self.camera_fps = 30
        self.camera_res = "1080p"
        self.brush_rpm = 0
        self.suction_power = 0
        self.magnetic_hold = False
        
        self.network_connected = False
        self.telemetry_buffer = deque(maxlen=1000)
        self.position = (0, 0)

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
        self.brush_rpm = 0
        self.suction_power = 0
        
        if turbidity_high:
            print("[Robot] 고탁도/어두움 감지 -> LED 밝기 증강 및 노출/FPS 자동 보정")
            self.led_level = 90
            self.camera_fps = 15
        print("[Robot] LiDAR 거리 측정 초기화 완료 (Ready)")
        self.state = "SENSORS_READY"
        return True

    def approach_and_attach(self, lidar_distance, slip_detected=False):
        print(f"[Robot] 선체 거리 측정값: {lidar_distance}cm")
        if lidar_distance > 10:
            print("[Robot] 안전거리 만족 -> 저속 전진")
        
        if slip_detected:
            print("[Robot] Warning: 슬립/이탈 감지! 속도 감속 및 자력 재확보 시도")
            self.magnetic_hold = True
            print("[Robot] SlipRecovered: 자석 트랙 표면 재밀착 성공")
        else:
            self.magnetic_hold = True
            
        print("[Robot] ApproachReady: 선체 밀착 완료, 청소 준비 상태 도달")
        self.state = "ATTACHED"

    def set_cleaning_actuators(self, rpm=800, suction=100):
        self.brush_rpm = rpm
        self.suction_power = suction
        print(f"[Actuator] 브러시 가동: {self.brush_rpm} RPM, 흡입구: {self.suction_power}%")

    def handle_overload(self):
        print("[Actuator] 부하/전류 상승 감지 -> 출력 강하(감속)")
        self.brush_rpm = int(self.brush_rpm * 0.6)

    def handle_suction_clog(self):
        print("[Actuator] 흡입구 막힘 감지! -> 역세(Reverse Flush) 및 펄스 진동 루틴 가동")
        time.sleep(0.3)
        return False

    def log_telemetry(self, current_pos, progress):
        self.position = current_pos
        payload = {
            "time": time.time(),
            "pos": self.position,
            "progress": f"{progress:.1f}%",
            "rpm": self.brush_rpm,
            "suction": self.suction_power,
            "led": self.led_level
        }
        if self.network_connected:
            return f"[Telemetry UpLink] {payload}"
        else:
            self.telemetry_buffer.append(payload)
            return f"[Buffer Saved] 네트워크 두절. 로컬 큐에 저장 (대기 건수: {len(self.telemetry_buffer)})"

    def flush_buffer_to_server(self):
        print(f"[Sync] 무선 링크 복구! 로컬 버퍼 {len(self.telemetry_buffer)}건 일괄 전송 시작")
        while self.telemetry_buffer:
            self.telemetry_buffer.popleft()
        print("[Sync] 텔레메트리 동기화 완료")