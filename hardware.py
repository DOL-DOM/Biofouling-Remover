"""
라즈베리파이 GPIO 하드웨어 설정 / 제어 (gpiozero - Raspberry Pi OS 기본 포함)

[구성]  좌/우 궤도(체인) 모터 2개 + L298N 1개 + 전방 카메라
  ┌ 왼쪽 궤도  = L298N A채널 (연결도 그대로)
  │   IN1 = GPIO17 (물리 11번)   IN2 = GPIO27 (물리 13번)   ENA = GPIO18 (물리 12번)
  └ 오른쪽 궤도 = L298N B채널 (추가 연결)
      IN3 = GPIO22 (물리 15번)   IN4 = GPIO23 (물리 16번)   ENB = GPIO13 (물리 33번)
  · ENA / ENB 점퍼는 둘 다 제거 (PWM 속도 제어)
  · 라즈베리파이 GND(물리 6번) ↔ L298N GND ↔ 배터리(-) 공통 연결
  · OUT1/OUT2 → 왼쪽 모터,  OUT3/OUT4 → 오른쪽 모터

※ 핀 번호는 'GPIO 번호(BCM)' 입니다. (물리 핀 번호 아님)
※ 전진 시 어느 한쪽이 거꾸로 돌면 그 모터의 in1 / in2 숫자를 서로 바꾸세요.
※ 엔코더가 없으므로 이동/회전량은 '시간'으로 제어 → motor_test.py 로 실측 후 아래 값 보정
"""

HW_CONFIG = {
    "drive_left":  {"in1": 17, "in2": 27, "en": 18},
    "drive_right": {"in1": 22, "in2": 23, "en": 13},

    "forward_speed": 0.5,   # 전진/후진 PWM 듀티 (0~1)
    "turn_speed": 0.6,      # 제자리 회전 PWM 듀티
    "step_sec": 0.4,        # 전진 1스텝 시간(초)
    "turn_90_sec": 0.8,     # 제자리 90도 회전 시간(초)  ← 실측 보정 필수

    # 브러시 모터 (현재 없음). 생기면 두 번째 L298N에 연결 후 예) {"in1": 5, "in2": 6, "en": 12}
    "brush": None,
    "brush_max_rpm": 1000,
}


class Hardware:
    """GPIO가 없는 환경(PC/시뮬레이션)에서는 자동으로 비활성(모터 명령은 무시)."""

    def __init__(self, cfg=HW_CONFIG):
        self.cfg = cfg
        self.left = self.right = self.brush = None
        try:
            from gpiozero import Motor
        except Exception as e:
            print(f"[HW] gpiozero 사용 불가 ({e}) → 모터 없이 동작")
            return

        def motor(c):
            return Motor(forward=c["in1"], backward=c["in2"], enable=c["en"], pwm=True) if c else None

        try:
            self.left = motor(cfg["drive_left"])
            self.right = motor(cfg["drive_right"])
            self.brush = motor(cfg["brush"])
            print(f"[HW] GPIO 초기화 완료 - 왼쪽 궤도:{'O' if self.left else 'X'} "
                  f"오른쪽 궤도:{'O' if self.right else 'X'} 브러시:{'O' if self.brush else 'X'}")
        except Exception as e:
            print(f"[HW] GPIO 초기화 실패 ({e}) → 모터 없이 동작")
            self.close()

    @property
    def can_drive(self):
        return bool(self.left and self.right)

    def run(self, l_dir, r_dir, sec, speed, stop_flag):
        """l_dir/r_dir: +1 전진, -1 후진. sec 동안 구동 후 정지. stop_flag()가 True면 즉시 정지."""
        import time
        if not self.can_drive:
            return
        for m, d in ((self.left, l_dir), (self.right, r_dir)):
            m.forward(speed) if d > 0 else m.backward(speed)
        end = time.time() + sec
        while time.time() < end and not stop_flag():
            time.sleep(0.01)
        self.left.stop()
        self.right.stop()

    def set_brush_duty(self, duty):
        if self.brush:
            duty = max(0.0, min(1.0, duty))
            self.brush.forward(duty) if duty > 0 else self.brush.stop()

    def stop_all(self):
        for m in (self.left, self.right, self.brush):
            try:
                if m:
                    m.stop()
            except Exception:
                pass

    def close(self):
        self.stop_all()
        for m in (self.left, self.right, self.brush):
            try:
                if m:
                    m.close()
            except Exception:
                pass
