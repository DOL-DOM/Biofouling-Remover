"""
좌/우 궤도 모터 배선 확인 + 이동량 보정용 테스트
※ 처음에는 로봇을 받침대에 올려 궤도가 바닥에서 뜬 상태로 실행하세요.

실행:  python motor_test.py          # 기본 동작 테스트
       python motor_test.py calib    # 바닥에서 실행: 1스텝 전진 / 90도 회전 실측 (hardware.py 보정용)
"""
import sys
import time
from gpiozero import Motor
from hardware import HW_CONFIG as C

L = Motor(forward=C["drive_left"]["in1"], backward=C["drive_left"]["in2"], enable=C["drive_left"]["en"], pwm=True)
R = Motor(forward=C["drive_right"]["in1"], backward=C["drive_right"]["in2"], enable=C["drive_right"]["en"], pwm=True)


def run(label, l, r, sec, speed):
    print(f"[Test] {label} ({sec:.2f}s)")
    (L.forward if l > 0 else L.backward)(speed) if l else L.stop()
    (R.forward if r > 0 else R.backward)(speed) if r else R.stop()
    time.sleep(sec)
    L.stop(); R.stop()
    time.sleep(0.8)


try:
    if len(sys.argv) > 1 and sys.argv[1] == "calib":
        run("전진 1스텝 → 이동 거리를 재세요", 1, 1, C["step_sec"], C["forward_speed"])
        run("제자리 우회전 90도 목표 → 실제 각도를 확인하세요", 1, -1, C["turn_90_sec"], C["turn_speed"])
        print("[Calib] 90도보다 덜 돌면 turn_90_sec 를 늘리고, 더 돌면 줄이세요. (hardware.py)")
    else:
        run("왼쪽 궤도만 전진", 1, 0, 1.5, 0.5)
        run("오른쪽 궤도만 전진", 0, 1, 1.5, 0.5)
        run("양쪽 전진", 1, 1, 1.5, 0.5)
        run("양쪽 후진", -1, -1, 1.5, 0.5)
        run("제자리 우회전", 1, -1, 1.5, 0.6)
        run("제자리 좌회전", -1, 1, 1.5, 0.6)
        print("[Test] 어느 쪽이 거꾸로 돌면 hardware.py 에서 그 모터의 in1 / in2 숫자를 서로 바꾸세요.")
finally:
    L.stop(); R.stop(); L.close(); R.close()
    print("[Test] 정지 완료")
