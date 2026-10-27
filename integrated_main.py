"""
따개비 청소 로봇 통합 실행 파일 (전방 카메라 + 좌/우 궤도 로봇)

동작 방식 (화면을 보고 바로 반응하는 방식)
  1) 장애물  : 화면 하단 중앙(진행 경로)에 장애물 → "장애물 발견" → 정지·후진·방향 전환
  2) 따개비  : 구역이 보이면 가장 가까운 구역 쪽으로 조향 → 접근 → 구역 위를 지나가며 청소
  3) 탐색    : 아무것도 안 보이면 제자리 회전(45°씩)으로 주변 탐색, 한 바퀴마다 조금 전진
  4) 종료    : 연속으로 따개비 구역 0개가 일정 횟수/시간 확인되면 "청소를 끝냈습니다"

실행 예)
  python integrated_main.py --source picam      # 라즈베리파이 카메라 모듈(리본 케이블)
  python integrated_main.py --source 0          # USB 카메라
  python integrated_main.py --sim               # 하드웨어/모델 없이 가상 선체로 테스트

브라우저: http://<라즈베리파이IP>:8000  (실시간 감지 화면 + 종료 버튼)
강제 종료: 터미널에서 q 키 / Ctrl + C / 웹 '강제 종료' 버튼
"""
import argparse
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

from explorer import DeadReckoning, LanePlanner
from obstacle_detector import CameraObstacleDetector
from robot_controller import SubseaCleaningRobot
from vision_detector import cluster_detections, draw_detections
from web_stream import WebStreamServer

FRAME_W, FRAME_H = 640, 480

# =====================================================================
# 종료 제어 (q 키 / Ctrl+C / 웹 버튼 → 즉시 하드웨어 정지)
# =====================================================================
class StopController:
    def __init__(self):
        self.event = threading.Event()
        self.reason = None
        self._callbacks = []
        self._lock = threading.Lock()

    def on_stop(self, cb):
        self._callbacks.append(cb)

    def request(self, reason):
        with self._lock:
            if self.event.is_set():
                return
            self.reason = reason
            self.event.set()
        print(f"\n[STOP] 강제 종료 요청 감지: {reason}")
        for cb in self._callbacks:  # 메인 루프를 기다리지 않고 즉시 하드웨어 정지
            try:
                cb(reason)
            except Exception as e:
                print(f"[STOP] 정지 콜백 오류: {e}")

    @property
    def requested(self):
        return self.event.is_set()

    def sleep(self, sec):
        """중단 가능한 대기. 종료 요청 시 즉시 True 반환."""
        return self.event.wait(sec)


class KeyboardListener:
    """터미널에서 q 키를 누르면 (Enter 없이) 즉시 종료 요청."""

    def __init__(self, stop: StopController):
        self.stop = stop
        self._old_term = None

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            if os.name == "nt":
                import msvcrt
                while not self.stop.requested:
                    if msvcrt.kbhit() and msvcrt.getwch().lower() == "q":
                        self.stop.request("키보드 q 입력")
                    time.sleep(0.05)
            elif sys.stdin.isatty():
                import select, termios, tty
                fd = sys.stdin.fileno()
                self._old_term = termios.tcgetattr(fd)
                tty.setcbreak(fd)  # Ctrl+C 신호는 그대로 동작
                while not self.stop.requested:
                    r, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if r and sys.stdin.read(1).lower() == "q":
                        self.stop.request("키보드 q 입력")
            else:  # 터미널이 아닌 환경: 'q' + Enter
                for line in sys.stdin:
                    if line.strip().lower() == "q":
                        self.stop.request("키보드 q 입력")
                        break
        except Exception as e:
            print(f"[Key] 키보드 입력 감시 불가 ({e}) - Ctrl+C 또는 웹 버튼을 사용하세요.")

    def restore(self):
        if self._old_term is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term)
            self._old_term = None


class PiCameraSource:
    """라즈베리파이 카메라 모듈(CSI 리본 케이블)용. cv2.VideoCapture와 같은 read()/release() 제공."""

    def __init__(self, width=640, height=480):
        from picamera2 import Picamera2  # sudo apt install python3-picamera2
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}))  # RGB888 = OpenCV BGR 배열
        self.cam.start()
        time.sleep(1.0)  # 노출 안정화

    def read(self):
        try:
            return True, self.cam.capture_array()
        except Exception:
            return False, None

    def release(self):
        self.cam.stop()
        self.cam.close()




# =====================================================================
# 화면 합성
# =====================================================================
def annotate(frame, dets, clusters, target, obs, boxes, blocked, robot, info):
    vis = draw_detections(frame, dets, clusters)
    if obs:
        obs.draw(vis, boxes, blocked)
    h, w = vis.shape[:2]
    if target is not None:
        cx, cy = target["center"]
        cv2.line(vis, (w // 2, h), (cx, cy), (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 7, (0, 255, 255), -1)
    cv2.line(vis, (w // 2, h - 25), (w // 2, h), (255, 255, 255), 2)
    if blocked:
        cv2.putText(vis, "OBSTACLE AHEAD!", (w // 2 - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 230), 3)
    return vis


def render_dashboard(cam_vis, robot, info, sim_map=None):
    panels = [cam_vis] if sim_map is None else [cam_vis, sim_map]
    canvas = np.hstack(panels)
    bar = np.full((36, canvas.shape[1], 3), 20, dtype=np.uint8)
    txt = (f"State: {robot.state} | Action: {robot.action} | Zones: {info['zones']} | "
           f"Passes: {info['passes']} | Avoided: {info['avoided']} | Empty: {info['empty']}")
    # (한글은 OpenCV로 표시 불가 → 진행 상황은 웹 패널에 표시)
    cv2.putText(bar, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    return np.vstack([canvas, bar])


# =====================================================================
# 메인
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="따개비 청소 로봇 (전방 카메라 + 궤도)")
    p.add_argument("--source", default="0",
                   help="USB 카메라 번호(기본 0) / picam(라즈베리파이 카메라 모듈) / 영상·RTSP 경로")
    p.add_argument("--sim", action="store_true", help="가상 선체 시뮬레이션 모드 (카메라/모델/모터 불필요)")
    p.add_argument("--model", default="best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--cluster-dist", type=int, default=60, help="같은 구역으로 묶을 bbox 간 최대 간격(px)")
    p.add_argument("--min-cluster", type=int, default=1, help="구역으로 인정할 최소 따개비 수")
    # 주행
    p.add_argument("--steer-gain", type=float, default=35, help="조향: 화면 중심에서 끝까지 벗어났을 때 회전 각도(°)")
    p.add_argument("--center-tol", type=float, default=0.15, help="이 비율 이내로 중앙이면 직진")
    p.add_argument("--arrive-y", type=float, default=0.78, help="구역 중심이 화면 높이의 이 비율 아래로 오면 '도착'")
    p.add_argument("--pass-steps", type=int, default=5, help="구역 위 청소 통과 전진 스텝 수")
    p.add_argument("--track-limit", type=int, default=40, help="한 구역 추적 최대 판단 횟수 (넘으면 건너뜀)")
    p.add_argument("--avoid-turn", type=float, default=60, help="따개비 추적 중 장애물 회피 회전 각도(°)")
    # 작업 구역 (왕복 주행) - 로봇을 작업 구역의 '왼쪽 아래 모서리'에 앞을 보게 놓고 시작
    p.add_argument("--lane-length", type=float, default=20, help="한 줄 길이 (전진 스텝 수)")
    p.add_argument("--lane-spacing", type=float, default=9, help="줄 간격 (전진 스텝 수, 카메라 시야 폭보다 약간 좁게)")
    p.add_argument("--lanes", type=int, default=4, help="줄 개수 (오른쪽으로)")
    # 종료
    p.add_argument("--empty-count", type=int, default=10, help="종료 판정: 최소 연속 미검출 횟수")
    p.add_argument("--empty-sec", type=float, default=3.0, help="종료 판정: 최소 연속 미검출 시간(초)")
    p.add_argument("--max-minutes", type=float, default=0, help="안전용 최대 작동 시간(분, 0=무제한)")
    # 장애물
    p.add_argument("--no-cam-obstacle", action="store_true", help="카메라 기반 장애물 인식 끄기")
    p.add_argument("--obs-thresh", type=float, default=25, help="장애물: 바탕색과의 색 차이 기준 (클수록 둔감)")
    # 기타
    p.add_argument("--step-delay", type=float, default=0.05, help="판단 루프 사이 대기(초)")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main():
    args = parse_args()
    stop = StopController()

    # ---------- 영상 소스 / 검출기 / 로봇 ----------
    sim = None
    if args.sim:
        from simulation import SimulatedHull
        sim = source = SimulatedHull()
        robot = SubseaCleaningRobot(motion_sim=sim)
        infer = sim.detect
        print("[Mode] 시뮬레이션 모드 (가상 선체 / 전방 카메라 시점)")
    else:
        from vision_detector import BarnacleDetector
        detector = BarnacleDetector(model_path=args.model, conf_threshold=args.conf,
                                    cluster_distance=args.cluster_dist, min_cluster_size=args.min_cluster)
        if args.source == "picam":
            source = PiCameraSource(FRAME_W, FRAME_H)
        else:
            src = int(args.source) if args.source.isdigit() else args.source
            source = cv2.VideoCapture(src)
            if not source.isOpened():
                print(f"[Error] 영상 소스를 열 수 없습니다: {args.source}")
                print("        라즈베리파이 카메라 모듈(리본 케이블)이라면 --source picam 으로 실행하세요.")
                return 1
        from hardware import Hardware
        hw = Hardware()
        if not hw.can_drive:
            print("[HW] 좌/우 궤도 모터가 모두 준비되지 않아 로봇은 움직이지 않습니다 (화면 인식만 동작).")
        robot = SubseaCleaningRobot(hardware=hw)
        infer = detector.infer_frame

    obs = None if args.no_cam_obstacle else CameraObstacleDetector(diff_thresh=args.obs_thresh)
    pose = DeadReckoning()
    robot.motion_listeners.append(pose.on_motion)
    lanes = LanePlanner(args.lane_length, args.lane_spacing, args.lanes)

    # ---------- 종료 경로 3가지 연결 ----------
    stop.on_stop(lambda reason: robot.emergency_stop(reason))
    signal.signal(signal.SIGINT, lambda *_: stop.request("Ctrl + C"))
    keyboard = KeyboardListener(stop)
    web = WebStreamServer(port=args.port, on_stop=stop.request)
    stop.on_stop(lambda reason: web.add_event(f"강제 종료: {reason}", "stop"))

    info = {"zones": 0, "passes": 0, "avoided": 0, "empty": "0"}
    last_vis = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    def log(msg, kind=""):
        print(msg)
        web.add_event(msg.strip(" =\n"), kind)

    def publish(vis=None):
        nonlocal last_vis
        if vis is not None:
            last_vis = vis
        web.push_frame(render_dashboard(last_vis, robot, info, sim.render_map() if sim else None))
        web.update_status(zones=info["zones"], state=robot.state, action=robot.action,
                          passes=info["passes"], obstacles=info["avoided"], empty_checks=info["empty"],
                          plan=info.get("plan", "-"))

    def look():
        """카메라 1프레임 → 따개비 구역 + 장애물 판단 + 화면 표시."""
        ok, frame = source.read()
        if not ok or frame is None:
            return None
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        dets, _, status = infer(frame)
        if status == "LOW_CONFIDENCE" and robot.led_level < 90:
            log("[Vision] 신뢰도 낮음 감지 -> LED 증강 및 보정")
            robot.boost_led()
        clusters = cluster_detections(dets, args.cluster_dist, args.min_cluster)
        boxes, blocked, obs_cx = obs.detect(frame, dets) if obs else ([], False, None)
        # 가장 가까운 구역 = 화면에서 가장 아래쪽에 있는 구역
        target = max(clusters, key=lambda c: c["bbox"][3]) if clusters else None
        info["zones"] = len(clusters)
        vis = annotate(frame, dets, clusters, target, obs, boxes, blocked, robot, info)
        publish(vis)
        return {"clusters": clusters, "target": target, "blocked": blocked, "obs_cx": obs_cx}

    def avoid(obs_cx):
        obstacle_alert("정지 후 회피합니다")
        robot.backward(1)
        # 장애물이 오른쪽에 있으면 왼쪽으로, 왼쪽에 있으면 오른쪽으로 회전
        robot.turn(-args.avoid_turn if (obs_cx or FRAME_W // 2) >= FRAME_W // 2 else args.avoid_turn)
        publish()

    def obstacle_alert(what):
        info["avoided"] += 1
        print("\n" + "=" * 54)
        log(f"⚠️  장애물 발견! {what}", "obs")
        print("=" * 54)

    def clean_pass(count):
        nonlocal chase
        chase = 0
        robot.state = "CLEANING"
        log(f"[Clean] 구역 진입 → 위를 지나가며 청소 (따개비 {count}개)")
        robot.set_brush(850)
        for _ in range(args.pass_steps):
            if not robot.forward(1):
                break
            w = look()
            if w is None or w["blocked"] or stop.requested:
                break
        robot.set_brush(0)
        info["passes"] += 1
        log(f"[Clean] 청소 통과 완료 (누적 {info['passes']}회) → 잔여 따개비 재확인")

    finished_normally = False
    try:
        web.start()
        keyboard.start()
        print("[Info] 강제 종료: q 키 / Ctrl + C / 웹 화면의 '강제 종료' 버튼\n")

        robot.boot_and_connect()
        robot.init_sensors(turbidity_high=False)
        robot.approach_and_attach(lidar_distance=12, slip_detected=False)
        robot.state = "RUNNING"

        empty_count, empty_since, prev_n, last_report = 0, None, None, 0.0
        zones_this_sweep, track_avoids, ignore_until, tick, track = 0, 0, -1, 0, None
        chase, skipped = 0, 0
        started = time.time()
        lanes.start_lane(pose)
        log(f"[Plan] 왕복 청소 시작: 줄 길이 {args.lane_length} × {args.lanes}줄 (간격 {args.lane_spacing})")

        while not stop.requested:
            tick += 1
            if args.max_minutes and time.time() - started > args.max_minutes * 60:
                log(f"[Safety] 최대 작동 시간({args.max_minutes}분) 도달 - 작동을 중지합니다.", "stop")
                break

            v = look()
            if v is None:
                log("[Vision] 프레임 수신 실패 - 재시도합니다.")
                if stop.sleep(0.5):
                    break
                continue

            n = len(v["clusters"])
            if n > 0 and (n != prev_n or time.time() - last_report > 5):
                log(f"\n따개비 인식완료, {n}개 구역 존재")
                prev_n, last_report = n, time.time()

            # ---- 1) 따개비 구역이 보임 → 추적 / 청소 ----
            if n > 0 and tick > ignore_until:
                empty_count, empty_since = 0, None
                info["empty"] = "0"
                zones_this_sweep += 1
                cx, cy = v["target"]["center"]
                err = (cx - FRAME_W / 2) / (FRAME_W / 2)       # -1(왼쪽) ~ +1(오른쪽)
                track = track or {}
                track.update(cy=cy / FRAME_H, err=err, lost=0, count=v["target"]["count"])
                chase += 1

                if chase > args.track_limit:                   # 청소 통과 없이 너무 오래 추적 → 포기
                    log("[Clean] 접근이 어려운 구역 → 건너뛰고 왕복 주행 계속", "obs")
                    ignore_until, track, chase = tick + 20, None, 0
                    skipped += 1
                elif v["blocked"]:                             # 가는 길에 장애물 → 회피
                    avoid(v["obs_cx"])
                    track_avoids += 1
                    if track_avoids >= 3:
                        log("[Clean] 장애물 때문에 접근할 수 없는 구역 → 건너뛰고 왕복 주행 계속", "obs")
                        ignore_until, track_avoids, track, chase = tick + 20, 0, None, 0
                        skipped += 1
                else:
                    robot.state = "TRACKING"
                    if cy >= args.arrive_y * FRAME_H and abs(err) < 0.3:
                        clean_pass(v["target"]["count"])
                        track = None
                    elif abs(err) > args.center_tol:
                        robot.turn(err * args.steer_gain)
                    else:
                        robot.forward(1)

            # ---- 1-1) 추적하던 구역이 화면에서 사라짐 ----
            elif track is not None:
                track["lost"] += 1
                chase += 1
                if track["cy"] > 0.55 and abs(track["err"]) < 0.4:
                    # 가까이 다가가서 화면 아래로 사라진 것 → 바로 위를 지나가며 청소
                    clean_pass(track["count"])
                    track = None
                elif track["lost"] > 2:
                    track = None                              # 놓침 → 왕복 주행으로 복귀

            # ---- 2) 따개비 없음 → 왕복(지그재그) 주행으로 구역 훑기 ----
            else:
                robot.state = "SWEEPING"
                if n == 0:
                    empty_count += 1
                    empty_since = empty_since or time.time()
                    info["empty"] = str(empty_count)

                dy = lanes.lane_y - pose.y
                if abs(dy) > 1.5:                     # 줄에서 벗어남(청소 후 / 줄 이동) → 옆으로 이동
                    want = 90.0 if dy > 0 else 270.0
                    turn = pose.heading_error(want)
                    if abs(turn) > 10:
                        robot.turn(turn)
                    elif v["blocked"]:
                        obstacle_alert("옆 줄로 가는 길이 막혀 현재 위치에서 줄을 시작합니다")
                        lanes.lane_y = pose.y
                    else:
                        robot.forward(1)
                else:
                    turn = pose.heading_error(lanes.heading)
                    if abs(turn) > 10:
                        robot.turn(turn)
                    elif lanes.lane_finished(pose) or v["blocked"]:
                        if v["blocked"]:
                            obstacle_alert("이 줄은 여기까지 청소하고 다음 줄로 이동합니다")
                        sweep_done = lanes.next_lane(pose)
                        if sweep_done:
                            elapsed = time.time() - (empty_since or time.time())
                            log(f"[Plan] 구역 전체 1회 훑기 완료 - 이번 스윕에서 따개비 구역 확인 {zones_this_sweep}회")
                            if (zones_this_sweep == 0 and empty_count >= args.empty_count
                                    and elapsed >= args.empty_sec):
                                print("\n" + "=" * 54)
                                log("청소를 끝냈습니다. 작동을 중지합니다.", "end")
                                if skipped:
                                    log(f"[참고] 접근이 어려워 건너뛴 구역 {skipped}회 - 수동 점검 권장", "obs")
                                print("=" * 54)
                                finished_normally = True
                                break
                            log("[Plan] 따개비가 남아 있었으므로 반대 방향으로 다시 훑습니다.")
                            zones_this_sweep = 0
                        else:
                            log(f"[Plan] 다음 줄로 이동 ({lanes.progress()})")
                    else:
                        robot.forward(1)

            robot.log_telemetry(n, info["passes"])
            info["plan"] = lanes.progress()
            publish()
            if stop.sleep(args.step_delay):
                break

    finally:
        # ---------- 안전 종료 ----------
        if stop.requested:
            robot.emergency_stop(stop.reason)
            final, kind = f"강제 종료되었습니다 ({stop.reason}). 하드웨어 안전 정지 완료.", "stop"
        else:
            robot.safe_shutdown()
            final = "청소를 끝냈습니다. 작동을 중지합니다." if finished_normally else "작동을 중지합니다."
            kind = "end" if finished_normally else "stop"
        try:
            publish()
            web.update_status(finished=True, final_message=final, final_kind=kind)
            time.sleep(1.2)  # 브라우저가 마지막 상태를 받아갈 시간
        except Exception:
            pass
        keyboard.restore()
        if getattr(robot, "hw", None):
            robot.hw.close()
        source.release()
        web.stop()
        if sim:
            print(f"[Sim] 남은 따개비 {len(sim.barnacles)}개 / 충돌 {sim.collisions}회")
        print(f"[Exit] {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
