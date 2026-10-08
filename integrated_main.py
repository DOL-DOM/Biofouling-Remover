import cv2
import numpy as np
import time
from vision_detector import BarnacleDetector
from coordinate_mapper import CoordinateMapper
from optimized_path_planner import OptimizedPathPlanner
from robot_controller import SubseaCleaningRobot

def main():
    GRID_SIZE = 20
    FRAME_W, FRAME_H = 640, 480
    
    detector = BarnacleDetector(model_path="best.pt", conf_threshold=0.25)
    mapper = CoordinateMapper(frame_width=FRAME_W, frame_height=FRAME_H, grid_size=GRID_SIZE)
    planner = OptimizedPathPlanner(width=GRID_SIZE, height=GRID_SIZE, robot_radius=0)
    robot = SubseaCleaningRobot()

    robot.boot_and_connect()
    robot.init_sensors(turbidity_high=False)
    robot.approach_and_attach(lidar_distance=12, slip_detected=False)

    obstacles = [(6, 6), (6, 7), (6, 8), (14, 12), (14, 13)]
    for ox, oy in obstacles:
        planner.set_obstacle(ox, oy)

    test_frame = np.full((FRAME_H, FRAME_W, 3), (120, 80, 50), dtype=np.uint8)
    barnacle_pixels = [(120, 150), (140, 160), (320, 240), (450, 380), (480, 390)]
    for bx, by in barnacle_pixels:
        cv2.circle(test_frame, (bx, by), 15, (200, 210, 220), -1)
        cv2.circle(test_frame, (bx, by), 6, (60, 70, 80), -1)

    print("\n[Vision] 카메라 프레임 YOLO 추론 시작...")
    detections, dirty_area, status = detector.infer_frame(test_frame)

    if status == "LOW_CONFIDENCE":
        print("[Vision] 신뢰도 낮음 감지 -> LED 증강 및 보정 요청")
        robot.init_sensors(turbidity_high=True)
        detections, dirty_area, status = detector.infer_frame(test_frame)

    print(f"[Vision] 검출 완료: {len(detections)}개 따개비 검출, 오염 면적: {dirty_area}px")

    barnacle_grid_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for det in detections:
        cx, cy = det["center"]
        gx, gy = mapper.pixel_to_grid(cx, cy)
        if planner.inflated_grid[gy, gx] == 0:
            barnacle_grid_map[gy, gx] = 2

    if np.sum(barnacle_grid_map == 2) == 0:
        for bx, by in barnacle_pixels:
            gx, gy = mapper.pixel_to_grid(bx, by)
            barnacle_grid_map[gy, gx] = 2

    print("\n[AI Path Planner] 최적 커버리지 및 A* 우회 경로 계산 중...")
    cleaning_path = planner.plan_coverage_path(orientation='horizontal')
    robot.set_cleaning_actuators(rpm=850, suction=100)

    CELL_PX = 32
    sim_w = GRID_SIZE * CELL_PX
    sim_h = GRID_SIZE * CELL_PX
    dashboard_w = 280
    total_w = sim_w + dashboard_w

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('yolo_integrated_cleaning.mp4', fourcc, 12, (total_w, sim_h))

    total_steps = len(cleaning_path)
    for step_idx, pos in enumerate(cleaning_path):
        cur_x, cur_y = pos

        if barnacle_grid_map[cur_y, cur_x] == 2:
            barnacle_grid_map[cur_y, cur_x] = 0

        if step_idx == 20:
            robot.handle_overload()
        elif step_idx == 40:
            robot.handle_suction_clog()

        progress = ((step_idx + 1) / total_steps) * 100
        robot.log_telemetry(pos, progress)

        canvas = np.ones((sim_h, total_w, 3), dtype=np.uint8) * 35

        for gy in range(GRID_SIZE):
            for gx in range(GRID_SIZE):
                rect = (gx * CELL_PX, gy * CELL_PX, CELL_PX, CELL_PX)
                if planner.raw_grid[gy, gx] == 1:
                    cv2.rectangle(canvas, rect, (50, 50, 200), -1)
                elif barnacle_grid_map[gy, gx] == 2:
                    cv2.circle(canvas, (gx * CELL_PX + CELL_PX // 2, gy * CELL_PX + CELL_PX // 2), 6, (0, 165, 255), -1)
                cv2.rectangle(canvas, rect, (65, 65, 65), 1)

        for p_idx in range(len(cleaning_path) - 1):
            p1 = (cleaning_path[p_idx][0] * CELL_PX + CELL_PX // 2, cleaning_path[p_idx][1] * CELL_PX + CELL_PX // 2)
            p2 = (cleaning_path[p_idx+1][0] * CELL_PX + CELL_PX // 2, cleaning_path[p_idx+1][1] * CELL_PX + CELL_PX // 2)
            cv2.line(canvas, p1, p2, (40, 120, 40), 2)

        rx = cur_x * CELL_PX + CELL_PX // 2
        ry = cur_y * CELL_PX + CELL_PX // 2
        cv2.circle(canvas, (rx, ry), 9, (255, 190, 0), -1)
        cv2.circle(canvas, (rx, ry), 11, (255, 255, 255), 2)

        ui_x = sim_w + 15
        cv2.putText(canvas, "YOLO-AI ROBOT DASHBOARD", (ui_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(canvas, f"State: {robot.state}", (ui_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, f"YOLO Model: YOLO26n (Barnacle)", (ui_x, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 180), 1)
        cv2.putText(canvas, f"Detected: {len(detections)} items", (ui_x, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        cv2.putText(canvas, f"Pos: ({cur_x}, {cur_y})", (ui_x, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, f"Progress: {progress:.1f}%", (ui_x, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, f"Brush RPM: {robot.brush_rpm}", (ui_x, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, f"Suction: {robot.suction_power}%", (ui_x, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.rectangle(canvas, (ui_x, 240), (ui_x + 240, 255), (70, 70, 70), -1)
        cv2.rectangle(canvas, (ui_x, 240), (ui_x + int(240 * (progress / 100)), 255), (0, 255, 0), -1)

        out.write(canvas)

    out.release()
    print("[Complete] 시뮬레이션 완료: 'yolo_integrated_cleaning.mp4' 파일이 저장되었습니다.")

if __name__ == "__main__":
    main()