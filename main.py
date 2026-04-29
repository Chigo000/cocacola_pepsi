import argparse
import os
import time

import cv2

from src.config import AppConfig
from src.counter import ConveyorCounter
from src.detector import YOLODetector
from src.visualizer import draw_dashboard, draw_tracks_and_boxes


def parse_args():
    parser = argparse.ArgumentParser(description="Beverage can classification and counting on conveyor")
    parser.add_argument("--source", type=str, default="0", help="0 for webcam or path to video")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLO model")
    parser.add_argument("--save", action="store_true", help="Save output video")
    return parser.parse_args()


def open_capture(source: str):
    if source.isdigit():
        cam_index = int(source)
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    return cap


def main():
    args = parse_args()
    config = AppConfig(model_path=args.model)

    if not os.path.exists(config.model_path):
        raise FileNotFoundError(
            f"Không tìm thấy model: {config.model_path}. Hãy copy file best.pt vào thư mục dự án."
        )

    detector = YOLODetector(config)
    counter = ConveyorCounter(config)

    cap = open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError("Không thể mở webcam/video source.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 30.0

    writer = None
    if args.save:
        os.makedirs(config.output_dir, exist_ok=True)
        output_path = os.path.join(config.output_dir, f"result_{int(time.time())}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"[INFO] Đang lưu video tại: {output_path}")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections, annotated = detector.track(frame)
        line_x1 = int(frame.shape[1] * config.count_line_x_ratio)
        line_x2 = int(frame.shape[1] * config.count_line_x_ratio_2)
        counter.update(detections, line_x1, line_x2)

        view = frame.copy()
        draw_tracks_and_boxes(view, detections, counter, line_x1, line_x2)

        current_time = time.time()
        fps_value = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        draw_dashboard(view, counter.get_summary(), fps_value)

        if config.display_scale != 1.0:
            view = cv2.resize(
                view,
                None,
                fx=config.display_scale,
                fy=config.display_scale,
                interpolation=cv2.INTER_LINEAR,
            )

        cv2.imshow(config.window_name, view)

        if writer is not None:
            writer.write(cv2.resize(view, (frame_width, frame_height)))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
