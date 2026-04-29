import cv2

from src.utils import CANONICAL_DISPLAY_NAMES



def draw_tracks_and_boxes(frame, detections, counter, line_x1: int, line_x2: int):
    h, w = frame.shape[:2]

    cv2.line(frame, (line_x1, 0), (line_x1, h), (0, 255, 255), 2)
    cv2.line(frame, (line_x2, 0), (line_x2, h), (255, 255, 0), 2)
    cv2.putText(
        frame,
        "COUNT LINE",
        (max(10, line_x1 - 55), 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "COUNT LINE",
        (max(10, line_x2 - 55), 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["center"]
        track_id = det["track_id"]
        class_name = det["class_name"]
        conf = det["conf"]

        state = counter.get_track_state(track_id) if track_id >= 0 else None
        direction = state.last_direction if state else "UNKNOWN"
        event_text = state.last_count_event if state else "NONE"

        color = counter.config.box_colors.get(class_name, (255, 255, 255))
        display_name = CANONICAL_DISPLAY_NAMES.get(class_name, class_name)

        # Ve bounding box va tam
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)

        label_1 = f"ID:{track_id} {display_name} {conf:.2f}"
        event_text = state.last_count_event if state else "NONE"

        if event_text == "IN":
            label_2 = f"{direction} | +1"
        elif event_text == "OUT":
            label_2 = f"{direction} | -1"
        else:
            label_2 = f"{direction}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_1 = 0.48
        font_scale_2 = 0.42
        thickness = 1
        pad_x = 6
        pad_y = 6
        line_gap = 10

        (w1, h1), _ = cv2.getTextSize(label_1, font, font_scale_1, thickness)
        (w2, h2), _ = cv2.getTextSize(label_2, font, font_scale_2, thickness)

        text_w = max(w1, w2)
        text_h = h1 + h2 + line_gap + pad_y * 2

        bg_x1 = max(0, x1)
        bg_y1 = y1 - text_h - 12

        if bg_y1 < 0:
            bg_y1 = min(h - text_h - 1, y2 + 12)

        bg_x2 = min(w - 1, bg_x1 + text_w + pad_x * 2)
        bg_y2 = min(h - 1, bg_y1 + text_h)

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)

        text_x = bg_x1 + pad_x
        text_y1 = bg_y1 + pad_y + h1
        text_y2 = text_y1 + line_gap + h2

        cv2.putText(frame, label_1, (text_x, text_y1), font, font_scale_1, color, thickness)
        cv2.putText(frame, label_2, (text_x, text_y2), font, font_scale_2, color, thickness)

def draw_dashboard(frame, summary, fps_value: float):
    overlay = frame.copy()

    x1, y1, x2, y2 = 10, 10, 330, 155
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    y = 30
    cv2.putText(
        frame,
        "DASHBOARD",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    y += 25
    cv2.putText(
        frame,
        f"FPS: {fps_value:.2f}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    y += 25
    for class_name, info in summary["per_class"].items():
        text = f"{info['display_name']}: {info['cans']} lon | {info['packs']} loc | {info['cases']} thung"

        color = (255, 255, 255)
        if class_name == "cocacola":
            color = (255, 0, 0)
        elif class_name == "pepsi":
            color = (0, 0, 255)
        elif class_name == "other":
            color = (0, 165, 255)

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )
        y += 22

    total = summary["total"]
    total_text = f"Tong: {total['cans']} lon | {total['packs']} loc | {total['cases']} thung"
    cv2.putText(
        frame,
        total_text,
        (20, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )