from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

from src.config import AppConfig
from src.utils import CANONICAL_DISPLAY_NAMES, compute_units


@dataclass
class TrackState:
    class_name: str
    centers: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=20))
    seen_frames: int = 0
    last_direction: str = "UNKNOWN"
    last_count_event: str = "NONE"


class ConveyorCounter:
    def __init__(self, config: AppConfig):
        self.config = config
        self.track_states: Dict[int, TrackState] = {}
        self.counts = defaultdict(int)
        for name in config.canonical_classes:
            self.counts[name] = 0

    def update(self, detections, line_x1, line_x2):
        active_track_ids = set()

        for det in detections:
            track_id = det["track_id"]
            if track_id < 0:
                continue

            active_track_ids.add(track_id)
            center = det["center"]
            class_name = det["class_name"]

            if track_id not in self.track_states:
                self.track_states[track_id] = TrackState(
                    class_name=class_name,
                    centers=deque(maxlen=self.config.max_history),
                )

            state = self.track_states[track_id]
            state.class_name = class_name
            state.centers.append(center)
            state.seen_frames += 1
            state.last_direction = self._estimate_direction(state)

            event = self._get_count_event(state, line_x1, line_x2)

            if event == "IN":
                self.counts[state.class_name] += 1
                state.last_count_event = "IN"

            elif event == "OUT":
                self.counts[state.class_name] = max(0, self.counts[state.class_name] - 1)
                state.last_count_event = "OUT"

        self._cleanup_lost_tracks(active_track_ids)

    def _estimate_direction(self, state: TrackState) -> str:
        if len(state.centers) < 2:
            return "UNKNOWN"

        x_first = state.centers[0][0]
        x_last = state.centers[-1][0]
        dx = x_last - x_first

        if dx > self.config.direction_margin_px:
            return "LEFT_TO_RIGHT"
        if dx < -self.config.direction_margin_px:
            return "RIGHT_TO_LEFT"
        return "STATIONARY"

    def _get_count_event(self, state: TrackState, line_x1: int, line_x2: int) -> str:
        if state.seen_frames < self.config.min_tracked_frames:
            return "NONE"
        if len(state.centers) < 2:
            return "NONE"

        prev_x = state.centers[-2][0]
        curr_x = state.centers[-1][0]

        crossed_line1_lr = prev_x < line_x1 <= curr_x
        crossed_line2_lr = prev_x < line_x2 <= curr_x

        crossed_line1_rl = prev_x > line_x1 >= curr_x
        crossed_line2_rl = prev_x > line_x2 >= curr_x

        if state.last_direction == "LEFT_TO_RIGHT":
            if (crossed_line1_lr or crossed_line2_lr) and state.last_count_event != "IN":
                return "IN"

        if state.last_direction == "RIGHT_TO_LEFT":
            if (crossed_line1_rl or crossed_line2_rl) and state.last_count_event != "OUT":
                return "OUT"

        return "NONE"

    def _cleanup_lost_tracks(self, active_track_ids):
        lost_ids = [track_id for track_id in self.track_states if track_id not in active_track_ids]
        for track_id in lost_ids:
            del self.track_states[track_id]

    def get_summary(self) -> Dict:
        per_class = {}
        total_cans = 0
        for class_name in self.config.canonical_classes:
            cans = int(self.counts[class_name])
            units = compute_units(cans)
            total_cans += cans
            per_class[class_name] = {
                "display_name": CANONICAL_DISPLAY_NAMES.get(class_name, class_name),
                **units,
            }

        total_units = compute_units(total_cans)
        return {
            "per_class": per_class,
            "total": {
                "display_name": "Tong cong",
                **total_units,
            },
        }

    def get_track_state(self, track_id: int):
        return self.track_states.get(track_id)
