from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class AppConfig:
    model_path: str = "best.pt"
    confidence: float = 0.6
    iou: float = 0.45
    device: int = 0
    imgsz: int = 960

    count_line_x_ratio: float = 0.55
    count_line_x_ratio_2: float = 0.65
    min_tracked_frames: int = 2
    max_history: int = 20
    direction_margin_px: int = 8

    window_name: str = "Beverage Conveyor Monitoring"
    output_dir: str = "output"
    display_scale: float = 1.0

    canonical_classes: Tuple[str, ...] = ("cocacola", "pepsi", "other")

    box_colors: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            "cocacola": (255, 0, 0),
            "pepsi": (0, 0, 255),
            "other": (0, 165, 255),
        }
    )
