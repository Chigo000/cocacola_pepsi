from typing import Dict


CANONICAL_DISPLAY_NAMES = {
    "cocacola": "Coca-Cola",
    "pepsi": "Pepsi",
    "other": "Loai khac",
}

def compute_units(cans: int) -> Dict[str, int]:
    cases = cans // 24
    remaining_after_cases = cans % 24
    packs = remaining_after_cases // 6
    remaining_cans = remaining_after_cases % 6
    return {
        "cans": cans,
        "packs": packs,
        "cases": cases,
        "remaining_cans": remaining_cans,
    }


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
