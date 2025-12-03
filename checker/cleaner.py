# checker/cleaner.py
from typing import List, Dict, Any, Tuple, Optional, Iterable
import copy

def _point_inside(p: Dict[str, float], frame_size: Tuple[int, int]) -> bool:
    W, H = frame_size
    return isinstance(p, dict) and 0 <= p.get("x", -1) < W and 0 <= p.get("y", -1) < H

def _bbox_inside_frame(bbox: Iterable[float], frame_size: Tuple[int, int]) -> bool:
    try:
        x1, y1, x2, y2 = bbox
    except Exception:
        return False
    W, H = frame_size
    return 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H

def _valid_detection(d: Dict[str, Any], frame_size: Tuple[int, int], keep_labels: Optional[set], conf_min: float) -> bool:
    if d.get("conf", 0.0) < conf_min:
        return False
    if keep_labels is not None and d.get("label") not in keep_labels:
        return False
    return _bbox_inside_frame(d.get("bbox", []), frame_size)

def _valid_hand(h: Dict[str, Any], frame_size: Tuple[int, int], required_points: Iterable[str], keep_labels: Optional[set]) -> bool:
    if keep_labels is not None and h.get("label") not in keep_labels:
        return False
    for k in required_points:
        if k not in h or not _point_inside(h[k], frame_size):
            return False
    return True

def filter_frame(
    frame: Dict[str, Any],
    frame_size: Tuple[int, int] = (1280, 720),
    det_conf_min: float = 0.70,
    det_keep_labels: Optional[set] = None,   # p.ej. {"person"}
    hand_required_points: Iterable[str] = ("wrist", "index_tip"),
    hand_keep_labels: Optional[set] = {"Left", "Right"},
    drop_empty: bool = True,
) -> Optional[Dict[str, Any]]:
    out = copy.deepcopy(frame)

    if "detections" in frame:
        dets = frame.get("detections", [])
        det_keep_labels = det_keep_labels if det_keep_labels is not None else {"person"}
        cleaned_dets = [
            d for d in dets
            if _valid_detection(d, frame_size, det_keep_labels, det_conf_min)
        ]
        out["detections"] = cleaned_dets

    if "hands" in frame:
        hands = frame.get("hands", [])
        cleaned_hands = [
            h for h in hands
            if _valid_hand(h, frame_size, hand_required_points, hand_keep_labels)
        ]
        out["hands"] = cleaned_hands

    if drop_empty:
        filtered_keys = []
        if "detections" in frame:
            filtered_keys.append(bool(out.get("detections")))
        if "hands" in frame:
            filtered_keys.append(bool(out.get("hands")))
        if filtered_keys and not any(filtered_keys):
            return None

    return out

def filter_stream(frames: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
    result = []
    for fr in frames:
        f = filter_frame(fr, **kwargs)
        if f is not None:
            result.append(f)
    return result
