from typing import List, Dict, Any, Iterable, Optional, Set, Tuple

def bbox_inside_frame(bbox: Iterable[float], frame_size: Tuple[int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    W, H = frame_size
    return 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H

def filter_detections(
    data: List[Dict[str, Any]],
    conf_min: float = 0.70,
    keep_labels: Optional[Set[str]] = None,
    frame_size: Tuple[int, int] = (1280, 720),
    drop_empty_frames: bool = True
) -> List[Dict[str, Any]]:
    if keep_labels is None:
        keep_labels = {"person"}
    out = []
    for frame in data:
        dets = frame.get("detections", [])
        cleaned = []
        for d in dets:
            if d.get("conf", 0.0) < conf_min:
                continue
            if d.get("label") not in keep_labels:
                continue
            if not bbox_inside_frame(d.get("bbox", []), frame_size):
                continue
            cleaned.append(d)
        if cleaned or not drop_empty_frames:
            out.append({
                "frame_idx": frame.get("frame_idx"),
                "time_sec": frame.get("time_sec"),
                "detections": cleaned
            })
    return out
