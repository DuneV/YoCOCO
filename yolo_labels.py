# yolo_labels.py

import math
from typing import List, Dict, Optional, Tuple
import os, json
import cv2
import torch
from ultralytics import YOLO

class YOLOVideoDetectorLite:
    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
        half: Optional[bool] = None,
        augment: bool = False,
        batch_size: int = 8,
        classes: Optional[List[int]] = None
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.augment = augment
        self.batch_size = max(1, int(batch_size))
        self.classes = classes
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if half is None:
            half = (self.device == "cuda")
        self.half = half and (self.device == "cuda")

    @staticmethod
    def _safe_fps(cap: cv2.VideoCapture, fallback: float = 25.0) -> float:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isinf(fps) or math.isnan(fps) or fps <= 1.0:
            return fallback
        return fps

    @staticmethod
    def _compute_stride(in_fps: float, target_fps: float) -> int:
        target_fps = max(target_fps, 0.1)
        return max(1, int(round(in_fps / target_fps)))

    @staticmethod
    def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float):
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return (x1, y1, w, h)

    def _predict_batch(self, frames: List):
        results = self.model.predict(
            source=frames,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
            augment=self.augment,
            classes=self.classes,
        )
        return results

    def detect(
        self,
        video_path: str,
        frame_fps: Optional[float] = None,
        bbox_format: str = "xyxy",
        return_timestamps: bool = True,
        save_video: bool = False,
        output_path: Optional[str] = None,
        draw_thickness: int = 2,
        draw_labels: bool = True,
        draw_conf: bool = True,
    ) -> Tuple[List[Dict], Optional[str]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")
        in_fps = self._safe_fps(cap, fallback=25.0)
        stride = 1 if frame_fps is None else self._compute_stride(in_fps, frame_fps)
        out_fps = in_fps if frame_fps is None else frame_fps
        results_out: List[Dict] = []
        batch_frames: List = []
        batch_meta: List[Tuple[int, float, 'frame_bgr']] = []
        idx_abs = 0
        frame_idx_kept = 0
        writer = None
        out_path_final = None

        def _ensure_writer(frame):
            nonlocal writer, out_path_final
            if not save_video:
                return
            if writer is not None:
                return
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if not output_path:
                base = video_path
                if '.' in base:
                    base_wo_ext = base.rsplit('.', 1)[0]
                else:
                    base_wo_ext = base
                out_path_final = base_wo_ext + "_annot.mp4"
            else:
                out_path_final = output_path
            writer = cv2.VideoWriter(out_path_final, fourcc, out_fps, (w, h), True)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx_abs % stride == 0:
                t_sec = (idx_abs / in_fps) if return_timestamps else None
                batch_frames.append(frame)
                batch_meta.append((frame_idx_kept, t_sec, frame))
                frame_idx_kept += 1
                if len(batch_frames) >= self.batch_size:
                    self._flush_batch_and_draw(
                        batch_frames, batch_meta, results_out, bbox_format,
                        draw_thickness, draw_labels, draw_conf,
                        ensure_writer=lambda fr: _ensure_writer(fr),
                        writer_ref=lambda: writer
                    )
                    if writer is not None:
                        for _, _, fr_annot in batch_meta:
                            writer.write(fr_annot)
                    batch_frames.clear()
                    batch_meta.clear()
            idx_abs += 1

        if batch_frames:
            self._flush_batch_and_draw(
                batch_frames, batch_meta, results_out, bbox_format,
                draw_thickness, draw_labels, draw_conf,
                ensure_writer=lambda fr: _ensure_writer(fr),
                writer_ref=lambda: writer
            )
            if writer is not None:
                for _, _, fr_annot in batch_meta:
                    writer.write(fr_annot)
            batch_frames.clear()
            batch_meta.clear()

        cap.release()
        if writer is not None:
            writer.release()
        return results_out, out_path_final

    def _flush_batch_and_draw(
        self,
        batch_frames: List,
        batch_meta: List[Tuple[int, float, 'frame_bgr']],
        results_out: List[Dict],
        bbox_format: str,
        draw_thickness: int,
        draw_labels: bool,
        draw_conf: bool,
        ensure_writer,
        writer_ref
    ):
        preds = self._predict_batch(batch_frames)
        for r, meta in zip(preds, batch_meta):
            fidx, tsecs, frame = meta
            frame_dets = []
            names = r.names
            if hasattr(r, "boxes") and r.boxes is not None:
                for b in r.boxes:
                    cls = int(b.cls[0])
                    label = names.get(cls, str(cls))
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                    if bbox_format == "xywh":
                        bbox_out = list(self._xyxy_to_xywh(x1, y1, x2, y2))
                    else:
                        bbox_out = [x1, y1, x2, y2]
                    frame_dets.append({"label": label, "conf": conf, "bbox": bbox_out})
                    if writer_ref() is not None or ensure_writer is not None:
                        ensure_writer(frame)
                        p1 = (int(x1), int(y1))
                        p2 = (int(x2), int(y2))
                        cv2.rectangle(frame, p1, p2, (0, 255, 0), draw_thickness)
                        if draw_labels:
                            text = label
                            if draw_conf:
                                text = f"{label} {conf:.2f}"
                            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            x, y = p1
                            y = max(th + 4, y)
                            x2 = x + tw + 4
                            y2 = y + baseline
                            cv2.rectangle(frame, (x, y - th - 4), (x2, y2), (0, 255, 0), -1)
                            cv2.putText(frame, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            results_out.append({
                "frame_idx": fidx,
                "time_sec": tsecs,
                "detections": frame_dets
            })

def print_all_results(results, decimals=2, show_time=True):
    for item in results:
        t = f"{item['time_sec']:.2f}s" if (show_time and item['time_sec'] is not None) else "-"
        print(f"\nframe {item['frame_idx']} (t={t}) -> {len(item['detections'])} dets")
        for j, det in enumerate(item['detections'], 1):
            label = det['label']
            conf = det['conf']
            bbox = det['bbox']
            print(f"  {j:>3}. {label} conf={conf:.{decimals}f} bbox={bbox}")

if __name__ == "__main__":
    model_path = "models/yolov8n.pt"
    video_path = "data_temp/M0jKmFfnlhE.mp4"
    det = YOLOVideoDetectorLite(
        model_path=model_path,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device=None,
        half=None,
        augment=False,
        batch_size=8,
        classes=None
    )
    results, out_video = det.detect(
        video_path=video_path,
        frame_fps=None,
        bbox_format="xyxy",
        return_timestamps=True,
        save_video=True,
        output_path=None,
        draw_thickness=2,
        draw_labels=True,
        draw_conf=True
    )
    out_json = (os.path.splitext(out_video or video_path)[0] + "_detections.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Video anotado:", out_video)
    print_all_results(results)

