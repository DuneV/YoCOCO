import math, os, json
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

KP = {
    "left":  {"shoulder": 5, "elbow": 7, "wrist": 9},
    "right": {"shoulder": 6, "elbow": 8, "wrist": 10},
}

class YOLOPoseArmsDetector:
    def __init__(self,
                 model_path: str = "yolov8n-pose.pt",
                 conf: float = 0.25,
                 iou: float = 0.45,
                 imgsz: int = 640,
                 device: Optional[str] = None,
                 half: Optional[bool] = None,
                 batch_size: int = 8,
                 draw: bool = True,
                 draw_thickness: int = 2):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.batch_size = max(1, int(batch_size))
        self.draw = draw
        self.draw_thickness = draw_thickness
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
    def _angle_3pts(a, b, c):
        bax = a[0] - b[0]; bay = a[1] - b[1]
        bcx = c[0] - b[0]; bcy = c[1] - b[1]
        na = math.hypot(bax, bay); nb = math.hypot(bcx, bcy)
        if na == 0 or nb == 0:
            return None
        dot = bax * bcx + bay * bcy
        cosang = max(-1.0, min(1.0, dot / (na * nb)))
        return math.degrees(math.acos(cosang))

    def _predict_batch(self, frames: List[np.ndarray]):
        return self.model.predict(
            source=frames,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False
        )

    def detect(self,
               video_path: str,
               frame_fps: Optional[float] = None,
               save_video: bool = False,
               output_path: Optional[str] = None,
               return_timestamps: bool = True) -> Tuple[List[Dict], Optional[str]]:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        in_fps = self._safe_fps(cap, fallback=25.0)
        stride = 1 if frame_fps is None else self._compute_stride(in_fps, frame_fps)
        out_fps = in_fps if frame_fps is None else frame_fps

        writer = None
        out_path_final = None
        results_out: List[Dict] = []
        idx_abs = 0
        kept_idx = 0
        batch_frames, batch_meta = [], []

        def _ensure_writer(frame):
            nonlocal writer, out_path_final
            if not save_video or writer is not None: return
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if not output_path:
                base = video_path
                base_wo_ext = base.rsplit('.', 1)[0] if '.' in base else base
                out_path_final = base_wo_ext + "_pose.mp4"
            else:
                out_path_final = output_path
            writer = cv2.VideoWriter(out_path_final, fourcc, out_fps, (w, h), True)

        def _flush():
            nonlocal batch_frames, batch_meta, results_out, writer, kept_idx
            if not batch_frames: return
            preds = self._predict_batch(batch_frames)
            for r, (fidx, tsec, frame) in zip(preds, batch_meta):
                arms = None
                if r.keypoints is not None and getattr(r.keypoints, "xy", None) is not None:
                    # shape: [num_personas, 17, 2]
                    persons = r.keypoints.xy
                    # confidencias por kp (si está disponible)
                    kpc = getattr(r.keypoints, "conf", None)  # [num_personas, 17]
                    # Toma a la persona con mayor área de bbox (opcional: podrías iterar todas)
                    best = None; best_area = -1
                    if r.boxes is not None and len(r.boxes) == len(persons):
                        for i, box in enumerate(r.boxes.xyxy):
                            x1, y1, x2, y2 = box.tolist()
                            area = max(0, x2-x1) * max(0, y2-y1)
                            if area > best_area:
                                best_area = area; best = i
                    else:
                        best = 0 if len(persons) else None

                    if best is not None and len(persons):
                        P = persons[best]  # (17,2)
                        C = kpc[best] if (kpc is not None and len(kpc) > best) else None

                        def get_kp(idx):
                            x, y = float(P[idx][0]), float(P[idx][1])
                            c = float(C[idx]) if C is not None else None
                            return {"x": int(round(x)), "y": int(round(y)), "conf": c}

                        Ls, Le, Lw = KP["left"]["shoulder"], KP["left"]["elbow"], KP["left"]["wrist"]
                        Rs, Re, Rw = KP["right"]["shoulder"], KP["right"]["elbow"], KP["right"]["wrist"]

                        left = {"shoulder": get_kp(Ls), "elbow": get_kp(Le), "wrist": get_kp(Lw)}
                        right= {"shoulder": get_kp(Rs), "elbow": get_kp(Re), "wrist": get_kp(Rw)}

                        # ángulos de codo
                        L_ang = self._angle_3pts(
                            (left["shoulder"]["x"], left["shoulder"]["y"]),
                            (left["elbow"]["x"],   left["elbow"]["y"]),
                            (left["wrist"]["x"],   left["wrist"]["y"])
                        )
                        R_ang = self._angle_3pts(
                            (right["shoulder"]["x"], right["shoulder"]["y"]),
                            (right["elbow"]["x"],    right["elbow"]["y"]),
                            (right["wrist"]["x"],    right["wrist"]["y"])
                        )
                        left["elbow"]["elbow_angle_deg"]  = L_ang
                        right["elbow"]["elbow_angle_deg"] = R_ang
                        arms = {"left": left, "right": right}

                        if self.draw:
                            _ensure_writer(frame)
                            # dibuja segmentos hombro–codo–muñeca
                            for side in ("left","right"):
                                s = arms[side]["shoulder"]; e = arms[side]["elbow"]; wkp = arms[side]["wrist"]
                                pS = (s["x"], s["y"]); pE = (e["x"], e["y"]); pW = (wkp["x"], wkp["y"])
                                cv2.line(frame, pS, pE, (0,255,0), self.draw_thickness)
                                cv2.line(frame, pE, pW, (0,255,0), self.draw_thickness)
                                for p in (pS, pE, pW):
                                    cv2.circle(frame, p, 4, (0,255,0), -1)
                                ang = arms[side]["elbow"].get("elbow_angle_deg")
                                if ang is not None:
                                    txt = f"{side[:1].upper()}-elbow: {ang:.1f}°"
                                    cv2.putText(frame, txt, (pE[0]+6, pE[1]-6),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                                    cv2.putText(frame, txt, (pE[0]+6, pE[1]-6),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                results_out.append({"frame_idx": fidx, "time_sec": tsec, "arms": arms})
                if writer is not None and self.draw:
                    writer.write(frame)

            batch_frames.clear(); batch_meta.clear()

        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx_abs % stride == 0:
                t_sec = (idx_abs / in_fps) if return_timestamps else None
                batch_frames.append(frame)
                batch_meta.append((kept_idx, t_sec, frame))
                kept_idx += 1
                if len(batch_frames) >= self.batch_size:
                    _flush()
            idx_abs += 1

        _flush()
        cap.release()
        if writer is not None:
            writer.release()
        return results_out, out_path_final

if __name__ == "__main__":
    video_path = "data_temp/M0jKmFfnlhE.mp4"

    det = YOLOPoseArmsDetector(
        model_path="models/yolov8n-pose.pt",  
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device=None,   # auto: cuda si disponible
        half=None,     # True si GPU FP16
        batch_size=8,
        draw=True,
        draw_thickness=2
    )

    results, out_video = det.detect(
        video_path=video_path,
        frame_fps=10.0,      
        save_video=True,  
        output_path=None,
        return_timestamps=True
    )

    out_json = (os.path.splitext(out_video or video_path)[0] + "_pose_arms.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Video anotado:", out_video)
    print("JSON:", out_json)
    for it in results[:5]:
        print(it)