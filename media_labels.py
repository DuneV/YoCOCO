import math
from typing import List, Dict, Optional, Tuple
import os, json
import cv2
import numpy as np
import mediapipe as mp

POSE = mp.solutions.pose.PoseLandmark
ARM_IDX = {
    "left":  {"shoulder": int(POSE.LEFT_SHOULDER), "elbow": int(POSE.LEFT_ELBOW), "wrist": int(POSE.LEFT_WRIST)},
    "right": {"shoulder": int(POSE.RIGHT_SHOULDER),"elbow": int(POSE.RIGHT_ELBOW),"wrist": int(POSE.RIGHT_WRIST)},
}

class MediaPipeArmsDetector:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        draw: bool = True,            
        draw_thickness: int = 2
    ):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.draw = draw
        self.draw_thickness = draw_thickness
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

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
    def _to_xy_pixel(landmark, w, h):
        """Convierte NormalizedLandmark a coordenadas de píxeles (x,y) y retorna también z y visibility."""
        x = min(max(int(round(landmark.x * w)), 0), w - 1)
        y = min(max(int(round(landmark.y * h)), 0), h - 1)
        return x, y, float(landmark.z), float(landmark.visibility)

    @staticmethod
    def _angle_3pts(a, b, c):
        """Ángulo ABC en grados con puntos (x,y)."""
        bax = a[0] - b[0]; bay = a[1] - b[1]
        bcx = c[0] - b[0]; bcy = c[1] - b[1]
        dot = bax * bcx + bay * bcy
        na = math.hypot(bax, bay); nb = math.hypot(bcx, bcy)
        if na == 0 or nb == 0:
            return None
        cosang = max(-1.0, min(1.0, dot / (na * nb)))
        return math.degrees(math.acos(cosang))

    def _extract_arm_points(self, landmarks, w, h):
        """Devuelve dict con puntos del brazo izq/der (xy pixel + z + visibility) y ángulos del codo."""
        arms = {}
        for side in ("left", "right"):
            sh = landmarks[ARM_IDX[side]["shoulder"]]
            el = landmarks[ARM_IDX[side]["elbow"]]
            wr = landmarks[ARM_IDX[side]["wrist"]]
            sx, sy, sz, sv = self._to_xy_pixel(sh, w, h)
            ex, ey, ez, ev = self._to_xy_pixel(el, w, h)
            wx, wy, wz, wv = self._to_xy_pixel(wr, w, h)
            angle_elbow = self._angle_3pts((sx, sy), (ex, ey), (wx, wy))  # hombro - codo - muñeca

            arms[side] = {
                "shoulder": {"x": sx, "y": sy, "z": sz, "visibility": sv},
                "elbow":    {"x": ex, "y": ey, "z": ez, "visibility": ev, "elbow_angle_deg": angle_elbow},
                "wrist":    {"x": wx, "y": wy, "z": wz, "visibility": wv}
            }
        return arms

    def _draw_arms(self, frame, arms: Dict, color=(0, 255, 0)):
        for side in ("left", "right"):
            s = arms[side]["shoulder"]; e = arms[side]["elbow"]; w = arms[side]["wrist"]
            pS = (int(s["x"]), int(s["y"]))
            pE = (int(e["x"]), int(e["y"]))
            pW = (int(w["x"]), int(w["y"]))
            cv2.line(frame, pS, pE, color, self.draw_thickness)
            cv2.line(frame, pE, pW, color, self.draw_thickness)
            for p in (pS, pE, pW):
                cv2.circle(frame, p, 4, color, -1)
            ang = e.get("elbow_angle_deg")
            if ang is not None:
                txt = f"{side[:1].upper()}-elbow: {ang:.1f}°"
                cv2.putText(frame, txt, (pE[0]+6, pE[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(frame, txt, (pE[0]+6, pE[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def detect(
        self,
        video_path: str,
        frame_fps: Optional[float] = None,
        save_video: bool = False,
        output_path: Optional[str] = None,
        return_timestamps: bool = True
    ) -> Tuple[List[Dict], Optional[str]]:
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

        def _ensure_writer(frame):
            nonlocal writer, out_path_final
            if not save_video or writer is not None:
                return
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if not output_path:
                base = video_path
                base_wo_ext = base.rsplit('.', 1)[0] if '.' in base else base
                out_path_final = base_wo_ext + "_arms.mp4"
            else:
                out_path_final = output_path
            writer = cv2.VideoWriter(out_path_final, fourcc, out_fps, (w, h), True)

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if idx_abs % stride != 0:
                idx_abs += 1
                continue

            h, w = frame_bgr.shape[:2]
            # MediaPipe trabaja en RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.pose.process(frame_rgb)

            arms = None
            if res.pose_landmarks:
                landmarks = res.pose_landmarks.landmark
                arms = self._extract_arm_points(landmarks, w, h)
                if self.draw:
                    _ensure_writer(frame_bgr)
                    # Dibuja malla pose 
                    self.mp_drawing.draw_landmarks(
                        frame_bgr, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style()
                    )
                    self._draw_arms(frame_bgr, arms, color=(0, 255, 0))

            t_sec = (idx_abs / in_fps) if return_timestamps else None
            results_out.append({
                "frame_idx": kept_idx,
                "time_sec": t_sec,
                "arms": arms 
            })

            if writer is not None and self.draw:
                writer.write(frame_bgr)

            kept_idx += 1
            idx_abs += 1

        cap.release()
        if writer is not None:
            writer.release()

        return results_out, out_path_final


def print_arm_results(results, show_time=True):
    for item in results:
        t = f"{item['time_sec']:.2f}s" if (show_time and item['time_sec'] is not None) else "-"
        arms = item['arms']
        print(f"\nframe {item['frame_idx']} (t={t})")
        if not arms:
            print("  Sin detección de brazos")
            continue
        for side in ("left", "right"):
            e = arms[side]["elbow"]
            ang = e.get("elbow_angle_deg")
            print(f"  {side.capitalize()} -> hombro={arms[side]['shoulder']}, codo={e}, muñeca={arms[side]['wrist']}")
            if ang is not None:
                print(f"    Ángulo codo: {ang:.1f}°")


if __name__ == "__main__":
    video_path = "data_temp/M0jKmFfnlhE.mp4"

    det = MediaPipeArmsDetector(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        draw=True,
        draw_thickness=2
    )

    results, out_video = det.detect(
        video_path=video_path,
        frame_fps=None,          
        save_video=True,
        output_path=None,
        return_timestamps=True
    )

    out_json = (os.path.splitext(out_video or video_path)[0] + "_arms.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Video anotado:", out_video)
    print_arm_results(results)
