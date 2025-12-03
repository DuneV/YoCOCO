# media_lite.py

import math, cv2, json, os
from typing import List, Dict, Optional, Tuple
import mediapipe as mp

HANDS = mp.solutions.hands
IDX_WRIST = 0 

class MediaPipeHandsArmsLite:
    def __init__(self,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 draw: bool = True,
                 draw_thickness: int = 2):
        self.hands = HANDS.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.draw = draw
        self.draw_thickness = draw_thickness

    @staticmethod
    def _safe_fps(cap, fallback=25.0):
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fallback if not fps or fps <= 1 else fps

    @staticmethod
    def _compute_stride(in_fps, target_fps):
        return max(1, int(round(in_fps / max(target_fps, 0.1))))

    @staticmethod
    def _to_xy(ln, w, h):
        x = int(round(min(max(ln.x * w, 0), w - 1)))
        y = int(round(min(max(ln.y * h, 0), h - 1)))
        return x, y, float(ln.z)

    def detect(self, video_path: str,
               frame_fps: Optional[float] = None,
               save_video: bool = False,
               output_path: Optional[str] = None,
               return_timestamps: bool = True) -> Tuple[List[Dict], Optional[str]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        in_fps = self._safe_fps(cap)
        stride = 1 if frame_fps is None else self._compute_stride(in_fps, frame_fps)
        out_fps = in_fps if frame_fps is None else frame_fps

        writer = None
        out_path_final = None
        results = []
        i_abs, i_kept = 0, 0

        def ensure_writer(frame):
            nonlocal writer, out_path_final
            if not save_video or writer is not None: return
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path_final = (output_path or (os.path.splitext(video_path)[0] + "_hands.mp4"))
            writer = cv2.VideoWriter(out_path_final, fourcc, out_fps, (w, h), True)

        while True:
            ok, frame = cap.read()
            if not ok: break
            if i_abs % stride:
                i_abs += 1; continue

            h, w = frame.shape[:2]
            res = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            hands_out = []
            if res.multi_hand_landmarks:
                for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness or []):
                    wrist_x, wrist_y, wrist_z = self._to_xy(lm.landmark[IDX_WRIST], w, h)
                    idx_tip = lm.landmark[HANDS.HandLandmark.INDEX_FINGER_TIP]
                    ix, iy, iz = self._to_xy(idx_tip, w, h)
                    label = handed.classification[0].label  

                    hands_out.append({
                        "label": label,
                        "wrist": {"x": wrist_x, "y": wrist_y, "z": wrist_z},
                        "index_tip": {"x": ix, "y": iy, "z": iz}
                    })

                    if self.draw:
                        ensure_writer(frame)
                        cv2.circle(frame, (wrist_x, wrist_y), 5, (0,255,0), -1)
                        cv2.putText(frame, f"{label} wrist", (wrist_x+6, wrist_y-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                        cv2.putText(frame, f"{label} wrist", (wrist_x+6, wrist_y-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            t = (i_abs / in_fps) if return_timestamps else None
            results.append({"frame_idx": i_kept, "time_sec": t, "hands": hands_out})

            if writer is not None and self.draw:
                writer.write(frame)

            i_kept += 1; i_abs += 1

        cap.release()
        if writer is not None: writer.release()
        return results, out_path_final

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

    det = MediaPipeHandsArmsLite(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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

    out_json = (os.path.splitext(out_video or video_path)[0] + "_hands.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # print("Video anotado:", out_video)
    # print("JSON:", out_json)

    # for it in results[:5]: 
    #     print(it)