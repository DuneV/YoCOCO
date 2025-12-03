import cv2 as cv
import numpy as np
import json, os
from pathlib import Path

class OpenPoseTFArms:
    BODY_PARTS = {
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
    }

    POSE_PAIRS_ALL = [
        ["Neck","RShoulder"],["Neck","LShoulder"],["RShoulder","RElbow"],
        ["RElbow","RWrist"],["LShoulder","LElbow"],["LElbow","LWrist"],
        ["Neck","RHip"],["RHip","RKnee"],["RKnee","RAnkle"],["Neck","LHip"],
        ["LHip","LKnee"],["LKnee","LAnkle"],["Neck","Nose"],["Nose","REye"],
        ["REye","REar"],["Nose","LEye"],["LEye","LEar"]
    ]

    POSE_PAIRS_ARMS = [
        ["RShoulder","RElbow"],["RElbow","RWrist"],
        ["LShoulder","LElbow"],["LElbow","LWrist"]
    ]

    ARM_PARTS = ["RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist"]

    def __init__(
        self,
        model_path="models/graph_opt.pb",
        in_size=(656, 368),
        heatmap_thr=0.15,
        nms_window=5,
        gauss_kernel=(5, 5),
        multiscales=(0.75, 1.0, 1.25),
        force_cpu=True,
        arms_only=True
    ):
        self.model_path = model_path
        self.inW, self.inH = in_size
        self.HEATMAP_THRESHOLD = heatmap_thr
        self.NMS_WINDOW = nms_window
        self.GAUSS_KERNEL = gauss_kernel
        self.SCALES = multiscales
        self.force_cpu = force_cpu
        self.arms_only = arms_only

        self._ensure_model()
        self.net = cv.dnn.readNetFromTensorflow(self.model_path)
        self._pick_backend_target()

        if self.arms_only:
            self.target_parts = self.ARM_PARTS
            self.pose_pairs = self.POSE_PAIRS_ARMS
        else:
            self.target_parts = [k for k in self.BODY_PARTS.keys() if k != "Background"]
            self.pose_pairs = self.POSE_PAIRS_ALL

        self.target_indices = [self.BODY_PARTS[p] for p in self.target_parts]

    def _ensure_model(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"No se encontró el modelo en '{self.model_path}'.")

    def _pick_backend_target(self):
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        if self.force_cpu:
            return
        try:
            cv.ocl.setUseOpenCL(True)
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
        except Exception:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def _nms_argmax_2d(self, heatmap):
        k = self.NMS_WINDOW if self.NMS_WINDOW % 2 == 1 else self.NMS_WINDOW + 1
        hm = cv.normalize(heatmap, None, 0.0, 1.0, cv.NORM_MINMAX)
        pooled = cv.dilate(hm, cv.getStructuringElement(cv.MORPH_RECT, (k, k)))
        mask = (hm == pooled) & (hm >= self.HEATMAP_THRESHOLD)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None, 0.0
        vals = hm[ys, xs]
        i = int(np.argmax(vals))
        return (int(xs[i]), int(ys[i])), float(vals[i])

    def _forward_multiscale(self, frame):
        acc = None
        for s in self.SCALES:
            fr = cv.resize(frame, None, fx=s, fy=s, interpolation=cv.INTER_LINEAR)
            blob = cv.dnn.blobFromImage(
                fr, 1.0, (self.inW, self.inH),
                (127.5,127.5,127.5), swapRB=True, crop=False
            )
            self.net.setInput(blob)
            out = self.net.forward()[:, :19, :, :].astype(np.float32)
            if acc is None:
                acc = np.zeros_like(out, dtype=np.float32)
            acc += out
        acc /= float(len(self.SCALES))
        return acc

    def process_frame(self, frame):
        Hf, Wf = frame.shape[:2]
        t0 = cv.getTickCount()
        out = self._forward_multiscale(frame)
        t1 = cv.getTickCount()
        ms = (t1 - t0) * 1000.0 / cv.getTickFrequency()

        out = out[:, :19, :, :]
        Hm, Wm = out.shape[2], out.shape[3]

        points = {name: None for name in self.target_parts}
        confs  = {name: 0.0  for name in self.target_parts}

        for name in self.target_parts:
            idx = self.BODY_PARTS[name]
            heatMap = out[0, idx, :, :]
            if self.GAUSS_KERNEL:
                heatMap = cv.GaussianBlur(heatMap, self.GAUSS_KERNEL, 0)
            peak, conf = self._nms_argmax_2d(heatMap)
            if peak is not None:
                x = int((Wf * peak[0]) / Wm)
                y = int((Hf * peak[1]) / Hm)
                points[name] = (x, y)
                confs[name]  = float(conf)

        canvas = frame
        for a, b in self.pose_pairs:
            pa, pb = points.get(a), points.get(b)
            if pa and pb:
                cv.line(canvas, pa, pb, (0, 255, 0), 3)
                cv.circle(canvas, pa, 3, (0, 0, 255), cv.FILLED)
                cv.circle(canvas, pb, 3, (0, 0, 255), cv.FILLED)

        cv.putText(canvas, f'{ms:.2f} ms', (10, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)

        keypoints = []
        for name in self.target_parts:
            pt = points[name]
            keypoints.append({
                "part": name,
                "x": int(pt[0]) if pt else None,
                "y": int(pt[1]) if pt else None,
                "conf": float(confs[name])
            })
        return canvas, ms, keypoints

    def run_video_with_json(self, input_path, output_path, json_path=None, output_fps=None, fourcc="mp4v"):
        cap = cv.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {input_path}")

        ok, first = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("No se pudo leer el primer frame.")

        H, W = first.shape[:2]
        in_fps = cap.get(cv.CAP_PROP_FPS)
        if not in_fps or in_fps <= 1.0:
            in_fps = 20.0
        fps_out = output_fps if output_fps else in_fps

        writer = cv.VideoWriter(
            output_path,
            cv.VideoWriter_fourcc(*fourcc),
            fps_out,
            (W, H)
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"No se pudo abrir el writer: {output_path}")

        results = []
        frame_idx = 0

        frame_out, ms, keypoints = self.process_frame(first)
        writer.write(frame_out)
        results.append({
            "frame_idx": frame_idx,
            "time_sec": frame_idx / in_fps,
            "speed_ms": ms,
            "keypoints": keypoints
        })
        frame_idx += 1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_out, ms, keypoints = self.process_frame(frame)
            writer.write(frame_out)
            results.append({
                "frame_idx": frame_idx,
                "time_sec": frame_idx / in_fps,
                "speed_ms": ms,
                "keypoints": keypoints
            })
            frame_idx += 1

        cap.release()
        writer.release()

        if json_path is None:
            base = os.path.splitext(output_path or input_path)[0]
            json_path = base + "_arms.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results, output_path, json_path


if __name__ == "__main__":
    filename = "data_temp/M0jKmFfnlhE.mp4"
    output_filename = "data_temp/M0jKmFfnlhE_out.mp4"

    pose = OpenPoseTFArms(
        model_path="models/graph_opt.pb",
        in_size=(656, 368),
        heatmap_thr=0.15,
        nms_window=5,
        gauss_kernel=(5, 5),
        multiscales=(0.75, 1.0, 1.25),
        force_cpu=True,
        arms_only=True
    )

    results, out_video, out_json = pose.run_video_with_json(
        input_path=filename,
        output_path=output_filename,
        json_path=None,
        output_fps=None,
        fourcc="mp4v"
    )
    print("Video:", out_video)
    print("JSON:", out_json)
