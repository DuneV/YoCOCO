# interactions_knn.py

from __future__ import annotations

import os
import math
import json
import uuid
import csv
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Iterable, DefaultDict
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import SpectralClustering, KMeans
# from sklearn.metrics import sidlhouette_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

DB_SCHEMA         = os.getenv("DB_SCHEMA", "aura_info")
DB_HOST           = os.getenv("DATABASE_HOST")
DB_PORT           = int(os.getenv("DATABASE_PORT", os.getenv("PORT", "5432")))
DB_NAME           = os.getenv("DATABASE_NAME")
DB_USER           = os.getenv("DATABASE_USERNAME")
DB_PASSWORD       = os.getenv("DATABASE_PASSWORD")

OBJECT_LABELS     = set(json.loads(os.getenv("OBJECT_LABELS", "null"))) if os.getenv("OBJECT_LABELS") else None
WRIST_LABELS      = set(json.loads(os.getenv("WRIST_LABELS", '["Left","Right"]')))
PROXIMITY_PX      = float(os.getenv("PROXIMITY_PX", "20.0"))   # distancia máx al bbox para considerar "cerca"

# segmentación
MIN_SEG_FRAMES    = int(os.getenv("MIN_SEG_FRAMES", "3"))
GAP_TOLERANCE     = int(os.getenv("GAP_TOLERANCE", "1"))       # frames permitidos de ruptura sin cortar segmento

# clustering
CLUSTER_METHOD    = os.getenv("CLUSTER_METHOD", "kmeans")    # spectral | kmeans

N_NEIGHBORS       = int(os.getenv("N_NEIGHBORS", "10"))        # k para grafo k-NN
AUTO_N_CLUSTERS   = os.getenv("AUTO_N_CLUSTERS", "1") == "1"   # probar 2..max
MAX_CLUSTERS      = int(os.getenv("MAX_CLUSTERS", "8"))
FIXED_CLUSTERS    = int(os.getenv("FIXED_CLUSTERS", "5"))      # si AUTO_N_CLUSTERS=0
PCA_VARIANCE      = float(os.getenv("PCA_VARIANCE", "0.95"))

# guardado
USE_DB            = os.getenv("USE_DB", "1") == "1"
WRITE_CSV         = os.getenv("WRITE_CSV", "1") == "1"
OUT_DIR           = os.getenv("OUT_DIR", "data_temp")

# supervised (opcional)
LABELS_CSV        = os.getenv("LABELS_CSV", "")   # CSV con columnas: segment_id, label
SAVE_MODELS       = os.getenv("SAVE_MODELS", "1") == "1"
MODELS_DIR        = os.getenv("MODELS_DIR", "models_motion")

VIDEO_ANALYSIS_ID = os.getenv("VIDEO_ANALYSIS_ID", "").strip()
VIDEO_NAME        = os.getenv("VIDEO_NAME", "").strip()
USE_LATEST        = os.getenv("USE_LATEST", "1") == "1"  # si no pasas id ni name, toma el último


from db.db import Database
from sqlalchemy import text as sql_text
from db_models import VideoAnalysis  # usa tu Base y el modelo que ya tienes


def point_in_bbox(x: float, y: float, bbox: Iterable[float]) -> bool:
    '''
        Me retorna si el punto de la coordenada esta dentro de mi bbox
    '''
    try:
        x1, y1, x2, y2 = bbox
    except Exception:
        return False
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def point_to_rect_distance(x: float, y: float, bbox: Iterable[float]) -> float:
    """Distancia euclídea de un punto al rectángulo (0 si está dentro)."""
    try:
        x1, y1, x2, y2 = bbox
    except Exception:
        return float("inf")
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2) # retorna 0 si esta por dentro
    return math.hypot(dx, dy) # norma del vector



def bbox_center(bbox: Iterable[float]) -> Tuple[float, float]:
    '''
        Centro del bbox
    '''
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def vel(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float, float]: # podemos ver otras aproximaciones de la velocidad
    """Devuelve (vx, vy, v) con v = norma."""
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    v = math.hypot(vx, vy)
    return vx, vy, v

@dataclass
class HandPoint:
    x: float
    y: float
    z: float
    label: str  

@dataclass
class Detection:
    label: str
    conf: float
    bbox: Tuple[float, float, float, float] 

@dataclass
class FrameData:
    frame_idx: int
    time_sec: float
    hands: List[HandPoint]
    dets: List[Detection]


def load_video_analysis(db: Database) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Devuelve (video_id_string, yolo_objects, arms_pose) de la fila seleccionada."""
    with db.session() as s:
        q = s.query(VideoAnalysis)
        if VIDEO_ANALYSIS_ID:
            row = q.filter(VideoAnalysis.id == VIDEO_ANALYSIS_ID).first()
        elif VIDEO_NAME:
            row = q.filter(VideoAnalysis.video_name == VIDEO_NAME).first()
        else:
            if USE_LATEST:
                row = q.order_by(VideoAnalysis.created_at.desc()).first()
            else:
                row = q.first()
        if not row:
            raise RuntimeError("No se encontró registro en aura_info.video_analyses con los criterios dados.")
        return str(row.id), row.yolo_objects, row.arms_pose
    

def align_frames(yolo: List[Dict[str, Any]], hands: List[Dict[str, Any]]) -> List[FrameData]:
    """Une por frame_idx (se asume misma fps/índices)."""
    map_y = {it["frame_idx"]: it for it in yolo}
    map_h = {it["frame_idx"]: it for it in hands}
    keys = sorted(set(map_y.keys()) | set(map_h.keys()))
    out: List[FrameData] = []
    for k in keys:
        y = map_y.get(k)
        h = map_h.get(k)
        if not y and not h:
            continue
        t = (y or h).get("time_sec", None)
        dets: List[Detection] = []
        hands_list: List[HandPoint] = []
        if y:
            for d in y.get("detections", []):
                bbox = tuple(map(float, d.get("bbox", [0, 0, 0, 0])))
                dets.append(Detection(label=str(d.get("label","")), conf=float(d.get("conf",0.0)), bbox=bbox))
        if h:
            for hh in h.get("hands", []):
                wr = hh.get("wrist", {})
                hands_list.append(HandPoint(
                    x=float(wr.get("x", -1)), y=float(wr.get("y", -1)), z=float(wr.get("z", 0.0)),
                    label=str(hh.get("label",""))
                ))
        out.append(FrameData(frame_idx=k, time_sec=float(t if t is not None else -1.0), hands=hands_list, dets=dets))
    return out


# ====== Interacciones mano-objeto ======
@dataclass
class InteractionEvent:
    frame_idx: int
    time_sec: float
    wrist: str
    wrist_xy: Tuple[float, float]
    obj_label: str
    bbox: Tuple[float, float, float, float]
    inside: bool
    near: bool
    dist: float

def compute_interactions(
    frames: List[FrameData],
    obj_labels: Optional[set] = OBJECT_LABELS,
    wrist_labels: Optional[set] = WRIST_LABELS,
    proximity_px: float = PROXIMITY_PX
) -> List[InteractionEvent]:
    events: List[InteractionEvent] = []
    for fr in frames:
        if not fr.hands or not fr.dets:
            continue
        for h in fr.hands:
            if wrist_labels and h.label not in wrist_labels:
                continue
            for d in fr.dets:
                if obj_labels is not None and d.label not in obj_labels:
                    continue
                inside = point_in_bbox(h.x, h.y, d.bbox)
                dist = 0.0 if inside else point_to_rect_distance(h.x, h.y, d.bbox)
                near = inside or (dist <= proximity_px)
                if near:
                    events.append(InteractionEvent(
                        frame_idx=fr.frame_idx,
                        time_sec=fr.time_sec,
                        wrist=h.label,
                        wrist_xy=(h.x, h.y),
                        obj_label=d.label,
                        bbox=d.bbox,
                        inside=inside,
                        near=near,
                        dist=dist
                    ))
    return events


# ====== Segmentación de movimiento (por par muñeca/objeto) ======
@dataclass
class MotionSegment:
    segment_id: str
    video_id: str
    wrist: str
    obj_label: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    frames: List[int]

def build_segments(
    frames: List[FrameData],
    events: List[InteractionEvent],
    min_len: int = MIN_SEG_FRAMES,
    gap_tol: int = GAP_TOLERANCE
) -> List[MotionSegment]:
    # agrupa por (wrist, obj_label)
    by_pair: DefaultDict[Tuple[str,str], List[InteractionEvent]] = defaultdict(list)
    for ev in events:
        by_pair[(ev.wrist, ev.obj_label)].append(ev)

    segments: List[MotionSegment] = []

    for (wrist, obj), lst in by_pair.items():
        lst = sorted(lst, key=lambda e: e.frame_idx)
        run: List[InteractionEvent] = []
        last_frame = None
        for ev in lst:
            if not run:
                run = [ev]; last_frame = ev.frame_idx
                continue
            gap = ev.frame_idx - (last_frame if last_frame is not None else ev.frame_idx)
            if gap <= (gap_tol + 1):
                run.append(ev)
            else:
                # cierra tramo
                if len(run) >= min_len:
                    seg = _make_segment(frames, wrist, obj, run)
                    if seg: segments.append(seg)
                run = [ev]
            last_frame = ev.frame_idx
        if run and len(run) >= min_len:
            seg = _make_segment(frames, wrist, obj, run)
            if seg: segments.append(seg)

    return segments

def _make_segment(frames: List[FrameData], wrist: str, obj: str, run: List[InteractionEvent]) -> Optional[MotionSegment]:
    start = run[0].frame_idx
    end   = run[-1].frame_idx
    times = [ev.time_sec for ev in run if ev.time_sec is not None]
    st = float(times[0]) if times else -1.0
    et = float(times[-1]) if times else -1.0
    segment_id = str(uuid.uuid4())
    frames_list = [ev.frame_idx for ev in run]
    # video_id se setea después (lo pasaremos al persistir)
    return MotionSegment(
        segment_id=segment_id, video_id="",
        wrist=wrist, obj_label=obj,
        start_frame=start, end_frame=end,
        start_time=st, end_time=et,
        frames=frames_list
    )


# ====== Features por segmento ======
def build_frame_index(frames: List[FrameData]) -> Dict[int, FrameData]:
    return {f.frame_idx: f for f in frames}

def extract_segment_features(
    frames: List[FrameData],
    segments: List[MotionSegment]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extrae features simples:
    - duración (frames y tiempo)
    - velocidades media/max (por muñeca)
    - distancia media/min al centro del bbox más cercano del objeto (en frames del segmento)
    - proporción de frames "inside"
    - desplazamiento total muñeca (dx, dy, path_length)
    """
    idx = build_frame_index(frames)
    rows = []
    feat_cols = [
        "duration_frames", "duration_sec",
        "speed_mean", "speed_max",
        "path_len",
        "dist_center_mean", "dist_center_min",
        "inside_ratio",
        "dx_end_start", "dy_end_start"
    ]
    for seg in segments:
        if not seg.frames:
            continue
        fs = sorted(seg.frames)
        f0, f1 = fs[0], fs[-1]
        # recolecta puntos de la muñeca y bbox del objeto por frame
        pts: List[Tuple[float, float]] = []
        centers: List[Tuple[float, float]] = []
        inside_flags: List[int] = []

        for fr in fs:
            frd = idx.get(fr)
            if not frd: 
                continue
            # punto de la muñeca
            wp = None
            for h in frd.hands:
                if h.label == seg.wrist:
                    wp = (h.x, h.y)
                    break
            if not wp:
                continue
            # toma el bbox del objeto: si múltiples del mismo label, coge el más cercano a la muñeca
            cand = [d.bbox for d in frd.dets if d.label == seg.obj_label]
            if not cand:
                continue
            # el más cercano al punto de la muñeca
            dists = [point_to_rect_distance(wp[0], wp[1], bb) for bb in cand]
            bb = cand[int(np.argmin(dists))]
            c = bbox_center(bb)
            inside = 1 if point_in_bbox(wp[0], wp[1], bb) else 0

            pts.append(wp)
            centers.append(c)
            inside_flags.append(inside)

        if len(pts) < 2:
            continue

        # velocidades y path length
        speeds = []
        path_len = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            vx, vy, v = vel(a, b)
            speeds.append(v)
            path_len += v

        # distancias a centro bbox
        dists_center = [math.hypot(px - cx, py - cy) for (px, py), (cx, cy) in zip(pts, centers)]
        dist_center_mean = float(np.mean(dists_center)) if dists_center else float("nan")
        dist_center_min  = float(np.min(dists_center)) if dists_center else float("nan")

        # inside ratio
        inside_ratio = float(np.mean(inside_flags)) if inside_flags else 0.0

        # duración
        duration_frames = len(pts)
        duration_sec    = max(0.0, float(seg.end_time - seg.start_time)) if (seg.end_time >= 0 and seg.start_time >= 0) else float("nan")

        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]

        rows.append({
            "segment_id": seg.segment_id,
            "wrist": seg.wrist,
            "obj_label": seg.obj_label,
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "duration_frames": duration_frames,
            "duration_sec": duration_sec,
            "speed_mean": float(np.mean(speeds)) if speeds else 0.0,
            "speed_max":  float(np.max(speeds)) if speeds else 0.0,
            "path_len":   path_len,
            "dist_center_mean": dist_center_mean,
            "dist_center_min":  dist_center_min,
            "inside_ratio": inside_ratio,
            "dx_end_start": dx,
            "dy_end_start": dy,
        })

    df = pd.DataFrame(rows)
    return df, feat_cols


# ====== Clustering no supervisado ======
def choose_n_clusters(X: np.ndarray, method: str, max_k: int = 8, n_neighbors: int = 10) -> int:
    """Devuelve k por máxima silueta en 2..max_k."""
    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        if method == "kmeans":
            labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(X)
        else:
            # spectral con grafo k-NN sim
            nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(X)-1), metric="euclidean")
            nbrs.fit(X)
            A = nbrs.kneighbors_graph(X, mode="connectivity")
            labels = SpectralClustering(n_clusters=k, affinity='precomputed_nearest_neighbors',
                                        n_neighbors=min(n_neighbors, len(X)-1), random_state=42).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, metric="euclidean")
        if score > best_score:
            best_score, best_k = score, k
    return best_k

def cluster_segments(
    seg_df: pd.DataFrame,
    feature_cols: List[str],
    method: str = CLUSTER_METHOD,
    n_neighbors: int = N_NEIGHBORS,
    auto_n: bool = AUTO_N_CLUSTERS,
    max_k: int = MAX_CLUSTERS,
    fixed_k: int = FIXED_CLUSTERS,
    pca_var: float = PCA_VARIANCE
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if seg_df.empty:
        return seg_df.assign(cluster=-1), {"empty": True}

    X = seg_df[feature_cols].fillna(seg_df[feature_cols].mean()).to_numpy(dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=min(len(feature_cols), len(seg_df)-1), svd_solver="full")
    Xp = pca.fit_transform(Xs)
    # reduce hasta var explicada mínima
    if pca_var < 1.0:
        total = np.cumsum(pca.explained_variance_ratio_)
        kdim = int(np.searchsorted(total, pca_var) + 1)
        Xp = Xp[:, :max(2, kdim)]

    if auto_n:
        k = choose_n_clusters(Xp, method=method, max_k=max_k, n_neighbors=n_neighbors)
    else:
        k = max(2, fixed_k)

    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = model.fit_predict(Xp)
    else:
        # spectral con afinidad de k-NN
        labels = SpectralClustering(
            n_clusters=k, affinity='nearest_neighbors',
            n_neighbors=min(n_neighbors, max(2, len(Xp)-1)),
            assign_labels='kmeans', random_state=42
        ).fit_predict(Xp)

    out = seg_df.copy()
    out["cluster"] = labels

    info = {
        "method": method,
        "k": int(k),
        "n_segments": int(len(out)),
        "feature_cols": feature_cols,
        "explained_var": float(np.sum(pca.explained_variance_ratio_[:Xp.shape[1]])),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    return out, info


# ====== Supervisado (opcional) ======
def optional_supervised_training(seg_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    results = {}
    if not LABELS_CSV or not os.path.exists(LABELS_CSV):
        return results

    # Espera un CSV con columnas: segment_id,label
    labels_map = {}
    with open(LABELS_CSV, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            labels_map[r["segment_id"]] = r["label"]

    df = seg_df.copy()
    df["label"] = df["segment_id"].map(labels_map)
    df = df.dropna(subset=["label"])
    if df.empty:
        return results

    X = df[feature_cols].fillna(df[feature_cols].mean()).to_numpy(dtype=float)
    y = df["label"].astype(str).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # KNN clasificador
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])
    knn.fit(X_train, y_train)
    acc_knn = float(knn.score(X_test, y_test))
    results["knn_acc"] = acc_knn

    # SVM
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, gamma="scale"))
    ])
    svm.fit(X_train, y_train)
    acc_svm = float(svm.score(X_test, y_test))
    results["svm_acc"] = acc_svm

    if SAVE_MODELS:
        os.makedirs(MODELS_DIR, exist_ok=True)
        try:
            import joblib
            joblib.dump(knn, os.path.join(MODELS_DIR, "knn_motion.pkl"))
            joblib.dump(svm, os.path.join(MODELS_DIR, "svm_motion.pkl"))
        except Exception as e:
            print(f"[WARN] No se pudieron guardar modelos: {e}")

    return results


# ====== Persistencia en DB ======
def ensure_tables(db: Database):
    # Crea tablas si no existen (en el esquema configurado)
    with db.engine.connect() as conn:
        conn.execute(sql_text(f"""
        CREATE TABLE IF NOT EXISTS "{DB_SCHEMA}".motion_segments (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            video_analysis_id UUID NOT NULL,
            segment_id TEXT NOT NULL UNIQUE,
            wrist TEXT NOT NULL,
            obj_label TEXT NOT NULL,
            start_frame INT NOT NULL,
            end_frame   INT NOT NULL,
            start_time  DOUBLE PRECISION,
            end_time    DOUBLE PRECISION,
            features    JSONB,
            cluster     INT
        );
        """))
        conn.execute(sql_text(f"""
        CREATE TABLE IF NOT EXISTS "{DB_SCHEMA}".motion_clusters (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            video_analysis_id UUID NOT NULL,
            method TEXT,
            k INT,
            n_segments INT,
            info JSONB
        );
        """))
        conn.commit()

def save_segments_and_clusters(
    db: Database,
    video_id: str,
    seg_clusters: pd.DataFrame,
    cluster_info: Dict[str, Any],
    feature_cols: List[str]
):
    ensure_tables(db)

    # Inserta resumen clusters
    with db.engine.begin() as conn:
        conn.execute(sql_text(f"""
        INSERT INTO "{DB_SCHEMA}".motion_clusters (video_analysis_id, method, k, n_segments, info)
        VALUES (:vid, :method, :k, :n_segments, :info::jsonb)
        """), {
            "vid": video_id,
            "method": cluster_info.get("method"),
            "k": cluster_info.get("k"),
            "n_segments": cluster_info.get("n_segments"),
            "info": json.dumps(cluster_info)
        })

    # Inserta segmentos (upsert por segment_id)
    with db.engine.begin() as conn:
        for _, r in seg_clusters.iterrows():
            feats = r[feature_cols].to_dict()
            conn.execute(sql_text(f"""
            INSERT INTO "{DB_SCHEMA}".motion_segments
              (video_analysis_id, segment_id, wrist, obj_label, start_frame, end_frame, start_time, end_time, features, cluster)
            VALUES
              (:vid, :sid, :w, :obj, :sf, :ef, :st, :et, :feats::jsonb, :cl)
            ON CONFLICT (segment_id) DO UPDATE SET
              wrist = EXCLUDED.wrist,
              obj_label = EXCLUDED.obj_label,
              start_frame = EXCLUDED.start_frame,
              end_frame = EXCLUDED.end_frame,
              start_time = EXCLUDED.start_time,
              end_time = EXCLUDED.end_time,
              features = EXCLUDED.features,
              cluster = EXCLUDED.cluster
            """), {
                "vid": video_id,
                "sid": r["segment_id"],
                "w": r["wrist"],
                "obj": r["obj_label"],
                "sf": int(r["start_frame"]),
                "ef": int(r["end_frame"]),
                "st": float(r["start_time"]) if pd.notna(r["start_time"]) else None,
                "et": float(r["end_time"]) if pd.notna(r["end_time"]) else None,
                "feats": json.dumps(feats),
                "cl": int(r["cluster"]) if pd.notna(r["cluster"]) else None
            })


# ====== Export CSV ======
def export_csvs(video_id: str, seg_df: pd.DataFrame, clustered_df: pd.DataFrame):
    os.makedirs(OUT_DIR, exist_ok=True)
    seg_df.to_csv(os.path.join(OUT_DIR, f"{video_id}_segments_raw.csv"), index=False)
    clustered_df.to_csv(os.path.join(OUT_DIR, f"{video_id}_segments_clustered.csv"), index=False)


# ====== MAIN ======
def main():
    # DB
    db = Database(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
        sslmode="require", models_module="db_models", schema=DB_SCHEMA
    )

    video_id, yolo_objects, arms_pose = load_video_analysis(db)
    print(f"[INFO] VideoAnalysis: {video_id}  frames_yolo={len(yolo_objects)} frames_hands={len(arms_pose)}")

    frames = align_frames(yolo_objects, arms_pose)
    events = compute_interactions(frames, obj_labels=OBJECT_LABELS, wrist_labels=WRIST_LABELS, proximity_px=PROXIMITY_PX)
    print(f"[INFO] Interacciones detectadas: {len(events)}")

    segments = build_segments(frames, events, min_len=MIN_SEG_FRAMES, gap_tol=GAP_TOLERANCE)
    if not segments:
        print("[WARN] No se detectaron segmentos. Revisa PROXIMITY_PX/MIN_SEG_FRAMES/GAP_TOLERANCE/OBJECT_LABELS.")
        return

    # asigna video_id en segmentos
    for s in segments:
        s.video_id = video_id

    seg_df, feature_cols = extract_segment_features(frames, segments)
    if seg_df.empty:
        print("[WARN] No se pudieron extraer features (muy pocos puntos).")
        return

    clustered_df, info = cluster_segments(
        seg_df, feature_cols,
        method=CLUSTER_METHOD,
        n_neighbors=N_NEIGHBORS,
        auto_n=AUTO_N_CLUSTERS,
        max_k=MAX_CLUSTERS,
        fixed_k=FIXED_CLUSTERS,
        pca_var=PCA_VARIANCE
    )

    print(f"[CLUSTER] method={info.get('method')} k={info.get('k')} n={info.get('n_segments')} var≈{info.get('explained_var'):.2f}")
    print(clustered_df[["cluster","segment_id","wrist","obj_label","duration_frames","path_len","inside_ratio"]].groupby("cluster").size())

    # Supervisado opcional
    sup = optional_supervised_training(clustered_df, feature_cols)
    if sup:
        print(f"[SUP] Acc KNN={sup.get('knn_acc', float('nan')):.3f}  Acc SVM={sup.get('svm_acc', float('nan')):.3f}")

    # Persistencia
    if USE_DB:
        save_segments_and_clusters(db, video_id, clustered_df, info, feature_cols)
    if WRITE_CSV:
        export_csvs(video_id, seg_df, clustered_df)

    # “Motivos” recurrentes (top clusters)
    counts = Counter(clustered_df["cluster"])
    top = counts.most_common(3)
    print("[TOP] clusters más recurrentes:", top)
    for cl, n in top:
        dfc = clustered_df[clustered_df["cluster"] == cl].copy()
        rep = dfc.sort_values(["duration_frames","inside_ratio","path_len"], ascending=[False,False,False]).head(3)
        print(f"  - cluster {cl} (n={n}) ejemplos:\n", rep[["segment_id","wrist","obj_label","duration_frames","inside_ratio","path_len"]].to_string(index=False))

if __name__ == "__main__":
    main()
