# main.py
import re, glob

from db.db import Database
from dotenv import load_dotenv
import os, json, re, time, datetime as dt
import cv2
from typing import List, Dict, Any, Tuple, Optional, Iterable
import math
from db_models import Base, VideoAnalysis
from youconnection import youtube_search, enrich_videos
from transaction import download_video_pytube
from yolo_labels import YOLOVideoDetectorLite
from media_lite import MediaPipeHandsArmsLite
from sqlalchemy import text as sql_text


load_dotenv()

OUT_DIR      = os.getenv("OUT_DIR", "data_temp")
WITH_AUDIO   = os.getenv("DOWNLOAD_WITH_AUDIO", "0") == "1"

YOLO_MODEL   = os.getenv("YOLO_MODEL", "models/yolov8n.pt")
YOLO_CONF    = float(os.getenv("CONF_THRESHOLD", "0.25"))
YOLO_IOU     = float(os.getenv("IOU_THRESHOLD", "0.45"))
YOLO_IMGSZ   = int(os.getenv("YOLO_IMGSZ", "640"))
YOLO_BATCH   = int(os.getenv("YOLO_BATCH", "8"))

MP_MAX_HANDS = int(os.getenv("MP_MAX_HANDS", "2"))
MP_DET_CONF  = float(os.getenv("MP_DET_CONF", "0.5"))
MP_TRK_CONF  = float(os.getenv("MP_TRK_CONF", "0.5"))
MP_FPS_OUT   = float(os.getenv("MP_FRAME_FPS", "10.0"))

USE_DB       = os.getenv("USE_DB", "1") == "1"
SKIP_ALREADY_PROCESSED = os.getenv("SKIP_ALREADY_PROCESSED", "1") == "1"

CLEAN_DET_CONF_MIN = float(os.getenv("CLEAN_DET_CONF_MIN", "0.30"))
DET_KEEP_LABELS    = set(json.loads(os.getenv("DET_KEEP_LABELS", '["person"]')))
HAND_KEEP_LABELS   = set(json.loads(os.getenv("HAND_KEEP_LABELS", '["Left","Right"]')))
HAND_REQUIRED_POINTS = tuple(json.loads(os.getenv("HAND_REQUIRED_POINTS", '["wrist","index_tip"]')))
DROP_EMPTY         = os.getenv("DROP_EMPTY", "1") == "1"
FRAME_SIZE_DEFAULT = tuple(map(int, os.getenv("FRAME_SIZE_DEFAULT", "1280,720").split(",")))  # "1280,720"
CLEAN_AFTER_SAVE = os.getenv("CLEAN_AFTER_SAVE", "1") == "1"
CLEAN_ON_SKIP    = os.getenv("CLEAN_ON_SKIP", "1") == "1"


GATE_ENABLE   = os.getenv("GATE_ENABLE", "1") == "1"  
MIN_DUR       = int(os.getenv("MIN_DUR_SEC", "15"))
MAX_DUR       = int(os.getenv("MAX_DUR_SEC", "300"))
ALLOW_VERTICAL= os.getenv("ALLOW_VERTICAL", "0") == "1"
RELEVANT_KW   = [k.strip() for k in os.getenv(
    "RELEVANT_KW",
    "cocina,cocinar,kitchen,coffee,café,barista,espresso,pour over,planchar,iron,laundry,ropa,organización,organize,fold clothes"
).split(",") if k.strip()]
BAD_KW_EXTRA  = [k.strip() for k in os.getenv(
    "BAD_KW",
    "timelapse,time-lapse,compilation,highlights,trailer,edit,music video,mv"
).split(",") if k.strip()]

KEEP_ANNOTATED = os.getenv("KEEP_ANNOTATED", "0") == "1"  

MAX_VIDEOS_PER_RUN = int(os.getenv("MAX_VIDEOS_PER_RUN", "999999"))
SLEEP_BETWEEN      = float(os.getenv("SLEEP_BETWEEN", "0.0"))

YT_QUERY     = os.getenv("YT_QUERY", "cocina barista espresso planchar ropa organización manos tutorial")
YT_DURATION  = os.getenv("YT_DURATION", "any")  # any|short|medium|long
YT_MAX       = int(os.getenv("YT_MAX", "40"))
YT_DAYS_BACK = int(os.getenv("YT_DAYS_BACK", "365"))


db_host = os.getenv("DATABASE_HOST")
db_port = os.getenv("DATABASE_PORT") or os.getenv("PORT")
db_name = os.getenv("DATABASE_NAME")
db_username = os.getenv("DATABASE_USERNAME")
db_password = os.getenv("DATABASE_PASSWORD")

db = Database(
    host=db_host,
    port=int(db_port) if db_port else None,
    database=db_name,
    user=db_username,
    password=db_password,
    sslmode="require",
    models_module="db_models",
    schema="aura_info",
)
db.create_all()
os.makedirs(OUT_DIR, exist_ok=True)
_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_\-]{6,})")

def youtube_id_from_url(url: str) -> str | None:
    m = _YT_ID_RE.search(url or "")
    return m.group(1) if m else None

def cleanup_by_video_id(video_id: str):
    """Borra TODO lo que empiece por el ID en OUT_DIR: mp4, json, .part, etc."""
    pattern = os.path.join(OUT_DIR, f"{video_id}*")
    for p in glob.glob(pattern):
        try:
            os.remove(p)
            print(f"[CLEAN] {p}")
        except Exception as e:
            print(f"[WARN] No se pudo borrar {p}: {e}")

def cleanup_by_filename(filename: str):
    """Borra derivados por prefijo (id = nombre sin extensión)."""
    base = os.path.splitext(filename)[0]
    cleanup_by_video_id(base)

def cleanup_from_db_catalog(db):
    print("[CLEANUP] Inicial por DB…")
    with db.engine.connect() as conn:
        rows = conn.execute(sql_text(f'SELECT video_name FROM "{db.schema}".video_analyses')).all()
    for (fname,) in rows:
        if fname:
            cleanup_by_filename(fname)


_YT_RE = re.compile(r"(?:v=|/v/|youtu\.be/|shorts/)([A-Za-z0-9_\-]{6,})")

def youtube_id_from_url(url: str) -> Optional[str]:
    m = _YT_RE.search(url)
    return m.group(1) if m else None

def build_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"

def get_frame_size(path: str) -> Tuple[int, int]:
    try:
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        pass
    return FRAME_SIZE_DEFAULT

def safe_unlink(*paths: Optional[str]):
    for p in paths:
        if not p: 
            continue
        try:
            if os.path.exists(p):
                os.remove(p)
                print(f"[CLEAN] borrado: {p}")
        except Exception as e:
            print(f"[WARN] No se pudo borrar {p}: {e}")

def already_processed(url: str) -> bool:
    if not SKIP_ALREADY_PROCESSED:
        return False
    q = sql_text("SELECT 1 FROM aura_info.video_analyses WHERE summary->>'url' = :u LIMIT 1")
    with db.engine.connect() as conn:
        r = conn.execute(q, {"u": url}).first()
        if r is not None and CLEAN_ON_SKIP:
            vid = youtube_id_from_url(url)
            if vid:
                cleanup_by_video_id(vid)
        return r is not None


import copy
from typing import Iterable

def _point_inside(p: Dict[str, float], frame_size: Tuple[int, int]) -> bool:
    W, H = frame_size
    return isinstance(p, dict) and 0 <= p.get("x", -1) < W and 0 <= p.get("y", -1) < H

def _bbox_inside_frame(bbox: Iterable[float], frame_size: Tuple[int, int]) -> bool:
    try: x1, y1, x2, y2 = bbox
    except Exception: return False
    W, H = frame_size
    return 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H

def _valid_detection(d: Dict[str, Any], frame_size: Tuple[int, int],
                     keep_labels: Optional[set], conf_min: float) -> bool:
    
    if d.get("conf", 0.0) < conf_min:
        return False

    if keep_labels:
        if d.get("label") not in keep_labels:
            return False

    return _bbox_inside_frame(d.get("bbox", []), frame_size)

def point_bbox_collision(px, py, bbox):
    """Devuelve True si un punto (px, py) está dentro de un bbox."""
    try:
        x1, y1, x2, y2 = bbox
        return (x1 <= px <= x2) and (y1 <= py <= y2)
    except:
        return False

def point_to_bbox_distance(px, py, bbox):
    """Distancia mínima del punto al bounding box."""
    x1, y1, x2, y2 = bbox
    
    if point_bbox_collision(px, py, bbox):
        return 0.0
    
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.sqrt(dx*dx + dy*dy)

def find_closest_object(px, py, detections):
    """Encuentra el objeto más cercano a un punto."""
    min_d = float("inf")
    closest = None
    for obj in detections:
        d = point_to_bbox_distance(px, py, obj.get("bbox", []))
        if d < min_d:
            min_d = d
            closest = obj
    return closest, min_d

def is_grasping(wrist, index_tip, obj_bbox, grasp_threshold=15):
    """
    Detecta un 'agarre' simple:
    - wrist cerca del objeto
    - index_tip cerca del objeto
    - ambos en lados cercanos (como presionando)
    """
    wx, wy = wrist
    ix, iy = index_tip

    dw = point_to_bbox_distance(wx, wy, obj_bbox)
    di = point_to_bbox_distance(ix, iy, obj_bbox)

    return dw < grasp_threshold and di < grasp_threshold

def grasp_probability(wrist, index_tip, obj_bbox, max_d=80):
    """
    Probabilidad simple de agarre: 1 = muy probable, 0 = nada.
    Basado en distancia normalizada al objeto.
    """
    wx, wy = wrist
    ix, iy = index_tip

    dw = point_to_bbox_distance(wx, wy, obj_bbox)
    di = point_to_bbox_distance(ix, iy, obj_bbox)

    score = 1 - min((dw + di) / (2 * max_d), 1.0)
    return round(score, 3)

def is_touching(distance, threshold=10):
    """Determina si una mano está 'tocando' un objeto (por distancia)."""
    return distance <= threshold

def _valid_hand(h: Dict[str, Any], frame_size: Tuple[int, int], required_points: Iterable[str], keep_labels: Optional[set]) -> bool:
    if keep_labels is not None and h.get("label") not in keep_labels: return False
    for k in required_points:
        if k not in h or not _point_inside(h[k], frame_size): return False
    return True

def filter_frame(
    frame: Dict[str, Any],
    frame_size: Tuple[int, int],
    det_conf_min: float = CLEAN_DET_CONF_MIN,
    det_keep_labels: Optional[set] = None,
    hand_required_points: Iterable[str] = HAND_REQUIRED_POINTS,
    hand_keep_labels: Optional[set] = HAND_KEEP_LABELS,
    drop_empty: bool = DROP_EMPTY,
) -> Optional[Dict[str, Any]]:
    out = copy.deepcopy(frame)

    if "detections" in frame:
        dets = frame.get("detections", [])
        det_keep_labels = det_keep_labels if det_keep_labels is not None else DET_KEEP_LABELS
        out["detections"] = [
            d for d in dets
            if _valid_detection(d, frame_size, det_keep_labels, det_conf_min)
        ]

    if "hands" in frame:
        hands = frame.get("hands", [])
        out["hands"] = [
            h for h in hands
            if _valid_hand(h, frame_size, hand_required_points, hand_keep_labels)
        ]
    dets = out.get("detections", [])
    hands = out.get("hands", [])

    # Aquí se guardan las interacciones
    out["interactions"] = []

    for h in hands:

        hand_label = h.get("label")             # “Left” / “Right”
        wrist = h.get("wrist", {})
        idx   = h.get("index_tip", {})

        if "x" not in wrist or "x" not in idx:
            continue

        wx, wy = wrist["x"], wrist["y"]
        ix, iy = idx["x"], idx["y"]

        closest_obj_idx, dist_idx = find_closest_object(ix, iy, dets)

        closest_obj_wrist, dist_wrist = find_closest_object(wx, wy, dets)

        if closest_obj_idx and closest_obj_wrist:
            if dist_idx < dist_wrist:
                best_obj = closest_obj_idx
                best_dist = dist_idx
                interaction_point = "index_tip"
            else:
                best_obj = closest_obj_wrist
                best_dist = dist_wrist
                interaction_point = "wrist"
        else:
            continue

        touching = is_touching(best_dist, threshold=20)
        grasping = is_grasping((wx,wy), (ix,iy), best_obj["bbox"])
        grasp_prob = grasp_probability((wx,wy), (ix,iy), best_obj["bbox"])
        
        out["interactions"].append({
            "hand": hand_label,
            "interaction_point": interaction_point,
            "object_label": best_obj.get("label"),
            "distance_px": best_dist,
            "touching": touching,
            "grasping": grasping,
            "grasp_probability": grasp_prob,
            "wrist_xy": (wx, wy),
            "index_xy": (ix, iy),
            "object_bbox": best_obj["bbox"],
        })
    if drop_empty:
        has_det = "detections" in frame and bool(out.get("detections"))
        has_hnd = "hands" in frame and bool(out.get("hands"))
        return out if (has_det or has_hnd) else None

    return out

def filter_stream(frames: List[Dict[str, Any]], frame_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    result = []
    for fr in frames:
        f = filter_frame(fr, frame_size=frame_size)
        if f is not None:
            result.append(f)
    return result

def is_relevant_by_text(text: str) -> bool:
    t = (text or "").lower()
    if any(bad in t for bad in BAD_KW_EXTRA):
        return False
    return any(g in t for g in (kw.lower() for kw in RELEVANT_KW))

def passes_gates(info: dict) -> bool:
    if not GATE_ENABLE:
        return True
    if info.get("is_short"):                           # shorts
        return False
    if (not ALLOW_VERTICAL) and info.get("is_vertical"):
        return False
    dur = int(info.get("duration_sec") or 0)
    if dur < MIN_DUR or dur > MAX_DUR:
        return False
    sn = info.get("snippet", {}) or {}
    text = f"{sn.get('title','')} {sn.get('description','')}"
    if info.get("has_bad_kw") or not is_relevant_by_text(text):
        return False
    return True

def run_yolo(video_path: str, fps_target: Optional[float]) -> Tuple[Optional[str], str]:
    det = YOLOVideoDetectorLite(
        model_path=YOLO_MODEL,
        conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ, batch_size=YOLO_BATCH,
    )
    results, out_video = det.detect(
        video_path=video_path,
        frame_fps=fps_target,
        bbox_format="xyxy",
        return_timestamps=True,
        save_video=True,
    )
    base = os.path.splitext(out_video or video_path)[0]
    out_json = base + "_detections.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return out_video, out_json

def run_mp(video_path: str, fps_target: Optional[float]) -> Tuple[Optional[str], str]:
    mpdet = MediaPipeHandsArmsLite(
        max_num_hands=MP_MAX_HANDS,
        min_detection_confidence=MP_DET_CONF,
        min_tracking_confidence=MP_TRK_CONF,
        draw=True
    )
    results, out_video = mpdet.detect(
        video_path=video_path,
        frame_fps=fps_target,
        save_video=True,
        return_timestamps=True
    )
    base = os.path.splitext(out_video or video_path)[0]
    out_json = base + "_hands.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return out_video, out_json

def process_one(url: str, enriched_info: Optional[dict] = None):
    # 1) Evitar reprocesar si ya está en la DB
    if SKIP_ALREADY_PROCESSED and already_processed(url):
        print(f"[SKIP] Ya procesado: {url}")
        return

    # 2) Descargar video con pytubefix
    path = download_video_pytube(url, OUT_DIR)
    if not path:
        print(f"[SKIP] No se pudo descargar el video: {url}")
        return

    base_name = os.path.basename(path)
    print(f"[RUN] Procesando archivo: {base_name}")

    # 3) Correr YOLO + MediaPipe
    target_fps = MP_FPS_OUT

    yolo_video, yolo_json = run_yolo(path, target_fps)
    mp_video,   mp_json   = run_mp(path,  target_fps)

    # 4) Cargar resultados brutos
    with open(yolo_json, "r", encoding="utf-8") as f:
        yolo_res = json.load(f)
    with open(mp_json, "r", encoding="utf-8") as f:
        mp_res = json.load(f)

    # 5) Limpiar detecciones / manos
    frame_size = get_frame_size(path)
    yolo_clean = filter_stream(yolo_res, frame_size=frame_size)
    mp_clean   = filter_stream(mp_res,   frame_size=frame_size)

    def last_t(arr): 
        return (arr[-1].get("time_sec") if arr else 0) or 0

    frame_count  = min(len(yolo_clean), len(mp_clean))
    duration_sec = int(round(max(last_t(yolo_clean), last_t(mp_clean))))

    # 6) Guardar en DB
    if USE_DB:
        with db.session() as s:
            va = VideoAnalysis(
                video_name=base_name,
                frame_fps=int(target_fps) if target_fps else None,
                duration_sec=duration_sec,
                frame_count=frame_count,
                yolo_objects=yolo_clean,
                arms_pose=mp_clean,
                summary={
                    "source": "youtube",
                    "url": url,
                    "notes": "YOLO+MediaPipe aligned; cleaned"
                },
                meta={
                    "yolo": {
                        "model": YOLO_MODEL,
                        "conf": YOLO_CONF,
                        "iou": YOLO_IOU,
                        "imgsz": YOLO_IMGSZ,
                        "batch": YOLO_BATCH
                    },
                    "mediapipe": {
                        "max_hands": MP_MAX_HANDS,
                        "det_conf": MP_DET_CONF,
                        "trk_conf": MP_TRK_CONF,
                        "fps_out": MP_FPS_OUT
                    },
                    "filters": {
                        "clean_det_conf_min": CLEAN_DET_CONF_MIN,
                        "det_keep_labels": list(DET_KEEP_LABELS),
                        "hand_keep_labels": list(HAND_KEEP_LABELS),
                        "hand_required_points": list(HAND_REQUIRED_POINTS),
                        "drop_empty": DROP_EMPTY
                    },
                    "metadata_gate": enriched_info or {}
                }
            )
            s.add(va)

    # 7) Limpieza de archivos temporales
    if KEEP_ANNOTATED:
        safe_unlink(path, yolo_json, mp_json)
    else:
        safe_unlink(path, yolo_json, mp_json, yolo_video, mp_video)

    print(f"[OK] Procesado y guardado: {url}")



def process_ids(video_ids: List[str]):
    if not video_ids:
        print("[INFO] Lista de IDs vacía.")
        return
    
    enriched = enrich_videos(video_ids) or {}

    accepted: List[str] = []
    for vid in video_ids:
        info = enriched.get(vid)
        if info is None:
            print(f"[GATE] sin metadata -> skip: {vid}")
            continue
        if passes_gates(info):
            accepted.append(vid)
        else:
            print(f"[GATE] no pasa filtros -> skip: {vid}")

    total = 0
    for vid in accepted:
        if total >= MAX_VIDEOS_PER_RUN:
            print(f"[STOP] Alcanzado MAX_VIDEOS_PER_RUN={MAX_VIDEOS_PER_RUN}")
            break
        url = build_url(vid)
        try:
            process_one(url, enriched_info=enriched.get(vid))
            total += 1
            if SLEEP_BETWEEN > 0:
                time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            print(f"[ERROR] Falló {url}: {e}")

    print(f"[DONE] Procesados {total}/{len(accepted)} aceptados (de {len(video_ids)} IDs)")


def collect_video_ids_from_queries(
    queries: List[str],
    days_back: int = YT_DAYS_BACK,
    max_items_per_query: int = YT_MAX,
    duration: str = YT_DURATION,
) -> List[str]:
    published_after_iso = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_back)).isoformat()
    all_ids: List[str] = []
    for q in queries:
        items = youtube_search(q, published_after_iso, max_items_per_query, duration)
        ids = [it["id"]["videoId"] for it in items if "id" in it and "videoId" in it["id"]]
        all_ids.extend(ids)
    seen, dedup = set(), []
    for x in all_ids:
        if x not in seen:
            dedup.append(x); seen.add(x)
    return dedup

def collect_ids_one_attempt(queries, days_back, max_items_per_query, duration, order, region, lang):
    """
    Ahora ignora region/lang/days porque SCRAPING NO LO SOPORTA.
    Pero mantiene la firma para tu pipeline.
    """
    all_ids = []
    for q in queries:
        items = youtube_search(
            q,
            max_items=max_items_per_query,
            order=order,
            duration=duration,
        )
        ids = [it["id"]["videoId"] for it in items if "id" in it and "videoId" in it["id"]]
        all_ids.extend(ids)

    # Dedup
    seen, dedup = set(), []
    for x in all_ids:
        if x not in seen:
            dedup.append(x)
            seen.add(x)

    print(f"[IDS] Recolectados {len(dedup)} IDs (sin API)")
    return dedup


def batch_enrich(ids):
    """
    Procesa enrichment en lotes de 50 IDs.
    Devuelve un dict con metadata para TODOS los videos que responda la API.
    """
    enriched_total = {}

    chunk_size = 50
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i+chunk_size]

        meta = enrich_videos(chunk) or {}
        enriched_total.update(meta)

        print(f"[ENRICH] Batch {i//chunk_size + 1}: "
              f"{len(chunk)} IDs → {len(meta)} enriquecidos")

    return enriched_total

def passes_gates_relaxed(info, level):
    """
    level 0 = estricto
    level 1 = medio (acepta sin metadata)
    level 2 = relajado (casi todo pasa)
    """

    if not info:
        return level >= 1  

    if level == 0:
        return passes_gates(info)

    dur = int(info.get("duration_sec", 0))
    if dur < 5 or dur > 3600:
        return False

    return True



def smart_query_variants(base_queries):
    """Pequeños cambios de wording ES/EN y sinónimos (> aumenta recall)."""
    extras = [
        "tutorial", "cómo", "how to", "POV", "cámara fija", "fixed camera",
        "plano cenital", "top down", "hands visible", "manos visibles",
        "paso a paso", "step by step", "slow", "sin música", "asmr"
    ]
    variants = []
    for q in base_queries:
        variants.append(q)
        for w in extras:
            variants.append(f"{q} {w}")
    # dedup manteniendo orden
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def relax_gates_level(info, level):
    """Level 0: strict (tus gates actuales); 1: amplio; 2: casi sin gate."""
    if level >= 2:
        return True  # procesa todo
    # Copiamos tu lógica de passes_gates pero la aflojamos por 'level'
    if info.get("is_short") and level == 0:
        return False
    if (not ALLOW_VERTICAL) and info.get("is_vertical") and level == 0:
        return False
    dur = int(info.get("duration_sec") or 0)
    lo = MIN_DUR if level == 0 else max(5, MIN_DUR // 3)
    hi = MAX_DUR if level == 0 else max(MAX_DUR, 3600)
    if dur < lo or dur > hi:
        return False
    # texto relevante
    sn = info.get("snippet", {}) or {}
    text = f"{sn.get('title','')} {sn.get('description','')}"
    # en level 1 ignoramos BAD_KW extra y pedimos match laxo
    if level == 0:
        if info.get("has_bad_kw"): return False
        return is_relevant_by_text(text)
    else:
        # basta con que tenga una señal leve
        return True

def process_until_target_with_small_changes(
    base_queries,
    target_accepts=50,
    max_attempts=12,
    per_query_limit=25,
):
    day_windows = [90, 365, 1825]
    durations = ["medium", "any", "long", "short"]
    orders = ["relevance", "date", "viewCount"]
    regions = [None, "CO", "MX", "ES", "US"]
    langs = ["es", "en", None]
    gate_levels = [0, 1, 2]

    queries = smart_query_variants(base_queries)
    processed_this_run = 0
    seen_ids = set()

    attempt = 0

    for glevel in gate_levels:
        print(f"\n===== GATE LEVEL {glevel} =====")

        for days in day_windows:
            for dur in durations:
                for ord_ in orders:
                    for reg in regions:
                        for lang in langs:

                            if processed_this_run >= target_accepts:
                                break
                            if attempt >= max_attempts:
                                break

                            attempt += 1
                            print(f"\n--- Attempt {attempt}/{max_attempts} ---")
                            print(f"Config: days={days}, dur={dur}, order={ord_}, region={reg}, lang={lang}")

                            # 1) Recolectar IDs
                            ids = collect_ids_one_attempt(
                                queries, days, per_query_limit, dur, ord_, reg, lang
                            )

                            # 2) Filtrar los que ya vimos
                            ids = [i for i in ids if i not in seen_ids]
                            seen_ids.update(ids)

                            if not ids:
                                print("[SKIP] No hay nuevos IDs.")
                                continue

                            print(f"[NEW IDS] {len(ids)} nuevos IDs.")

                            # 3) Enrichment MASIVO
                            enriched = batch_enrich(ids)

                            # 4) Gate relajado
                            accepted = [
                                vid for vid in ids
                                if passes_gates_relaxed(enriched.get(vid), glevel)
                            ]

                            print(f"[GATE] Aceptados: {len(accepted)} / {len(ids)}")

                            # 5) Procesar videos aceptados
                            for vid in accepted:
                                if processed_this_run >= target_accepts:
                                    break

                                url = build_url(vid)
                                try:
                                    process_one(url, enriched_info=enriched.get(vid))
                                    processed_this_run += 1
                                    print(f"[OK] Procesados: {processed_this_run}/{target_accepts}")

                                except Exception as e:
                                    print(f"[ERROR] {url}: {e}")

                        if processed_this_run >= target_accepts:
                            break
                    if processed_this_run >= target_accepts:
                        break
                if processed_this_run >= target_accepts:
                    break
            if processed_this_run >= target_accepts:
                break

    print(f"\n[SMART] Procesados {processed_this_run} videos tras {attempt} intentos.")

# ===== main.py (añade debajo de collect_video_ids_from_queries o en cualquier parte util) =====

def collect_ids_one_attempt(queries, days_back, max_items_per_query, duration, order, region, lang):
    published_after_iso = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_back)).isoformat()
    all_ids = []
    for q in queries:
        items = youtube_search(
            q, published_after_iso, max_items_per_query,
            duration=duration, order=order, region_code=region, relevance_language=lang
        )
        ids = [it["id"]["videoId"] for it in items if "id" in it and "videoId" in it["id"]]
        all_ids.extend(ids)
    # dedup preservando orden
    seen, dedup = set(), []
    for x in all_ids:
        if x not in seen:
            dedup.append(x); seen.add(x)
    return dedup

def smart_query_variants(base_queries):
    """Pequeños cambios de wording ES/EN y sinónimos (> aumenta recall)."""
    extras = [
        "tutorial", "cómo", "how to", "POV", "cámara fija", "fixed camera",
        "plano cenital", "top down", "hands visible", "manos visibles",
        "paso a paso", "step by step", "slow", "sin música", "asmr"
    ]
    variants = []
    for q in base_queries:
        variants.append(q)
        for w in extras:
            variants.append(f"{q} {w}")
    # dedup manteniendo orden
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def relax_gates_level(info, level):
    """Level 0: strict (tus gates actuales); 1: amplio; 2: casi sin gate."""
    if level >= 2:
        return True  # procesa todo
    # Copiamos tu lógica de passes_gates pero la aflojamos por 'level'
    if info.get("is_short") and level == 0:
        return False
    if (not ALLOW_VERTICAL) and info.get("is_vertical") and level == 0:
        return False
    dur = int(info.get("duration_sec") or 0)
    lo = MIN_DUR if level == 0 else max(5, MIN_DUR // 3)
    hi = MAX_DUR if level == 0 else max(MAX_DUR, 3600)
    if dur < lo or dur > hi:
        return False
    # texto relevante
    sn = info.get("snippet", {}) or {}
    text = f"{sn.get('title','')} {sn.get('description','')}"
    # en level 1 ignoramos BAD_KW extra y pedimos match laxo
    if level == 0:
        if info.get("has_bad_kw"): return False
        return is_relevant_by_text(text)
    else:
        # basta con que tenga una señal leve
        return True

def process_until_target_with_small_changes(
    base_queries,
    target_accepts=50,
    max_attempts=12,
    per_query_limit=25
):
    """Sigue buscando con pequeños cambios hasta alcanzar 'target_accepts' procesados."""
    # palancas pequeñas que vamos rotando:
    day_windows = [90, 365, 1825]                        # 3 meses, 1 año, 5 años
    durations   = ["medium", "any", "long", "short"]     # prueba primero "medium"
    orders      = ["relevance", "date", "viewCount"]     # cambia orden
    regions     = [None, "CO", "MX", "ES", "US"]         # región
    langs       = ["es", "en", None]                     # idioma de relevancia
    gate_levels = [0, 1, 2]                              # relaja en el tiempo

    queries = smart_query_variants(base_queries)
    processed_this_run = 0
    seen_ids = set()

    # Enriquecimiento por lotes y gating progresivo
    attempt = 0
    for glevel in gate_levels:
        for days in day_windows:
            for dur in durations:
                for ord_ in orders:
                    for reg in regions:
                        for lang in langs:
                            if attempt >= max_attempts or processed_this_run >= target_accepts:
                                break
                            attempt += 1
                            ids = collect_ids_one_attempt(
                                queries, days, per_query_limit, dur, ord_, reg, lang
                            )
                            # quita los ya vistos en esta corrida
                            ids = [i for i in ids if i not in seen_ids]
                            seen_ids.update(ids)

                            if not ids:
                                continue

                            # enriquecemos y aplicamos gate relajado
                            enriched = enrich_videos(ids) or {}
                            accepted = []
                            for vid in ids:
                                info = enriched.get(vid)
                                if not info:
                                    continue
                                if relax_gates_level(info, glevel):
                                    accepted.append(vid)

                            # procesa en orden
                            for vid in accepted:
                                if processed_this_run >= target_accepts:
                                    break
                                url = build_url(vid)
                                try:
                                    process_one(url, enriched_info=enriched.get(vid))
                                    processed_this_run += 1
                                    if SLEEP_BETWEEN > 0:
                                        time.sleep(SLEEP_BETWEEN)
                                except Exception as e:
                                    print(f"[ERROR] Falló {url}: {e}")

                        if attempt >= max_attempts or processed_this_run >= target_accepts:
                            break
                    if attempt >= max_attempts or processed_this_run >= target_accepts:
                        break
                if attempt >= max_attempts or processed_this_run >= target_accepts:
                    break
            if attempt >= max_attempts or processed_this_run >= target_accepts:
                break

    print(f"[SMART] Procesados {processed_this_run} videos tras {attempt} intentos pequeños.")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    db.create_all()

    if os.getenv("CLEANUP_ON_START", "0") == "1":
        cleanup_from_db_catalog(db)

    ids_env = os.getenv("YT_IDS", "").strip()
    if ids_env:
        ids = [x.strip() for x in ids_env.split(",") if x.strip()]
        print(f"[RUN] IDs recibidos: {len(ids)}")
        process_ids(ids)
    else:
        topics_env = os.getenv(
            "YT_TOPICS",
            "cocina manos, barista espresso, planchar ropa, organización de ropa, fold clothes kitchen"
        )
        base_queries = [q.strip() for q in topics_env.split(",") if q.strip()]

        target_accepts = int(os.getenv("TARGET_ACCEPTS", "50"))
        max_attempts   = int(os.getenv("MAX_ATTEMPTS", "12"))
        per_query_lim  = int(os.getenv("PER_QUERY_LIMIT", "25"))

        process_until_target_with_small_changes(
            base_queries,
            target_accepts=target_accepts,
            max_attempts=max_attempts,
            per_query_limit=per_query_lim
        )




