# main.py

from db.db import Database
from dotenv import load_dotenv
import os, json
import datetime as dt
from db_models import Base, VideoAnalysis
from youconnection import youtube_search, enrich_videos
from transaction import is_youtube_url, download_video_only, download_video
from yolo_labels import YOLOVideoDetectorLite, print_all_results as print_yolo
from media_lite import MediaPipeHandsArmsLite

YT_QUERY     = os.getenv("YT_QUERY", "personas operando manos visibles tutorial")
YT_DURATION  = os.getenv("YT_DURATION", "any")
YT_MAX       = int(os.getenv("YT_MAX", "20"))
YT_DAYS_BACK = int(os.getenv("YT_DAYS_BACK", "365"))
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

load_dotenv()

db_host = os.getenv("DATABASE_HOST")
db_port = os.getenv("PORT")
db_name = os.getenv("DATABASE_NAME")

db_username = os.getenv("DATABASE_USERNAME")
db_password = os.getenv("DATABASE_PASSWORD")


db = Database(
    host=db_host,
    port=db_port,
    database=db_name,
    user=db_username,
    password=db_password,
    sslmode="require",     
    models_module="db_models",
    schema="aura",
)
db.create_all()
os.makedirs(OUT_DIR, exist_ok=True)

def run_yolo(video_path):
    det = YOLOVideoDetectorLite(
        model_path=YOLO_MODEL,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        batch_size=YOLO_BATCH,
    )
    results, out_video = det.detect(
        video_path=video_path,
        frame_fps=None,
        bbox_format="xyxy",
        return_timestamps=True,
        save_video=True,
    )
    base = os.path.splitext(out_video or video_path)[0]
    out_json = base + "_detections.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return out_video, out_json

def run_mp(video_path):
    mpdet = MediaPipeHandsArmsLite(
        max_num_hands=MP_MAX_HANDS,
        min_detection_confidence=MP_DET_CONF,
        min_tracking_confidence=MP_TRK_CONF,
        draw=True
    )
    results, out_video = mpdet.detect(
        video_path=video_path,
        frame_fps=MP_FPS_OUT,
        save_video=True,
        return_timestamps=True
    )
    base = os.path.splitext(out_video or video_path)[0]
    out_json = base + "_hands.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return out_video, out_json