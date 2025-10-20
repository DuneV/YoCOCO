# transaction.py

import os, tempfile, cv2, json
from yt_dlp import YoutubeDL

CONF = float(os.getenv("CONF_THRESHOLD", "0.25"))
FRAME_FPS = float(os.getenv("FRAME_SAMPLE_FPS", "1"))
MAX_HEIGHT = int(os.getenv("MAX_HEIGHT", "720"))

def is_youtube_url(url: str) -> bool:
    return any(url.lower().startswith(p) for p in (
        "https://www.youtube.com/", "http://www.youtube.com/",
        "https://youtu.be/", "http://youtu.be/"
    ))

def download_video_only(url: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)

    fmt = (
        f"bv*[ext=mp4][vcodec^=avc1][height<={MAX_HEIGHT}]/"
        f"bv*[ext=mp4][height<={MAX_HEIGHT}]/"
        f"bestvideo[height<={MAX_HEIGHT}]/"
        f"bestvideo"
    )

    ydl_opts = {
        "outtmpl": os.path.join(outdir, "%(id)s.%(ext)s"),
        "format": fmt,         
        "quiet": True,
        "noprogress": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        return path

def download_video(url: str, outdir: str) -> str:
    fmt = (
        f"bv*[vcodec^=avc1][height<={MAX_HEIGHT}]+ba/"
        f"b[height<={MAX_HEIGHT}]/"
        f"bv*[height<={MAX_HEIGHT}]+ba/"
        f"best[height<={MAX_HEIGHT}]"
    )

    os.makedirs(outdir, exist_ok=True) 

    ydl_opts = {
        "outtmpl": os.path.join(outdir, "%(id)s.%(ext)s"),
        "format": fmt,
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "quiet": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        if not path.lower().endswith(".mp4"):
            path = os.path.splitext(path)[0] + ".mp4"
        return path

VIDEO_PATH = "https://www.youtube.com/watch?v=M0jKmFfnlhE"

print(is_youtube_url(VIDEO_PATH))
download_video_only(VIDEO_PATH, 'data_temp')

