# youconnection.py
import os, time, requests, datetime as dt
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


def iso8601_duration_to_seconds(s: str) -> int:
    h = m = sec = 0
    s = s.replace('PT', '')
    num = ''
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            if ch == 'H': h = int(num or 0)
            if ch == 'M': m = int(num or 0)
            if ch == 'S': sec = int(num or 0)
            num = ''
    return h * 3600 + m * 60 + sec


def youtube_search(
    query: str,
    published_after: str | None = None,
    max_items: int = 50,
    duration: str = "any",              # any|short|medium|long
    order: str = "relevance",           # relevance|date|viewCount|rating|title|videoCount
    region_code: str | None = None,     # "CO","MX","ES","US", etc.
    relevance_language: str | None = None  # "es"|"en"|None
):
    items = []
    page_token = None
    while True:
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": min(50, max_items - len(items)),
            "key": API_KEY,
            "order": order,
            "safeSearch": "none",
            "videoDefinition": "high",
            "videoDimension": "2d",
            "videoEmbeddable": "true",
            "videoSyndicated": "true",
            "eventType": "completed",
        }
        if published_after:
            params["publishedAfter"] = published_after
        if duration in ("short", "medium", "long"):
            params["videoDuration"] = duration
        if region_code:
            params["regionCode"] = region_code
        if relevance_language:
            params["relevanceLanguage"] = relevance_language
        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items.extend(data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token or len(items) >= max_items:
            break
        time.sleep(0.1)

    return items[:max_items]


def looks_like_short(snippet, duration_sec: int) -> bool:
    if duration_sec <= 61:
        return True
    text = f"{snippet.get('title','')} {snippet.get('description','')}".lower()
    tags = [t.lower() for t in snippet.get('tags', [])] if snippet.get('tags') else []
    short_kw = ("#shorts", "#short", " shorts ", "[shorts]", " tiktok ", " reels ")
    if any(k in text for k in short_kw) or any("short" in t for t in tags):
        return True
    thumbs = snippet.get("thumbnails", {})
    best = max(thumbs.values(), key=lambda t: (t.get("width", 0) * t.get("height", 0)), default={})
    w, h = best.get("width", 0), best.get("height", 0)
    if h > w and (h - w) >= 80:
        return True
    return False


def enrich_videos(video_ids: list[str]) -> dict:
    if not video_ids:
        return {}
    out: dict[str, dict] = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        params = {
            "part": "contentDetails,statistics,snippet",
            "id": ",".join(chunk),
            "key": API_KEY
        }
        r = requests.get(VIDEOS_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for it in data.get("items", []):
            vid = it["id"]
            sn = it["snippet"]
            cd = it["contentDetails"]
            st = it.get("statistics", {})
            dur = iso8601_duration_to_seconds(cd.get("duration", "PT0S"))
            views = int(st.get("viewCount", 0))

            is_short = looks_like_short(sn, dur)
            definition = cd.get("definition")
            dimension = cd.get("dimension")
            projection = cd.get("projection")
            live_flag = sn.get("liveBroadcastContent")
            thumbs = sn.get("thumbnails", {})
            best_th = max(thumbs.values(), key=lambda t: (t.get("width", 0) * t.get("height", 0)), default={})
            w, h = best_th.get("width", 0), best_th.get("height", 0)
            is_vertical = h > w
            title_desc = f"{sn.get('title','')} {sn.get('description','')}".lower()
            bad_keywords = (
                "timelapse", "time-lapse", "compilation", "montage", "highlights",
                "trailer", "edit", "music video", "mv", "shorts #shorts"
            )
            good_keywords = (
                "tutorial", "how to", "demostración", "demo", "setup",
                "pov", "fixed camera", "estático", "estabilizado"
            )
            has_bad = any(k in title_desc for k in bad_keywords)
            has_good = any(k in title_desc for k in good_keywords)

            out[vid] = {
                "duration_sec": dur,
                "view_count": views,
                "snippet": sn,
                "definition": definition,
                "dimension": dimension,
                "projection": projection,
                "live": live_flag,
                "is_vertical": is_vertical,
                "has_bad_kw": has_bad,
                "has_good_kw": has_good,
                "is_short": is_short,
            }
        time.sleep(0.1)
    return out


if __name__ == "__main__":
    # ejemplo simple para validar la fecha aware (evita utcnow deprecado)
    published_after_iso = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=365)).isoformat()
    print("OK", published_after_iso)
