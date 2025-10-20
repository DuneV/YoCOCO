# youconnection.py
import os, time, math, requests, datetime as dt
from urllib.parse import urlencode
from psycopg2 import connect
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
PG_DSN  = os.getenv("PG_DSN") 

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

def iso8601_duration_to_seconds(s):
    h = m = sec = 0
    s = s.replace('PT','')
    num = ''
    for ch in s:
        if ch.isdigit(): num += ch
        else:
            if ch == 'H': h = int(num or 0)
            if ch == 'M': m = int(num or 0)
            if ch == 'S': sec = int(num or 0)
            num = ''
    return h*3600 + m*60 + sec

def youtube_search(query, published_after=None, max_items=50, duration="any"):
    items = []
    page_token = None
    while True:
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": min(50, max_items - len(items)),
            "key": API_KEY,
            "order": "date",
            "relevanceLanguage": "es",
            "safeSearch": "none",
            "videoDefinition": "high",       
            "videoDimension": "2d",         
            "videoEmbeddable": "true",
            "videoSyndicated": "true",
            "eventType": "completed",
        }
        if published_after:
            params["publishedAfter"] = published_after
        if duration in ("short","medium","long"):
            params["videoDuration"] = duration
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

def looks_like_short(snippet, duration_sec):
    if duration_sec <= 61:
        return True
    text = f"{snippet.get('title','')} {snippet.get('description','')}".lower()
    tags = [t.lower() for t in snippet.get('tags', [])] if snippet.get('tags') else []
    short_kw = ("#shorts", "#short", " shorts ", "[shorts]", " tiktok ", " reels ")
    if any(k in text for k in short_kw) or any("short" in t for t in tags):
        return True
    thumbs = snippet.get("thumbnails", {})
    best = max(thumbs.values(), key=lambda t: (t.get("width",0)*t.get("height",0)), default={})
    w, h = best.get("width", 0), best.get("height", 0)
    if h > w and (h - w) >= 80: 
        return True

    return False

def enrich_videos(video_ids):
    if not video_ids: return {}
    out = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        params = {
            "part": "contentDetails,statistics,snippet",  # puedes añadir recordingDetails si te sirve
            "id": ",".join(chunk),
            "key": API_KEY
        }
        r = requests.get(VIDEOS_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for it in data.get("items", []):
            vid   = it["id"]
            sn    = it["snippet"]
            cd    = it["contentDetails"]
            st    = it.get("statistics", {})
            dur   = iso8601_duration_to_seconds(cd.get("duration", "PT0S"))
            views = int(st.get("viewCount", 0))
            
            is_short = looks_like_short(sn, dur)

            definition = cd.get("definition")              # "hd" | "sd"
            dimension  = cd.get("dimension")               # "2d" | "3d"
            projection = cd.get("projection")              # "rectangular" | "360"
            live_flag  = sn.get("liveBroadcastContent")    # "none" | "live" | "upcoming"
            thumbs = sn.get("thumbnails", {})
            best_th = max(thumbs.values(), key=lambda t:(t.get("width",0)*t.get("height",0)), default={})
            w, h = best_th.get("width", 0), best_th.get("height", 0)
            is_vertical = h > w
            title_desc = f"{sn.get('title','')} {sn.get('description','')}".lower()
            bad_keywords = ("timelapse","time-lapse","compilation","montage","highlights","trailer","edit","music video","mv","shorts #shorts")
            good_keywords = ("tutorial","how to","demostración","demo","setup","pov","fixed camera","estático","estabilizado")
            has_bad  = any(k in title_desc for k in bad_keywords)
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

def store_results(dsn, query, published_after_iso, duration, max_results, search_items, enriched):
    with connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO aura.video_searches (query, published_after, duration_filter, max_results)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (query, published_after_iso, duration, max_results))
        search_id = cur.fetchone()[0]

        for it in search_items:
            vid = it["id"]["videoId"]
            sn  = it["snippet"]
            ext = enriched.get(vid, {})
            title = sn["title"]
            channel = sn["channelTitle"]
            published_at = sn["publishedAt"]
            url = f"https://www.youtube.com/watch?v={vid}"
            cur.execute("""
                INSERT INTO aura.video_results
                  (search_id, video_id, title, channel_title, published_at, duration_sec, view_count, url, raw)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (search_id, video_id) DO NOTHING
            """, (search_id, vid, title, channel, published_at,
                  ext.get("duration_sec"), ext.get("view_count"), url, Json(it)))
        return search_id

if __name__ == "__main__":
    query = os.getenv("YT_QUERY", "automatización con IA finanzas")
    
    published_after_iso = (dt.datetime.utcnow() - dt.timedelta(days=365)).isoformat("T") + "Z"
    duration = os.getenv("YT_DURATION", "any")  # any|short|medium|long
    max_results = int(os.getenv("YT_MAX", "25"))
    results = youtube_search(query, published_after_iso, max_results, duration)
    ids = [x["id"]["videoId"] for x in results]
    enriched = enrich_videos(ids)
    results_no_shorts = []
    for it in results:
        vid = it["id"]["videoId"]
        e = enriched.get(vid)
        if not e: 
            continue
        if e.get("is_short"):
            continue
        results_no_shorts.append(it)
    # sid = store_results(PG_DSN, query, published_after_iso, duration, max_results, results, enriched)
    print( "videos:", results)
