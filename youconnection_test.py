# ============================================================
#   YOUTUBE SEARCH + METADATA (SIN API, SOLO yt-dlp)
# ============================================================
import time
from yt_dlp import YoutubeDL

# ------------------------------------------------------------
# Buscar videos sin API (scraping oficial de yt-dlp)
# ------------------------------------------------------------
def youtube_search(query: str, max_items=25, order="relevance", duration="any"):
    """
    Realiza una búsqueda en YouTube SIN API usando yt-dlp.
    Devuelve una lista de items con id, título, duración, etc.
    """

    # duration mapping
    duration_map = {
        "any": "",
        "short": "short",
        "medium": "medium",
        "long": "long",
    }
    dur_kw = duration_map.get(duration, "")

    # La sintaxis de yt-dlp:
    #   ytsearchN:<query>
    #   ytsearchdateN:<query> -> orden por fecha
    #   ytsearchN:<query> <keywords>
    if order == "date":
        search_expr = f"ytsearchdate{max_items}:{query}"
    else:
        search_expr = f"ytsearch{max_items}:{query}"

    # Si hay filtro de duración, lo agregamos como keyword
    if dur_kw:
        search_expr = f"{search_expr} {dur_kw}"

    ydl_opts = {
        "quiet": True,
        "extract_flat": False,
        "extractor_args": {
            "youtube": {
                "player_client": ["default"]
            }
        }
    }

    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(search_expr, download=False)
        entries = result.get("entries", [])

    items = []
    for e in entries:
        if not e or not e.get("id"):
            continue

        # normalizamos formato estilo API
        items.append({
            "id": {"videoId": e["id"]},
            "snippet": {
                "title": e.get("title"),
                "description": e.get("description"),
                "thumbnails": e.get("thumbnails"),
                "tags": e.get("tags", []),
                "channelTitle": e.get("channel"),
            },
            "duration_sec": e.get("duration", 0),
            "is_short": looks_like_short_ydl(e),
            "is_vertical": looks_like_vertical_ydl(e),
        })

    return items[:max_items]


# ------------------------------------------------------------
# Helpers para identificar shorts y vertical
# ------------------------------------------------------------
def looks_like_short_ydl(info):
    dur = info.get("duration") or 0
    if dur <= 61:
        return True
    title = (info.get("title") or "").lower()
    if "#shorts" in title or "short" in title:
        return True
    return False


def looks_like_vertical_ydl(info):
    w = info.get("width") or 0
    h = info.get("height") or 0
    return h > w


def enrich_videos(video_ids: list[str]) -> dict:
    """
    Extrae metadata completa de cada video usando yt-dlp.
    Reemplaza por completo la API oficial.
    """
    out = {}
    ydl_opts = {
        "quiet": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["default"]
            }
        }
    }
    
    with YoutubeDL(ydl_opts) as ydl:
        for vid in video_ids:
            try:
                url = f"https://www.youtube.com/watch?v={vid}"
                info = ydl.extract_info(url, download=False)

                # normalizamos como tu API anterior
                out[vid] = {
                    "duration_sec": info.get("duration", 0),
                    "view_count": info.get("view_count", 0),
                    "snippet": {
                        "title": info.get("title"),
                        "description": info.get("description"),
                        "tags": info.get("tags"),
                        "thumbnails": info.get("thumbnails"),
                        "channelTitle": info.get("channel"),
                    },
                    "definition": info.get("resolution"),
                    "dimension": info.get("resolution"),
                    "projection": info.get("projection"),
                    "live": info.get("live_status") in ("is_live", "was_live"),
                    "is_vertical": looks_like_vertical_ydl(info),
                    "is_short": looks_like_short_ydl(info),
                    "has_bad_kw": detect_bad_kw(info),
                    "has_good_kw": detect_good_kw(info),
                }

            except Exception as e:
                print(f"[WARN] enrich error {vid}: {e}")
                continue

            time.sleep(0.1)

    return out


# ------------------------------------------------------------
# Detectores de palabras clave
# ------------------------------------------------------------
BAD_KW = ["timelapse", "compilation", "highlights", "edit", "trailer", "mv"]
GOOD_KW = ["tutorial", "how to", "setup", "pov", "fixed camera", "estático"]

def detect_bad_kw(info):
    text = f"{info.get('title','')} {info.get('description','')}".lower()
    return any(k in text for k in BAD_KW)

def detect_good_kw(info):
    text = f"{info.get('title','')} {info.get('description','')}".lower()
    return any(k in text for k in GOOD_KW)
