import os
from pytubefix import YouTube

MAX_HEIGHT = int(os.getenv("MAX_HEIGHT", "720"))
# =====================================================
#   HELPERS PARA PYTUBEFIX
# =====================================================

def _get_int_resolution(stream) -> int | None:
    """
    Convierte '720p' -> 720, '1080p' -> 1080, etc.
    Devuelve None si no se puede parsear.
    """
    res = getattr(stream, "resolution", None)
    if not res:
        return None
    # res suele ser algo como '720p'
    digits = "".join(ch for ch in res if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _pick_best_stream(yt: YouTube, max_height: int | None = None):
    """
    Elige el mejor stream progresivo MP4, idealmente <= max_height.
    - Primero intenta streams progresivos mp4 ordenados por resolución desc.
    - Si max_height está definido, escoge el mejor <= max_height.
    - Si no hay ninguno dentro del límite, usa el más alto disponible.
    """
    streams = (
        yt.streams
        .filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
    )

    if not streams:
        return None

    if max_height is not None:
        for s in streams:
            h = _get_int_resolution(s)
            # Si no tiene resolución clara, lo consideramos candidato
            if h is None or h <= max_height:
                return s

    # Fallback: el más alto de todos
    return streams.first()


# =====================================================
#   DESCARGA VIDEO + AUDIO CON PYTUBEFIX (MP4)
# =====================================================

def download_video_pytube(url: str, outdir: str) -> str | None:
    """
    Descarga el video de YouTube en la mayor calidad posible para los modelos.

    Estrategia:
    1) Intentar: solo video, mp4, ordenado por resolución desc.
    2) Si no hay: solo video (cualquier formato) por resolución desc.
    3) Si tampoco: progresivo mp4 (video+audio) por resolución desc.
    4) Si nada: devuelve None.

    Siempre devuelve la ruta del archivo o None si falla.
    """
    os.makedirs(outdir, exist_ok=True)

    try:
        yt = YouTube(url)
        print(f"[pytubefix] Título: {yt.title}")
    except Exception as e:
        print(f"[pytubefix][ERROR] No se pudo inicializar YouTube para {url}: {e}")
        return None

    # 1) Solo video, mp4, mejor resolución
    streams = yt.streams.filter(only_video=True, file_extension="mp4").order_by("resolution").desc()

    # 2) Si no hay mp4, cualquier solo-video
    if not streams:
        streams = yt.streams.filter(only_video=True).order_by("resolution").desc()

    # 3) Si no hay solo-video, usar progresivo mp4 (video+audio)
    if not streams:
        streams = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc()

    if not streams:
        print(f"[pytubefix][SKIP] No se encontró ningún stream descargable para {url}")
        return None

    stream = streams.first()
    res = getattr(stream, "resolution", None)
    mime = getattr(stream, "mime_type", None)
    subtype = getattr(stream, "subtype", "mp4") or "mp4"

    print(f"[pytubefix] Usando stream: {res} ({mime})")

    # Nombre de archivo estable: ID del video + extensión adecuada
    filename = f"{yt.video_id}.{subtype}"

    try:
        filepath = stream.download(output_path=outdir, filename=filename)
        print(f"[pytubefix][OK] Descarga completada: {filepath}")
        return filepath
    except Exception as e:
        print(f"[pytubefix][ERROR] Falló la descarga de {url}: {e}")
        return None
