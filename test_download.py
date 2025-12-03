from pytubefix import YouTube

url = "https://www.youtube.com/watch?v=M0jKmFfnlhE"

try:
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    print(f"Descargando: {yt.title}...")
    stream.download()
    print("¡Descarga completada!")
except Exception as e:
    print(f"Error: {e}")