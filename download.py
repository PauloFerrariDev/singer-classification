from pytube import Playlist, YouTube
import yt_dlp
from io import BytesIO
import ffmpeg
import os
import json

# Sample JSON data (as a string)
# json_singers = '''
# [
#     {"name": "ENG_HAWAII", "url": "https://youtube.com/playlist?list=PLcEo8wAwxOpIHtOj5lRtgulwQey4x2MfE&si=KhyEfIjx0ihw7B1Y"},
#     {"name": "CAPITAL_INICIAL", "url": "https://youtube.com/playlist?list=PLcEo8wAwxOpIJVdOClCzrXHn6VJRg9rsE&si=C2k_XKjVhuy9el2p"},
#     {"name": "O_RAPPA", "url": "https://youtube.com/playlist?list=PLcEo8wAwxOpKvr7QmMPt4HrlNWzs1HpUS&si=ubACMhW-FxSVAkfF"},
#     {"name": "CASSIA_ELLER", "url": "https://youtube.com/playlist?list=PLcEo8wAwxOpKpAz1HbcP6hDMUkbpFRbzJ&si=ZeQZaDWKcaUYjMwB"},
#     {"name": "RITA_LEE", "url": "https://youtube.com/playlist?list=PLcEo8wAwxOpIlTzO9HuLv-b4TBBhjFQbo&si=VmVcYrt6rgVJVRdB"}
# ]
# '''
    # {"name": "ZELIA_DUNCAN", "url": "https://youtube.com/playlist?list=PLcEo8wAwxOpJMT5sQKG9vpbxrGJ0pCPFb&si=kaYIizhn7xcgiFsU"},
    # {"name": "TIM_MAIA", "url": "https://youtube.com/playlist?list=PLcEo8wAwxOpLcy6jZf1rWmh-m57ENXtsq&si=h1JgVBfl9J0WNBx5"},
    # {"name": "ZE_RAMALHO", "url": "https://youtube.com/playlist?list=PLcEo8wAwxOpJKbQ0iRJkXFvt1WH6_klJs&si=-wRuJ1ZUGfpyd4vZ"},
    # {"name": "ELIS_REGINA", "url": "https://youtube.com/playlist?list=PLcEo8wAwxOpLURx0AY8Cl0NFFCSQFZ7SE&si=qvis1UWOgiafPvkv"},
json_singers = '''
[
    {"name": "PITTY", "url": "https://youtube.com/playlist?list=PLcEo8wAwxOpLeH2SW5g8vHFnHe5NWQ6rr&si=yXSBjT95k9uhwKO_"}
]
'''

# Parse the JSON data
singers = json.loads(json_singers)
playlist_size = 30

#* Buffering audio stream using Pytube
def audio_buffer(url:str):
    yt = YouTube(url)
    stream = yt.streams.get_audio_only()
    print("TITLE:", stream.title, "FILESIZE:", stream.filesize, "TYPE:", stream.type, "SUBTYPE:", stream.subtype)
    buffer = BytesIO()
    stream.stream_to_buffer(buffer)
    return buffer

#* Process audio buffer using ffmpeg
def process(buffer:BytesIO, output_file:str, ss="01:30"):
    process = (
            ffmpeg
            .input('pipe:', ss=ss, t="30", ac=2, format="mp4")
            .output(output_file, ac=1, format="wav")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    try:
        process.communicate(input=buffer.getbuffer(), timeout=None)
        process.wait() # Wait for ffmpeg process to finish
    except Exception as e:
        print("Error:", e)

#* DOWNLOAD WITH yt_dlp
def download_audio(url, output_path='.', num='0'):
    print(f"DOWNLOAD: {url} IN {output_path}")
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_path}/{num}.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

#* Create directory for singer's audios
def create_directory(dir:str):
    if not os.path.exists(dir): # checking if the directory exist or not     
        os.makedirs(dir) # if the directory is not present then create it

#* Iterate over playlists array
def playlists_handler():
    for singer in singers:
        output_path = f"./musicas/{singer['name']}"
        create_directory(output_path)
        pl = Playlist(singer['url'])[:playlist_size]
        print('pl', pl)
        print('pl len', len(pl))
        for i in range(0, playlist_size):
            url = pl[i]
            download_audio(url, output_path, i+1)           

#* Run script
print("\n*** START ***")
print('singers', singers)
playlists_handler()
print("*** END ***\n")
