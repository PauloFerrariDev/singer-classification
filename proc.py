# [x] carregar musica com sr 48k
# [x] obter quantidade total (int) de trechos de excerpt_duration segundos que o tempo da musica fornece
# [x] colocar para tocar cada trecho, dar offset de excerpt_duration segundos entre cada trecho
# [x] para cada trecho verificar se tem voz (1) ou nao (0)
# [x] salvar num txt para caso precise depois

import librosa
import math
import sounddevice as sd
import numpy as np
import os

def create_directory(dir:str):
    if not os.path.exists(dir): # checking if the directory exist or not     
        os.makedirs(dir) # if the directory is not present then create it

#! Essa funcao sera usada para carregar os .txt para extrair as features
def loadMusicExcerptsTxt(path:str):
    numbers = np.loadtxt(path, dtype=int)
    print("Loaded numbers:", numbers)

def inputMusicNumber():
    while(1):
        user_input = input(f"\nInput music file number: ")
        try:
            number = int(user_input)
            return number
        except ValueError:
                print(f"Invalid input: '{user_input}' is not a number. Please try again.")

voice_key = 1 #! NAO ALTERAR ESSE PARAMETRO!!!
noise_key = 0 #! NAO ALTERAR ESSE PARAMETRO!!!
excerpt_duration = 5 #! NAO ALTERAR ESSE PARAMETRO!!!
sr = 48000 #! NAO ALTERAR ESSE PARAMETRO!!!
offset = sr * excerpt_duration #! NAO ALTERAR ESSE PARAMETRO!!!
musics_folder = 'musicas' # alterar de acordo com seu diretorio
excerpts_folder = 'trechos' # alterar de acordo com seu diretorio
artist = 'CAPITAL INICIAL' # alterar para cada cantor analisado
excerpt_dir = f"./{excerpts_folder}/{artist}"
create_directory(excerpt_dir)
while(1):
    music_num = inputMusicNumber()
    music_path = f"./{musics_folder}/{artist}/{music_num}.mp3"
    excerpt_path = f"{excerpt_dir}/{music_num}.txt"
    excerpt_class = []
    audio, *_ = librosa.load(path=music_path, sr=sr)
    duration = math.floor(len(audio) / sr)
    remainder = duration % excerpt_duration
    intervals = int((duration - remainder) / excerpt_duration)
    print(f"Analyzing the file:", music_path)
    print(f"The remainder of {duration} divided by {excerpt_duration} is {remainder}")
    print(f"Duration: {duration} seconds")
    print(f"Intervals: {intervals} intervals of {excerpt_duration} seconds")
    for i in range(intervals):
        start = i*offset
        end = (i+1)*offset
        excerpt = audio[start:end] # trecho
        print("Excerpt ", i+1)
        sd.play(excerpt, sr)
        sd.wait()
        while(1):
            user_input = input(f"Excerpt Classification. [{voice_key}] Voice | [{noise_key}] Noise: ")
            try:
                number = int(user_input)
                if number == voice_key or number == noise_key:
                    excerpt_class.append(number)
                    break
                else:
                    print("Use one of the following options! [{voice_key}] Voice | [{noise_key}] Noise")
            except ValueError:
                print(f"Invalid input: '{user_input}' is not a number. Please try again.")
    np.savetxt(excerpt_path, np.array(excerpt_class), fmt='%d')
    

#! TESTE
# arr = [1,2,3,4,5,6,7,8]
# create_directory(excerpt_dir)
# np.savetxt(excerpt_path, np.array(arr), fmt='%d')
# loadMusicExcerptsTxt(excerpt_path)