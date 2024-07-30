import numpy as np
import math
import librosa
from librosa import feature
import sounddevice as sd
import time
import os
import filter
from create_csv import create_csv_file

n_mfcc=20
playlist_size=30
singers = [
  "CAPITAL_INICIAL",
  "CASSIA_ELLER",
  "ELIS_REGINA",
  "ENG_HAWAII",
  "O_RAPPA",
  "PITTY",
  "RITA_LEE",
  "TIM_MAIA",
  "ZE_RAMALHO",
  "ZELIA_DUNCAN",
]

def bpm(audio, sr):
    bpm = feature.tempo(y=audio, sr=sr)[0]
    return bpm

def stft(audio, sr):
    y = feature.chroma_stft(y=audio, sr=sr)
    m = np.mean(y)
    v = np.var(y)
    return m, v

def cqt(audio, sr):
    y = feature.chroma_cqt(y=audio, sr=sr)
    m = np.mean(y)
    v = np.var(y)
    return m, v

def cens(audio, sr):
    y = feature.chroma_cens(y=audio, sr=sr)
    m = np.mean(y)
    v = np.var(y)
    return m, v

def contrast(audio, sr):
    y = feature.spectral_contrast(y=audio, sr=sr)
    m = np.mean(y)
    v = np.var(y)
    return m, v

def centroid(audio, sr):
    y = feature.spectral_centroid(y=audio, sr=sr)[0]
    m = np.mean(y)
    v = np.var(y)
    return m, v

def bandwidth(audio, sr):
    y = feature.spectral_bandwidth(y=audio, sr=sr)[0]
    m = np.mean(y)
    v = np.var(y)
    return m, v

def melspectrogram(audio, sr):
    y = feature.melspectrogram(y=audio, sr=sr)[0]
    m = np.mean(y)
    v = np.var(y)
    return m, v

def rms(audio):
    y = feature.rms(y=audio)[0]
    m = np.mean(y)
    v = np.var(y)
    return m, v

def mfcc(audio, sr):
    y = feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    m = np.mean(y, axis=1)
    v = np.var(y, axis=1)
    return m, v

#* CREATE MEAN AND VAR COLUMNS HEADER FOR EACH FEATURE
def create_data_table_header():
    header = ['singer','filename','sample','bpm','stft_mean','stft_var','cqt_mean','cqt_var','cens_mean','cens_var','contrast_mean','contrast_var','centroid_mean','centroid_var','bandwidth_mean','bandwidth_var','melspectrogram_mean','melspectrogram_var','rms_mean','rms_var']
    for i in range(1, n_mfcc+1):
        header = np.append(header, ['mfcc%s_mean'%i, 'mfcc%s_var'%i])
    return header

#* CREATE MEAN AND VAR COLUMNS HEADER FOR EACH FEATURE
def create_data_table_header_clean():
    header = ['singer','bpm','stft_mean','stft_var','cqt_mean','cqt_var','cens_mean','cens_var','contrast_mean','contrast_var','centroid_mean','centroid_var','bandwidth_mean','bandwidth_var','melspectrogram_mean','melspectrogram_var','rms_mean','rms_var']
    for i in range(1, n_mfcc+1):
        header = np.append(header, ['mfcc%s_mean'%i, 'mfcc%s_var'%i])
    return header

#* CREATE MEAN AND VAR COLUMNS FOR EACH FEATURE
def create_instance(singer, filename, sample, audio, sr):    
    instance = np.array([singer,filename,sample,bpm(audio,sr),*stft(audio,sr),*cqt(audio,sr),
                         *cens(audio,sr),*contrast(audio,sr),*centroid(audio,sr),
                         *bandwidth(audio,sr),*melspectrogram(audio,sr),*rms(audio)])
    m, v = mfcc(audio,sr)
    for i in range(0, n_mfcc):
        instance = np.append(instance, [m[i], v[i]])
    return instance

#* CREATE MEAN AND VAR COLUMNS FOR EACH FEATURE
def create_instance_clean(singer, audio, sr):    
    instance = np.array([singer,bpm(audio,sr),*stft(audio,sr),*cqt(audio,sr),
                         *cens(audio,sr),*contrast(audio,sr),*centroid(audio,sr),
                         *bandwidth(audio,sr),*melspectrogram(audio,sr),*rms(audio)])
    m, v = mfcc(audio,sr)
    for i in range(0, n_mfcc):
        instance = np.append(instance, [m[i], v[i]])
    return instance

#! CREATE MEAN AND VAR COLUMNS FOR EACH FEATURE
def get_audio_features(audio, sr):
    features = np.array([bpm(audio,sr),*stft(audio,sr),*cqt(audio,sr),
                         *cens(audio,sr),*contrast(audio,sr),*centroid(audio,sr),
                         *bandwidth(audio,sr),*melspectrogram(audio,sr),*rms(audio)])
    m, v = mfcc(audio,sr)
    for i in range(0, n_mfcc):
        features = np.append(features, [m[i], v[i]])
    return features

def run_features_script():
    print("\n*** START ***")
    # Record the start time
    start_time = time.time()
    header = create_data_table_header()
    csvfile, writer = create_csv_file("new_dataset.csv")
    writer.writerow(header)          
    for singer in singers:
        singer_dir = f"./audios_gravados/{singer}"
        for num in range(0, playlist_size):
            audio_path = f"{singer_dir}/audio-{num}.wav"
            print(audio_path)
            audio, sr, _ = filter.audio_data(audio_path)
            audio_bpf, *_ = filter.bandpass_filter(audio, sr)
            # audio_lpf, *_ = filter.lowpass_filter(audio, sr)
            # audio_hpf, *_ = filter.highpass_filter(audio, sr)
            audio_uniform_noise = filter.add_uniform_noise(audio, noise_level=0.11)
            audio_normal_noise = filter.add_normal_noise(audio, noise_level=0.11)
            uniform_noise_bpf, *_ = filter.bandpass_filter(audio_uniform_noise, sr)
            normal_noise_bpf, *_ = filter.bandpass_filter(audio_normal_noise, sr)
            instances = [create_instance(singer, num, 'original', audio, sr),
                        create_instance(singer, num, 'bandpass', audio_bpf, sr),
                        # create_instance(singer, num, 'lowpass', audio_lpf, sr),
                        # create_instance(singer, num, 'highpass', audio_hpf, sr),
                        # create_instance(singer, num, 'uniform_noise', audio_uniform_noise, sr),
                        # create_instance(singer, num, 'normal_noise', audio_normal_noise, sr),
                        create_instance(singer, num, 'uniform_noise_bandpass', uniform_noise_bpf, sr),
                        create_instance(singer, num, 'normal_noise_bandpass', normal_noise_bpf, sr),
                        ]
            for instance in instances:
                writer.writerow(instance)
    csvfile.close()
    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
    print("*** END ***\n")

#! criar pasta "datasets" para armazenar os CSV das features de cada artista
#! percorrer as 30 musicas do cantor e usar np.loadtxt para carregar as classficacoes
#! carregar a musica correspondente e percorrer cada trecho
#! para cada trecho extrair as features e atribuir a 'singer' o Nome do Artista ou 'NOISE'

def create_directory(dir:str):
    if not os.path.exists(dir): # checking if the directory exist or not     
        os.makedirs(dir) # if the directory is not present then create it

def clean_audio_data(y):
    # Check if there are any NaN or Inf values
    if not np.isfinite(y).all():
        print("Audio buffer contains NaN or Inf values.")
        # Replace NaN with zero and Inf with finite large value
        y = np.where(np.isnan(y), 0, y)
        y = np.where(np.isinf(y), np.finfo(y.dtype).max, y)
    return y

def extract_features():
    print("\n*** START ***")
    # Record the start time
    start_time = time.time()
    voice_key = 1 #! NAO ALTERAR ESSE PARAMETRO!!!
    excerpt_duration = 5 #! NAO ALTERAR ESSE PARAMETRO!!!
    sr = 48000 #! NAO ALTERAR ESSE PARAMETRO!!!
    offset = sr * excerpt_duration #! NAO ALTERAR ESSE PARAMETRO!!!
    artist = 'ZELIA_DUNCAN' # alterar para cada cantor analisado
    musics_dir = f'./musicas/{artist}' # alterar de acordo com seu diretorio
    excerpts_dir = f'./trechos/{artist}' # alterar de acordo com seu diretorio
    dataset_dir = './datasets' # alterar de acordo com seu diretorio
    create_directory(dataset_dir)
    csvfile, writer = create_csv_file(f'{dataset_dir}/{artist}.csv')
    header = create_data_table_header()
    writer.writerow(header)
    for file_num in range(1, playlist_size + 1): #! ALTERAR
        excerpt_path = f'{excerpts_dir}/{file_num}.txt'
        music_path = f'{musics_dir}/{file_num}.mp3'
        audio, *_ = librosa.load(path=music_path, sr=sr, dtype=np.float32, mono=True) #? CARREGA MUSICA
        # sd.play(audio, sr)
        # sd.wait()
        excerpts_class = np.loadtxt(fname=excerpt_path, dtype=int)
        duration = math.floor(len(audio) / sr)
        remainder = duration % excerpt_duration
        intervals = int((duration - remainder) / excerpt_duration)
        print(f"\nAnalyzing the file:", music_path)
        print('sr', sr, '\naudio', audio, len(audio))
        print(f"The remainder of {duration} divided by {excerpt_duration} is {remainder}")
        print(f"Duration: {duration} seconds")
        print(f"Intervals: {intervals} intervals of {excerpt_duration} seconds")
        for i in range(intervals):
            next = i+1
            start = i*offset
            end = next*offset
            singer = artist if excerpts_class[i] == voice_key else 'NOISE'
            print(f'\nExcerpt {next} Class: {singer}')
            excerpt = clean_audio_data(audio[start:end]) #? trecho da musica
            print("excerpt:", excerpt, len(excerpt))
            # Check if all elements are zero, then ignore excerpt
            if np.all(excerpt == 0):
                print("The array contains only zeros")
                continue #! next iteration
            # sd.play(excerpt, sr)
            # sd.wait()
            excerpt_un = filter.add_uniform_noise(excerpt, noise_level=0.07)
            print("excerpt_un:", excerpt_un, len(excerpt_un))
            # sd.play(excerpt_un, sr)
            # sd.wait()
            excerpt_nn = filter.add_normal_noise(excerpt, noise_level=0.07)
            print("excerpt_nn:", excerpt_nn, len(excerpt_nn))
            # sd.play(excerpt_nn, sr)
            # sd.wait()
            excerpt_bp, *_ = filter.bandpass_filter(excerpt, sr)
            excerpt_bp = filter.audio_normalized(clean_audio_data(excerpt_bp))
            print("excerpt_bp:", excerpt_bp, len(excerpt_bp))
            # sd.play(excerpt_bp, sr)
            # sd.wait()
            excerpt_bp_un, *_ = filter.bandpass_filter(excerpt_un, sr)
            excerpt_bp_un = filter.audio_normalized(clean_audio_data(excerpt_bp_un))
            print("excerpt_bp_un:", excerpt_bp_un, len(excerpt_bp_un))
            # sd.play(excerpt_bp_un, sr)
            # sd.wait()
            excerpt_bp_nn, *_ = filter.bandpass_filter(excerpt_nn, sr)
            excerpt_bp_nn = filter.audio_normalized(clean_audio_data(excerpt_bp_nn))
            print("excerpt_bp_nn:", excerpt_bp_nn, len(excerpt_bp_nn))
            # sd.play(excerpt_bp_nn, sr)
            # sd.wait()            
            filename = f'{artist}/{file_num}.mp3 - trecho {next}'
            instances = [
                        create_instance(singer, filename, 'original', excerpt, sr),
                        create_instance(singer, filename, 'original_uniform_noise', excerpt_un, sr),
                        create_instance(singer, filename, 'original_normal_noise', excerpt_nn, sr),
                        create_instance(singer, filename, 'bandpass', excerpt_bp, sr),
                        create_instance(singer, filename, 'bandpass_uniform_noise', excerpt_bp_un, sr),
                        create_instance(singer, filename, 'bandpass_normal_noise', excerpt_bp_nn, sr)
                        ]
            for instance in instances:
                writer.writerow(instance)
    # Close write in CSV file             
    csvfile.close()
    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time} seconds")
    print("*** END ***\n")
            
#* Run script
# run_features_script()
extract_features()