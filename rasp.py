import Orange
import pickle
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import time
import csv
import warnings

warnings.filterwarnings("ignore") # Suppress all warnings

n_mfcc=20
playlist_size=30

def bpm(audio, sr):
    bpm = librosa.feature.tempo(y=audio, sr=sr)[0]
    return bpm

def stft(audio, sr):
    y = librosa.feature.chroma_stft(y=audio, sr=sr)
    m = np.mean(y)
    v = np.var(y)
    return m, v

def cqt(audio, sr):
    y = librosa.feature.chroma_cqt(y=audio, sr=sr)
    m = np.mean(y)
    v = np.var(y)
    return m, v

def cens(audio, sr):
    y = librosa.feature.chroma_cens(y=audio, sr=sr)
    m = np.mean(y)
    v = np.var(y)
    return m, v

def contrast(audio, sr):
    y = librosa.feature.spectral_contrast(y=audio, sr=sr)
    m = np.mean(y)
    v = np.var(y)
    return m, v

def centroid(audio, sr):
    y = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    m = np.mean(y)
    v = np.var(y)
    return m, v

def bandwidth(audio, sr):
    y = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    m = np.mean(y)
    v = np.var(y)
    return m, v

def melspectrogram(audio, sr):
    y = librosa.feature.melspectrogram(y=audio, sr=sr)[0]
    m = np.mean(y)
    v = np.var(y)
    return m, v

def rms(audio):
    y = librosa.feature.rms(y=audio)[0]
    m = np.mean(y)
    v = np.var(y)
    return m, v

def mfcc(audio, sr):
    y = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    m = np.mean(y, axis=1)
    v = np.var(y, axis=1)
    return m, v

def create_csv_file(filename):
    # open new file for writing - will erase file if it already exists -
    csvfile = open(filename, 'w', newline='', encoding='utf-8')
    # make a new variable - c - for Python's CSV writer object -
    writer = csv.writer(csvfile)
    return csvfile, writer

def record_audio(duration=30, sample_rate=48000):
    print("Start Recording")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Finished Recording")
    return audio, sample_rate

def play_audio(audio, sr, duration=0):
    if (len(audio) < sr*duration):
        print("Play Audio: Error")
        return
    print("Start Playing")
    sd.play(audio, sr)
    sd.wait()
    print("Finished Playing")

def audio_normalized(audio):
    max = np.max(np.absolute(audio))
    audio_n = audio / max
    return audio_n

def cleanAudioData(y):
    # Check if there are any NaN or Inf values
    if not np.isfinite(y).all():
        print("Audio buffer contains NaN or Inf values.")
        # Replace NaN with zero and Inf with finite large value
        y = np.where(np.isnan(y), 0, y)
        y = np.where(np.isinf(y), np.finfo(y.dtype).max, y)
    return y

def create_data_table_header_clean():
    header = ['singer','bpm','stft_mean','stft_var','cqt_mean','cqt_var','cens_mean','cens_var','contrast_mean','contrast_var','centroid_mean','centroid_var','bandwidth_mean','bandwidth_var','melspectrogram_mean','melspectrogram_var','rms_mean','rms_var']
    for i in range(1, n_mfcc+1):
        header = np.append(header, ['mfcc%s_mean'%i, 'mfcc%s_var'%i])
    return header

def create_instance_clean(singer, audio, sr):    
    instance = np.array([singer,bpm(audio,sr),*stft(audio,sr),*cqt(audio,sr),
                         *cens(audio,sr),*contrast(audio,sr),*centroid(audio,sr),
                         *bandwidth(audio,sr),*melspectrogram(audio,sr),*rms(audio)])
    m, v = mfcc(audio,sr)
    for i in range(0, n_mfcc):
        instance = np.append(instance, [m[i], v[i]])
    return instance

print("\n--- START ---")
start_time = time.time() # Record the start time

excerpt_duration = 3 #! NAO ALTERAR ESSE PARAMETRO!!!
sr = 48000 #! NAO ALTERAR ESSE PARAMETRO!!!
offset = sr * excerpt_duration
duration = 60 #? alterar de acordo com o tempo total de amostragem
intervals = int(duration / excerpt_duration)
data_path = './datasets/DATA.csv' #? alterar de acordo com seu diretorio
music_path = './files/DATA.wav' #? alterar de acordo com seu diretorio
model_path = "./modelos/orange_NN_model_5singers.pkcls" #? alterar de acordo com seu diretorio
csvfile, writer = create_csv_file(data_path)
header = create_data_table_header_clean()
writer.writerow(header)

# audio, *_ = librosa.load(path=music_path, sr=sr)

print('START RECORDING', duration, 'SECONDS...')
audio, *_ = record_audio(duration=duration, sample_rate=sr)
audio = np.array(np.array(audio).flat) #! essa linha esta correta!
audio = audio_normalized(cleanAudioData(audio))
print('SAVE AUDIO IN .WAV FILE')
sf.write(file=music_path, data=audio, samplerate=sr)

for i in range(intervals):
    next = i+1
    start = i*offset
    end = next*offset
    singer = f'SAMPLE_{next}'
    excerpt = audio[start:end]
    print(f'{singer}: EXTRACTING FEATURES...')
    play_audio(excerpt, sr)
    instance = create_instance_clean(singer, audio=excerpt, sr=sr)
    writer.writerow(instance)

# Close write in CSV file             
csvfile.close()

# Load the saved model
with open(model_path, "rb") as file:
    model = pickle.load(file)

domain = model.domain
classes = model.domain.class_var.values
# print('domain', domain, len(domain))
print('classes:', classes, ' - len:', len(classes))

# Load your data to make predictions. Assume you have a data file 'data.csv' to predict
data = Orange.data.Table(data_path)
# print('data:', data)

# Make predictions
predictions = model(data)
classes_count = {}

# Print predictions
for prediction in predictions:
    p = int(prediction)
    c = classes[p]
    if c in classes_count:
        classes_count[c] += 1
    else:
        classes_count[c] = 1
    print('Prediction:', p, '- Classe:', c)

# Sort dictionary by descending order
classes_count_sorted = sorted(classes_count.items(), key=lambda item: item[1], reverse=True)
print('SUM FOR EACH CLASS')
# Show each key: value
for key, value in classes_count_sorted:
    print(f"{key}: {value}")

# Class with higher sum
key = classes_count_sorted[0][0]
value = classes_count_sorted[0][1]
if key=='NOISE' and len(classes_count_sorted) > 1:
    key = classes_count_sorted[1][0]
    value = classes_count_sorted[1][1]

print(f'*** SINGER: {key} ***')

end_time = time.time() # Record the end time
elapsed_time = end_time - start_time # Calculate the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")
print("--- END ---\n")