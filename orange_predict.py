import Orange
import pickle
import numpy as np
import soundfile as sf
import librosa
import time
from filter import record_audio, play_audio, audio_normalized
from features import cleanAudioData, create_data_table_header_clean, create_instance_clean
from create_csv import create_csv_file
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

print("\n--- START ---")
# Record the start time
start_time = time.time()

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
# audio = cleanAudioData(audio)
audio = audio_normalized(cleanAudioData(audio))
# play_audio(audio, sr)
# print('audio:', audio, 'len:', len(audio))
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

# Record the end time
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
# Print the elapsed time
print(f"Elapsed time: {elapsed_time} seconds")
print("--- END ---\n")