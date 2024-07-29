import Orange
import pickle
import numpy as np
import soundfile as sf
from filter import record_audio, play_audio, audio_normalized
from features import get_audio_features, cleanAudioData, create_directory, create_data_table_header_clean, create_instance_clean
from create_csv import create_csv_file

excerpt_duration = 5 #! NAO ALTERAR ESSE PARAMETRO!!!
sr = 48000 #! NAO ALTERAR ESSE PARAMETRO!!!
offset = sr * excerpt_duration
duration = 30 #? alterar de acordo com o tempo total de amostragem
intervals = int(duration / excerpt_duration)
data_path = './datasets/DATA.csv' #? alterar de acordo com seu diretorio
csvfile, writer = create_csv_file(data_path)
header = create_data_table_header_clean()
writer.writerow(header)

print('START RECORDING', duration, 'SECONDS...')
audio, *_ = record_audio(duration=duration, sample_rate=sr)
audio = np.array(np.array(audio).flat) #! essa linha esta correta!
audio = audio_normalized(cleanAudioData(audio))
play_audio(audio, sr)
print('audio:', audio, 'len:', len(audio))
print('Save audio in .WAV file')
sf.write('./files/DATA.wav', data=audio, samplerate=sr)

for i in range(intervals):
    next = i+1
    start = i*offset
    end = next*offset
    singer = f'SAMPLE_{next}'
    excerpt = audio[start:end]
    print('ANALYZING', singer)
    play_audio(excerpt, sr)
    print('EXTRACTING EXCERPT FEATURES...')
    instance = create_instance_clean(singer, audio=excerpt, sr=sr)
    writer.writerow(instance)

# Close write in CSV file             
csvfile.close()

# Load the saved model
model_path = "./orange_NN_model.pkcls"
with open(model_path, "rb") as file:
    model = pickle.load(file)

domain = model.domain
classes = model.domain.class_var.values
# print('domain', domain, len(domain))
print('classes', classes, 'len:', len(classes))

# Load your data to make predictions
# Assume you have a data file 'data.csv' to predict
data = Orange.data.Table(data_path)
# print('data:', data)

# Make predictions
predictions = model(data)

# Print predictions
for prediction in predictions:
    p = int(prediction)
    print('prediction:', p, '- classe:', classes[p])

# # # Example NumPy array (replace with your actual data)
# # Assume this is a 2D array where rows are samples and columns are features
# numpy_data = np.array([audio_features])
# print('numpy_data:', numpy_data, 'len:', len(numpy_data))

# # # Convert NumPy array to Orange data Table
# domain = model.domain
# orange_data = Orange.data.Table(domain, numpy_data)

# # Make predictions
# predictions = model(orange_data)

# # Print predictions
# for prediction in predictions:
#     print(prediction)
