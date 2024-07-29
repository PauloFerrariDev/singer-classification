import numpy as np
import joblib
import librosa
import features
from features import singers
import filter
from random import randint
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

######### LOAD MODEL #########
# SINGER_MODEL = joblib.load('nn_model.pkl')

# Carregar o modelo treinado
model = load_model('dataset_complete_with_recordings.h5')

# Carregar o scaler e o label encoder salvos
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def identify_singer(audio, sr):    
    audio_bpf, *_ = filter.bandpass_filter(audio, sr)
    filter.play_audio(audio_bpf, sr)

    # L = len(audio)
    # t = np.linspace(start=0, stop=L/sr, num=L) # samples
    # plt.subplot(2,1,1)
    # plt.plot(t,audio)
    # plt.subplot(2,1,2)
    # plt.plot(t,audio_bpf)
    # plt.show()

    audio_features = features.get_audio_features(audio_bpf, sr)
    audio_features = audio_features.reshape(1, -1)
    # singer_predicted = SINGER_MODEL.predict(audio_features)

    # Fazer previsões no novo conjunto de dados
    X_novo = scaler.transform(audio_features)
    y_novo_pred = model.predict(X_novo)
    y_novo_pred_classes = y_novo_pred.argmax(axis=-1)
    y_novo_pred_labels = label_encoder.inverse_transform(y_novo_pred_classes)

    print("Identified singer:", y_novo_pred_labels)

def run_singer_classifier_audios_script():
    for singer in singers:       
        num = randint(0, 29)
        audio_path = f"./audios/{singer}/audio-{num}.wav"
        print("Identifying the singer of:", audio_path)
        audio, sr = librosa.load(audio_path)
        identify_singer(audio, sr)

def run_singer_classifier_recording_script():
    duration = 30
    audio, sr = filter.record_audio(duration)
    audio = np.array(np.array(audio).flat) # essa linha esta correta!
    audio = filter.audio_normalized(audio)

    # L = len(audio)
    # t = np.linspace(start=0, stop=L/sr, num=L) # samples
    # plt.subplot(2,1,1)
    # plt.plot(t,audio)
    # plt.subplot(2,1,2)
    # plt.plot(t,audio_n)
    # plt.show()

    print('Audio:', audio)
    print('Audio length:', len(audio))
    print('Audio sample rate:', sr)
    filter.play_audio(audio, sr, duration)
    identify_singer(audio, sr)

# run_singer_classifier_audios_script()
run_singer_classifier_recording_script()
    