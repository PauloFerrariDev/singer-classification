import numpy as np
from scipy.signal import butter, lfilter, freqz, spectrogram
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd

two_pi = 2*np.pi
order = 6 # max order for bandpass Butterworth filter without error
Wn = np.array([100, 270])*two_pi # cutoff frequencies [rad/s]

def audio_data(audio_path: str):
    audio , sr = librosa.load(audio_path) # sr = sampling rate [sample/s]
    duration = audio.size / sr
    t = np.linspace(0, duration, num=audio.size) # samples
    return audio, sr, t

def audio_normalized(audio):
    max = np.max(np.absolute(audio))
    audio_n = audio / max
    return audio_n

def lowpass_filter(audio, sr):    
    b, a = butter(N=order, Wn=Wn[0], btype='lowpass', analog=False, output='ba', fs=sr)
    y = lfilter(b, a, audio)
    return y, b, a

def bandpass_filter(audio, sr):    
    b, a = butter(N=order, Wn=Wn, btype='bandpass', analog=False, output='ba', fs=sr)
    y = lfilter(b, a, audio)
    return y, b, a

def highpass_filter(audio, sr):    
    b, a = butter(N=order, Wn=Wn[1], btype='highpass', analog=False, output='ba', fs=sr)
    y = lfilter(b, a, audio)
    return y, b, a

def add_uniform_noise(audio_data, noise_level=0.05, t=[]):
    """
    Adds uniform noise to an audio file.
    
    Parameters:
    audio_path (str): Path to the input audio file.
    noise_level (float): The amplitude of the uniform noise to be added. 
                         A higher value means more noise. Default is 0.05.
    output_path (str): Path to save the output noisy audio file. Default is 'noisy_audio.wav'.
    """
       
    # Ensure the audio data is in float format
    # audio_data = audio_data.astype(np.float32)
    
    # Generate uniform noise
    noise = np.random.uniform(low=-noise_level, high=noise_level, size=audio_data.shape)
    
    # if(len(t)>0):
    #     plt.plot(t,noise)
    #     plt.title("uniform noise")
    #     plt.show()

    # Add the noise to the audio data
    noisy_audio = audio_data + noise
    
    # Clip the values to stay within the valid range
    if np.issubdtype(audio_data.dtype, np.integer):
        min_val = np.iinfo(audio_data.dtype).min
        max_val = np.iinfo(audio_data.dtype).max
    else:
        min_val = -1.0
        max_val = 1.0
    noisy_audio = np.clip(noisy_audio, min_val, max_val)
    
    # Convert back to original data type
    noisy_audio = noisy_audio.astype(audio_data.dtype)
    return noisy_audio

def add_normal_noise(audio_data, noise_level=0.05, t=[]):
    """
    Adds Normal noise to an audio file.
    
    Parameters:
    audio_path (str): Path to the input audio file.
    noise_level (float): Standard deviation of the Normal noise to be added. 
                         A higher value means more noise. Default is 0.05.
    output_path (str): Path to save the output noisy audio file. Default is 'noisy_audio.wav'.
    """
          
    # Ensure the audio data is in float format
    # audio_data = audio_data.astype(np.float32)
    
    # Generate Normal noise
    noise = np.random.normal(loc=0.0, scale=noise_level, size=audio_data.shape)
    
    # if(len(t)>0):
    #     plt.plot(t,noise)
    #     plt.title("normal noise")
    #     plt.show()

    # Add the noise to the audio data
    noisy_audio = audio_data + noise
    
    # Clip the values to stay within the valid range
    if np.issubdtype(audio_data.dtype, np.integer):
        min_val = np.iinfo(audio_data.dtype).min
        max_val = np.iinfo(audio_data.dtype).max
    else:
        min_val = -1.0
        max_val = 1.0
    noisy_audio = np.clip(noisy_audio, min_val, max_val)
    
    # Convert back to original data type
    noisy_audio = noisy_audio.astype(audio_data.dtype)
    return noisy_audio

def plot_filtered_audio(audio, filtered_audio, sr, b, a, t, filter_type):
    #* Plot the frequency response.
    w, h = freqz(b, a, fs=sr, worN=8000)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(w/two_pi, np.abs(h), 'b')
    plt.plot(Wn[0]/two_pi, np.sqrt(2)/2, 'ko')
    plt.axvline(Wn[0]/two_pi, color='r')
    plt.plot(Wn[1]/two_pi, np.sqrt(2)/2, 'ko')
    plt.axvline(Wn[1]/two_pi, color='r')
    plt.xlim(0, 500)
    plt.title("%s Filter Frequency Response"%filter_type)
    plt.xlabel('Frequency [Hz]')
    #* Filter the data, and plot both the original and filtered signals.
    plt.subplot(3, 1, 2)
    plt.plot(t, audio, 'b-')
    plt.title('Audio')
    plt.xlabel('Time [sec]')
    plt.subplot(3, 1, 3)
    plt.plot(t, filtered_audio, 'g-')
    plt.title('Filtered Audio')
    plt.xlabel('Time [sec]')
    plt.subplots_adjust(hspace=0.9)
    #* Spectrogram
    f1, _, sxx1 = spectrogram(audio, sr)
    f2, _, sxx2 = spectrogram(filtered_audio, sr)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(f1/two_pi, sxx1, 'b-')
    plt.title("Spectrogram (audio)")
    plt.xlabel('Frequency [Hz]')
    plt.xlim(0, 500)
    plt.subplot(2, 1, 2)
    plt.plot(f2/two_pi, sxx2, 'g-')
    plt.title("Spectrogram (filtered audio)")
    plt.xlabel('Frequency [Hz]')
    plt.xlim(0, 500)
    plt.subplots_adjust(hspace=0.6)
    plt.show()

def record_audio(duration=30, sample_rate=22050, device=14):
    print("Start Recording")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32', device=device)
    sd.wait()
    print("Finished Recording")
    return audio, sample_rate

def play_audio(audio, sr, duration=0, device=4):
    if (len(audio) < sr*duration):
        print("Play Audio: Error")
        return
    print("Start Playing")
    sd.play(audio, sr, device=device)
    sd.wait()
    print("Finished Playing")

#* Main function
def run_filter_script():
    print("\n*** START ***")
    audio_path_test = './audios/rita-lee/audio-10.wav'
    audio, sr, t = audio_data(audio_path_test)
    # audio_uniform = add_uniform_noise(audio, noise_level=0.15, t=t)
    # audio_normal = add_normal_noise(audio, noise_level=0.15, t=t)
    audio_bpf, *_ = bandpass_filter(audio, sr)
    print('Audio:', audio)
    print('Audio BDF:', audio_bpf)
    print('Audio length:', len(audio))
    print('Audio sample rate:', sr)
    plt.subplot(2,1,1)
    plt.plot(t,audio)
    plt.subplot(2,1,2)
    plt.plot(t,audio_bpf)
    plt.show()
    #* Filter the audio
    # audio_lp, blp, alp = lowpass_filter(audio, sr)
    # audio_bp, bbp, abp = bandpass_filter(audio, sr)
    # audio_hp, bhp, ahp = highpass_filter(audio, sr)
    #* Plot filtered audio and its frequency response
    # plot_filtered_audio(audio, audio_lp, sr, blp, alp, t, "Lowpass")
    # plot_filtered_audio(audio, audio_bp, sr, bbp, abp, t, "Bandpass")
    # plot_filtered_audio(audio, audio_hp, sr, bhp, ahp, t, "Highpass")
    #* Play audio and audio filtered
    # play_audio(audio=audio, sr=sr)
    # play_audio(audio=audio_uniform, sr=sr)
    # play_audio(audio=audio_normal, sr=sr)
    # play_audio(audio=audio_bp, sr=sr)
    # play_audio(audio=audio_hp, sr=sr)
    print("*** END ***\n")

#* Run script
# run_filter_script()
a, sr = record_audio(duration=10, sample_rate=48000)
play_audio(a, sr)