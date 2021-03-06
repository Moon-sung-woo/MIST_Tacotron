from scipy.io.wavfile import write
import librosa
import numpy as np
import os
import glob

def make_sr16(wav_file):
    sr = 16000
    max_wav_value=32768.0
    trim_fft_size = 1024
    trim_hop_size = 256
    trim_top_db = 23

    silence_audio_size = trim_hop_size * 3

    data, sampling_rate = librosa.core.load(wav_file, sr)
    data = data / np.abs(data).max() * 0.999
    data_ = librosa.effects.trim(data, top_db=trim_top_db, frame_length=trim_fft_size, hop_length=trim_hop_size)[0]
    data_ = data_ * max_wav_value
    data_ = np.append(data_, [0.] * silence_audio_size)
    data_ = data_.astype(dtype=np.int16)
    write(wav_file, sr, data_)

    print('finish')

path = 'inference_audio'

file_list = os.listdir(path)

#print(file_list)

for file in file_list:
    file_path = os.path.join(path, file)
    wav_file_list = glob.glob(file_path + '/*.wav')
    for wav_file in wav_file_list:
        make_sr16(wav_file)

