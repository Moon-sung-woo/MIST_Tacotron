import numpy as np
import torch
from scipy.io.wavfile import read
from f0.yin import compute_yin

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def get_f0(audio, sampling_rate=22050, frame_length=1024,
           hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
    f0, harmonic_rates, argmins, times = compute_yin(
        audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
        harm_thresh)
    pad = int((frame_length / hop_length) / 2)
    f0 = [0.0] * pad + f0 + [0.0] * pad

    f0 = np.array(f0, dtype=np.float32)
    return f0

if __name__ == "__main__":

    #audio_path = '../dataset/ang/wav/acriil_ang_00000002.wav'
    audio_path = '../test_file/gst_nem.wav'
    audio, sampling_rate = load_wav_to_torch(audio_path)
    print(audio)
    print(sampling_rate)
    f0 = get_f0(audio, sampling_rate=16000)
    print(f0)
    print(f0.shape)
