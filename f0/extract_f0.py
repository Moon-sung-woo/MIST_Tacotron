import numpy as np
import torch
from scipy.io.wavfile import read
from f0.yin import compute_yin
import matplotlib.pyplot as plt

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
    org_audio_path = '../inference_audio/single_normal/acriil_neu_00002169.wav'
    GST_audio_path = '../inference_audio/single_normal/VAE_result/audio_acriil_neu_00002169.wav'
    save_file_name = 'single_VAE_result'
    plt_legend = ['Original', 'VAE']

    org_audio, org_sampling_rate = load_wav_to_torch(org_audio_path)
    GST_org_audio, GST_sampling_rate = load_wav_to_torch(GST_audio_path)

    org_f0 = get_f0(org_audio, sampling_rate=org_sampling_rate)
    GST_f0 = get_f0(GST_org_audio, sampling_rate=GST_sampling_rate)

    plt.plot(org_f0, linewidth=2.5)
    plt.plot(GST_f0, linewidth=2.5, linestyle='-.')
    plt.legend(plt_legend, fontsize=17, loc=1)
    plt.xlabel('time')
    plt.ylabel('F0')
    plt.savefig('../save_img/{}.png'.format(save_file_name))
    plt.show()
    # print(f0)

