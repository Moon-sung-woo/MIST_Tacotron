from common.layers import TacotronSTFT
from common.utils import load_wav_to_torch
import torch

def load_mel(path):
    stft = TacotronSTFT()
    audio, sampling_rate = load_wav_to_torch(path)
    if sampling_rate != 16000:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / 32768.0 # hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    #melspec = melspec.cuda()
    melspec = torch.squeeze(melspec, 0)
    return melspec

print(load_mel('emotion_dataset/Personality/pfa/wav/pfa00013.wav'))

# file_path = 'filelists/sum_multy_filelist.txt'
# save_path = 'filelists/sum_multy_filelist2.txt'
#
# f = open(file_path, 'r', encoding='utf-8')
# lines = f.readlines()
# f.close()
# wf = open(save_path, 'w', encoding='utf-8')
# for line in lines:
#     line = line.replace('\ufeff', '')
#     wf.write(line)
#
# wf.close()