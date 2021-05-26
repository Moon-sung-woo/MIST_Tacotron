# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

from tacotron2.text import text_to_sequence
import models
import torch
import argparse
import os
import numpy as np
from scipy.io.wavfile import write
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import PIL

import sys

import time
import dllogger as DLLogger
from dllogger import StdOutBackend, JSONStreamBackend, Verbosity

from common.utils import load_wav_to_torch
from common.layers import TacotronSTFT
from waveglow.denoiser import Denoiser

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='full path to the input text (phareses separated by new line)')
    parser.add_argument('-o', '--output', required=True,
                        help='output folder to save audio (file per phrase)')
    parser.add_argument('--suffix', type=str, default="", help="output filename suffix")
    parser.add_argument('--tacotron2', type=str,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('--waveglow', type=str,
                        help='full path to the WaveGlow model checkpoint file')
    parser.add_argument('-s', '--sigma-infer', default=0.9, type=float)
    parser.add_argument('-d', '--denoising-strength', default=0.01, type=float)
    parser.add_argument('-sr', '--sampling-rate', default=16000, type=int,
                        help='Sampling rate')

    parser.add_argument('--E', type=int, default=256)
    parser.add_argument('--ref_enc_filters', nargs='*', default=[32, 32, 64, 64, 128, 128])
    parser.add_argument('--ref_enc_size', nargs='*', default=[3, 3])
    parser.add_argument('--ref_enc_strides', nargs='*', default=[2, 2])
    parser.add_argument('--ref_enc_pad', nargs='*', default=[1, 1])
    parser.add_argument('--ref_enc_gru_size', type=int, default=256 // 2)

    # Style Token Layer
    parser.add_argument('--token_num', type=int, default=10)
    parser.add_argument('--num_heads', type=int, default=8)

    parser.add_argument('--n_mels', type=int, default=80)

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument('--fp16', action='store_true',
                        help='Run inference with mixed precision')
    run_mode.add_argument('--cpu', action='store_true',
                        help='Run inference on CPU')

    parser.add_argument('--log-file', type=str, default='nvlog.json',
                        help='Filename for logging')
    parser.add_argument('--include-warmup', action='store_true',
                        help='Include warmup')
    parser.add_argument('--stft-hop-length', type=int, default=256,
                        help='STFT hop length for estimating audio length from mel size')

    parser.add_argument('--ref_mel', type=str, required=True, #default='emotion_dataset/emotional-to-emotional/neb/wav/neb00209.wav',
                        help='style ref mel')
    parser.add_argument('--speaker_id', type=str, required=True,
                        help='speaker_id')

    parser.add_argument('--n_emotions', type=int, default=4) #총 4개 감정
    parser.add_argument('--emotion_embedding_dim', type=int, default=128)

    parser.add_argument('--n_speakers', type=int, default=40) #총 40명
    parser.add_argument('--speaker_embedding_dim', type=int, default=128)

    parser.add_argument('--emotion_id', type=int, default=1)
    parser.add_argument('--png_path', type=str, required=True)
    parser.add_argument('--z_latent_dim', type=int, default=32)


    return parser


def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def load_and_setup_model(model_name, parser, checkpoint, fp16_run, cpu_run, forward_is_infer=False):
    model_parser = models.model_parser(model_name, parser, add_help=False)
    model_args, _ = model_parser.parse_known_args()

    model_config = models.get_model_config(model_name, model_args)
    model = models.get_model(model_name, model_config, cpu_run=cpu_run,
                             forward_is_infer=forward_is_infer)

    if checkpoint is not None:
        if cpu_run:
            state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['state_dict']
        else:
            state_dict = torch.load(checkpoint)['state_dict']
        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)

        model.load_state_dict(state_dict)

    if model_name == "WaveGlow":
        model = model.remove_weightnorm(model)

    model.eval()

    if fp16_run:
        model.half()

    return model


# taken from tacotron2/data_function.py:TextMelCollate.__call__
def pad_sequences(batch):
    # Right zero-pad all one-hot text sequences to max input length
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x) for x in batch]),
        dim=0, descending=True)
    max_input_len = input_lengths[0]

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        text = batch[ids_sorted_decreasing[i]]
        text_padded[i, :text.size(0)] = text

    return text_padded, input_lengths


def prepare_input_sequence(texts, cpu_run=False):

    d = []
    for i,text in enumerate(texts):
        d.append(torch.IntTensor(
            text_to_sequence(text, ['korean_cleaners'])[:]))

    text_padded, input_lengths = pad_sequences(d)
    if not cpu_run:
        text_padded = text_padded.cuda().long()
        input_lengths = input_lengths.cuda().long()
    else:
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()

    return text_padded, input_lengths

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

def image_loader(img_path):
    loader = transforms.Compose([
        transforms.ToTensor()  # torch.Tensor 형식으로 변경 [0, 255] → [0, 1]
    ])
    image = PIL.Image.open(img_path).convert('RGB')
    image = loader(image)
    return image

def create_speaker_lookup_table(audiopaths_and_text):
    speaker_list = [x[3] for x in audiopaths_and_text]
    speaker_ids = np.sort(np.unique(speaker_list))
    d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
    return d

def load_filepaths_and_text(dataset_path, filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        def split_line(root, line):
            parts = line.strip().split(split)
            if len(parts) > 4:
                raise Exception(
                    "incorrect line format for file: {}".format(filename))
            path = os.path.join(root, parts[0])
            text = parts[1]
            emotion = parts[2]
            speaker = parts[3]
            return path, text, emotion, speaker
        filepaths_and_text = [split_line(dataset_path, line) for line in f]
    return filepaths_and_text


class MeasureTime():
    def __init__(self, measurements, key, cpu_run=False):
        self.measurements = measurements
        self.key = key
        self.cpu_run = cpu_run

    def __enter__(self):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self.cpu_run:
            torch.cuda.synchronize()
        self.measurements[self.key] = time.perf_counter() - self.t0


def main():
    """
    Launches text to speech (inference).
    Inference is executed on a single GPU or CPU.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 Inference')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    log_file = os.path.join(args.output, args.log_file)
    DLLogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_file),
                            StdOutBackend(Verbosity.VERBOSE)])
    for k,v in vars(args).items():
        DLLogger.log(step="PARAMETER", data={k:v})
    DLLogger.log(step="PARAMETER", data={'model_name':'Tacotron2_PyT'})

    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     args.fp16, args.cpu, forward_is_infer=True) # forward is infer를 해줌으로써 tacotron model의 infer로 간다.
    waveglow = load_and_setup_model('WaveGlow', parser, args.waveglow,
                                    args.fp16, args.cpu, forward_is_infer=True)
    denoiser = Denoiser(waveglow)
    if not args.cpu:
        denoiser.cuda()

    jitted_tacotron2 = torch.jit.script(tacotron2)

    texts = []
    id_list = []
    try:
        f = open(args.input, 'r')
        texts = f.readlines()
    except:
        print("Could not read file")
        sys.exit(1)

    #-------------------------------------------------------------------------------------------------------------------
    file_path = 'filelists/multi_train_file_list.txt'
    s_filelist = load_filepaths_and_text('/', file_path)
    speaker_table = create_speaker_lookup_table(s_filelist)

    ref_mel = load_mel(args.ref_mel)
    id_list.append(args.emotion_id)
    emotion_id = torch.LongTensor(id_list).cuda()
    style_png = image_loader(args.png_path)

    input_speaker_id = args.speaker_id
    speaker_id = speaker_table[input_speaker_id]
    speaker_id = torch.LongTensor([speaker_id]).cuda()
    print('emotion_id : ', emotion_id)
    print('speaker_id : ', speaker_id)
    #-------------------------------------------------------------------------------------------------------------------


    if args.include_warmup:
        sequence = torch.randint(low=0, high=80, size=(1, 50)).long()
        input_lengths = torch.IntTensor([sequence.size(1)]).long()
        if not args.cpu:
            sequence = sequence.cuda()
            input_lengths = input_lengths.cuda()
        for i in range(3):
            with torch.no_grad():
                mel, mel_lengths, _ = jitted_tacotron2(sequence, input_lengths, ref_mel, emotion_id, style_png, speaker_id)
                _ = waveglow(mel)

    measurements = {}

    sequences_padded, input_lengths = prepare_input_sequence(texts, args.cpu)

    with torch.no_grad(), MeasureTime(measurements, "tacotron2_time", args.cpu):
        mel, mel_lengths, alignments = jitted_tacotron2(sequences_padded, input_lengths, ref_mel, emotion_id, style_png, speaker_id)

    with torch.no_grad(), MeasureTime(measurements, "waveglow_time", args.cpu):
        audios = waveglow(mel, sigma=args.sigma_infer)
        audios = audios.float()
    with torch.no_grad(), MeasureTime(measurements, "denoiser_time", args.cpu):
        audios = denoiser(audios, strength=args.denoising_strength).squeeze(1)

    print("Stopping after",mel.size(2),"decoder steps")
    tacotron2_infer_perf = mel.size(0)*mel.size(2)/measurements['tacotron2_time']
    waveglow_infer_perf = audios.size(0)*audios.size(1)/measurements['waveglow_time']

    DLLogger.log(step=0, data={"tacotron2_items_per_sec": tacotron2_infer_perf})
    DLLogger.log(step=0, data={"tacotron2_latency": measurements['tacotron2_time']})
    DLLogger.log(step=0, data={"waveglow_items_per_sec": waveglow_infer_perf})
    DLLogger.log(step=0, data={"waveglow_latency": measurements['waveglow_time']})
    DLLogger.log(step=0, data={"denoiser_latency": measurements['denoiser_time']})
    DLLogger.log(step=0, data={"latency": (measurements['tacotron2_time']+measurements['waveglow_time']+measurements['denoiser_time'])})

    print(args.png_path.split('/')[-1][:-4])
    file_name_num = args.png_path.split('/')[-1][:-4]

    # print(args.ref_mel.split('/')[-1][:-4])
    # file_name_num = args.ref_mel.split('/')[-1][:-4]

    for i, audio in enumerate(audios):

        plt.imshow(alignments[i].float().data.cpu().numpy().T, aspect="auto", origin="lower")
        figure_path = os.path.join(args.output, "alignment_{}.png".format(file_name_num))
        plt.savefig(figure_path)

        audio = audio[:mel_lengths[i]*args.stft_hop_length]
        audio = audio/torch.max(torch.abs(audio))
        # audio_path = os.path.join(args.output, "audio_"+str(i)+args.suffix+".wav")
        audio_path = os.path.join(args.output, "audio_{}.wav".format(file_name_num))
        write(audio_path, args.sampling_rate, audio.cpu().numpy())

    DLLogger.flush()

if __name__ == '__main__':
    main()
