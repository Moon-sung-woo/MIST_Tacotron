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

import random
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import PIL

import common.layers as layers
from common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu
from tacotron2.text import text_to_sequence

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, dataset_path, audiopaths_and_text, args, emotion_ids=None, speaker_ids=None):
        self.audiopaths_and_text = load_filepaths_and_text(dataset_path, audiopaths_and_text)
        self.text_cleaners = args.text_cleaners
        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        self.load_mel_from_disk = args.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.sampling_rate, args.mel_fmin,
            args.mel_fmax)
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

        self.emotion_ids = emotion_ids
        if emotion_ids is None:
            self.emotion_ids = self.create_emotion_lookup_table(self.audiopaths_and_text)

        self.speaker_ids = speaker_ids
        if speaker_ids is None:
            self.speaker_ids = self.create_speaker_lookup_table(self.audiopaths_and_text)

    # ===============mel style transfer 관련 부분===============
    def get_img_path(self, wav_path): # 단일화자 할 때
        wav_path = wav_path.split('/')
        wav_path[3] = 'img'
        wav_path[-1] = wav_path[-1].replace('.wav', '.png')
        img_path = '/'.join(wav_path)
        return img_path

    def get_img_path2(self, wav_path): #멀티스피커 할 때
        wav_path = wav_path.split('/')
        wav_path[4] = 'img'
        wav_path[-1] = wav_path[-1].replace('.wav', '.png')
        img_path = '/'.join(wav_path)
        return img_path

    def image_loader(self, img_path):
        loader = transforms.Compose([
            transforms.ToTensor()  # torch.Tensor 형식으로 변경 [0, 255] → [0, 1]
        ])
        image = PIL.Image.open(img_path).convert('RGB')
        image = loader(image)
        return image

    def get_img(self, audiopath):
        # img_path = self.get_img_path(audiopath) #이미지 경로 받아오기
        img_path = self.get_img_path2(audiopath)  # 멀티 할때 경로 받아오기
        #이미지 불러오기

        image = self.image_loader(img_path)
        return image
    # ===============mel style transfer 관련 종료===============

    # ===============emotion id 관련 부분===============
    def create_speaker_lookup_table(self, audiopaths_and_text):
        speaker_list = [x[3] for x in audiopaths_and_text]
        speaker_ids = np.sort(np.unique(speaker_list))
        d = {speaker_ids[i]: i for i in range(len(speaker_ids))}
        return d

    def create_emotion_lookup_table(self, audiopaths_and_text):
        emotion_list = [x[2] for x in audiopaths_and_text]
        emotion_ids = np.sort(np.unique(emotion_list))
        d = {emotion_ids[i]: i for i in range(len(emotion_ids))}
        return d

    def get_emotion_id(self, emotion_id):
        return torch.IntTensor([self.emotion_ids[emotion_id]])

    def get_speaker_id(self, speaker_id):
        # ('data_function.py  ====> get_speaker : ',self.speaker_ids[speaker_id])
        return torch.IntTensor([self.speaker_ids[speaker_id]])

    # ===============speaker id 관련 부분 종료===============


    # ===============mel style transfer 관련 부분===============
    def get_style_mel_path(self, wav_path):
        wav_path = wav_path.split('/')
        wav_path[6] = 'mel_npy'
        wav_path[-1] = wav_path[-1].replace('.wav', '.npy')
        img_path = '/'.join(wav_path)
        return img_path

    def get_style_mel(self, audiopath):

        # style mel 경로 가지고 오기
        style_mel = self.get_style_mel_path(audiopath)

        # style mel 불러오는 거로 수정해야함
        style = np.load(style_mel)
        style = torch.Tensor(style)
        return style
    # ===============mel style transfer 관련 종료===============


    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text, emotion, speaker = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2], audiopath_and_text[3]
        # print('data_function.py ====> txt : ', text)
        # print('data_function.py ====> speaker : ', speaker)

        len_text = len(text)
        text = self.get_text(text)
        # print('2: ', text)
        mel = self.get_mel(audiopath)
        emotion_id = self.get_emotion_id(emotion)
        # style_img = self.get_img(audiopath)
        style_mel = self.get_style_mel(audiopath)
        speaker_id = self.get_speaker_id((speaker))


        # print('data_function.py ====> emotion_id : ', emotion_id)
        # print('data_function.py ====> speaker_id : ', speaker_id)
        return (text, mel, len_text, emotion_id, style_mel, speaker_id)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length

        # print('batch : ', batch)
        # print('======batch len======', len(batch))


        # 이미지 모아서 확인해보기
        style_mel_list = []
        for m in range(len(batch)):
            style_mel = batch[m][4]
            style_mel_list.append(style_mel)
            # print('style_mel.shape : ========> ', style_mel.shape)

        shape_list = [x.shape for x in style_mel_list]
        # print('shape_list : ', shape_list)
        # mel이 차원수 값 80
        mel_level = shape_list[0][0]
        mel_length = max(shape_list, key=lambda x: x[1])[1] # mel 길이

        # input_길이 확인하고 최대 input길이 확인하기 위해서[100, 80, 50, 30]
        # ids_sorted_decreasing : 입력으로 들어온 text들의 길이를 정렬할 때 순서 확인용[2,3,0,1]
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        # print('input_lengths : ', input_lengths)
        # print("ids_sorted_decreasing : ", ids_sorted_decreasing)

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        emotion_ids = torch.LongTensor(len(batch))
        style_img = torch.FloatTensor(len(batch), mel_level, mel_length)
        style_img.zero_()
        speaker_ids = torch.LongTensor(len(batch))


        # print('data_function, TextMelCollate1 ====> ', style_img)
        # print('data_function, TextMelCollate1 shape ====> ', style_img.shape)
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
            emotion_ids[i] = batch[ids_sorted_decreasing[i]][3]
            img = batch[ids_sorted_decreasing[i]][4]
            # print('img.shape test check : ', img.shape[1])
            style_img[i, :, :img.shape[1]] = img
            speaker_ids[i] = batch[ids_sorted_decreasing[i]][5]


        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)

        # for i in range(len(ids_sorted_decreasing)):
        #     print('mel_padded.shape : ====> ', mel_padded[i].shape)
        # for j in range(len(ids_sorted_decreasing)):
        #     # print('style_img.shape : =======> ', style_img[j].shape)

        # print('data_function, TextMelCollate2 ====> ', style_img)
        # print('data_function, TextMelCollate1 shape ====> ', style_img.shape)
        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, len_x, emotion_ids, style_img, speaker_ids

def batch_to_gpu(batch):
    text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, len_x, emotion_ids, style_img, speaker_ids = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    #print('to gpu wjs data_function.py batch_to_gpu ===> ', emotion_ids)
    emotion_ids = to_gpu(emotion_ids).long()
    style_img = to_gpu(style_img).float()
    speaker_ids = to_gpu(speaker_ids).long()

    #print('data_function.py batch_to_gpu ===> ', style_img)

    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths, emotion_ids, style_img, speaker_ids)
    y = (mel_padded, gate_padded)
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
