# 필요한 PyTorch 라이브러리 불러오기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from common.layers import TacotronSTFT
from common.utils import load_wav_to_torch

import copy
import PIL
import os
from PIL import Image
import matplotlib.pyplot as plt
import librosa.display

# GPU 장치 사용 설정
GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU

import time


# 이미지를 불러와 다운받아 텐서(Tensor) 객체로 변환하는 함수
def image_loader(img_path):
    loader = transforms.Compose([
        transforms.ToTensor()  # torch.Tensor 형식으로 변경 [0, 255] → [0, 1]
    ])
    image = PIL.Image.open(img_path).convert('RGB')
    # 네트워크 입력에 들어갈 이미지에 배치 목적의 차원(dimension) 추가
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)  # GPU로 올리기


def get_png_name(pt_path):
    # pt_path = ''test_file/nem00397.wav''
    pt_path = pt_path.split('/')
    png_name = pt_path[1].replace('.wav', '.png')
    pt_path[1] = png_name
    pt_path[0] = 'test_single_mel1_img'
    style_png_name = '/'.join(pt_path)

    return style_png_name

# torch.Tensor 형태의 이미지를 화면에 출력하는 함수
def imshow(tensor):
    # matplotlib는 CPU 기반이므로 CPU로 옮기기
    image = tensor.cpu().clone()
    # torch.Tensor에서 사용되는 배치 목적의 차원(dimension) 제거
    image = image.squeeze(0)
    # PIL 객체로 변경
    image = transforms.ToPILImage()(image)
    # 이미지를 화면에 출력(matplotlib는 [0, 1] 사이의 값이라고 해도 정상적으로 처리)
    plt.imshow(image)
    plt.show()


def eshow(image):
    img = Image.open(image)
    img.show()

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def gram_matrix(input):
    # a는 배치 크기, b는 특징 맵의 개수, (c, d)는 특징 맵의 차원을 의미
    a, b, c, d = input.size()
    # 논문에서는 i = 특징 맵의 개수, j = 각 위치(position)
    features = input.view(a * b, c * d)
    # 행렬 곱으로 한 번에 Gram 내적 계산 가능
    G = torch.mm(features, features.t())
    # Normalize 목적으로 값 나누기
    return G.div(a * b * c * d)


# 스타일 손실(style loss) 계산을 위한 클래스 정의
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# 스타일 손실(style loss)을 계산하는 함수
def get_style_losses(cnn, style_img, noise_image):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)
    style_losses = []

    # 가장 먼저 입력 이미지가 입력 정규화(input normalization)를 수행하도록
    model = nn.Sequential(normalization)

    # 현재 CNN 모델에 포함되어 있는 모든 레이어를 확인하며
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # 설정한 style layer까지의 결과를 이용해 style loss를 계산
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 마지막 style loss 이후의 레이어는 사용하지 않도록
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses


def style_reconstruction(cnn, style_img, input_img, iters):
    model, style_losses = get_style_losses(cnn, style_img, input_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print("[ Start ]")
    #imshow(input_img)

    flag = True
    now_score = 100000000

    # 하나의 값만 이용하기 위해 배열 형태로 사용
    run = [0]
    while flag:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0

            for sl in style_losses:
                style_score += sl.loss

            style_score *= 1e6
            style_score.backward()
            run[0] += 1

            nonlocal flag, now_score

            if style_score.item() < 1:
                # print(f"[ Step: {run[0]} / Style loss: {style_score.item()}]")
                # imshow(input_img)
                flag = False

            if style_score.item() <= now_score:
                now_score = style_score.item()

            return style_score


        optimizer.step(closure)

    # 결과적으로 이미지의 각 픽셀의 값이 [0, 1] 사이의 값이 되도록 자르기
    input_img.data.clamp_(0, 1)

    return input_img

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


# 뉴럴 네트워크 모델을 불러옵니다.

cnn = models.vgg19(pretrained=True).features.to(device).eval()

style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# style transfer
root_path = 'test_single_audio'
file_list = os.listdir(root_path)

for test_file in file_list:
    wav_path = os.path.join(root_path, test_file)
    print('wav_path : ', wav_path)

    # 저장할 png file 이름
    png_name = get_png_name(wav_path)
    print('png_name : ', png_name)

    # 기존 .pt파일을 png파일로 저장
    m = load_mel(wav_path)
    m = m.numpy() # 이미지로 만들기 위해 넘파이로 변경
    a, b = m.shape

    librosa.display.specshow(m)
    plt.Figure()
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig(png_name, bbox_inches='tight', pad_inches=0)
    plt.close()

    # 이미지 resize
    img = Image.open(png_name)
    shape_check_img = image_loader(png_name)
    resize_image = img.resize((int(b * 1.5), shape_check_img.shape[2]))
    resize_image.save(png_name)
    target_image = image_loader(png_name)

    # 콘텐츠 이미지와 동일한 크기의 노이즈 이미지 준비하기
    input_img = torch.empty_like(target_image).uniform_(0, 1).to(device)

    # style transfer 시작
    # style reconstruction 수행
    print('style 추출 중')
    output = style_reconstruction(cnn, style_img=target_image, input_img=input_img, iters=1000)
    print('style 추출 끝남')

    # style transfer한 이미지 저장
    # matplotlib는 CPU 기반이므로 CPU로 옮기기
    image = output.cpu().clone()
    # torch.Tensor에서 사용되는 배치 목적의 차원(dimension) 제거
    image = image.squeeze(0)
    # PIL 객체로 변경
    image = transforms.ToPILImage()(image)
    # 이미지를 화면에 출력(matplotlib는 [0, 1] 사이의 값이라고 해도 정상적으로 처리)

    fig = plt.figure()
    plt.imshow(image)

    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    plt.savefig(png_name, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    final_temp_img = Image.open(png_name)
    final_img_shape = image_loader(png_name)
    final_img = final_temp_img.resize((int(b * 1.5), output.shape[2]))
    final_img.save(png_name)
