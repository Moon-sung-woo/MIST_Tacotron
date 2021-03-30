from make_img import *

# 뉴럴 네트워크 모델을 불러옵니다.

cnn = models.vgg19(pretrained=True).features.to(device).eval()

style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

wave_path = 'dataset/fea/wav/acriil_fea_00002987.wav'
save_path = 'test_file/test_mel.png'
