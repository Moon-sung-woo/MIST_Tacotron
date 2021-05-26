from f0.extract_f0 import *
import numpy as np
import os
import glob

def infer_change(org_f0, infer_f0):
    gap = len(org_f0) - len(infer_f0)
    temp_list = []
    for i in range(gap):
        temp_list.append(0)
    temp_list = np.array(temp_list)
    # print(temp_list)
    infer_f0 = np.append(infer_f0, temp_list)
    return infer_f0

def get_GPE(org_f0, infer_f0):
    sum_v = 0
    sum_error = 0
    for i, f0 in enumerate(org_f0):
        if f0>0:
            sum_v +=1
            if infer_f0[i] >= f0*1.2 or infer_f0[i] <= f0*0.8:
                sum_error += 1
    GPE = sum_error/len(infer_f0)*100
    return GPE

def get_VDE(org_f0, infer_f0):
    sum_error = 0
    for i, f0 in enumerate(org_f0):
        if f0 > 0 and infer_f0[i] ==0:
            sum_error += 1
        elif f0 == 0 and infer_f0[i] >0:
            sum_error += 1
    VDE = sum_error/len(infer_f0)*100
    return VDE

def get_FFE(GPE, VDE):
    return GPE + VDE


# root = '../inference_audio'
# file_list = os.listdir(root)
# real_file_list = []
# for file in file_list:
#     if file[0] == 'm':
#         real_file_list.append(file)
#
# for file in real_file_list:
#     org_wav_path_list = []
#     result_list = []
#     file_path = os.path.join(root, file) # multi_emotion
#     all_file_list = glob.glob(file_path + '/*')
#     print(file)
#     for f in all_file_list:
#         if os.path.isdir(f):
#             result_list.append(f)
#         else:
#             org_wav_path_list.append(f)
#     org_wav_path_list = sorted(org_wav_path_list)
#     #print(result_list)
#     # print(org_wav_path)
#     for result in result_list:
#         result_name = result.split('/')[-1]
#         if result_name == 'GST_result':
#             result_wav_list = glob.glob(result + '/*.wav')
#             result_wav_list = sorted(result_wav_list)
#             GST_wav_file_list = result_wav_list
#         elif result_name == 'VAE_result':
#             result_wav_list = glob.glob(result + '/*.wav')
#             result_wav_list = sorted(result_wav_list)
#             VAE_wav_file_list = result_wav_list
#         elif result_name == 'mist_result':
#             result_wav_list = glob.glob(result + '/*.wav')
#             result_wav_list = sorted(result_wav_list)
#             mist_wav_file_list = result_wav_list
#     print('org : ', org_wav_path_list)
#     print('GST : ', GST_wav_file_list)
#     print('VAE : ', VAE_wav_file_list)
#     print('mist : ', mist_wav_file_list)
#
#     for i, org in enumerate(org_wav_path_list):
#         GST_wav_path = GST_wav_file_list[i]
#         VAE_wav_path = VAE_wav_file_list[i]
#         mist_wav_path = mist_wav_file_list[i]
#
#         org_audio, org_sampling_rate = load_wav_to_torch(org)
#         GST_audio, GST_sampling_rate = load_wav_to_torch(GST_wav_path)
#         VAE_audio, VAE_sampling_rate = load_wav_to_torch(VAE_wav_path)
#         mist_audio, mist_sampling_rate = load_wav_to_torch(mist_wav_path)

# org_audio_path = '../inference_audio/multi_style/avb00328.wav'
# infer_audio_path = '../inference_audio/multi_style/GST_result/audio_avb00328.wav'
#
# org_audio, org_sampling_rate = load_wav_to_torch(org_audio_path)
# infer_audio, infer_sampling_rate = load_wav_to_torch(infer_audio_path)
#
# org_f0 = get_f0(org_audio, sampling_rate=org_sampling_rate)
# infer_f0 = get_f0(infer_audio, sampling_rate=infer_sampling_rate)
#
# if len(org_f0) > len(infer_f0):
#     infer_f0 = infer_change(org_f0, infer_f0)

org_file_path = '../inference_audio/single_emotion'
infer_file_path = '../inference_audio/single_emotion/mist1_result'

org_wav_file_list = glob.glob(org_file_path + '/*.wav')
infer_wav_file_list = glob.glob(infer_file_path + '/*.wav')

org_wav_file_list = sorted(org_wav_file_list)
infer_wav_file_list = sorted(infer_wav_file_list)
print(org_wav_file_list)
print(infer_wav_file_list)
gpe_result = []
vde_result = []
ffe_result = []
for i in range(len(org_wav_file_list)):
    org_audio_path = org_wav_file_list[i]
    infer_audio_path = infer_wav_file_list[i]

    org_audio, org_sampling_rate = load_wav_to_torch(org_audio_path)
    infer_audio, infer_sampling_rate = load_wav_to_torch(infer_audio_path)

    org_f0 = get_f0(org_audio, sampling_rate=org_sampling_rate)
    infer_f0 = get_f0(infer_audio, sampling_rate=infer_sampling_rate)

    if len(org_f0) > len(infer_f0):
        infer_f0 = infer_change(org_f0, infer_f0)

    gpe = get_GPE(org_f0, infer_f0)
    gpe_result.append(gpe)
    vde = get_VDE(org_f0, infer_f0)
    vde_result.append(vde)
    ffe = get_FFE(gpe, vde)
    ffe_result.append(ffe)
print(gpe_result)
print(vde_result)
print(ffe_result)