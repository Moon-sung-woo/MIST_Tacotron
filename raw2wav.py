import os
import glob
from tqdm import tqdm


file_path = 'dataset'
file_list_path = 'filelists/train_data.txt'
emotion_list = glob.glob(os.path.join(file_path, '*'))

f = open(file_list_path, 'w')

for emotion in tqdm(emotion_list):
    wav_path = emotion + "/wav"
    raw_path = emotion + "/raw"
    txt_path = emotion + "/txt"

    try:
        if not os.path.exists(wav_path):
            os.makedirs(wav_path)
    except:
        pass

    raw_file_list = glob.glob(os.path.join(raw_path, '*'))

    for raw_file in raw_file_list:
        file_name = raw_file.split('/')[3]
        wav_file = wav_path + "/" + file_name.replace(".raw", ".wav")
        txt_file = txt_path + "/" + file_name.replace(".raw", ".txt")

        # file_list 작성
        rf = open(txt_file, 'r', encoding='utf-8')
        line = rf.readline()
        if line[-1:] == '\n':
            f.write(wav_file + '|' + line)
        else:
            f.write(wav_file + '|' + line + '\n')
        print(wav_file + '|' + line)
        rf.close()

        # wav file 생성
        os.system("sox -r 16k -e signed-integer -b 16 -c 1 {} {}".format(raw_file, wav_file))

f.close()