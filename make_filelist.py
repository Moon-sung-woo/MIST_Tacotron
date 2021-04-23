import os
import glob

def get_folder_list(root):
    return os.listdir(root)

def get_transcript(wav_path):
    transcript_path = wav_path.replace('/wav', '/transcript')
    transcript_path = transcript_path.replace('.wav', '.txt')

    sf = open(transcript_path, 'r', encoding='utf-8')
    line = sf.readline()
    sf.close()

    return line

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


root = 'emotion_dataset'
save_path = 'filelists/sum_multy_filelist.txt'

#f = open(save_path, 'w', encoding='utf-8')

folder_list = get_folder_list(root)
print(folder_list)

for folder in folder_list:
    if folder == 'emotional-to-emotional' or folder == 'plain-to-emotional':
        folder = os.path.join(root, folder)
        s_folder_list = get_folder_list(folder)

        for s_folder in s_folder_list:
            wav_path_list = sorted(glob.glob(os.path.join(folder, s_folder, 'wav') + '/*.wav'))
            print(os.path.join(folder, s_folder, 'mel'))
            createFolder(os.path.join(folder, s_folder, 'img'))

            for wav_path in wav_path_list[:100]:
                # print(wav_path)
                transcript = get_transcript(wav_path)
                # print(transcript)
                #f.write(wav_path + '|' + transcript[:-1] + '|' + '0|' + s_folder + '\n')
            for wav_path in wav_path_list[100:200]:
                # print(wav_path)
                transcript = get_transcript(wav_path)
                # print(transcript)
                #f.write(wav_path + '|' + transcript[:-1] + '|' + '1|' + s_folder + '\n')
            for wav_path in wav_path_list[200:300]:
                # print(wav_path)
                transcript = get_transcript(wav_path)
                # print(transcript)
                #f.write(wav_path + '|' + transcript[:-1] + '|' + '2|' + s_folder + '\n')
            for wav_path in wav_path_list[300:]:
                # print(wav_path)
                transcript = get_transcript(wav_path)
                # print(transcript)
                #f.write(wav_path + '|' + transcript[:-1] + '|' + '3|' + s_folder + '\n')

    else:
        folder = os.path.join(root, folder)
        s_folder_list = get_folder_list(folder)

        for s_folder in s_folder_list:
            wav_path_list = sorted(glob.glob(os.path.join(folder, s_folder, 'wav') + '/*.wav'))
            createFolder(os.path.join(folder, s_folder, 'img'))
            for wav_path in wav_path_list:
                # print(wav_path)
                transcript = get_transcript(wav_path)
                # print(transcript)
                #f.write(wav_path + '|' + transcript[:-1] + '|' + '0|' + s_folder + '\n')

#f.close()