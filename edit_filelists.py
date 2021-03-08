import random

file_name = 'filelists/train_data.txt'

train_file = 'filelists/train_file_list3.txt'
test_file = 'filelists/test_file_list3.txt'
val_file = 'filelists/vla_file_list3.txt'

f = open(file_name, 'r', encoding='utf-8')
script_list = []

count = 0

while True:
    line = f.readline()
    if not line: break
    #print(line)
    script_list.append(line)

f.close()

random.shuffle(script_list)
script_len = len(script_list)

def make_filelist(file, script, start, end):
    rf = open(file, 'w', encoding='utf-8')
    for line in script[start:end]:
        #print(line)
        if '\ufeff' in line:
            print(line)
            line = line.replace('\ufeff', '')
        rf.write(line)
    rf.close()

make_filelist(train_file, script_list, 0, -600)
make_filelist(test_file, script_list, -600, -100)
make_filelist(val_file, script_list, -100, int(script_len+1))