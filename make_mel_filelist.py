file_path = 'filelists/train_file_list4.txt'
save_path = 'filelists/train_mel_file_list.txt'

f = open(file_path, 'r', encoding='utf-8')
wf = open(save_path, 'w', encoding='utf-8')

while True:
    line = f.readline()
    if not line: break

    sp_line = line.split('/')
    sp_line[2] = 'mel'
    script = '/'.join(sp_line)
    script = script.replace('.wav', '.pt')
    wf.write(script)
    print(script)

wf.close()
f.close()