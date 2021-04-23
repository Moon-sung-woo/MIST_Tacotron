file_path = 'filelists/sum_multy_filelist.txt'
save_path = 'filelists/sum_mel_multy_filelist.txt'

f = open(file_path, 'r', encoding='utf-8')
wf = open(save_path, 'w', encoding='utf-8')

while True:
    line = f.readline()
    if not line: break


    sp_line = line.split('/')
    sp_line[3] = 'mel'
    script = '/'.join(sp_line)
    script = script.replace('.wav', '.pt')
    wf.write(script)
    print(script)

    # sp_line = line.split('/')
    # sp_line[2] = 'mel'
    # script = '/'.join(sp_line)
    # script = script.replace('.wav', '.pt')
    # wf.write(script)
    # print(script)

wf.close()
f.close()