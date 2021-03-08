# 출력결과를 콘솔창 실시간출력 & 텍스트 파일로 동시에 저장하기

import sys

class Logger(object):

    def __init__(self):
        log_filepath = 'GST_Tacotron_log_test.txt'
        self.terminal = sys.stdout

        self.log = open(log_filepath, "a")

    def write(self, message):

        self.terminal.write(message)

        self.log.write(message)

    def flush(self):

        #this flush method is needed for python 3 compatibility.

        #this handles the flush command by doing nothing.

        #you might want to specify some extra behavior here.

        pass