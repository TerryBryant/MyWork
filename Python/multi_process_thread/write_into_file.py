# This is a python multiprocessing program
# Firstly read content from 'filenames.txt', then grep content from 'xx.txt'
# If content found, then write to the 'output.txt'
# Pay attention that the file lock is different from value lock
# By TerryBryant, 2019/06/20
from __future__ import print_function
import os
from multiprocessing import Process
import fcntl    # Notice that windows platform doesn't has this library, also notice that the last word is l, not arabic number 1

root_path = os.getcwd()


def run_proc(lines_):
    for line in lines_:
        line = line.strip()
        res = os.popen('grep %s xx.txt' % line[:-1]).read()

        if len(res) > 0:
            with open(os.path.join(root_path, 'output.txt'), 'a+') as fw:
                    fcntl.flock(fw, fcntl.LOCK_EX)
                    fw.write(res[index1:index-1] + ' ' + res[index + index2: index + index22 + 4] + '\n')
                
            print('Find one!')
                
                    
            

if __name__ == '__main__':
    num_processor = 10      # num of processors you want to use
    with open(os.path.join(root_path, 'filenames.txt'), 'r') as f:
        lines = f.readlines()

    lines_len = len(lines)
    portion = lines_len // num_processor

    record = []
    for i in range(num_processor):
        p = Process(target=run_proc, args=(lines[i * portion: (i + 1) * portion],))
        p.start()
        record.append(p)

    for r in record:
        r.join()

