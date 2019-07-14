# This is a python multiprocessing program
# Firstly read content from 'filenames.txt', then grep content from 'xx.txt'
# If content found, then write to the 'output.txt'
# Pay attention that the file lock is different from value lock
# By TerryBryant, 2019/06/20
# Update in 2019/07/14, use pool to create new processor seems to be faster, see more from below website
# https://www.ellicium.com/python-multiprocessing-pool-process/
from __future__ import print_function
import os
import multiprocessing
import fcntl    # Notice that windows platform doesn't have this library, also notice that the last word is l, not arabic number 1

root_path = os.getcwd()


def run_proc(lines_):
    for line in lines_:
        line = line.strip()
        res = os.popen('grep %s xx.txt' % line[:-1]).read()

        if len(res) > 0:
            with open(os.path.join(root_path, 'output.txt'), 'a+') as fw:
                    fcntl.flock(fw, fcntl.LOCK_EX)
                    fw.write(res + '\n')
                
            print('Find one!')
                
                    
            

if __name__ == '__main__':		 # Notie that multi-processor program must run in the '__main__' function
    num_processor = 10      # num of processors you want to use
    with open(os.path.join(root_path, 'filenames.txt'), 'r') as f:
        lines = f.readlines()

    portion = len(lines) // num_processor
	
	pool = multiprocessing.Pool(num_processor + 1)	# pool takes less time to create 
	for i in range(num_processor + 1):
	if i == num_processor:
		pool.map_async(run_proc, (lines[i * portion:],))	# you can use 'get()' method to retrieve the return value
	else:
		pool.map_async(run_proc, (lines[i * portion: (i + 1) * portion],))

#     record = []
#     for i in range(num_processor + 1):
#         if i == num_processor:
#             p = multiprocessing.Process(target=run_proc, args=(lines[i * portion:],))
#         else:
#             p = multiprocessing.Process(target=run_proc, args=(lines[i * portion: (i + 1) * portion],))
#         p.start()
#         record.append(p)

#     for r in record:
#         r.join()

