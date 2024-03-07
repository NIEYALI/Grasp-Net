import os
import glob
import random

path = '../myModel/test'

files = glob.glob(os.path.join(path, '*', '*.urdf'))
random.shuffle(files)

txt = open(path + '/list.txt', 'w+')
for f in files:
    fname = os.path.basename(f)
    pre_fname = os.path.basename(os.path.dirname(f))
    txt.write(pre_fname + '/' + fname[:-5] + '\n')
txt.close()
print('done')

    
