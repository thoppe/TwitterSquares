import glob
import math
import os
import codecs
import json
import re, shutil
import random
from unidecode import unidecode
import sys

name = sys.argv[1]
save_dest = "tsne/images/{}".format(name)
load_dest = "data/profile_image/{}".format(name)

if not os.path.exists(save_dest):

    os.system('mkdir -p "{}"'.format(save_dest))

F_IN = sorted(glob.glob(os.path.join(load_dest, '*')))
F_IN = [x for x in F_IN if os.stat(x).st_size>0]

random.shuffle(F_IN)

for f_img in F_IN[:180]:
    f_base = os.path.basename(f_img)
    shutil.copyfile(f_img, os.path.join(save_dest, f_base))

n_found_files = len(glob.glob(os.path.join(save_dest, '*')))

n_size =  (math.floor(n_found_files**0.5))
n_size = min(12, n_size)

print (name, len(F_IN), n_size)

cmd = "python grid.py -s {} -d '{}' -p tsne/ --name '{}.jpg'".format(
    n_size, save_dest, name)

os.system(cmd)

os.system('eog "tsne/{}.jpg"'.format(name))
