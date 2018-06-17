"""Render Twitter Squares

Usage:
  render.py <term> <n_images>

Options:
  -h --help       Show this screen.

"""

import glob
import math
import os
import shutil
import random
from unidecode import unidecode
import sys
from tqdm import tqdm
import cv2
import docopt
from docopt import docopt

dargs = docopt(__doc__)

total_images = int(dargs["<n_images>"])
max_image_row_size = 20

VGG_size = 224

name = dargs["<term>"]
load_dest = f"data/profile_image/{name}"
subimage_dest = f"data/processed_image/{name}"
figure_dest = "figures/"

if not os.path.exists(subimage_dest):
    os.system(f'mkdir -p "{subimage_dest}"')

if not os.path.exists(figure_dest):
    os.system(f'mkdir -p "{figure_dest}"')

F_IN = sorted(glob.glob(os.path.join(load_dest, '*')))
F_IN = [x for x in F_IN if os.stat(x).st_size>0]

if len(F_IN) < total_images:
    msg = f"Not enough images for {name}, {len(F_IN)}/{total_images}"
    raise ValueError(msg)

# Resize all the images to the base shape of (VGG_size,VGG_size)
# Center crop non-square images

for f0 in tqdm(F_IN):
    f1 = os.path.join(subimage_dest, os.path.basename(f0)) + '.jpg'
    if os.path.exists(f1):
        continue
    
    img = cv2.imread(f0)

    if img is None:
        os.remove(f0)
        continue
    
    x,y,c = img.shape

    if x > y:
        dx = (x - y)//2
        img = img[dx:dx+y, :, :]
    if y > x:
        dy = y - x
        img = img[:, dy:dy+x, :]

    img = cv2.resize(img, (VGG_size,VGG_size))
    x,y,c = img.shape
    assert(x==y==VGG_size)

    cv2.imwrite(f1, img)
    print ("Saved", f1)



n_found_files = len(glob.glob(os.path.join(subimage_dest, '*')))
n_size = total_images

print (name, len(F_IN), n_size)

cmd = f"python grid.py -s {n_size} -d '{subimage_dest}' -p {figure_dest} --name '{name}.jpg'"

print(cmd)


val = os.system(cmd)
if not val:
    os.system(f'eog "figures/{name}.jpg"')
