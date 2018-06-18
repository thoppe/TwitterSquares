"""Render Twitter Squares

Usage:
  render.py <term> <n_images>

Options:
  -h --help       Show this screen.

"""

import glob
import os
import sys
import random
from tqdm import tqdm
import numpy as np
import cv2

from docopt import docopt
dargs = docopt(__doc__)

total_images = int(dargs["<n_images>"])
max_image_row_size = 20

VGG_size = 224

name = dargs["<term>"]
load_dest = f"data/profile_image/{name}"
subimage_dest = f"data/subimage/{name}"
activations_dest = f"data/activations/{name}"

figure_dest = "figures/"

def resize_and_crop(f0):
    # Resize all the images to the base shape of (VGG_size,VGG_size)
    # Center crop non-square images

    f1 = os.path.join(subimage_dest, os.path.basename(f0)) + '.jpg'
    if os.path.exists(f1):
        return False
    
    img = cv2.imread(f0)

    if img is None:
        os.remove(f0)
        return False
    
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
    #print ("Saved", f1)


_clf = None

def compute_activations(f0):

    f1 = os.path.join(activations_dest, os.path.basename(f0)) + '.txt'
    if os.path.exists(f1):
        return False

    global _clf
    if _clf is None:
        print("Importing VGG model")
        from model import VGG_model as VGG
        _clf = VGG()

    img = cv2.imread(f0)
    img = img[:,:,::-1]  # BGR to RGB
    ax = _clf.predict(img)

    np.savetxt(f1, ax)


# Create any missing directories
for d in [subimage_dest, figure_dest, activations_dest]:
    if not os.path.exists(d):
        os.system(f'mkdir -p "{d}"')

F_IN = set(sorted(glob.glob(os.path.join(load_dest, '*'))))

# Remove all zero-byte files
for f in list(F_IN):
    if os.stat(f).st_size==0:
        print(f"Removing zero-byte file {f}")
        os.remove(f)
        F_IN.remove(f)

for f0 in tqdm(F_IN):
    resize_and_crop(f0)

F_IN = set(sorted(glob.glob(os.path.join(subimage_dest, '*'))))
for f0 in tqdm(F_IN):
    compute_activations(f0)


F_IN = set(sorted(glob.glob(os.path.join(activations_dest, '*'))))
if len(F_IN) < total_images:
    msg = f"Not enough images for {name}, {len(F_IN)}/{total_images}"
    raise ValueError(msg)

#n_found_files = len(glob.glob(os.path.join(subimage_dest, '*')))
#n_size = total_images
#print (name, len(F_IN), n_size)

cmd = f"python grid.py -s {total_images} -d '{subimage_dest}' -p {figure_dest} --name '{name}.jpg' -a {activations_dest}"

print(cmd)

val = os.system(cmd)
if not val:
    os.system(f'eog "figures/{name}.jpg"')
