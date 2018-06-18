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
square_n = int(np.sqrt(total_images))

if square_n**2 != total_images:
    raise ValueError(f"<n_images={total_images}> must be a square number!")

max_image_row_size = 20

#model_img_size = 224
model_img_size = 299

name = dargs["<term>"]
load_dest = f"data/profile_image/{name}"
subimage_dest = f"data/subimage/{name}"
activations_dest = f"data/activations/{name}"

figure_dest = "figures/"

def resize_and_crop(f0):
    # Resize all the images to the base shape of (model_img_size,model_img_size)
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

    img = cv2.resize(img, (model_img_size,model_img_size))
    x,y,c = img.shape
    assert(x==y==model_img_size)

    cv2.imwrite(f1, img)
    #print ("Saved", f1)


def load_image_data():

    F_INPUT = sorted(glob.glob(os.path.join(subimage_dest, '*')))
    random.shuffle(F_INPUT)
    F_INPUT = F_INPUT[:total_images]

    IMG, ACT = [], []
    for f0 in tqdm(F_INPUT):
        f1 = os.path.join(activations_dest, os.path.basename(f0))+'.txt'
        assert(os.path.exists(f1))
        img = cv2.imread(f0)

        IMG.append(img)
        ACT.append(np.loadtxt(f1))

    IMG = np.array(IMG)
    ACT = np.array(ACT)
    
    return IMG, ACT



_clf = None   # Only import the model if we need to score something
def compute_activations(f0):

    f1 = os.path.join(activations_dest, os.path.basename(f0)) + '.txt'
    if os.path.exists(f1):
        return False

    global _clf
    if _clf is None:
        print("Importing classification model")
        from model import layer_model
        _clf = layer_model()

    img = cv2.imread(f0)
    img = img[:,:,::-1]  # BGR to RGB
    ax = _clf.predict(img)

    np.savetxt(f1, ax)


if __name__ == "__main__":

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

    print(f"Largest model possible {int(np.floor(len(F_IN)**0.5)**2)}")

    F_IN = set(sorted(glob.glob(os.path.join(subimage_dest, '*'))))
    for f0 in tqdm(F_IN):
        compute_activations(f0)

    # Check to make sure we have enough images
    F_IN = set(sorted(glob.glob(os.path.join(activations_dest, '*'))))
    if len(F_IN) < total_images:
        msg = f"Not enough images for {name}, {len(F_IN)}/{total_images}"
        raise ValueError(msg)
    
    IMG, ACT = load_image_data()

    from grid import generate_tsne, fit_to_grid
    print("Generating tSNE coordinates")
    X = generate_tsne(ACT)

    print("Running Jonker-Volgenan")
    img = fit_to_grid(IMG, X, square_n, out_res=model_img_size)

    f_img_save = os.path.join(figure_dest, f"{name}.jpg")
    cv2.imwrite(f_img_save, img)
    print (f"Saved output image to {f_img_save}")

    os.system(f'eog "figures/{name}.jpg"')
