# Adapted from
# https://github.com/prabodhhere/tsne-grid/blob/master/tsne_grid.py

import numpy as np
import os, argparse
import tensorflow as tf
from PIL import Image
from lapjv import lapjv
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

from tensorflow.python.keras.preprocessing import image

from tqdm import tqdm

from model import VGG_model
import cv2
import random


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', type=int, help="number of small images in a row/column in output image")
parser.add_argument('-d', '--dir', type=str, help="source directory for images")
parser.add_argument('-r', '--res', type=int, default=224, help="width/height of output square image")
parser.add_argument('-n', '--name', type=str, default='tsne_grid.jpg', help='name of output image file')
parser.add_argument('-p', '--path', type=str, default='./', help="destination directory for output image")
parser.add_argument('-x', '--per', type=int, default=50, help="tsne perplexity")
parser.add_argument('-i', '--iter', type=int, default=5000, help="number of iterations in tsne algorithm")

args = parser.parse_args()
out_res = args.res
out_name = args.name
out_dim = args.size

if not out_dim:
    raise ValueError("Set -s size")

to_plot = np.square(out_dim)
perplexity = args.per
tsne_iter = args.iter

if out_dim == 1:
    raise ValueError("Output grid dimension 1x1 not supported.")

if os.path.exists(args.dir):
    in_dir = args.dir
else:
    raise argparse.ArgumentTypeError(f"'{args.dir}' not a valid directory.")

if os.path.exists(args.path):
    out_dir = args.path
else:
    raise argparse.ArgumentTypeError(f"'{args.path}' not a valid directory.")


def load_img(in_dir):
    pred_img = [f for f in os.listdir(in_dir)
                    if os.path.isfile(os.path.join(in_dir, f))]
    img_collection = []

    random.shuffle(pred_img)
    pred_img = pred_img[:out_dim]
    
    for idx, img in enumerate(pred_img):

        img = cv2.imread(os.path.join(in_dir, img))
        img = img[:,:,::-1]  # BGR to RGB
        img_collection.append(img)

    if (out_dim > len(img_collection)):
        msg = f"Cannot fit {len(img_collection)} images in {out_dim}x{out_dim} grid"
        raise ValueError(msg)

    return img_collection

def get_activations(clf, img_collection):
    activations = []
    for idx, img in enumerate(tqdm(img_collection)):
        activations.append(clf.predict(img))

    return activations

def generate_tsne(activations):
    tsne = TSNE(
        perplexity=perplexity, n_components=2,
        init='random', n_iter=tsne_iter
    )
    X_2d = tsne.fit_transform(np.array(activations)[0:to_plot,:])
    X_2d -= X_2d.min(axis=0)
    X_2d /= X_2d.max(axis=0)
    return X_2d

def save_tsne_grid(img_collection, X_2d, out_res, out_dim):
    grid = np.dstack(np.meshgrid(
        np.linspace(0, 1, out_dim),
        np.linspace(0, 1, out_dim))).reshape(-1, 2)
    cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]
    out = np.ones((out_dim*out_res, out_dim*out_res, 3))

    for pos, img in zip(grid_jv, img_collection[0:to_plot]):
        h_range = int(np.floor(pos[0]* (out_dim - 1) * out_res))
        w_range = int(np.floor(pos[1]* (out_dim - 1) * out_res))
        out[
            h_range:h_range + out_res,
            w_range:w_range + out_res]  = image.img_to_array(img)

    im = image.array_to_img(out)
    im.save(out_dir + out_name, quality=100)

def main():
    clf = VGG_model()
    img_collection = load_img(in_dir)

    activations = get_activations(clf, img_collection)
    print (activations[0])

    print("Generating 2D representation.")
    X_2d = generate_tsne(activations)
    print("Generating image grid.")
    save_tsne_grid(img_collection, X_2d, out_res, int(np.sqrt(out_dim)))

if __name__ == '__main__':
    main()
