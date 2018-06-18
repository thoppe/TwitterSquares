# Adapted from
# https://github.com/prabodhhere/tsne-grid/blob/master/tsne_grid.py

import numpy as np
from lapjv import lapjv
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import cv2

from tqdm import tqdm
import random

def generate_tsne(activations, perplexity=50, tsne_iter=5000):
    tsne = TSNE(
        perplexity=perplexity,
        n_components=2,
        n_iter=tsne_iter,
        init='random', 
    )
    X_2d = tsne.fit_transform(np.array(activations))
    X_2d -= X_2d.min(axis=0)
    X_2d /= X_2d.max(axis=0)
    return X_2d

def fit_to_grid(img_collection, X_2d, out_dim, out_res=224):
    grid = np.dstack(np.meshgrid(
        np.linspace(0, 1, out_dim),
        np.linspace(0, 1, out_dim))).reshape(-1, 2)
    
    cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    
    grid_jv = grid[col_asses]
    out = np.ones((out_dim*out_res, out_dim*out_res, 3))

    for pos, img in zip(grid_jv, img_collection):
        h_range = int(np.floor(pos[0]* (out_dim - 1) * out_res))
        w_range = int(np.floor(pos[1]* (out_dim - 1) * out_res))
        out[
            h_range:h_range + out_res,
            w_range:w_range + out_res]  = img

    return out

