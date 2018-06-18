# Adapted from
# https://github.com/prabodhhere/tsne-grid/blob/master/tsne_grid.py

import numpy as np
from lapjv import lapjv
from scipy.spatial.distance import cdist

def generate_tsne(activations, perplexity=50, tsne_iter=5000):
    # Run tSNE in parallel if the proper library is installed

    args = {
        "perplexity" : perplexity,
        "n_components" : 2,
        "n_iter" : tsne_iter,
        "init" : "random",    
    }

    try:
        from MulticoreTSNE import MulticoreTSNE as TSNE
        args["n_jobs"] = -1
    except ModuleNotFoundError:
        from sklearn.manifold import TSNE

    X = TSNE(**args).fit_transform(np.array(activations))
    X -= X.min(axis=0)
    X /= X.max(axis=0)
    return X


def fit_to_grid(IMG, X_2d, n, out_res=224):
    grid = np.dstack(np.meshgrid(
        np.linspace(0, 1, n),
        np.linspace(0, 1, n))).reshape(-1, 2)
    
    cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    
    grid_jv = grid[col_asses]
    out = np.ones((n*out_res, n*out_res, 3))

    for pos, img in zip(grid_jv, IMG):
        h_range = int(np.floor(pos[0]* (n - 1) * out_res))
        w_range = int(np.floor(pos[1]* (n - 1) * out_res))
        out[
            h_range:h_range + out_res,
            w_range:w_range + out_res]  = img

    return out

