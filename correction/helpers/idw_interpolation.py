import numpy as np
import torch
from scipy.spatial import cKDTree as KDTree
from correction.helpers.math import gauss_function


class Invdisttree:
    def __init__(self, X, q, leaf_size=10, n_near=6, eps=0, dist_mode='gaussian'):
        self.ix = None
        self.distances = None
        self.dist_mode = dist_mode
        self.X = X
        self.k = 1

        self.tree = KDTree(X, leafsize=leaf_size)  # build the tree
        self.calc_interpolation_weights(q, n_near, eps)

    def calc_interpolation_weights(self, q, n_near=6, eps=0):
        q = np.asarray(q)
        self.distances, self.ix = self.tree.query(q, k=n_near, eps=eps)
        if np.where(self.distances < 1e-10)[0].size != 0:
            print('Zeros in indices!')
        self.weights = self.calc_dist(self.distances)
        self.weights = self.weights / np.sum(self.weights, axis=-1, keepdims=True)
        self.weights = torch.from_numpy(self.weights).to('cuda')
        print(self.weights.shape, 'weights.shape')

    def calc_dist(self, dist):
        if self.dist_mode == 'inverse':
            return 1 / dist
        elif self.dist_mode == 'gaussian':
            sigma_squared = np.square(self.distances.max()) / 9 / self.k
            print(sigma_squared)
            return gauss_function(dist, sigma_squared=sigma_squared)

    def __call__(self, z):
        ans1 = (z[..., self.ix]*self.weights).sum(-1)
        return ans1


if __name__ == "__main__":
    N = 58800
    Ndim = 2
    Nask = 36  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
    Nnear = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    leafsize = 10
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    p = 1  # weights ~ 1 / distance**p
    cycle = .25
    seed = 1

    np.random.seed(seed)
    np.set_printoptions(3, threshold=100, suppress=True)  # .3f

    print("\nInvdisttree:  N %d  Ndim %d  Nask %d  Nnear %d  leafsize %d  eps %.2g  p %.2g" % (
        N, Ndim, Nask, Nnear, leafsize, eps, p))


    def terrain(x):
        """ ~ rolling hills """
        return np.sin((2 * np.pi / cycle) * np.mean(x, axis=-1))


    known = np.random.uniform(size=(N, Ndim)) ** .5  # 1/(p+1): density x^p
    z = terrain(known)
    ask = np.random.uniform(size=(Nask, Ndim))

    invdisttree1 = Invdisttree(known, ask, dist_mode='inverse')
    invdisttree2 = Invdisttree(known, ask, dist_mode='gaussian')
    print(invdisttree1.weights, '\n', invdisttree2.weights)
    ans1 = invdisttree1(z)
    ans2 = invdisttree2(z)

    print("average distances to nearest points: %s" % \
          np.mean(invdisttree1.distances, axis=0))
    # see Wikipedia Zipf's law
    err1 = np.abs(terrain(ask) - ans1)
    err2 = np.abs(terrain(ask) - ans2)

    print("average |terrain() - interpolated|: %.2g %.2g" % (np.mean(err1), np.mean(err2)))
    print(terrain(ask), '\n', ans1, '\n', ans2)
