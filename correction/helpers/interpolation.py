import torch
import numpy as np
from correction.helpers.math import gauss_function
from scipy.spatial import cKDTree as KDTree
from correction.config.config import cfg

def get_distances_matrix(a, b):
    r = torch.mm(a.t(), b)
    diag1 = torch.mm(a.t(), a).diag().unsqueeze(1).expand_as(r)
    diag2 = torch.mm(b.t(), b).diag().unsqueeze(0).expand_as(r)
    return (diag1 + diag2 - 2 * r).sqrt()


def get_nearest_neighbour(a, b, n_neighbours=1, indices_only=True):
    """
    a: tensor with shape 2 x m
    b: tensor with shape 2 x n
    out: tensor with shape m
    """
    D = get_distances_matrix(a, b)
    out = torch.topk(D, n_neighbours, 1, largest=False)

    return out.indices, out.values


class Interpolator(torch.nn.Module):
    def __init__(self, interp_points, values_points, mode='nearest'):
        super().__init__()
        indices, values = get_nearest_neighbour(interp_points, values_points)
        self.nearest_neighbour = indices
        print(self.nearest_neighbour)
        self.nearest_neighbour_values = values
        self.wrf_grid = values_points
        self.stations_grid = interp_points

    def calc_bilinear_coefs(self):
        pass

    def forward(self, values):
        out = values[..., self.nearest_neighbour].clone()
        return out


class InvDistTree(torch.nn.Module):
    def __init__(self, x, q, leaf_size=10, n_near=6, eps=0, dist_mode='gaussian'):
        super().__init__()
        self.ix = None
        self.distances = None
        self.dist_mode = dist_mode
        self.x = x
        self.k = 1

        self.tree = KDTree(x, leafsize=leaf_size)  # build the tree
        self.calc_interpolation_weights(q, n_near, eps)

    def calc_interpolation_weights(self, q, n_near=6, eps=0):
        q = np.asarray(q)
        self.distances, self.ix = self.tree.query(q, k=n_near, eps=eps)
        if np.where(self.distances < 1e-10)[0].size != 0:
            print('Zeros in indices!')
        self.weights = self.calc_dist_coefs(self.distances)
        self.weights = self.weights / torch.sum(self.weights, dim=-1, keepdim=True)
        self.weights = self.weights.type(torch.float).to(cfg.GLOBAL.DEVICE)
        print(self.weights.shape, 'weights.shape')

    def calc_dist_coefs(self, dist):
        if self.dist_mode == 'inverse':
            return 1 / dist
        elif self.dist_mode == 'gaussian':
            sigma_squared = np.square(self.distances.max()) / 9 / self.k
            print(sigma_squared)
            return gauss_function(dist, sigma_squared=sigma_squared)
        elif self.dist_mode == 'LinearNN':  # todo
            pass

    def __call__(self, z):
        ans1 = (z[..., self.ix]*self.weights).sum(-1)
        return ans1


if __name__ == "__main__":
    wrf_coords = torch.rand(2, 21*28)
    stations_coords = torch.rand(2, 32)
    interpolator = Interpolator(stations_coords, wrf_coords)
    wrf_data = torch.rand(4, 1, 3, 21, 28)
    s = wrf_data.shape
    wrf_data = interpolator(wrf_data.view(*s[:-2], s[-1] * s[-2]))
    print(wrf_data.shape, wrf_data[..., 2, :].shape)

    a = torch.ones([4, 1, 3, 36, 1])[..., 2, :]
    b = torch.ones([4, 1, 36, 4])[..., :, 1]
    print(torch.nn.MSELoss()(a, b))
    print(a.shape, b.shape)

