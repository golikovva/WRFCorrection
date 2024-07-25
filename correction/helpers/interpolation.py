import torch
import numpy as np
from correction.helpers.math import gauss_function
from scipy.spatial import cKDTree as KDTree


class InvDistTree(torch.nn.Module):
    def __init__(self, x, q, leaf_size=10, n_near=6, eps=0, dist_mode='gaussian', sigma_squared=None, device='cpu'):
        super().__init__()

        self.ix = None
        self.weights = None
        self.distances = None
        self.dist_mode = dist_mode
        self.x = x
        self.k = 1
        self.device = device
        self.tree = KDTree(x, leafsize=leaf_size)  # build the tree
        self.calc_interpolation_weights(q, n_near, eps, sigma_squared)

    def calc_interpolation_weights(self, q, n_near=6, eps=0, sigma_squared=None):
        q = np.asarray(q)
        self.distances, self.ix = self.tree.query(q, k=n_near, eps=eps)
        if np.where(self.distances < 1e-10)[0].size != 0:
            print('Zeros in indices!')
        self.weights = self.calc_dist_coefs(self.distances, sigma_squared)
        self.weights = self.weights / torch.sum(self.weights, dim=-1, keepdim=True)
        self.weights = self.weights.type(torch.float).to(self.device)

    def calc_dist_coefs(self, dist, sigma_squared=None):
        if self.dist_mode == 'inverse':
            return torch.from_numpy(1 / dist)
        elif self.dist_mode == 'gaussian':
            sigma_squared = sigma_squared if sigma_squared else np.square(self.distances.max()) / 9 / self.k
            print(sigma_squared, 'sigma_squared')
            return gauss_function(dist, sigma_squared=sigma_squared)
        elif self.dist_mode == 'LinearNN':  # todo
            pass

    def __call__(self, z):
        ans1 = (z[..., self.ix] * self.weights).sum(-1)
        return ans1

    def calc_input_tensor_mask(self, mask_shape, distance_criterion=0.15, fill_value=0):
        s = mask_shape
        assert s[-1] * s[-2] == self.distances.shape[0], "mask shape should be compatible with calculated distances"
        mask = torch.ones([s[-1] * s[-2]])
        mask[np.where(self.distances.mean(-1) > distance_criterion)] = fill_value
        mask = mask.reshape(*s).to(self.device)
        return mask

    def to(self, device):
        self.device = device
        if self.weights is not None:
            self.weights = self.weights.to(device)
        return self


def interpolate_input_to_scat(input_data, scatter_data, interpolator, i, start_date, input_mask,
                              return_counts=False):
    scat_wind = scatter_data[:, :, :2].clone()
    scat_mask = scatter_data[:, :, 2]
    scat_time = scatter_data[:, :, 3]

    seq_len = 4
    start_date = int(start_date)
    i_date1 = (start_date + i * 3600)[:, None, None].to(scatter_data.device)  # время первого элемента
    wrf_on_scat_grid = interpolator(input_data.flatten(-2, -1))

    scat_dates = (scat_time * scat_mask * input_mask).flatten(-2, -1)
    scat_wind = (scat_wind * scat_mask[:, :, None] * input_mask).flatten(-2, -1)
    counts = torch.zeros_like(scat_dates[:, 0])
    wrfs = []
    scatters = []
    for t in range(seq_len - 1):
        wrf0 = wrf_on_scat_grid[t]  # ([12, 2, 56760]) wrf0 shape ([bs, c, h*w])
        wrf1 = wrf_on_scat_grid[t + 1]

        date0 = i_date1 + 3600 * t
        date1 = date0 + 3600 * 1
        dates_in_range = torch.logical_and(scat_dates > date0,
                                           scat_dates < date1)
        if s := (dates_in_range[:, 0] * dates_in_range[:, 1]).sum() > 0:
            print('dates_in_range asc desc intercepts', s)

        scat_wind_dated = torch.nansum((dates_in_range[:, :, None] * scat_wind), dim=1)
        alpha = (scat_dates - date0) / (date1 - date0)
        alpha = dates_in_range * alpha  # ([12, 2, 2, 56760]) alpha shape ([bs, sl, file, h*w])
        alpha = alpha.sum(dim=1)
        dates_in_range = dates_in_range.sum(dim=1)
        counts += dates_in_range
        wrf_interpolated = (wrf0 + (wrf1 - wrf0) * alpha[:, None]) * dates_in_range[:, None]
        wrf_interpolated = wrf_interpolated.nan_to_num()
        wrfs.append(wrf_interpolated)
        scatters.append(scat_wind_dated)

    res = (torch.stack(wrfs, 1), torch.stack(scatters, 1))
    if return_counts:
        res += (counts,)
    return _unpack_tuple(res)


def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x


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
        self.nearest_neighbour_values = values
        self.wrf_grid = values_points
        self.stations_grid = interp_points

    def calc_bilinear_coefs(self):
        pass

    def forward(self, values):
        out = values[..., self.nearest_neighbour].clone()
        return out


def create_mask_by_nearest_to_nans(wind, coords, fill_value=0):
    nn_ids = InvDistTree(coords, coords, n_near=9).ix
    s = wind.shape
    num_neighbour_nans = np.isnan(wind.reshape(*s[:-2], -1)[..., nn_ids]).sum(-1)
    mask = np.ones_like(wind.reshape(*s[:-2], -1))
    mask[np.where(num_neighbour_nans > 0)] = fill_value
    return mask.reshape(*s)


if __name__ == "__main__":
    wrf_coords = torch.rand(2, 21 * 28)
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
