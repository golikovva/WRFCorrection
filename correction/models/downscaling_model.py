import torch
import torch.nn as nn
import numpy as np
from correction.helpers.interpolation import get_nearest_neighbour, Interpolator


class LinearDownscaling(nn.Module):
    def __init__(self, metadata, n_neighbours=4):
        super().__init__()

        wrf_grid_coords = torch.from_numpy(np.stack([metadata['wrf_xx'].flatten(),
                                                     metadata['wrf_xx'].flatten()]))
        stations_coords = torch.from_numpy(metadata['Coords']).t()
        print(stations_coords.shape, wrf_grid_coords.shape, 'grids shapes')
        print(stations_coords.dtype, wrf_grid_coords.dtype, 'grids dtypes')
        self.n_neighbours = n_neighbours
        self.interpolator = Interpolator(wrf_grid_coords.float(), stations_coords.float()).float()
        self.model = nn.Linear(n_neighbours, 1)

    def forward(self, input):

        return output


if __name__ == "__main__":
    model = LinearDownscaling()
