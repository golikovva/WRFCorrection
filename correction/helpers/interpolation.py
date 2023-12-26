import torch


def get_nearest_neighbour(a, b, n_neighbours=1, indices_only=True):
    """
    a: tensor with shape 2 x m
    b: tensor with shape 2 x n
    out: tensor with shape m
    """
    r = torch.mm(a.t(), b)
    r1 = torch.mm(a.t(), a)
    diag1 = r1.diag().unsqueeze(1)
    diag1 = diag1.expand_as(r)
    r2 = torch.mm(b.t(), b)
    diag2 = r2.diag().unsqueeze(0)
    diag2 = diag2.expand_as(r)
    D = (diag1 + diag2 - 2 * r).sqrt()
    out = torch.topk(D, n_neighbours, 1, largest=False)

    return out.indices, out.values


class Interpolator(torch.nn.Module):
    def __init__(self, values_points, interp_points):
        super().__init__()
        indices, values = get_nearest_neighbour(interp_points, values_points)
        self.nearest_neighbour = indices
        self.nearest_neighbour_values = values
        self.wrf_grid = values_points
        self.stations_grid = interp_points

    def forward(self, values):
        out = values[..., self.nearest_neighbour].clone()
        return out


if __name__ == "__main__":
    a1 = torch.rand(2, 5)
    a2 = torch.rand(2, 12)
    c = get_nearest_neighbour(a1, a2)
    print(c)
