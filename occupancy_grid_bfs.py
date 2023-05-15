import sys
from warnings import warn

import torch
import torch.nn.functional as F
from torch.nn import Parameter


class BFS(torch.nn.Module):

    def __init__(self, *, ndim=2, kernel_size=3, mode='connect-4'):
        super().__init__()
        # Transition kernels to some state.
        ks = [kernel_size]*ndim
        kd = kernel_size**ndim
        nums = torch.arange(kd)
        kernels = (nums.view(1, *ks) == nums.view(kd, *[1]*ndim)).int()
        kernels = kernels[:, None].float()
        assert kernel_size & 1
        nums = torch.arange(kernel_size).float() - kernel_size // 2
        coords = torch.meshgrid(*[nums for i in range(ndim)], indexing='ij')
        dists = torch.norm(torch.stack(coords), dim=0)
        costs = (kernels*dists).sum(dim=(-1, -2)).squeeze()
        if mode == 'connect-4':
            keep = costs <= 1.0
            kernels = kernels[keep]
            costs = costs[keep]
        elif mode == 'connect-8':
            pass
        else:
            raise ValueError(mode)
        self.ndim               = ndim
        self.kernel_size        = kernel_size
        self.kernels            = Parameter(kernels)
        self.costs              = Parameter(costs)
        self.LARGE_VAL_UNSET    = torch.as_tensor(1e10)
        self.LARGE_VAL_OCCUPIED = torch.as_tensor(2e10)

    @torch.no_grad()
    def forward(self, *, occupied, source):
        "BFS in tensor of occupancy grids"
        batch_shape = occupied.shape[:-self.ndim]
        spat_shape  = occupied.shape[-self.ndim:]
        spat_shape_pad = [d + 2 for d in spat_shape]
        grids_pad = torch.full((*batch_shape, *spat_shape_pad),
                               self.LARGE_VAL_UNSET,
                               device=self.kernels.device,
                               dtype=self.kernels.dtype)

        spat_range = (..., *(slice(1, -1) for i in range(self.ndim)))

        grids = grids_pad[spat_range]
        grids[(..., *source)] = 0
        grids[occupied] = self.LARGE_VAL_OCCUPIED

        grids_next_pad = grids_pad.clone()
        grids_next = grids_next_pad[spat_range]

        N = torch.prod(torch.as_tensor(spat_shape)).item()

        for i in range(N):

            conv = F.conv2d(grids_pad, self.kernels, self.costs)

            torch.amin(conv, dim=1, keepdim=True, out=grids_next)

            grids_next[occupied] = self.LARGE_VAL_OCCUPIED

            if torch.all(grids_next == grids):
                break

            grids,     grids_next     = grids_next,     grids
            grids_pad, grids_next_pad = grids_next_pad, grids_pad

        else:
            warn('max iterations reached')

        grids[grids == self.LARGE_VAL_OCCUPIED] = torch.inf
        grids[occupied] = torch.nan

        return grids


def main(args=sys.argv[1:]):
    torch.set_printoptions(linewidth=200)

    [device] = args if args else ['cuda']

    bfs = BFS(mode='connect-8').to(device=device)

    occupied = torch.rand(size=(2000, 1, 121, 121), device=device) < 0.1

    _ = bfs(occupied=occupied, source=(5, 4))


if __name__ == "__main__":
    main()
