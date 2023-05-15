# Torch Occupancy Grid BFS

This project implements Breadth-First Search (BFS) in an occupancy grid using
PyTorch convolutions. This is useful for finding the shortest path from a point
in an occupancy grid.

## Installation

To use this project, you need to have PyTorch installed. You can install it using pip:

```bash
pip install torch
pip install git+https://github.com/lericson/torch-occupancy-grid-bfs.git
```

## Usage and Performance

Occupancy grid BFS with an 8-connected grid using CUDA:

```python
bfs = BFS(mode='connect-8').cuda()
# Some test data
occupied = torch.rand(size=(2000, 1, 121, 121), device=device) < 0.1
grids = bfs(occupied=occupied, source=(50, 50))
```

This computes BFS in 2000 grids of 121 x 121 cells. It takes 5.091 seconds on
my computer with an NVIDIA Titan X, and 51.372 seconds(!) on the CPU (about 6
threads on an i7-6850K.)
