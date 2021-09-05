import torch
import numpy as np

def constant_input(bs, num_points):
    x = torch.cuda.FloatTensor(vertics_gen(num_points))
    x = x.transpose(0, 1).contiguous().unsqueeze(0)
    x = x.expand(bs, x.size(1), x.size(2)).contiguous()
    x = ((x - 0.5) * 2).contiguous()
    return x


def vertics_gen(num_points):
    num_points = num_points
    grain_x = 2 ** np.floor(np.log2(num_points) / 3) - 1
    grain_y = 2 ** np.ceil(np.log2(num_points) / 3) - 1
    grain_z = 2 ** np.ceil(np.log2(num_points) / 3) - 1
    vertices = []
    for i in range(int(grain_x + 1)):
        for j in range(int(grain_y + 1)):
            for k in range(int(grain_z + 1)):
                vertices.append([i / grain_x, j / grain_y, k / grain_z])

    return vertices


np.save("cubic.npy", constant_input(1, 2048).cpu().numpy())
