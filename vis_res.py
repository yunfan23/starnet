import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

def normalize_pc(pc):
    pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
    pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
    shift = ((pc_min + pc_max) / 2).view(1, 3)
    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
    pc = (pc - shift) / scale
    return pc

def plot_save(data, cate):
    data = torch.from_numpy(data)
    data = normalize_pc(data)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=3, zdir='y')
    ax.set_axis_off()
    if cate == "car":
        ax.view_init(0, 0)
        ax.set_ylim(-1, 1)
    else:
        ax.view_init(8, -85)

    filename = file.split(".npy")[0]
    plt.savefig(f"./images/{filename}.png")
    # print(f"{filename}.png")
    plt.close()


file = "./chair_gen_smp.npy"
data = np.load(file)
for idx in range(data.size(0)):
    plot_save(data[idx], "chair")
