import os
import sys

sys.path.append("/home/yunfan/workarea/stylegan_pcgen.v1/")

import h5py
import numpy as np
import torch
from trainers.utils.utils import ForkedPdb, farthest_point_sample

synsetid_to_cate = {
    '02691156': 'airplane',
    '02773838': 'bag',
    '02801938': 'basket',
    '02808440': 'bathtub',
    '02818832': 'bed',
    '02828884': 'bench',
    '02876657': 'bottle',
    '02880940': 'bowl',
    '02924116': 'bus',
    '02933112': 'cabinet',
    '02747177': 'can',
    '02942699': 'camera',
    '02954340': 'cap',
    '02958343': 'car',
    '03001627': 'chair',
    '03046257': 'clock',
    '03207941': 'dishwasher',
    '03211117': 'monitor',
    '04379243': 'table',
    '04401088': 'telephone',
    '02946921': 'tin_can',
    '04460130': 'tower',
    '04468005': 'train',
    '03085013': 'keyboard',
    '03261776': 'earphone',
    '03325088': 'faucet',
    '03337140': 'file',
    '03467517': 'guitar',
    '03513137': 'helmet',
    '03593526': 'jar',
    '03624134': 'knife',
    '03636649': 'lamp',
    '03642806': 'laptop',
    '03691459': 'speaker',
    '03710193': 'mailbox',
    '03759954': 'microphone',
    '03761084': 'microwave',
    '03790512': 'motorcycle',
    '03797390': 'mug',
    '03928116': 'piano',
    '03938244': 'pillow',
    '03948459': 'pistol',
    '03991062': 'pot',
    '04004475': 'printer',
    '04074963': 'remote_control',
    '04090263': 'rifle',
    '04099429': 'rocket',
    '04225987': 'skateboard',
    '04256520': 'sofa',
    '04330267': 'stove',
    '04530566': 'vessel',
    '04554684': 'washer',
    '02992529': 'cellphone',
    '02843684': 'birdhouse',
    '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

def get_fps(num_pts = 256, cate = "airplane"):
    root = "./data/ShapeNetCore.v2.PC15k/"
    # split = ['train', 'test', 'val']
    split = ['all']

    for s in split:
        point_clouds = []
        sub_dir = os.path.join(root, cate_to_synsetid[cate], s)
        h5_file = os.path.join(sub_dir, f"consolidated_{num_pts}.h5")
        if os.path.isfile(h5_file):
            continue
        for x in os.listdir(sub_dir):
            if not x.endswith(".npy"):
                continue
            rawdata = np.load(os.path.join(sub_dir, x))
            point_clouds.append(rawdata[np.newaxis, ...])
        point_clouds = np.concatenate(point_clouds)
        print(point_clouds.shape)

        batch_data = torch.from_numpy(point_clouds).cuda()
        idx = farthest_point_sample(batch_data, num_pts)
        fps = torch.gather(batch_data, 1, idx.unsqueeze(2).expand(-1, -1, batch_data.size(-1))).cpu().numpy()
        print(fps.shape)
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("group_name", data=fps)
        print(f"write to file {h5_file}")

if __name__ == "__main__":
    # for pts in [16, 32, 64, 128, 256, 512, 1024, 2048]:
    # cates = ["chair", "car"]
    cates = synsetid_to_cate.values()
    for cate in cates:
        print(cate)
        # if cate == "table" or cate == "lamp" or cate == "sofa":
            # continue
        for pts in [2048]:
            get_fps(pts, cate)

    print(len(cates))
