# This script is to prepare the shapenet dataset for the whole dataset
import glob
import os
import shutil

root = "./data/ShapeNetCore.v2.PC15k/"
files = glob.glob(f"{root}/*")
# print(files)

for file in files:
    dest = os.path.join(file, "all")
    if os.path.isfile(os.path.join(dest, "consolidated_2048.h5")):
        continue
    else:
        # print(file)
        os.makedirs(dest, exist_ok=True)
        for sub in ["train", "val", "test"]:
            src = os.path.join(file, sub)
            smp = len(glob.glob(f"{src}/*.npy"))
            print(f"{smp} samples in {src}")
            print(f"copying {src} to {dest}")
            shutil.copytree(src, dest, dirs_exist_ok=True)
        
        smp = len(glob.glob(f"{dest}/*.npy"))
        print(f"{smp} samples in {dest}")
