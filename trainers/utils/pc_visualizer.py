# Copyright (c) Yunfan Zhang All Rights Reserved.
# File Name: vis_pcd.py
# Author: Yunfan Zhang
# Mail: yunfan.zhang23@gmail.com
# Github: https://github.com/yunfan23
# Blog: http://www.zhangyunfan.tech/
# Created Time: 2021-05-21

import numpy as np
import open3d as o3d


def save_img_from_pc_3view(pcdata: np, out: str, logging: bool = False):
    """save_img_from_pc.

    :param pcdata:
    :type pcdata: np
    :param out:
    :type out: str
    """
    assert pcdata.shape[1] == 3
    assert pcdata.shape[0] >= 2048

    pts = pcdata
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pts)
    # print(pcl)
    vis = o3d.visualization.Visualizer()
    x_angle = [np.pi/6, -np.pi/6, np.pi/6]
    y_angle = [np.pi-np.pi/8, np.pi, np.pi+np.pi/8]
    for view in range(3):
        pcl.points = o3d.utility.Vector3dVector(pts)
        pcl_r = pcl.get_rotation_matrix_from_xyz((x_angle[view], y_angle[view], 0))
        pcl.rotate(pcl_r, center=(0, 0, 0))
        pcl.paint_uniform_color([0.5, 0.5, 1.])
        vis.create_window()
        vis.add_geometry(pcl)
        # this function must be called when geometry is changed
        vis.update_geometry(pcl)
        vis.poll_events()
        vis.update_renderer()
        # print(vis.get_view_control())
        vis.capture_screen_image(f"{out}_{view}.png")
        if logging:
            print(f"{out}_{view}.png is saved")
    vis.destroy_window()

def save_img_from_pc(pcdata: np, out: str, logging: bool = False):
    """save_img_from_pc.

    :param pcdata:
    :type pcdata: np
    :param out:
    :type out: str
    """
    assert pcdata.shape[1] == 3
    assert pcdata.shape[0] >= 2048
    # pts = np.load(pcd)
    # print(pts.shape)
    # idx = np.random.choice(pcdata.shape[0], 2048)
    # pts = pcdata[idx]
    # print(pts.shape)

    pts = pcdata
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pts)
    # print(pcl)
    vis = o3d.visualization.Visualizer()
    pcl_r = pcl.get_rotation_matrix_from_xyz((np.pi/6, np.pi-np.pi/6, 0))
    # pcl_r = pcl.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    # pcl_r = pcl.get_rotation_matrixfrom_xyz((np.pi / 2, 0, np.pi / 4))
    pcl.rotate(pcl_r, center=(0, 0, 0))
    pcl.paint_uniform_color([0.5, 0.5, 1.])
    # pcl.colors[:] = [1, 0, 0]
    vis.create_window()
    vis.add_geometry(pcl)
    # this function must be called when geometry is changed
    vis.update_geometry(pcl)
    vis.poll_events()
    vis.update_renderer()
    # print(vis.get_view_control())
    vis.capture_screen_image(f"{out}.png")
    if logging:
        print(f"{out}.png is saved")
    vis.destroy_window()

def save_img_from_np(
        npdata: np,
        fileName: str = "test",
        logging: bool = False
    ):
    """ Save Image from an Numpy Arrary

    Args:
        npdata (np): input numpy array
        fileName (str, optional): image name. Defaults to "test".
        logging (bool, optional): output logging or not. Defaults to False.
    """
    from PIL import Image
    rescaled = (255.0 / npdata.max() * (npdata - npdata.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    im.save(f"{fileName}.png")
    if logging:
        print(f"{fileName}.png is saved")


if __name__ == '__main__':
    path = '/home/yunfan/workarea/ShapeGF/data/ShapeNetCore.v2.PC15k/02691156/train/10155655850468db78d106ce0a280f87.npy'
    data = np.load(path)
    out = 'test'
    save_img_from_pc_3view(data, out)
