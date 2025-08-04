import numpy as np
import os
import sys
from typing import NamedTuple

import torch

import json
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(os.path.expanduser("~"))
from DNSplatter.nerfstudio.nerfstudio.process_data.colmap_utils import colmap_to_json

from matplotlib.colors import Normalize

from plyfile import PlyData

from utils.spc_types import read_spcpoints3D_text,read_spcimages_text

from utils.read_write_model import (
    rotmat2qvec,
    qvec2rotmat,
    write_cameras_text,
    read_cameras_text,
    detect_model_format,
    read_model,
    write_model
)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
    

## GET 2D AND 3D POINTS FROM TXT FILES
# images_path = '/home/jnolan9/DNSplatter/phomo_mod_cornelia/images_with_svec.txt'
# points3d_path = '/home/jnolan9/DNSplatter/phomo_mod_cornelia/points3D.txt'
# cameras_path = '/home/jnolan9/DNSplatter/phomo_mod_cornelia/cameras.txt'
images_path = './reconstruct_files/images.txt'
points3d_path = './reconstruct_files/points3D.txt'
cameras_path = './reconstruct_files/cameras.txt'

points3d = read_spcpoints3D_text(points3d_path)
images = read_spcimages_text(images_path)
cameras = read_cameras_text(cameras_path)


# ## CREATE JSON FILE
# recon_dir = ''
# model_path = Path('./reconstruct_files')
# write_model(cameras, images, points3d, model_path, ext=".bin")
# colmap_to_json(model_path,model_path)

## READ JSON FILE FOR GIVEN FRAME
img_ind = 2
json_path = '/home/jnolan9/2DGS/nerfData/transforms_train.json'
# json_path = '/home/jnolan9/2DGS/nerf/data/nerf_synthetic/lego/transforms_train.json'
with open(json_path) as json_file:
    contents = json.load(json_file)
    frames = contents["frames"]
    frame = frames[img_ind]
    img_name = frame['file_path']
    print(img_name)

    # NeRF 'transform_matrix' is a camera-to-world transform (from 2dgs dataset_readers.py)
    c2w = np.array(frame["transform_matrix"])
    c2w[:3, 1:3] *= -1 # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)

    w2c = np.linalg.inv(c2w)
    R_CW = w2c[:3,:3]
    R_CW = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    r_WC_C = w2c[:3, 3]
    # R_CW = np.array([[1,0,0],[0,1,0],[0,0,1]])

    # R_CW[[1,2]]=R_CW[[2,1]]
    # R_CW[1]=-R_CW[1]
    R_CW = R_CW.T

## GET NORMALS FROM PLY FILE
# 9934fd88-b - poster
# a6556f4f-1 - Vesta
path_to_ply = '/home/jnolan9/2DGS/2d-gaussian-splatting/output/a6556f4f-1/point_cloud/iteration_30000/point_cloud.ply'
# path_to_ply = '/home/jnolan9/2DGS/2d-gaussian-splatting/output/9934fd88-b/point_cloud/iteration_30000/point_cloud.ply'
plydata = PlyData.read(path_to_ply)
vertex_data = plydata['vertex']

qr = vertex_data['rot_0']
qx = vertex_data['rot_1']
qy = vertex_data['rot_2']
qz = vertex_data['rot_3']

r = np.stack([qr, qx, qy, qz], axis=1)
r = torch.tensor(r, dtype=torch.float32, device='cuda')
R = build_rotation(r)

x = vertex_data['x']
y = vertex_data['y']
z = vertex_data['z']
xyz_2dgs = np.stack([x, y, z], axis=1)
pcd_2dgs = xyz_2dgs
nvecs_2dgs = R[:, :, 2].detach().cpu().numpy() # third row of rotation matrix (source: https://github.com/hbb1/2d-gaussian-splatting/issues/210?utm_source=chatgpt.com)

## PLOT NORMALS

plt.rcParams['figure.dpi'] = 150
fx, fy, cx, cy = cameras[0].params
vis_normal = lambda normal: ((normal - np.nanmin(normal[:])) / 2)[..., ::-1]

# For lego dataset
# fx = 1111.1110311937682
# fy = 1111.1110311937682
# cx = 400
# cy = 400

# print(fx)
# print(fy)
# print(cx)
# print(cy)

# Get albedos and scale.
pcd_gt = np.array([p3d.xyz for p3d in points3d.values()])
alb_gt = np.array([p3d.albedo for p3d in points3d.values()])
nvecs_gt = np.array([p3d.nvec for p3d in points3d.values()])

# Get projected camera frame truth coordinates
R_CA = images[img_ind].qvec2rotmat()
r_AC_C = images[img_ind].tvec[..., None]
xyz_C_gt = R_CA @ pcd_gt.T + r_AC_C
x_C_gt = fx * xyz_C_gt[0] / xyz_C_gt[2] + cx
y_C_gt = fy * xyz_C_gt[1] / xyz_C_gt[2] + cy
xy_C_gt = np.vstack((x_C_gt[None, ...], y_C_gt[None, ...]))


# Get projected camera frame 2dgs coordinates
xyz_C_2dgs = R_CA @ pcd_2dgs.T + r_WC_C[..., None]
x_C_2dgs = fx * xyz_C_2dgs[0] / xyz_C_2dgs[2] + cx
y_C_2dgs = fy * xyz_C_2dgs[1] / xyz_C_2dgs[2] + cy
xy_C_2dgs = np.vstack((x_C_2dgs[None, ...], y_C_2dgs[None, ...]))


# Save figure of gt normal map
plt.close()
plt.scatter(xy_C_gt[0], xy_C_gt[1], s=0.15, c=vis_normal(nvecs_gt), marker=",")
# plt.axis("off")
plt.gca().invert_yaxis()
plt.gca().set_aspect("equal")
plt.savefig('renders/gt/normalMap_'+str(img_ind)+'_gt.png')

# Save figure of gt normal map
plt.close()
plt.scatter(xy_C_2dgs[0], xy_C_2dgs[1], s=0.15, c=vis_normal(nvecs_2dgs), marker=",")
# plt.axis("off")
plt.gca().set_aspect("equal")
plt.xlim(0,1024)
plt.ylim(0,1024)
# plt.xlim(0,800)
# plt.ylim(0,800)
plt.gca().invert_yaxis()
plt.savefig('renders/2dgs/normalMap_'+str(img_ind)+'_2dgs.png')