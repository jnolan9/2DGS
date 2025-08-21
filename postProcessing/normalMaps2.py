import numpy as np
import os
import sys
from typing import NamedTuple

import torch

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

sys.path.append(os.path.expanduser("~"))
from DNSplatter.nerfstudio.nerfstudio.process_data.colmap_utils import colmap_to_json

from matplotlib.colors import Normalize

from plyfile import PlyData

from PIL import Image

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

def colormap(img, cmap='jet'):
    import matplotlib.pyplot as plt
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    return img
    
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
img_ind = 8
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

    # Adjust rotation matrix
    R_CW = R_CW.T # undo transpose because not processed by 'glm' here  
    R_CW[:,[0,1,2]]=R_CW[:,[0,2,1]]
    R_CW[:,1] *= -1


## PLOT NORMALS
plt.rcParams['figure.dpi'] = 150
fx, fy, cx, cy = cameras[0].params
vis_normal = lambda normal: ((normal - np.nanmin(normal[:])) / 2)[..., ::-1]


## READ SAVED DATA
import pandas as pd
df = pd.read_csv('correlated_normals.csv')

xyz_W_2dgs = np.stack((df["pos_x_W_2dgs"].to_numpy(),df["pos_y_W_2dgs"].to_numpy(),df["pos_z_W_2dgs"].to_numpy()),axis=1)
nvecs_W_2dgs = np.stack((df["nvec_x_W_2dgs"].to_numpy(),df["nvec_y_W_2dgs"].to_numpy(),df["nvec_z_W_2dgs"].to_numpy()),axis=1)
xyz_W_gt = np.stack((df["pos_x_W_gt"].to_numpy(),df["pos_y_W_gt"].to_numpy(),df["pos_z_W_gt"].to_numpy()),axis=1)
nvecs_W_gt = np.stack((df["nvec_x_W_gt"].to_numpy(),df["nvec_y_W_gt"].to_numpy(),df["nvec_z_W_gt"].to_numpy()),axis=1)
theta = df["theta_gt_2dgs"].to_numpy()
dist = np.linalg.norm((xyz_W_2dgs-xyz_W_gt),axis=1)

# Get pixel coordinates for gt
xyz_C_gt = R_CW @ xyz_W_gt.T + r_WC_C[..., None]
nvecs_C_gt = R_CW @ nvecs_W_gt.T
x_C_gt = fx * xyz_C_gt[0] / xyz_C_gt[2] + cx
y_C_gt = fy * xyz_C_gt[1] / xyz_C_gt[2] + cy
xy_C_gt= np.vstack((x_C_gt[None, ...], y_C_gt[None, ...]))

# Get pixel coordinates for 2dgs
xyz_C_2dgs = R_CW @ xyz_W_2dgs.T + r_WC_C[..., None]
nvecs_C_2dgs = R_CW @ nvecs_W_2dgs.T
x_C_2dgs = fx * xyz_C_2dgs[0] / xyz_C_2dgs[2] + cx
y_C_2dgs = fy * xyz_C_2dgs[1] / xyz_C_2dgs[2] + cy
xy_C_2dgs = np.vstack((x_C_2dgs[None, ...], y_C_2dgs[None, ...]))

# Filter coordinates and differences by proximity in pixel coordinates
px_dist = np.linalg.norm((xy_C_2dgs-xy_C_gt),axis=0)
px_gt = xy_C_gt[:,px_dist<1]
px_2dgs = xy_C_2dgs[:,px_dist<1]
theta_filtered = theta[px_dist<1]

# Save figure of gt normal map
# plt.close()
# plt.scatter(x_C_2dgs,y_C_2dgs, c=theta, cmap="jet",marker=",")
# # plt.axis("off")
# plt.gca().set_aspect("equal")
# plt.xlim(0,1024)
# plt.ylim(0,1024)
# # plt.xlim(0,800)
# # plt.ylim(0,800)
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.savefig('renders/diff/normalMap_'+str(img_ind)+'_diff2.png')


# Save figure of gt normal map
plt.close()
plt.scatter(xy_C_gt[0], xy_C_gt[1], s=0.15, c=vis_normal((R_CW.T @ nvecs_C_gt).T), marker=",")
# plt.axis("off")
plt.gca().invert_yaxis()
plt.gca().set_aspect("equal")
plt.savefig('renders/gt/normalMap_'+str(img_ind)+'_gtW.png')

# Save figure of gt normal map
plt.close()
plt.scatter(xy_C_2dgs[0], xy_C_2dgs[1], s=0.15, c=vis_normal((R_CW.T @ nvecs_C_2dgs).T), marker=",")
# plt.axis("off")
plt.gca().set_aspect("equal")
plt.xlim(0,1024)
plt.ylim(0,1024)
# plt.xlim(0,800)
# plt.ylim(0,800)
plt.gca().invert_yaxis()
plt.savefig('renders/2dgs/normalMap_'+str(img_ind)+'_2dgsW.png')

px_nvecs_2dgs = nvecs_W_2dgs[px_dist<1,:].T
df_nvec_2dgs = pd.DataFrame({'x': np.floor(px_2dgs[0]), 'y': np.floor(px_2dgs[1]), 
                             'nx': px_nvecs_2dgs[0], 'ny': px_nvecs_2dgs[1], 'nz': px_nvecs_2dgs[2]})
sum_nx = df_nvec_2dgs.groupby(['x', 'y'])['nx'].sum().reset_index()
sum_ny = df_nvec_2dgs.groupby(['x', 'y'])['ny'].sum().reset_index()
sum_nz = df_nvec_2dgs.groupby(['x', 'y'])['nz'].sum().reset_index()
avg_nvec_2dgs = np.stack((sum_nx['nx'].to_numpy(),sum_ny['ny'].to_numpy(),sum_nz['nz'].to_numpy()),axis=0)
avg_nvec_2dgs = avg_nvec_2dgs / np.linalg.norm(avg_nvec_2dgs, axis=0, keepdims=True)
plt.close()
plt.scatter(sum_nx['x'], sum_nx['y'], s=0.15, c=vis_normal(avg_nvec_2dgs.T), marker=",")
plt.gca().set_aspect("equal")
plt.gca().invert_yaxis()
plt.savefig('renders/2dgs/normalMap_'+str(img_ind)+'_2dgsAvg.png')

px_nvecs_gt = nvecs_W_gt[px_dist<1,:].T
df_nvec_gt = pd.DataFrame({'x': np.floor(px_gt[0]), 'y': np.floor(px_gt[1]), 
                             'nx': px_nvecs_gt[0], 'ny': px_nvecs_gt[1], 'nz': px_nvecs_gt[2]})
sum_nx = df_nvec_gt.groupby(['x', 'y'])['nx'].sum().reset_index()
sum_ny = df_nvec_gt.groupby(['x', 'y'])['ny'].sum().reset_index()
sum_nz = df_nvec_gt.groupby(['x', 'y'])['nz'].sum().reset_index()
avg_nvec_gt = np.stack((sum_nx['nx'].to_numpy(),sum_ny['ny'].to_numpy(),sum_nz['nz'].to_numpy()),axis=0)
avg_nvec_gt = avg_nvec_gt / np.linalg.norm(avg_nvec_gt, axis=0, keepdims=True)
plt.close()
plt.scatter(sum_nx['x'], sum_nx['y'], s=0.15, c=vis_normal(avg_nvec_gt.T), marker=",")
plt.gca().set_aspect("equal")
plt.gca().invert_yaxis()
plt.savefig('renders/gt/normalMap_'+str(img_ind)+'_gtAvg.png')


# Create scatter plot of theta differecnes
plt.close()
plt.scatter(px_2dgs[0],px_2dgs[1], c=theta_filtered, cmap="jet",marker=",",s=0.05)
# plt.axis("off")
plt.gca().set_aspect("equal")
# plt.xlim(0,1024)
# plt.ylim(0,1024)
# plt.xlim(0,800)
# plt.ylim(0,800)
plt.gca().invert_yaxis()
plt.colorbar()
plt.savefig('renders/diff/scatter/normalMap_'+str(img_ind)+'_scatterdiff.png')

# Create an image averaging the theta differences in each pixel
df_avg = pd.DataFrame({'x': np.floor(px_2dgs[0]), 'y': np.floor(px_2dgs[1]), 't': theta_filtered})
# df_avg = df_avg[df_avg['t'] < 2]
df_avg = df_avg[(df_avg['x'] >= 0) & (df_avg['x'] <= 1024) & (df_avg['y'] >= 0) & (df_avg['y'] <= 1024)]
avg_t = df_avg.groupby(['x', 'y'])['t'].mean().reset_index()
pivot = avg_t.pivot(index='x', columns='y', values='t')
pivot=np.nan_to_num(pivot, nan=0)
pivot = np.array(pivot)
# pivot = pivot/np.max(pivot)
fig, ax = plt.subplots()
im = plt.imshow(pivot.T,cmap="jet")
# plt.xlim(0,1024)
# plt.ylim(0,1024)
plt.colorbar()
plt.savefig('renders/diff/avg/normalMap_'+str(img_ind)+'_avgdiff.png')

# plt.close()
# plt.scatter(avg_t['x'], avg_t['y'], s=0.15, c=avg_t.t, marker=",")
# plt.gca().set_aspect("equal")
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.savefig('renders/diff/avg/normalMap_'+str(img_ind)+'_avgdiff.png')


# python;  exec(open('normalMaps2.py').read())