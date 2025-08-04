import torch
import numpy as np
from plyfile import PlyData

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


path_to_ply = './output/b9a3251d-0/point_cloud/iteration_30000/point_cloud.ply'
plydata = PlyData.read(path_to_ply)
vertex_data = plydata['vertex']

qr = vertex_data['rot_0']
qx = vertex_data['rot_1']
qy = vertex_data['rot_2']
qz = vertex_data['rot_3']

r = np.stack([qr, qx, qy, qz], axis=1)
r = torch.tensor(r, dtype=torch.float32, device='cuda')
R = build_rotation(r)

normals = R[:, :, 2] # third row of rotation matrix (source: https://github.com/hbb1/2d-gaussian-splatting/issues/210?utm_source=chatgpt.com)
print(R[0, :, 2])
print(np.linalg.norm(R[0, :, 2].cpu().numpy()))
print(normals)

'''
for i in range(0,np.size(qr)):
    r = np.stack([qr[i], qx[i], qy[i], qz[i]], axis=1)
    R = build_rotation(r)
'''
