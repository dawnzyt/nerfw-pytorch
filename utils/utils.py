import numpy as np
import cv2
import torch


# 四元数=>旋转矩阵
def quat2rotmat(q):
    """
    单位四元数=>旋转矩阵
    q:(w,x,y,z), 用单位四元数表示即w=cosθ
    """
    return np.array([
        [1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
         2 * q[1] * q[2] - 2 * q[0] * q[3],
         2 * q[3] * q[1] + 2 * q[0] * q[2]],
        [2 * q[1] * q[2] + 2 * q[0] * q[3],
         1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
         2 * q[2] * q[3] - 2 * q[0] * q[1]],
        [2 * q[3] * q[1] - 2 * q[0] * q[2],
         2 * q[2] * q[3] + 2 * q[0] * q[1],
         1 - 2 * q[1] ** 2 - 2 * q[2] ** 2]])


def rotmat2quat(R):
    """
    旋转矩阵=>单位四元数
    """
    w = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


def quat_mult(Q, P):
    """
    四元数乘法: 对于标准的两个四元数Q(4,)=(q0,q)、P(4,)=(p0,p)
    QP=(p0q0-p·q, p0q+q0p+q×p) [ ×为叉乘 ]

    :param Q: (N,4)
    :param P: (N,4)
    :return:
    """
    mul_w = Q[:, :1] * P[:, :1] - np.sum(Q[:, 1:] * P[:, 1:], axis=1,keepdims=True)
    mul_xyz = Q[:, :1] * P[:, 1:] + P[:, :1] * Q[:, 1:] + np.cross(Q[:, 1:], P[:, 1:], axis=1)
    return np.hstack([mul_w, mul_xyz])


# 畸变校正的map
def get_rectify_map(img_size, dist, K):
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, img_size, 1)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, new_K, img_size, cv2.CV_32FC1)
    return mapx, mapy


# 根据内外参计算world系下所有ray direction和center
# 获得camera coordinate下所有光线的direction
# direction: ((x-cx)/sx,(y-cy)/sy,f) <=> ((x-cx)/fx, (y-cy)/fy, 1)
# x即j,y即i
# 方向直接c2w, 光心就是c2w×(0,0,0)即c2w的t_vec。
def get_rays_o_d(img_w, img_h, K, c2w):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y = np.meshgrid(np.arange(0, img_w, 1), np.arange(0, img_h, 1))
    x, y = np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)
    d = np.concatenate([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], axis=-1).reshape(-1, 3)
    d = d @ c2w[:, :3].T
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)  # 归一化, 相当于光线采样范围以o为中心[near,far]的球。
    o = c2w[:, -1].reshape(1, 3)
    o = np.tile(o, (len(d), 1))
    return torch.FloatTensor(o), torch.FloatTensor(d)


def get_corner_ray(img_w, img_h, K, c2w):
    '''
    获得world系下camera中心以及射向四个角的射线
    :param img_w:
    :param img_h:
    :param K:
    :param c2w:
    :return: ray_o:(3,), ray_d:(4,3)
    '''
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ray_o = c2w[:, -1]
    xy = np.array([[0, 0], [img_w - 1, 0], [img_w - 1, img_h - 1], [0, img_h - 1]])  # four corners
    x, y, z = (xy[:, :1] - cx) / fx, (xy[:, -1:] - cy) / fy, np.ones((4, 1))
    ray_d = np.hstack([x, y, z])
    ray_d = ray_d / np.linalg.norm(ray_d, axis=1, keepdims=True)
    # camera to world
    ray_d = ray_d @ c2w[:, :3].T
    return ray_o, ray_d


import torchvision.transforms as T


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x)  # get minimum depth
    ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    heat_img = cv2.applyColorMap(x, colormap=cmap)
    return heat_img

ply_header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {}
property list uchar int vertex_indices
property uchar red
property uchar green
property uchar blue
property uchar alpha
element edge {}
property int vertex1
property int vertex2
property uchar red
property uchar green
property uchar blue
end_header
'''