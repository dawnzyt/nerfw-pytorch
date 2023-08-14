import os.path
import sys
import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import *
import pickle


class CambridgeDataset(Dataset):
    def __init__(self, root_dir, scene='StMarysChurch', split='train', img_downscale=1, use_cache=False,
                 if_save_cache=True):
        self.scene = scene
        self.root_dir = root_dir
        self.split = split
        self.img_downscale = img_downscale
        self.img_size = (1920, 1080)  # fixed size of Cambridge nerf
        self.downscale_size = (self.img_size[0] // img_downscale, self.img_size[1] // img_downscale)
        self.true_downscale = self.img_size[0] / self.downscale_size[0]
        self.use_cache = use_cache
        self.if_save_cache = if_save_cache
        self.load_data()

    def load_data(self):
        """
        nvm format
        <Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
        <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
        """
        print('load reconstruction data of scene "{}", split: {}'.format(self.scene, self.split))
        base_dir = os.path.join(self.root_dir, self.scene)
        with open(os.path.join(base_dir, 'reconstruction.nvm'), 'r') as f:
            nvm_data = f.readlines()  # visualSFM nvm file
        with open(os.path.join(base_dir, 'dataset_train.txt'), 'r') as f:
            train_split = f.readlines()[3:]  # train split
        with open(os.path.join(base_dir, 'dataset_test.txt'), 'r') as f:
            test_split = f.readlines()[3:]  # test split

        # 1. 解析cameras
        self.N_views = int(nvm_data[2])
        self.N_points = int(nvm_data[4 + self.N_views])
        if self.use_cache:
            self.load_cache()
            return
        self.file2id = {}  # img name->id
        self.view_filename = {}  # id->img file name
        self.view_K = {}  # id->K
        self.view_c2w = {}  # id->c2w
        self.view_rectifymap = {}  # id->rectifyMap, the distortion correction
        self.view_w2c = {}  # id->w2c
        for i, data in enumerate(nvm_data[3:3 + self.N_views]):
            data = data.split()
            # data = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
            data[0] = data[0].replace('/', '/')
            self.file2id[data[0]] = i
            self.view_filename[i] = data[0]
            params = np.array(data[1:-1], dtype=float)
            # img_downscale: fx、fy、cx、cy、img_w、img_h ↓
            K = np.zeros((3, 3), dtype=float)
            K[0, 0] = K[1, 1] = params[0] / self.img_downscale
            K[0, 2] = self.downscale_size[0] / 2
            K[1, 2] = self.downscale_size[1] / 2
            K[2, 2] = 1
            self.view_K[i] = K
            bottom = np.zeros(shape=(1, 4))
            bottom[0, -1] = 1
            rotmat = quat2rotmat(params[1:5]/np.linalg.norm(params[1:5]))
            t_vec = -rotmat @ params[5:8].reshape(3, 1)  # visualSFM存储的c2w,即world系下的camera center的位置。rotmat是w2c...
            # t_vec = params[5:8].reshape(3, 1)
            w2c = np.vstack([np.hstack([rotmat, t_vec]), bottom])
            self.view_w2c[i] = w2c[:3]  # w2c
            c2w = np.linalg.inv(w2c)
            self.view_c2w[i] = c2w[:3]  # c2w

            self.view_rectifymap[i] = get_rectify_map((self.downscale_size[0], self.downscale_size[1]), params[-1], K)
        # 2. 解析points
        self.view_near = {}  # camera坐标系下最小深度, 用以nerf采样
        self.view_far = {}  # 最大深度
        points_str = nvm_data[5 + self.N_views:5 + self.N_points + self.N_views]
        self.points = np.vstack([np.array(x.split()[:6], dtype=float) for x in points_str])
        points_h = np.hstack([self.points[:, :3], np.ones(shape=(self.N_points, 1))])  # 齐次坐标
        for i in range(self.N_views):
            w2c = self.view_w2c[i]
            xyz_cam_i = points_h @ w2c.T
            xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2] > 0]
            self.view_near[i] = np.percentile(xyz_cam_i[:, 2], 0.1)  # 注意这里range[0,100]
            # 画出z的分布图,z∈(0,100)
            self.view_far[i] = np.percentile(xyz_cam_i[:, 2], 99.9)
        # 尺度放缩, 影响w2c、c2w的tvec, points以及near、far
        max_dep = np.max([v for v in self.view_far.values()])
        scale_factor = max_dep / 5
        self.scale_factor = scale_factor
        for i in range(self.N_views):
            self.view_c2w[i] = np.hstack([self.view_c2w[i][:, :3], self.view_c2w[i][:, 3:] / scale_factor])
            self.view_w2c[i] = np.hstack([self.view_w2c[i][:, :3], self.view_w2c[i][:, 3:] / scale_factor])
            self.view_near[i] = self.view_near[i] / scale_factor
            self.view_far[i] = self.view_far[i] / scale_factor
        self.points[:, :3] = self.points[:, :3] / scale_factor
        # 3. 解析train_split和test_split
        self.train_set = []  # train
        self.test_set = []  # test
        for data in train_split:
            data = data.split()
            data[0] = data[0].replace('/', '/')
            id = self.file2id[data[0]]
            self.train_set.append(id)
        for data in test_split:
            data = data.split()
            data[0] = data[0].replace('/', '/')
            id = self.file2id[data[0]]
            self.test_set.append(id)

        # 4. 生成all train view的rays
        if self.split == 'train':
            self.all_rays = []
            for i, id in tqdm.tqdm(enumerate(self.train_set), total=len(self.train_set), file=sys.stdout,
                                   desc="acquiring rays "):
                # distortion rectify
                rays = self.get_rays(id)
                self.all_rays += [rays]
            self.all_rays = torch.vstack(self.all_rays)
            print('all rays: ', self.all_rays.shape)

        # save cache
        if self.if_save_cache:
            self.save_cache()

    def get_rays(self, i):
        """
        rays:(N_rays, 12), position、direction、near、far、id、rgb; 其中pos、dir是在世界坐标系下,
        near、far方便光线采样, id则是match 对应的appearance/transient embedding

        :param i: view id
        :return:
        """
        # distortion rectify
        base_dir = os.path.join(self.root_dir, self.scene)
        img = Image.open(os.path.join(base_dir, self.view_filename[i])).convert(mode='RGB')
        img_w, img_h = img.size
        img_w, img_h = img_w // self.img_downscale, img_h // self.img_downscale
        img = img.resize((img_w, img_h), Image.LANCZOS)
        rect_img = np.array(img)
        # rect_img = cv2.remap(np.array(img), self.view_rectifymap[i][0], self.view_rectifymap[i][1],
        #                      cv2.INTER_LINEAR)
        # 获得世界坐标系下的光线的position和orientation
        c2w = self.view_c2w[i]
        rays_o, rays_d = get_rays_o_d(img_w, img_h, self.view_K[i], c2w)
        nears = self.view_near[i] * torch.ones((len(rays_o), 1))
        fars = self.view_far[i] * torch.ones((len(rays_o), 1))
        ids = i * torch.ones((len(rays_o), 1))
        rays = torch.hstack([rays_o, rays_d, nears, fars, ids, torch.FloatTensor(rect_img.reshape(-1, 3)) / 255])
        return rays

    def load_cache(self):
        # 1. load dicts
        dict_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'dicts.pkl')
        with open(dict_path, 'rb') as f:
            ds = pickle.load(f)
            self.train_set, self.test_set, self.file2id, self.view_filename, self.view_K, self.view_w2c, \
                self.view_c2w, _, self.view_near, self.view_far, self.scale_factor = ds
        # 2. load points
        points_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'sfm_points.npy')
        self.points = np.load(points_path)
        # 3. load all rays
        if self.split == 'train':
            ray_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'all_rays.npy')
            self.all_rays = torch.from_numpy(np.load(ray_path))
            print('all rays: ', self.all_rays.shape)
        # if self.split == 'train':
        #     self.all_rays=torch.zeros(size=(341913600,12),dtype=torch.float)
        print('cache load done...')

    def save_cache(self):
        if not os.path.exists(os.path.join(self.root_dir, self.scene, 'cache')):
            os.makedirs(os.path.join(self.root_dir, self.scene, 'cache'))
        # 1. 存dicts
        dict_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'dicts.pkl')
        with open(dict_path, 'wb') as f:
            pickle.dump([self.train_set, self.test_set, self.file2id, self.view_filename, self.view_K, self.view_w2c,
                         self.view_c2w, self.view_rectifymap, self.view_near, self.view_far, self.scale_factor], f)
        # 2. 转numpy存all_rays
        rays = self.all_rays.numpy()
        ray_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'all_rays.npy')
        np.save(ray_path, rays)

        # 3. 存points
        points_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'sfm_points.npy')
        np.save(points_path, self.points)
        print('cache save done...')

        # 4. 单独存scale_factor(供后续pose regressor缩放尺度用, 保证与nerf-w建模尺度一致)
        with open(os.path.join(self.root_dir, self.scene, 'scale_factor.txt'), 'w') as f:
            np.savetxt(f, np.c_[self.scale_factor], fmt='%.6f')

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'valid':  # 固定训练集某张图像进行validate, 有效不会过拟合因为train unit是ray而不是一张img
            return len(self.train_set)

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.all_rays[idx]
        elif self.split in ['valid', 'test']:
            return self.get_rays(self.train_set[idx])


if __name__ == '__main__':
    server_dir = '/root/autodl-tmp/dataset/Cambridge'
    scene = 'StMarysChurch'
    local_dir = 'E:\\dataset\\Cambridge'
    train_dataset = CambridgeDataset(root_dir=local_dir, scene=scene,
                                     split='train', img_downscale=3, use_cache=False, if_save_cache=True)
