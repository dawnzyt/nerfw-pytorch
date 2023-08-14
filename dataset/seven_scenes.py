import glob
import os.path
import pickle
from PIL import Image
import numpy as np
import sys
from torch.utils.data import Dataset
import torch
import tqdm
from utils.utils import *

'''
https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
Each sequence (seq-XX.zip) consists of 500-1000 frames. Each frame consists of three files:
Color: frame-XXXXXX.color.png (RGB, 24-bit, PNG)
Depth: frame-XXXXXX.depth.png (depth in millimeters, 16-bit, PNG, invalid depth is set to 65535).
Pose: frame-XXXXXX.pose.txt (camera-to-world, 4×4 matrix in homogeneous coordinates).
Principle point (320,240), Focal length (585,585).
'''


class SevenScenesDataset(Dataset):
    def __init__(self, root_dir, scene='fire', split='train', img_downscale=2, use_cache=False, if_save_cache=True):
        super().__init__()
        self.scene = scene
        self.root_dir = root_dir
        self.split = split
        self.img_downscale = img_downscale
        self.img_size = (640, 480)  # fixed size of Cambridge nerf
        self.downscale_size = (self.img_size[0] // img_downscale, self.img_size[1] // img_downscale)
        self.true_downscale = self.img_size[0] / self.downscale_size[0]
        self.use_cache = use_cache
        self.if_save_cache = if_save_cache
        self.K = np.zeros((3, 3), dtype=float)
        self.K[0, 0] = self.K[1, 1] = 585  # fixed
        self.K[0, 2], self.K[1, 2], self.K[2, 2] = 320, 240, 1
        self.K[0, 0], self.K[1, 1] = self.K[0, 0] / img_downscale, self.K[1, 1] / img_downscale
        self.K[0, 2], self.K[1, 2] = self.K[0, 2] / img_downscale, self.K[1, 2] / img_downscale
        self.load_data()
        pass

    def load_data(self):
        if self.use_cache:
            self.load_cache()
            return
        print('load reconstruction data of scene "{}", split: {}'.format(self.scene, self.split))
        base_dir = os.path.join(self.root_dir, self.scene)
        # train_split and test_split
        with open(os.path.join(base_dir, 'TrainSplit.txt'), 'r') as f:
            train_seq = f.readlines()
        with open(os.path.join(base_dir, 'TestSplit.txt'), 'r') as f:
            test_seq = f.readlines()
        train_seq = ['seq-0' + seq[-2] for seq in train_seq]  # seq-01 seq-02 ...
        test_seq = ['seq-0' + seq[-2] for seq in test_seq]  # ...
        # load
        self.N_views = 0
        self.file2id = {}
        self.view_filename = {}
        self.view_c2w = {}
        self.view_w2c = {}
        self.view_near = {}
        self.view_far = {}
        self.train_set = []
        self.test_set = []
        count_id = 0
        # 1. 解析训练、测试集views, poses, depth
        for seq in train_seq + test_seq:
            seq_dir = os.path.join(base_dir, seq)
            png_files = glob.glob(os.path.join(seq_dir, '*color.png'))
            dep_files = glob.glob(os.path.join(seq_dir, '*depth.png'))
            pose_files = glob.glob(os.path.join(seq_dir, '*.txt'))
            for png_file, dep_file, pose_file in zip(png_files, dep_files, pose_files):
                self.file2id[os.path.join(seq, png_file)] = count_id
                self.view_filename[count_id] = os.path.join(seq, png_file)
                c2w = np.loadtxt(os.path.join(seq_dir, pose_file), dtype=float)
                w2c = np.linalg.inv(c2w)
                self.view_c2w[count_id] = c2w[:3]
                self.view_w2c[count_id] = w2c[:3]
                depth = cv2.imread(os.path.join(seq_dir, dep_file), cv2.IMREAD_ANYDEPTH).astype(
                    float) / 1000  # 读取16bit深度图
                self.view_near[count_id] = np.percentile(depth[depth != 65.535], 0.1)
                self.view_far[count_id] = np.percentile(depth[depth != 65.535], 99.9)
                if seq in train_seq:
                    self.train_set.append(count_id)
                else:
                    self.test_set.append(count_id)
                count_id += 1
        # 2. scale to limit
        self.N_views = count_id
        max_dep = np.max([dep for dep in self.view_far.values()])
        self.scale_factor = max_dep / 4  # scale为[0,4]
        for i in range(self.N_views):
            self.view_c2w[i][:, -1] = self.view_c2w[i][:, -1] / self.scale_factor
            self.view_w2c[i][:, -1] = self.view_w2c[i][:, -1] / self.scale_factor
            self.view_far[i] = self.view_far[i] / self.scale_factor
            self.view_near[i] = self.view_near[i] / self.scale_factor
        # get all train set rays
        self.all_rays = []
        for i, id in tqdm.tqdm(enumerate(self.train_set), total=len(self.train_set), file=sys.stdout,
                               desc="acquiring rays "):
            rays = self.get_rays(id)
            self.all_rays += [rays]
        self.all_rays = torch.vstack(self.all_rays)
        print('all rays: ', self.all_rays.shape)
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
        img = np.array(img)
        # 获得世界坐标系下的光线的position和orientation
        c2w = self.view_c2w[i]
        rays_o, rays_d = get_rays_o_d(img_w, img_h, self.K, c2w)
        nears = self.view_near[i] * torch.ones((len(rays_o), 1))
        fars = self.view_far[i] * torch.ones((len(rays_o), 1))
        ids = i * torch.ones((len(rays_o), 1))
        rays = torch.hstack([rays_o, rays_d, nears, fars, ids, torch.FloatTensor(img.reshape(-1, 3)) / 255])
        return rays

    def load_cache(self):
        dict_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'dicts.pkl')
        with open(dict_path, 'rb') as f:
            ds = pickle.load(f)
            self.N_views, self.train_set, self.test_set, self.file2id, self.view_filename, self.view_w2c, \
                self.view_c2w, self.view_near, self.view_far, self.scale_factor = ds
        if self.split == 'train':
            ray_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'all_rays.npy')
            self.all_rays = torch.from_numpy(np.load(ray_path))
            print('all rays: ', self.all_rays.shape)
        # if self.split == 'train':
        #     self.all_rays=torch.zeros(size=(341913600,12),dtype=torch.float)
        print('cache load done...')

    def save_cache(self):
        dict_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'dicts.pkl')
        with open(dict_path, 'wb') as f:
            pickle.dump(
                [self.N_views, self.train_set, self.test_set, self.file2id, self.view_filename, self.view_w2c,
                 self.view_c2w, self.view_near, self.view_far, self.scale_factor], f)
        # 1. rays转numpy存
        rays = self.all_rays.numpy()
        ray_path = os.path.join(os.path.join(self.root_dir, self.scene), 'cache', 'all_rays.npy')
        np.save(ray_path, rays)
        print('cache save done...')

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.all_rays[idx]
        elif self.split in ['valid', 'test']:
            return self.get_rays(self.train_set[idx])

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        elif self.split == 'valid':  # 固定训练集某张图像进行validate, 有效不会过拟合因为train unit是ray而不是一张img
            return len(self.train_set)


if __name__ == '__main__':
    server_dir = '/root/autodl-tmp/dataset/7 scenes'
    scene = 'fire'
    local_dir = 'E:\\dataset\\7 scenes'
    dataset = SevenScenesDataset(server_dir, scene='fire', split='valid', img_downscale=2, use_cache=True,
                                 if_save_cache=True)
    print(1)