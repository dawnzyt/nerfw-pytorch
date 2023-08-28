##################################################################
################### generate novel views #########################
# 多卡DDP策略生成基于nerf的合成图片
# Input: novel_views相关数据, select_novel_views.py生成的novel_views.pkl
# Output: 保存nerf model生成的novel_views视图至数据集根目录下的seq_synthesis目录
# 注意:
# 1. 只使用到nerf的θ1和θ2, 即不考虑transient encoding;
# 2. appearance embedding从训练集中随机插值得到;

##################################################################
import os.path
import pickle
import sys

import PIL.Image
import numpy as np
import tqdm
from matplotlib import pyplot as plt

import torch.multiprocessing as mp
import torch.distributed as dist

from dataset.cambridge import CambridgeDataset
from dataset.seven_scenes import SevenScenesDataset
from model.nerfw_system import NeRFWSystem
from utils.utils import *


def init_processes(rank, world_size):
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23333',
                            rank=rank,
                            world_size=world_size)
    torch.manual_seed(0)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)


def generate(rank, world_size, opt, ckpt_path, novel_views_data, seq_dir_name='seq_synthesis'):
    init_processes(rank, world_size)
    # init
    # valid_dataset = CambridgeDataset(opt.data_root_dir, opt.scene, split='valid', img_downscale=opt.img_downscale,
    #                                  use_cache=True)
    valid_dataset = SevenScenesDataset(opt.data_root_dir, opt.scene, split='valid', img_downscale=opt.img_downscale,
                                     use_cache=True)
    nerf = NeRFWSystem(valid_dataset.N_views, 128, 256, opt.use_disp,
                       opt.perturb, opt.layers, opt.W, opt.N_xyz_freq, opt.N_dir_freq, opt.encode_a, opt.encode_t,
                       opt.a_dim, opt.t_dim, beta_min=opt.beta_min, lambda_u=opt.lambda_u)
    # load ckpt
    ckpt = torch.load(ckpt_path, map_location=f'cuda:{rank}')  # map
    nerf.load_state_dict(ckpt['nerf'])
    nerf = nerf.to(rank)
    print('GPU:{} start, load ckpt from {}'.format(rank, ckpt_path))
    # novel views data
    [img_size, Ks, c2ws, nears, fars, nearest_ids] = novel_views_data
    save_dir = os.path.join(opt.data_root_dir, opt.scene, seq_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 1. generate novel views
    for i in tqdm.tqdm(range(0, len(Ks), world_size), file=sys.stdout, desc='[gpu {}] generating...'.format(rank)):
        ###############################
        ## 生成synthesis view rays ####
        img_w, img_h = img_size
        rays_o, rays_d = get_rays_o_d(img_w, img_h, Ks[i], c2ws[i])
        near = nears[i]*15 / 16 * torch.ones((len(rays_o), 1))
        far = (fars[i] + nears[i] / 16) * torch.ones((len(rays_o), 1))
        ids = nearest_ids[i] * torch.ones((len(rays_o), 1))
        all_rays = torch.hstack([rays_o, rays_d, near, far, ids])
        ################################
        ## 随机插值appearance embedding ##
        idx = np.random.randint(0, len(valid_dataset.train_set) - 1, size=2)
        i1, i2 = valid_dataset.train_set[idx[0]], valid_dataset.train_set[idx[1]]
        a_emb1, a_emb2 = nerf.appearance_embedding.weight[i1], nerf.appearance_embedding.weight[i2]
        alpha = np.random.rand()
        a_emb = alpha * a_emb1 + (1 - alpha) * a_emb2  # 随机插值a_emb
        coarse_ray_rgbs = []
        fine_ray_rgbs = []
        fine_z = []
        chunk = 1024 * 3
        with torch.no_grad():
            rays_num = len(all_rays)
            for j in range(0, rays_num, chunk):
                rays = all_rays[j:j + chunk]
                rays=rays.to(rank)
                res_c, res_f, _ = nerf.forward(rays, custom_appearance_emb=None, test_time=False)
                coarse_ray_rgbs += [res_c['ray_rgb'].cpu()]
                fine_ray_rgbs += [res_f['ray_rgb'].cpu()]
                fine_z += [res_f['z'].cpu()]
            coarse_ray_rgbs = torch.vstack(coarse_ray_rgbs).view(img_h, img_w, 3)
            fine_ray_rgbs = torch.vstack(fine_ray_rgbs).view(img_h, img_w, 3)
            fine_z = torch.vstack(fine_z).view(img_h, img_w, 1)
        cc, fc, z = coarse_ray_rgbs.numpy(), fine_ray_rgbs.numpy(), fine_z.numpy()
        cc = (np.clip(cc, 0, 1) * 255).astype(np.uint8)
        fc = (np.clip(fc, 0, 1) * 255).astype(np.uint8)
        # nearest view
        img_path = os.path.join(opt.data_root_dir, opt.scene, valid_dataset.view_filename[nearest_ids[i]])
        nearest_img = PIL.Image.open(img_path).convert(mode='RGB')
        nearest_img = nearest_img.resize((img_w, img_h), PIL.Image.LANCZOS)
        nearest_img = np.array(nearest_img)
        # plt
        plt.figure(figsize=(16, 12))
        plt.subplot(2, 2, 1), plt.imshow(nearest_img), plt.title('nearest view')
        plt.subplot(2, 2, 2), plt.imshow(cc), plt.title('coarse')
        plt.subplot(2, 2, 3), plt.imshow(fc), plt.title('fine')
        plt.subplot(2, 2, 4), plt.imshow(visualize_depth(z)), plt.title('depth')
        plt.suptitle('synthesis view {}, nearest view [idx:{} id:{}, filename:{}]'.
                     format(i, valid_dataset.train_set.index(nearest_ids[i]), nearest_ids[i],
                            valid_dataset.view_filename[nearest_ids[i]]))
        plt.savefig(os.path.join(save_dir, 'figure_{}.png'.format(i)))
        cv2.imwrite(os.path.join(save_dir, 'synthesis_view_{}.png'.format(i)), cv2.cvtColor(fc, cv2.COLOR_RGB2BGR))

    # 2. save synthesis split information
    # dataset_synthesis.txt
    # Camera Position [X Y Z W P Q R], R为w2c, t为c2w即cam center
    if rank == 0:
        with open(os.path.join(opt.data_root_dir, opt.scene, 'dataset_synthesis.txt'), 'w') as f:
            for i in range(len(c2ws)):
                c2w = c2ws[i]
                bottom = np.zeros(shape=(1, 4))
                bottom[0, -1] = 1
                w2c = np.linalg.inv(np.vstack([c2w, bottom]))[:3]
                q = rotmat2quat(w2c[:, :3])
                f.write("%s %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (
                    os.path.join(seq_dir_name, f'synthesis_view_{i}.png'),
                    *list(c2w[:, -1] * valid_dataset.scale_factor), *list(q)))


if __name__ == '__main__':
    world_size = 1
    # resume from previous experiment: opt and ckpt
    opt_path = './runs/nerf/fire/opt.pkl'
    with open(opt_path, mode='rb') as f:
        opt = pickle.load(f)
    ckpt_path = '/root/zju/project/runs/nerf/fire/ckpt_epoch13_iter24000_psnr-23.091c-26.222f.pkl'
    # load novel views produced by `select_novel_views.py`
    cache_dir = os.path.join(opt.data_root_dir, opt.scene, 'cache')
    with open(os.path.join(cache_dir, 'novel_views.pkl'), mode='rb') as f:
        novel_views_data = pickle.load(f)
    mp.spawn(generate, args=(world_size, opt, ckpt_path, novel_views_data), nprocs=world_size, join=True)
