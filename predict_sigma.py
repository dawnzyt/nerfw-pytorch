'''
1. 利用sfm points和train camera center计算bounding box;
2. 生成bounding box内的meshgrid points;
3. predict sigma for each point, delete points whose sigma < sigma_threshold;
4. 保存得到: ①bounding_box.npy; ② predict_scene.npy:(n,4) [points,sigma]; ③ predict_scene.ply(以可视化)
'''
import os.path
import pickle
import sys

import tqdm
from dataset.cambridge import CambridgeDataset
from dataset.seven_scenes import SevenScenesDataset
from model.nerfw_system import NeRFWSystem
from utils.utils import *

# 参数为默认参数与《LENS: Localization enhanced by NeRF synthesis》一致
# link: https://readpaper.com/pdf-annotate/note?pdfId=4551871871071559681&noteId=1869358340360615936
sigma_threshold = 20  # scene体素密度阈值
resolution = 64  # 最短边resolution, 确定lambda_v
E_max = 0.1  # expanding distance, 单位m
bounding_box = []  # 包围所有train view poses的bounding box而非整个3D scene的bounding box
print('sigma_threshold:%d\nresolution:%d\nE_max:%dm' % (sigma_threshold, resolution, E_max))
if __name__ == '__main__':
    # 1. resume opt, dataset, nerf-w model
    opt_path = './runs/nerf/fire/opt.pkl'
    with open(opt_path, mode='rb') as f:
        opt = pickle.load(f)
    valid_dataset = SevenScenesDataset(opt.data_root_dir, opt.scene, split='valid', img_downscale=opt.img_downscale,
                                     use_cache=True,)
    # 保持尺度一致
    E_max = E_max / valid_dataset.scale_factor
    print('scale_factor: ', valid_dataset.scale_factor)
    nerf = NeRFWSystem(valid_dataset.N_views, opt.N_c, opt.N_f, opt.use_disp,
                       opt.perturb, opt.layers, opt.W, opt.N_xyz_freq, opt.N_dir_freq, opt.encode_a, opt.encode_t,
                       opt.a_dim, opt.t_dim, beta_min=opt.beta_min, lambda_u=opt.lambda_u)
    # load ckpt
    ckpt_path = '/root/zju/project/runs/nerf/fire/ckpt_epoch13_iter24000_psnr-23.091c-26.222f.pkl'
    ckpt = torch.load(ckpt_path)
    nerf.load_state_dict(ckpt['nerf'])
    nerf = nerf.cuda()

    # 2. get 3D meshgrid
    # camera centers
    centers = [valid_dataset.view_c2w[id][:, -1] for id in valid_dataset.train_set]
    centers = np.vstack(centers)
    # bounding box1: 包围重建三维点和train view的camera中心
    # min_x, max_x = np.percentile(points[:, 0], 0.01), np.percentile(points[:, 0], 99.99)
    # min_y, max_y = np.percentile(points[:, 1], 0.01), np.percentile(points[:, 1], 99.99)
    # min_z, max_z = np.percentile(points[:, 2], 0.01), np.percentile(points[:, 2], 99.99)
    # min_x = min(min_x, centers[:, 0].min())
    # min_y = min(min_y, centers[:, 1].min())
    # min_z = min(min_z, centers[:, 2].min())
    # max_x = max(max_x, centers[:, 0].max())
    # max_y = max(max_y, centers[:, 1].max())
    # max_z = max(max_z, centers[:, 2].max())
    min_x = np.percentile(centers[:, 0], 0.5)
    min_y = np.percentile(centers[:, 1], 0.5)
    min_z = np.percentile(centers[:, 2], 0.5)
    max_x = np.percentile(centers[:, 0], 99.5)
    max_y = np.percentile(centers[:, 1], 99.5)
    max_z = np.percentile(centers[:, 2], 99.5)
    # bounding box2: expand box1
    min_x -= E_max
    min_y -= E_max
    min_z -= E_max
    max_x += E_max
    max_y += E_max
    max_z += E_max
    bounding_box.append([min_x, max_x])
    bounding_box.append([min_y, max_y])
    bounding_box.append([min_z, max_z])
    bounding_box = np.array(bounding_box)
    print('bounding_box:', bounding_box * valid_dataset.scale_factor)
    # 生成mesh
    lambda_v = np.min([(bounding_box[0, 1] - bounding_box[0, 0]), (bounding_box[1, 1] - bounding_box[1, 0]),
                       (bounding_box[2, 1] - bounding_box[2, 0])]) / resolution
    mesh = np.mgrid[bounding_box[0, 0]:bounding_box[0, 1]:lambda_v, bounding_box[1, 0]:bounding_box[1, 1]:lambda_v,
           bounding_box[2, 0]:bounding_box[2, 1]:lambda_v]
    (n, m, l) = mesh.shape[1:]  # shape
    mesh_points = np.reshape(mesh, [3, -1]).T  # [n*m*l, 3]
    print(f'mesh shape: [{n},{m},{l}], points num: {mesh_points.shape[0]}')

    # 3. predict sigma
    chunk = 8 * 1024
    # predict sigma
    with torch.no_grad():
        x = torch.from_numpy(mesh_points).float()
        sigma = []
        for i in tqdm.tqdm(range(0, x.shape[0], chunk), desc='predicting sigma', file=sys.stdout,
                           total=x.shape[0] // chunk):
            x_chunk = x[i:i + chunk].cuda()
            x_embed = nerf.pos_emb.embed(x_chunk)
            sigma_chunk = nerf.fine_model.forward(x_embed, flag=0)
            sigma += [sigma_chunk.cpu().numpy()]
        sigma = np.vstack(sigma)
    sigma = np.squeeze(sigma, -1)
    # percent = 98.5
    # sigma_threshold = np.percentile(sigma, percent)
    # print(percent, '%')
    mesh_points = mesh_points[sigma > sigma_threshold]
    sigma = sigma[sigma > sigma_threshold]
    print('filtered points num: ', mesh_points.shape[0])

    # 4. save scene points, expanded bounding box
    cache_dir = os.path.join(opt.data_root_dir, opt.scene, 'cache')
    np.save(os.path.join(cache_dir, 'box_scene.npy'), np.hstack([mesh_points, sigma.reshape(-1, 1)]))
    np.save(os.path.join(cache_dir, 'bounding_box.npy'), bounding_box)

    path = os.path.join(opt.data_root_dir, opt.scene, 'box_scene.ply')
    with open(path, 'w') as f:
        f.write(ply_header.format(mesh_points.shape[0], 0, 0))
    with open(path, 'a') as f:
        x = np.hstack([mesh_points, np.zeros((len(sigma), 2)), 255 * np.ones((len(sigma), 1))])
        np.savetxt(f, np.c_[x], fmt='%.6f %.6f %.6f %d %d %d')
