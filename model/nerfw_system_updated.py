import os.path
import time
from tqdm import tqdm

import cv2
from model.nerfw_basemodel import NeRFW
from model.position_emb import PositionEmbedding
from model.custom_nerfw import CustomNeRFW
import torch
import torch.nn as nn
from einops import rearrange
from loss.nerfw_loss import ColorLoss, LikelihoodLoss
from utils.evaluation_metrics import psnr


class NeRFWSystem(nn.Module):
    def __init__(self, N_views=2017, N_c=64, N_f=128, use_disp=False, perturb=1.0
                 , layers=8, W=256, N_xyz_freq=10, N_dir_freq=4, encode_a=True, encode_t=True,
                 a_dim=48, t_dim=16, res_layer=[4], device='cpu',
                 beta_min=0.03, lambda_u=0.01, white_back=False
                 ):
        super().__init__()
        # init hyperparameters
        self.N_c = N_c
        self.N_f = N_f
        self.use_disp = use_disp
        self.perturb = perturb
        self.beta_min = beta_min
        self.white_back = white_back
        self.encode_a = encode_a  # fine
        self.encode_t = encode_t  # fine
        ###########################################################################
        ############## 这里可以任意设置coarse model的encode_a、encode_t #############
        self.coarse_model = CustomNeRFW(layers, W, N_xyz_freq, N_dir_freq, True, False, a_dim, t_dim, res_layer,
                                        N_views, beta_min, white_back)
        self.fine_model = CustomNeRFW(layers, W, N_xyz_freq, N_dir_freq, encode_a, encode_t, a_dim, t_dim, res_layer,
                                      N_views, beta_min, white_back)
        self.c_loss = ColorLoss()
        self.f_loss = LikelihoodLoss(lambda_u=lambda_u)

    def sample_uniform(self, near, far, N_samples):
        """
        根据near、far均匀采样z
        :param near: (n_rays,1)
        :param far: (n_rays,1)
        :param N_samples:
        :return:
        """
        n = len(near)
        u = torch.linspace(0, 1, N_samples, device=near.device)
        u = rearrange(u, "N -> 1 N").tile(n, 1)
        # 是否disp空间采样,disp与z成反比
        if self.use_disp:
            z = 1 / ((1 / near) * (1 - u) + (1 / far) * u)
        else:
            z = near * (1 - u) + far * u
        # add perturbation
        z_mid = (z[:, :-1] + z[:, 1:]) / 2
        z_upper = torch.hstack([z_mid, z[:, -1:]])
        z_lower = torch.hstack([z[:, :1], z_mid])
        z = self.perturb * torch.rand_like(z, device=near.device) * (z_upper - z_lower) + z_lower
        return z

    def sample_pdf(self, weights, z, N_samples, identical=False, eps=1e-5):
        """
        采样时, weights和pdf等价, 不过未归一, mid_w指weights离散分布每两点的中点, 以便通过面积积分计算cdf。
        :param weights: (n_rays, m)
        :param z: (n_rays, m)
        :param N_samples:
        :return:
        """
        n_rays, m = weights.shape
        mid_w = (weights[:, :-1] + weights[:, 1:]) / 2
        mid_w += eps  # 防止全0
        if not identical:
            u = torch.rand(size=(n_rays, N_samples), device=weights.device)
        else:
            u = torch.rand(size=(1, N_samples), device=weights.device).tile(n_rays, 1)
        pdf = mid_w / torch.sum(mid_w, dim=1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=1)
        cdf = torch.hstack([torch.zeros((n_rays, 1), device=weights.device), cdf])
        u = u.contiguous()  # 使得内存连续一致
        idx = torch.searchsorted(cdf, u, right=True)
        idx_lower = torch.clamp_min(idx - 1, 0)  # 确定根据cdf均匀采样点的position
        idx_upper = torch.clamp_max(idx, m - 1)

        # z=lower_z*(1-α) + upper_z*α
        # 这里还要考虑weights=0->两点之间的概率密度积分即Δcdf=0的情况,防止除0,且这种不采样他。
        # idx_gather = rearrange(torch.stack([idx_lower, idx_upper], dim=-1), "n_rays m a -> n_rays (m a)", a=2)
        # cdf_low_up = rearrange(torch.gather(cdf, dim=1, index=idx_gather), "n_rays (m a) -> n_rays m a", a=2)
        # z_low_up = rearrange(torch.gather(z, dim=1, index=idx_gather), "n_rays (m a) -> n_rays m a", a=2)
        # cdf_lower,cdf_up=cdf_low_up[:,:,0],cdf_low_up
        cdf_lower, cdf_upper = torch.gather(cdf, dim=1, index=idx_lower), torch.gather(cdf, dim=1, index=idx_upper)
        z_lower, z_upper = torch.gather(z, dim=1, index=idx_lower), torch.gather(z, dim=1, index=idx_upper)
        delta_cdf = cdf_upper - cdf_lower
        delta_cdf[delta_cdf < eps] = 1  # 当weight等于0时cdf=0即不取u决定的这个z_low和z_up之间中间的这个点了, 统一变为取z_low
        alpha = (u - cdf_lower) / delta_cdf
        fine_z = z_lower + (z_upper - z_lower) * alpha
        return fine_z

    def forward(self, rays, rays_rgb=None, cal_loss=False, custom_appearance_emb=None, test_time=False):
        """
        steps:
        1. 粗采样sample_uniform
        2. coarse model render, 计算static_sigma, static_rgb, weights of rgb.
        3. 细采样sample_pdf
        4. fine model render, 计算fstatic_sigma,fstatic_rgb...

        :param rays: (batch_size, 9), [position、direction、near、far、id]
        :param rays_rgb: (batch_size, 3), rgb
        :param cal_loss: True:计算loss(rays_rgb≠None), False:不计算loss
        :param custom_appearance_emb: 自定义appearance embedding
        :param test_time: True:测试时不encode transient
        :return: res_c, res_f, losses: all dicts
        """
        n = len(rays)
        rays_o, rays_d, near, far, ids = torch.split(rays, [3, 3, 1, 1, 1], dim=1)
        # 1. coarse sample uniform
        z = self.sample_uniform(near, far, self.N_c)  # sample coarse
        # 2. coarse rendering
        res_c = self.coarse_model.forward(rays_o, rays_d, z, ids, custom_appearance_emb, test_time)
        coarse_static_weights = res_c['static_weights']  # used for fine model sampling
        # 3. fine sample pdf
        N_all = self.N_c + self.N_f  # 把所有的采样点（N_c + N_f）一起输入到fine网络进行预测
        fine_z = self.sample_pdf(coarse_static_weights.detach(), torch.squeeze(z, -1), self.N_f)
        all_z = torch.sort(torch.hstack([z, fine_z]), dim=1)[0]  # sort
        # 4. fine rendering
        res_f = self.fine_model.forward(rays_o, rays_d, all_z, ids, custom_appearance_emb, test_time)
        # 5. calculate loss
        losses = dict() if cal_loss else None
        if cal_loss:
            # ①coarse loss
            loss_c = self.c_loss(res_c['ray_rgb'], rays_rgb)
            # ②fine loss
            fine_transient_ray_beta, fine_transient_sigma = res_f.get('transient_ray_beta', None), res_f.get(
                'transient_sigma', None)
            # encode_t: loss_f is negative likelihood loss
            # not encode_t: regular_loss=None, loss_f is simple color mse loss of fine model
            loss_f, regular_loss = self.f_loss(res_f['ray_rgb'], rays_rgb, fine_transient_ray_beta,
                                               fine_transient_sigma)  # fine loss
            c_psnr = psnr(res_c['ray_rgb'].detach(), rays_rgb)
            f_psnr = psnr(res_f['ray_rgb'].detach(), rays_rgb)
            losses['coarse'], losses['fine'], losses['fine_regular'] = loss_c, loss_f, regular_loss
            losses['c_psnr'], losses['f_psnr'] = c_psnr, f_psnr
        return res_c, res_f, losses

    def inference(self, all_rays, chunk):
        """
        推理阶段即render一个whole img的所有rays, 这里即all_rays, rays的format和forward时一致。
        Input: rays
        Output: complete image
        :param all_rays: [n_rays,9] [position、direction、near、far、id]
        :param chunk: 一次render光线数目
        :return:
        """
        coarse_ray_rgbs = []
        fine_static_ray_rgbs = []
        fine_transient_ray_rgbs = []
        fine_ray_rgbs = []
        fine_static_z = []
        with torch.no_grad():
            self.eval()
            rays_num = len(all_rays)
            for i in tqdm(range(0, rays_num, chunk), total=rays_num // chunk + 1,desc='inference'):
                rays = all_rays[i:i + chunk]
                res_c, res_f, _ = self.forward(rays)
                coarse_ray_rgbs += [res_c['ray_rgb']]
                if self.encode_t:
                    fine_static_ray_rgbs += [res_f['static_ray_rgb']]
                    fine_transient_ray_rgbs += [res_f['transient_ray_rgb']]
                fine_ray_rgbs += [res_f['ray_rgb']]
                fine_static_z += [res_f['static_z']]
            res = dict()
            res['cc'] = torch.vstack(coarse_ray_rgbs)
            if self.encode_t:
                res['fs'] = torch.vstack(fine_static_ray_rgbs)
                res['ft'] = torch.vstack(fine_transient_ray_rgbs)
            res['fc'] = torch.vstack(fine_ray_rgbs)
            res['z'] = torch.vstack(fine_static_z)
            self.train()
            return res
