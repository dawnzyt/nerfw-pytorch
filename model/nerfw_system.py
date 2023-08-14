import os.path
import time

import cv2
from model.nerfw_basemodel import NeRFW
from model.position_emb import PositionEmbedding
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
        self.c_loss = ColorLoss()
        self.f_loss = LikelihoodLoss(lambda_u=lambda_u)
        # init model
        in_xyz_dim, in_dir_dim = 3 * (2 * N_xyz_freq + 1), 3 * (2 * N_dir_freq + 1)
        self.coarse_model = NeRFW(layers, W, in_xyz_dim, in_dir_dim, False, False, res_layer=res_layer, device=device)
        self.fine_model = NeRFW(layers, W, in_xyz_dim, in_dir_dim, encode_a, encode_t, a_dim, t_dim, res_layer, device)
        self.coarse_model = self.coarse_model.to(device)
        self.fine_model = self.fine_model.to(device)
        self.pos_emb = PositionEmbedding(max_log_freq=N_xyz_freq - 1, N_freqs=N_xyz_freq)
        self.dir_emb = PositionEmbedding(max_log_freq=N_dir_freq - 1, N_freqs=N_dir_freq)
        self.encode_a = encode_a
        self.encode_t = encode_t
        self.a_dim = a_dim
        self.t_dim = t_dim
        if encode_a:
            self.appearance_embedding = nn.Embedding(N_views, a_dim)
        if encode_t:
            self.transient_embedding = nn.Embedding(N_views, t_dim)

        # load ckpt
        # if not is_train or ckpt_path:
        #     self.load_state_dict(torch.load(ckpt_path))
        # self.coarse_model.load_state_dict(torch.load(os.path.join(ckpt_path, 'coarse.pkl')))
        # self.fine_model.load_state_dict(torch.load(os.path.join(ckpt_path, 'fine.pkl')))
        # if encode_a:
        #     self.appearance_embedding.load_state_dict(torch.load(os.path.join(ckpt_path, 'a_emb.pkl')))
        # if encode_t:
        #     self.transient_embedding.load_state_dict(torch.load(os.path.join(ckpt_path, 't_emb.pkl')))

    def render(self, z, sigma, second_sigma=None, c=None, transmittance=None, only_weights=False):
        """
        render equation: I(s)=ΣTn*(1-exp(-σn*δn))*cn {1-N}, Tn=exp(Σ-σk*δk) {1- n-1}
        其中T(transmittance)是光传输率,σ为体素密度,δ即Δz（按z采样）。
        attention:
        1. 实际做最后一个δ为inf
        2. 光传输率第一项为1。
        3. weight即Tn(1-exp(-σnδn)), 可理解为从渲染方程中抽象出来的加权权重。
        :param z: (n_rays, n_samples)
        :param sigma: (n_rays, n_samples)
        :param second_sigma: (n_rays,n_samples), 可能是场景transient的sigma
        :param c: (n_rays, n_samples, m) 一种attribute, m任意,取决于nerf输出
        :param transmittance: (n_rays,n_samples), 复用transmittance
        :param only_weights: True-直接返回weights

        Outputs:
            transmittance,weights, render_result
            transmittance: 在all_sigma下计算得到的光传输率
            weights: 用sigma计算得到的alpha乘以transmittance得到的不透明度
        """
        n_rays = len(z)
        delta = torch.hstack([z[:, 1:] - z[:, :-1], 233 * torch.ones((n_rays, 1), device=z.device)])
        alpha = 1 - torch.exp(-sigma * delta)  # static或transient的alpha
        if transmittance is not None:  # 复用
            T = transmittance
        else:
            if second_sigma is not None:  # 计算merge transmittance
                all_sigma = sigma + second_sigma
                T = torch.cumprod(torch.exp(-all_sigma * delta), dim=1)
            else:
                T = torch.cumprod(1 - alpha, dim=1)
            T = torch.hstack([torch.ones((n_rays, 1), device=z.device), T[:, :-1]])
        weights = T * alpha
        if only_weights:
            return weights
        weights = rearrange(weights, "n_rays n_samples -> n_rays n_samples 1")
        render_c = torch.sum(weights * c, dim=1)
        return T, torch.squeeze(weights, -1), render_c

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
        z = self.sample_uniform(near, far, self.N_c)  # sample coarse

        # obtain xyz, embedded xyz, dir for coarse model
        rays_o = rearrange(rays_o, "n c -> n 1 c")
        rays_d = rearrange(rays_d, "n c -> n 1 c")
        z = rearrange(z, "n N_c -> n N_c 1")
        xyz = rays_o + rays_d * z
        xyz_emb = self.pos_emb.embed(xyz.view(-1, 3))
        dir_emb = self.dir_emb.embed(rays_d.tile(1, self.N_c, 1).view(-1, 3))
        # coarse model forward
        x = torch.hstack([xyz_emb, dir_emb])
        static_sigma, static_rgb = self.coarse_model.forward(x, flag=1)
        static_sigma, static_rgb = static_sigma.view(n, self.N_c), static_rgb.view(n, self.N_c, 3)
        _, weights, coarse_ray_rgb = self.render(torch.squeeze(z, -1), static_sigma, None, static_rgb)
        if self.white_back:  # 白色背景
            opacity = torch.sum(weights, dim=1, keepdim=True)  # 不透明度
            coarse_ray_rgb += 1 - opacity

        # fine sample by pdf, embed...
        N_all = self.N_c + self.N_f  # 把所有的采样点（N_c + N_f）一起输入到fine网络进行预测
        fine_z = self.sample_pdf(weights.detach(), torch.squeeze(z, -1), self.N_f)
        all_z = torch.sort(torch.hstack([z.squeeze(-1), fine_z]), dim=1)[0]  # 从小到大sort
        # all_z = torch.sort(fine_z, dim=1)[0]
        xyz = rays_o + rays_d * torch.unsqueeze(all_z, -1)
        xyz_emb = self.pos_emb.embed(xyz.view(-1, 3))
        dir_emb = self.dir_emb.embed(rays_d.tile(1, N_all, 1).view(-1, 3))
        ids = ids.long()
        if self.encode_a:
            if custom_appearance_emb is None:
                appearance_emb = self.appearance_embedding(ids).tile(1, N_all, 1).view(-1, self.a_dim)
            else: # custom appearance embedding:shape(a_dim,)
                appearance_emb=custom_appearance_emb.view(1,-1).tile(n*N_all,1)
        if self.encode_t and not test_time:
            transient_emb = self.transient_embedding(ids).tile(1, N_all, 1).view(-1, self.t_dim)
        a_t_emb = ([] if not self.encode_a else [appearance_emb]) +\
                  ([] if (not self.encode_t or test_time) else [transient_emb])
        x = torch.hstack([xyz_emb, dir_emb] + a_t_emb)

        # fine model forward
        if self.encode_t and not test_time:   # forward
            fstatic_sigma, fstatic_rgb, ftransient_sigma, ftransient_rgb, ftransient_beta = self.fine_model(x, flag=2)
            ftransient_sigma, ftransient_rgb, ftransient_beta = ftransient_sigma.view(n, N_all), \
                ftransient_rgb.view(n, N_all, 3), ftransient_beta.view(n, N_all, 1)  # reshape
        else:
            fstatic_sigma, fstatic_rgb = self.fine_model(x, flag=1)
        fstatic_sigma, fstatic_rgb = fstatic_sigma.view(n, N_all), fstatic_rgb.view(n, N_all, 3)  # reshape
        if self.encode_t and not test_time:   # render fine static scene:
            # all_T: 即整个场景包括static和transient的光传输率, 可复用
            all_T, fine_weights, fine_static_ray_rgb = \
                self.render(all_z, fstatic_sigma, ftransient_sigma, fstatic_rgb)  # render static
        else:
            _, fine_weights, fine_static_ray_rgb = \
                self.render(all_z, fstatic_sigma, None, fstatic_rgb)  # render fine static

        # render fine transient, rgb and beta
        if self.encode_t and not test_time:
            _, fine_transient_weights, rgb_beta = \
                self.render(all_z, ftransient_sigma, None, torch.cat([ftransient_rgb, ftransient_beta], -1), all_T)
            fine_transient_ray_rgb, transient_ray_beta = rgb_beta[:, :3], rgb_beta[:, -1:] + self.beta_min
        fine_ray_rgb = fine_static_ray_rgb
        if self.encode_t and not test_time:   # transient ray rgb color
            fine_ray_rgb = fine_static_ray_rgb + fine_transient_ray_rgb
        if self.white_back:
            if self.encode_t and not test_time:   # consider both sigma, the sum of static sigma and transient sigma
                merge_weights = self.render(all_z, fstatic_sigma + ftransient_sigma, None, None, all_T,
                                            only_weights=True)
                opacity = torch.sum(merge_weights, dim=1, keepdim=True)
            else:
                opacity = torch.sum(fine_weights, dim=1, keepdim=True)
            fine_ray_rgb += 1 - opacity

        # record, results
        res_c, res_f = dict(), dict()
        res_c['ray_rgb'] = coarse_ray_rgb
        if self.encode_t and not test_time:
            res_f['static_ray_rgb'] = fine_static_ray_rgb
            res_f['transient_ray_rgb'] = fine_transient_ray_rgb
            res_f['beta'] = transient_ray_beta
        res_f['ray_rgb'] = fine_ray_rgb
        # merge_weights = self.render(all_z, fstatic_sigma + ftransient_sigma, None, None, all_T,
        #                             only_weights=True)
        res_f['z'] = torch.sum(fine_weights * all_z, dim=1, keepdim=True)

        # calculate loss
        losses = dict() if cal_loss else None
        if cal_loss:
            loss_c = self.c_loss(coarse_ray_rgb, rays_rgb)  # coarse loss
            if not self.encode_t:
                transient_ray_beta, ftransient_sigma = None, None
            # encode_t: loss_f is negative likelihood loss
            # not encode_t: regular_loss=None, loss_f is simple color mse loss of fine model
            loss_f, regular_loss = self.f_loss(fine_ray_rgb, rays_rgb, transient_ray_beta, ftransient_sigma)  # fine loss
            c_psnr = psnr(coarse_ray_rgb.detach(), rays_rgb)
            f_psnr = psnr(fine_ray_rgb.detach(), rays_rgb)
            losses['coarse'], losses['fine'], losses['fine_regular'] = loss_c, loss_f + 3, regular_loss
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
        fine_z = []
        with torch.no_grad():
            rays_num = len(all_rays)
            for i in range(0, rays_num, chunk):
                rays = all_rays[i:i + chunk]
                res_c, res_f, _ = self.forward(rays)
                coarse_ray_rgbs += [res_c['ray_rgb']]
                if self.encode_t:
                    fine_static_ray_rgbs += [res_f['static_ray_rgb']]
                    fine_transient_ray_rgbs += [res_f['transient_ray_rgb']]
                fine_ray_rgbs += [res_f['ray_rgb']]
                fine_z += [res_f['z']]
            res = dict()
            res['cc'] = torch.vstack(coarse_ray_rgbs)
            if self.encode_t:
                res['fs'] = torch.vstack(fine_static_ray_rgbs)
                res['ft'] = torch.vstack(fine_transient_ray_rgbs)
            res['fc'] = torch.vstack(fine_ray_rgbs)
            res['z'] = torch.vstack(fine_z)
            return res

    def synthesis(self, img_size, ):
        """
        利用camera position and direction合成
        :return:
        """

    def save_model(self):
        pass
