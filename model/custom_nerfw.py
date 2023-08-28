from einops import rearrange

from model.nerfw_basemodel import NeRFW
import torch.nn as nn
import torch
import numpy as np

from model.position_emb import PositionEmbedding


# 在NeRFW basemodel基础上定义NeRF-W支撑的体渲染任务
# 1. 定义nerf-w的appearance embedding和transient embedding
# 2. 定义了voxel rendering 相关func:render和forward
class CustomNeRFW(nn.Module):
    def __init__(self, layers=8, W=256, N_xyz_freq=10, N_dir_freq=4, encode_a=True, encode_t=True, a_dim=48, t_dim=16,
                 res_layer=[4], N_views=2017, beta_min=0.03, white_back=False):
        super().__init__()
        # init
        self.encode_a = encode_a
        self.encode_t = encode_t
        self.a_dim = a_dim if encode_a else 0
        self.t_dim = t_dim if encode_t else 0
        self.in_xyz_dim = 3 * (2 * N_xyz_freq + 1)
        self.in_dir_dim = 3 * (2 * N_dir_freq + 1)
        self.model = NeRFW(layers, W, self.in_xyz_dim, self.in_dir_dim, encode_a, encode_t, a_dim, t_dim, res_layer)
        self.N_views = N_views
        self.beta_min = beta_min
        self.white_back = white_back
        # embeddings
        if encode_a:
            self.appearance_embedding = nn.Embedding(N_views, a_dim)
        if encode_t:
            self.transient_embedding = nn.Embedding(N_views, t_dim)
        # position embedding
        self.pos_emb = PositionEmbedding(max_log_freq=N_xyz_freq - 1, N_freqs=N_xyz_freq)
        self.dir_emb = PositionEmbedding(max_log_freq=N_dir_freq - 1, N_freqs=N_dir_freq)

    def render(self, z, sigma, second_sigma=None, c=None, transmittance=None, only_weights=False):
        """
        render equation: I(s)=ΣTn*(1-exp(-σn*δn))*cn {1-N}, Tn=exp(Σ-σk*δk) {1- n-1}
        其中T(transmittance)是光传输率,σ为体素密度,δ即Δz（按z采样）。
        attention:
        1. 实际做最后一个δ为inf
        2. 光传输率第一项为1。
        3. weight即Tn(1-exp(-σnδn)), 可理解为从渲染方程中抽象出来的加权权重。

        Inputs:
            z: (n_rays, n_samples)
            sigma: (n_rays, n_samples)
            second_sigma: (n_rays,n_samples), 可能是场景transient的sigma
            c: (n_rays, n_samples, m) 一种attribute, m任意,取决于nerf输出
            transmittance: (n_rays,n_samples), 复用transmittance
            only_weights: True-直接返回weights

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

    def forward(self, rays_o, rays_d, z, ids, custom_appearance_emb=None, test_time=False):
        """
        给出n_rays个ray，每个ray采样m个z值，基于体渲染的方法计算每个ray的rgb,以及transient的rgb、beta等
        Input:
            rays_o: (n_rays,3)
            rays_d: (n_rays,3)
            z: (n_rays,samples)
            ids: (n_rays,1), embedding ids
            custom_appearance_emb: (a_dim), custom appearance embedding
            test_time: test_time means absolutely no transient objects
        :return:
        """
        # 1. prepare
        n, m = z.shape[0], z.shape[1]
        rays_o = rearrange(rays_o, "n c -> n 1 c")
        rays_d = rearrange(rays_d, "n c -> n 1 c")
        xyz = rays_o + rays_d * z.unsqueeze(-1)
        dir = rays_d.tile(1, m, 1)
        xyz_emb = self.pos_emb.embed(xyz.view(-1, 3))
        dir_emb = self.dir_emb.embed(dir.view(-1, 3))
        ids = ids.long()
        if self.encode_a:
            if custom_appearance_emb is None:  # default appearance embedding:shape(N_views,a_dim)
                appearance_emb = self.appearance_embedding(ids).tile(1, m, 1).view(-1, self.a_dim)
            else:  # custom appearance embedding:shape(a_dim,)
                appearance_emb = custom_appearance_emb.view(1, -1).tile(n * m, 1)
        if self.encode_t and not test_time:
            transient_emb = self.transient_embedding(ids).tile(1, m, 1).view(-1, self.t_dim)
        a_t_emb = ([] if not self.encode_a else [appearance_emb]) + (
            [] if (not self.encode_t or test_time) else [transient_emb])
        x = torch.hstack([xyz_emb, dir_emb] + a_t_emb)

        # 2.volumetric rendering
        # ①forward
        if self.encode_t and not test_time:
            static_sigma, static_rgb, transient_sigma, transient_rgb, transient_beta = self.model(x, flag=2)
            transient_sigma, transient_rgb, transient_beta = transient_sigma.view(n, m), \
                transient_rgb.view(n, m, 3), transient_beta.view(n, m, 1)  # reshape
        else:
            static_sigma, static_rgb = self.model(x, flag=1)
        static_sigma, static_rgb = static_sigma.view(n, m), static_rgb.view(n, m, 3)  # reshape
        # ②render static scene
        if self.encode_t and not test_time:
            # all_T: 即整个场景包括static和transient的光传输率, 可复用
            all_T, static_weights, static_ray_rgb = \
                self.render(z, static_sigma, transient_sigma, static_rgb)  # render static
        else:
            _, static_weights, static_ray_rgb = \
                self.render(z, static_sigma, None, static_rgb)  # render fine static
        # ③render transient rgb and beta
        if self.encode_t and not test_time:
            _, transient_weights, rgb_beta = \
                self.render(z, transient_sigma, None, torch.cat([transient_rgb, transient_beta], -1), all_T)
            transient_ray_rgb, transient_ray_beta = rgb_beta[:, :3], rgb_beta[:, -1:] + self.beta_min
        # ④merge static and transient
        ray_rgb = static_ray_rgb
        if self.encode_t and not test_time:  # transient ray rgb color
            ray_rgb = static_ray_rgb + transient_ray_rgb
        # ⑤white background
        if self.white_back:
            if self.encode_t and not test_time:  # consider both sigma, the sum of static sigma and transient sigma
                merge_weights = self.render(z, static_sigma + transient_sigma, None, None, all_T,
                                            only_weights=True)
                opacity = torch.sum(merge_weights, dim=1, keepdim=True)
            else:
                opacity = torch.sum(static_weights, dim=1, keepdim=True)
            ray_rgb += 1 - opacity
        # 3. return
        res = dict()
        res['ray_rgb'] = ray_rgb
        res['static_weights'] = static_weights # maybe coarse model's output for fine z pdf sampling
        res['static_z'] = torch.sum(static_weights * z, dim=1, keepdim=True)
        if self.encode_t and not test_time:
            res['transient_sigma'] = transient_sigma
            res['static_ray_rgb'] = static_ray_rgb
            res['transient_ray_beta'] = transient_ray_beta
            res['transient_ray_rgb'] = transient_ray_rgb
            res['transient_weights'] = transient_weights
            res['transient_z'] = torch.sum(transient_weights * z, dim=1, keepdim=True)
        return res
