import torch
from torch import nn


# 基本的均方loss
class ColorLoss(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c
        self.lossfn = nn.MSELoss(reduction='mean')

    def forward(self, pred, y):
        return self.c * self.lossfn(pred, y)  # 均方loss再/2


# nerf-w考虑transient noise定义的似然loss
class LikelihoodLoss(nn.Module):
    def __init__(self, c=1, lambda_u=0.01):
        super().__init__()
        self.c = c
        self.lambda_u = lambda_u  # transient sigma的正则项
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred_c, y_c, beta, transient_sigma):
        """

        :param pred_c: (n_rays, 3) color
        :param y_c: (n_rays, 3)    ground truth color
        :param beta: (n_rays, 1) transient beta
        :param transient_sigma: (n_rays, N_samples) transient_sigma
        :return:
        """
        if beta is None and transient_sigma is None:  # without encoding transient objects
            return self.mse(pred_c, y_c), None
        likelihood_loss = self.mse(pred_c / beta, y_c / beta) / 2 + torch.mean(torch.log(beta))
        regular_loss = self.lambda_u * torch.mean(transient_sigma)
        return likelihood_loss, regular_loss
