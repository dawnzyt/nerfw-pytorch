import torch
from torch import nn


# position embedding, 增强模型对xyz、direction的感知
# x-> (x,sin(2^k*x),co(2^k*x)...), shape:(N_freqs+1,)
class PositionEmbedding:
    def __init__(self, max_log_freq, N_freqs=10, sample_log=True):
        """
        position embedding
        :param max_log_freq: log2(maximum frequency)
        :param N_freqs: number of frequencies
        :param sample_log: whether to sample in log space
        """
        self.funcs = [torch.sin, torch.cos]
        if sample_log:
            self.freqs = 2 ** torch.linspace(0, max_log_freq, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2 ** max_log_freq, N_freqs)

    def embed(self, x):
        """

        :param x: (n,m)
        :return: (n,m*(1+2*N_freqs))
        """
        res = [x]
        for freq in self.freqs:
            for func in self.funcs:
                res.append(func(freq * x))
        return torch.hstack(res)
