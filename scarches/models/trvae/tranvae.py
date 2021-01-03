import torch
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F
from typing import Optional
import numpy as np

from .trvae import trVAE
from .losses import mse, mmd, zinb, nb
from ._utils import one_hot_encoder,euclidean_dist


class tranVAE(trVAE):
    def __init__(self,
                 input_dim: int,
                 conditions: list,
                 cell_types: list,
                 landmarks_labeled: Optional[np.ndarray] = None,
                 landmarks_unlabeled: Optional[np.ndarray] = None,
                 **trvae_kwargs,
                 ):
        super().__init__(
            input_dim,
            conditions,
            **trvae_kwargs)
        self.n_cell_types = len(cell_types)
        self.cell_types = cell_types
        self.cell_type_encoder = {k: v for k, v in zip(cell_types, range(len(cell_types)))}
        self.landmarks_labeled = landmarks_labeled
        self.landmarks_unlabeled = landmarks_unlabeled

    def classify(self, x, c=None):
        latent = self.get_latent(x,c)
        distances = euclidean_dist(latent, self.landmarks_labeled)
        _, y_pred = torch.max(-distances, dim=1)
        return y_pred

    def forward(self, x=None, batch=None, sizefactor=None, celltype=None, labeled=None):
        x_log = torch.log(1 + x)
        z1_mean, z1_log_var = self.encoder(x_log, batch)
        z1 = self.sampling(z1_mean, z1_log_var)
        outputs = self.decoder(z1, batch)

        if self.recon_loss == "mse":
            recon_x, y1 = outputs
            recon_loss = mse(recon_x, x_log).sum(dim=-1).mean()
        elif self.recon_loss == "zinb":
            dec_mean_gamma, dec_dropout, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(one_hot_encoder(batch, self.n_conditions), self.theta)
            dispersion = torch.exp(dispersion)
            recon_loss = -zinb(x=x, mu=dec_mean, theta=dispersion, pi=dec_dropout).sum(dim=-1).mean()
        elif self.recon_loss == "nb":
            dec_mean_gamma, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view
            dispersion = F.linear(one_hot_encoder(batch, self.n_conditions), self.theta)
            dispersion = torch.exp(dispersion)
            recon_loss = -nb(x=x, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()

        z1_var = torch.exp(z1_log_var) + 1e-4
        kl_div = kl_divergence(
            Normal(z1_mean, torch.sqrt(z1_var)),
            Normal(torch.zeros_like(z1_mean), torch.ones_like(z1_var))
        ).sum(dim=1).mean()

        mmd_loss = 0
        if self.use_mmd:
            mmd_calculator = mmd(self.n_conditions, self.beta, self.mmd_boundary)
            if self.mmd_on == "z":
                mmd_loss = mmd_calculator(z1, batch)
            else:
                mmd_loss = mmd_calculator(y1, batch)

        return z1, recon_loss, kl_div, mmd_loss