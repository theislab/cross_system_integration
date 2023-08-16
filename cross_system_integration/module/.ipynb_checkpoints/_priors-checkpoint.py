import abc
from abc import abstractmethod
from typing import Tuple, Optional

import torch
from scvi.module.base import auto_move_data
from torch.distributions import Normal
from torch.distributions import kl_divergence


class Prior(torch.nn.Module, abc.ABC):

    @abstractmethod
    def kl(self, m_q, v_q, z):
        pass


class StandardPrior(Prior):

    def kl(self, m_q, v_q, z=None):
        # 1 x N
        return kl_divergence(Normal(m_q, v_q.sqrt()), Normal(torch.zeros_like(m_q), torch.ones_like(v_q))).sum(dim=1)


class VampPrior(Prior):
    # Adapted from https://github.com/jmtomczak/intro_dgm/main/vaes/vae_priors_example.ipynb
    # K - components, I - inputs, L - latent, N - samples

    def __init__(self, n_components, n_input, n_cov, encoder,
                 data: Optional[Tuple[torch.tensor, torch.tensor]] = None,
                 trainable_priors=True,
                 ):
        super(VampPrior, self).__init__()

        self.encoder = encoder

        # pseudoinputs
        if data is None:
            u = torch.rand(n_components, n_input)  # K * I
            u_cov = torch.zeros(n_components, n_cov)  # K * C
        else:
            u = data[0]
            u_cov = data[1]
        self.u = torch.nn.Parameter(u, requires_grad=trainable_priors)
        self.u_cov = torch.nn.Parameter(u_cov, requires_grad=trainable_priors)

        # mixing weights
        self.w = torch.nn.Parameter(torch.zeros(self.u.shape[0], 1, 1))  # K x 1 x 1

    def get_params(self):
        # u->encoder->mean, var
        z = self.encoder(x=self.u, cov=self.u_cov)
        return z['y_m'], z['y_v']  # (K x L), (K x L)

    def log_prob(self, z):
        # Mixture of gaussian computed on K x N x L
        z = z.unsqueeze(0)  # 1 x N x L

        # u->encoder->mean, var
        m_p, v_p = self.get_params()  # (K x L), (K x L)
        m_p = m_p.unsqueeze(1)  # K x 1 x L
        v_p = v_p.unsqueeze(1)  # K x 1 x L

        # mixing probabilities
        w = torch.nn.functional.softmax(self.w, dim=0)  # K x 1 x 1

        # sum of log_p across components weighted by w
        log_prob = Normal(m_p, v_p.sqrt()).log_prob(z) + torch.log(w)  # K x N x L
        log_prob = torch.logsumexp(log_prob, dim=0, keepdim=False)  # N x L

        return log_prob  # N x L

    def kl(self, m_q, v_q, z):
        return (Normal(m_q, v_q.sqrt()).log_prob(z) - self.log_prob(z)).sum(1)
