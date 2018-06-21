import torch
import torch.nn as nn


class VAELoss(nn.Module):
    """
            This criterion is an implementation of VAELoss
    """

    def __init__(self, size_average=False):
        super(VAELoss, self).__init__()
        self.size_average = size_average

    def forward(self, recon_x, x, mu, logvar):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        bsz = x.shape[0]
        reconst_err = (x - recon_x).pow(2).reshape(bsz, -1)
        reconst_err = 0.5 * torch.sum(reconst_err, dim=-1)

        # KL(q || p) = -log_sigma + sigma^2/2 + mu^2/2 - 1/2
        KL = (-logvar + logvar.exp() + mu.pow(2) - 1) * 0.5
        KL = torch.sum(KL, dim=-1)
        # loss = reconst_err + KL

        if self.size_average:
            reconst_err = torch.mean(reconst_err)
            KL = torch.mean(KL)
            # loss = torch.mean(loss)
        else:
            reconst_err = torch.sum(reconst_err)
            KL = torch.sum(KL)
            # loss = torch.sum(loss)

        return reconst_err, KL