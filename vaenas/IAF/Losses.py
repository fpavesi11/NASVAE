import torch
from torch import nn 
from torch.nn.modules.loss import _Loss
from math import pi
from vaenas.Utils import CustomReconLosslinear

class VAELoss(_Loss):
    def __init__(self, reduction='sum', cat_loss_weight=0.1, alpha=None, norm_constant=None):
        """_summary_

        Args:
            reduction (str, optional): Reduction of loss. Defaults to 'sum'.
            alpha (_type_, optional): If None, vanilla VAE is applied, if < 1, Bowman method
            of slow increasing towards 1 is applied, if > 1, Beta VAE is applied. Defaults to None.
        """
        super(VAELoss, self).__init__()
        self.cat_loss_weight = cat_loss_weight
        self.reduction = reduction
        self.alpha = alpha
        self.norm_constant = norm_constant
        self.method = 'vanilla'
        if alpha is not None:
            if alpha < 1:
                self.kl_weight = 0
                self.method = 'Bowman'
            else:
                self.kl_weight = alpha
                if self.norm_constant is not None:
                    self.kl_weight *= norm_constant
                self.method = 'Beta'
        else:
            self.kl_weight = 1
            
    @staticmethod
    def log_gauss(z, mu, std):
        return - 0.5 * (torch.pow(z - mu, 2) * torch.pow(std + 1e-8, -2) + 2 * torch.log(std + 1e-8) + torch.log(2 * torch.tensor(pi))).sum(1)
            
    def forward(self, y, y_true, z, eps, mu, log_var, log_det):
        # RECONSTRUCTION LOSS
        mseloss = CustomReconLosslinear(reduction='none')
        features_loss, category_loss = mseloss(y, y_true)
        reconstructon_loss = features_loss + self.cat_loss_weight * category_loss
        
        # KL DIVERGENCE POSTERIOR || PRIOR (I used this derivation, it should be correct)
        # NB the usage of 1 + etc ensures we account for k (sum(1) k times = k)
        #KL_div = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1, keepdim=True)
        # NB: recall to activate -log_det in overall loss
        
        # MC DENSITY DIFFERENCE APPROACH (IAF Paper)
        log_q_z = -0.5 * (log_var + eps.pow(2) + torch.log(2 * torch.tensor(pi))).sum(-1, keepdim=True) - log_det
        log_p_z = -0.5 * (z.pow(2) + torch.log(2 * torch.tensor(pi))).sum(-1, keepdim=True)
        KL_div = log_q_z - log_p_z

        # Bowman method
        if self.kl_weight < 1 and self.method=='Bowman':
            self.kl_weight += self.alpha
            if self.kl_weight > 1:
                self.kl_weight = 1
        
        overall_loss = reconstructon_loss + self.kl_weight * KL_div
        
        if self.reduction == 'sum':
            overall_loss = overall_loss.sum()
            reconstructon_loss = reconstructon_loss.sum()
            KL_div = KL_div.sum()
            log_det = log_det.sum()
        elif self.reduction == 'mean':
            overall_loss = overall_loss.mean()
            reconstructon_loss = reconstructon_loss.mean()
            KL_div = KL_div.mean()
            log_det = log_det.mean()
        else:
            raise ValueError(f'Illegal reduction {self.reduction}')
                    
        return overall_loss, reconstructon_loss, KL_div, log_det, features_loss, category_loss
