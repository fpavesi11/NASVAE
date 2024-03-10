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
            
    def forward(self, y, y_true, z, eps, mu, log_var, log_det):
        # RECONSTRUCTION LOSS
        mseloss = CustomReconLosslinear(reduction='none')
        features_loss, category_loss = mseloss(y, y_true)
        reconstructon_loss = features_loss + self.cat_loss_weight * category_loss
        
        # KL DIVERGENCE POSTERIOR || PRIOR
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
    
    

"""
SECOND VERSION OF VAE LOSS WITH FREE BITS
"""

class VAELossV2(_Loss):
    def __init__(self, cat_loss_weight=1, feat_loss_weight=1, reduction='sum', 
                 alpha=None, norm_constant=None, free_bits=None):
        """_summary_

        Args:
            reduction (str, optional): Reduction of loss. Defaults to 'sum'.
            alpha (_type_, optional): If None, vanilla VAE is applied, if < 1, Bowman method
            of slow increasing towards 1 is applied, if > 1, Beta VAE is applied. Defaults to None.
        """
        super().__init__()
        self.cat_loss_weight = cat_loss_weight
        self.feat_loss_weight = feat_loss_weight
    
        # Notice: free bits method makes cluster of one dimension only
        self.free_bits = free_bits
        
        self.reduction = reduction
        self.alpha = alpha
        self.norm_constant = norm_constant
        if alpha is not None:
            self.kl_weight_frac = 0
        else:
            self.kl_weight_frac = 1
            
        if norm_constant is not None:
            self.kl_weight = norm_constant
        else:
            self.kl_weight = 1
            
    def forward(self, y, y_true, z, eps, mu, log_var, log_det):
        # RECONSTRUCTION LOSS
        mseloss = CustomReconLosslinear(reduction='none')
        features_loss, category_loss = mseloss(y, y_true)
        reconstruction_loss = self.feat_loss_weight * features_loss + self.cat_loss_weight * category_loss
        
        # KL DIVERGENCE POSTERIOR || PRIOR
        log_q_z = -0.5 * (log_var + eps.pow(2) + torch.log(2 * torch.tensor(pi))).sum(-1, keepdim=True) - log_det
        log_p_z = -0.5 * (z.pow(2) + torch.log(2 * torch.tensor(pi))).sum(-1, keepdim=True)
        KL_div = log_q_z - log_p_z
        
        if self.kl_weight_frac < 1 and self.alpha is not None:
            self.kl_weight_frac += self.alpha
            if self.kl_weight_frac > 1:
                self.kl_weight_frac = 1
        passage_kl_weight = self.kl_weight_frac * self.kl_weight
        
        # Free bits
        if self.free_bits is not None:
            KL_div = torch.max(KL_div, torch.tensor(self.free_bits))
        
        overall_loss = reconstruction_loss + passage_kl_weight * KL_div
        
        if self.reduction == 'sum':
            overall_loss = overall_loss.sum()
            reconstruction_loss = reconstruction_loss.sum()
            KL_div = KL_div.sum()
            log_det = log_det.sum()
        elif self.reduction == 'mean':
            overall_loss = overall_loss.mean()
            reconstruction_loss = reconstruction_loss.mean()
            KL_div = KL_div.mean()
            log_det = log_det.mean()
        else:
            raise ValueError(f'Illegal reduction {self.reduction}')
                    
        return overall_loss, reconstruction_loss, KL_div, log_det, features_loss, category_loss
    

