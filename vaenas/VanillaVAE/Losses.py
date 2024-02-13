import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from vaenas.Utils import CustomReconLosslinear

"""
Vanilla VAE diagonal posterior loss
"""
class VAELoss(_Loss):
    def __init__(self, cat_loss_weight=1, features_loss_weight=1, reduction='sum', alpha=None, norm_constant=None):
        """_summary_

        Args:
            reduction (str, optional): Reduction of loss. Defaults to 'sum'.
            alpha (_type_, optional): If None, vanilla VAE is applied, if < 1, Bowman method
            of slow increasing towards 1 is applied, if > 1, Beta VAE is applied. Defaults to None.
        """
        super(VAELoss, self).__init__()
        self.cat_loss_weight = cat_loss_weight
        self.features_loss_weight = features_loss_weight
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
            
    def forward(self, y, y_true, mu, log_sigma):
        # RECONSTRUCTION LOSS
        mseloss = CustomReconLosslinear(reduction='none')
        features_loss, category_loss = mseloss(y, y_true)
        reconstructon_loss = (self.features_loss_weight * features_loss 
                              + self.cat_loss_weight * category_loss).sum()
        
        # KL DIVERGENCE POSTERIOR || PRIOR
        KL_div = -0.5 * (1 + log_sigma - mu ** 2 - log_sigma.exp()).sum()
        
        # Bowman method
        if self.kl_weight < 1 and self.method=='Bowman':
            self.kl_weight += self.alpha
            if self.kl_weight > 1:
                self.kl_weight = 1
                
                    
        return reconstructon_loss + KL_div * self.kl_weight, reconstructon_loss, KL_div, self.kl_weight
    
    
class VAELossFull(_Loss):
    def __init__(self, reduction='sum', cat_loss_weight=1, features_loss_weight=1, alpha=None, add_noise=False, norm_constant=None):
        """_summary_

        Args:
            reduction (str, optional): Reduction of loss. Defaults to 'sum'.
            alpha (_type_, optional): If None, vanilla VAE is applied, if < 1, Bowman method
            of slow increasing towards 1 is applied, if > 1, Beta VAE is applied. Defaults to None.
            add_noise (bool, optional): Adds random noise to L diagonal. Defaults to False.
        """
        super(VAELossFull, self).__init__()
        self.cat_loss_weight=cat_loss_weight
        self.features_loss_weight=features_loss_weight
        self.reduction = reduction
        self.add_noise = add_noise
        self.alpha = alpha
        self.norm_constant = norm_constant
        if alpha is not None:
            if alpha < 1:
                self.kl_weight = 0
                self.method = 'Bowman'
            else:
                self.kl_weight = alpha
                if norm_constant is not None:
                    self.kl_weight *= norm_constant
                self.method = 'Beta'
        else:
            self.kl_weight = 1
    
    def forward(self, y, y_true, mu, L_diag):
        # RECONSTRUCTION LOSS
        mseloss = CustomReconLosslinear(reduction='none')
        features_loss, category_loss = mseloss(y, y_true)
        reconstructon_loss = (self.features_loss_weight * features_loss 
                              + self.cat_loss_weight * category_loss).sum()
        
        # KL DIVERGENCE POSTERIOR || PRIOR
        #sigma = torch.bmm(L_diag, L_diag.transpose(1,2))
        
        # Adds noise if needed        
        if self.add_noise:
            noise_scale = 1e-6
            noise = torch.randn_like(mu) * noise_scale
            L_diag += torch.diag_embed(noise)
            
        muTmu = torch.sum(mu  * mu, dim=1, keepdim=True) #simplifies matmul
        #tr_sigma = sigma.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).unsqueeze(-1)
        tr_sigma = L_diag.pow(2).sum(dim=-1).sum(dim=-1).unsqueeze(-1)
        #tr_sigma =  sigma.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1)
        k = mu.size(1)
        # calculate determinant with the advantage of cholesky decomposition
        #log_det_sigma = L_diag.diagonal(offset=0, dim1=-1, dim2=-2).pow(2).prod(-1).log().unsqueeze(-1) 
        #log_det_sigma = 2 * L_diag.diagonal(offset=0, dim1=-1, dim2=-2).log().sum(-1).unsqueeze(-1)
        log_det_sigma = 2 * torch.log(L_diag.diagonal(dim1=-1, dim2=-2)).sum(-1).unsqueeze(-1)
        KL_div = 0.5*(muTmu + tr_sigma - k - log_det_sigma).sum()
        
        # Bowman method
        if self.kl_weight < 1 and self.method=='Bowman':
            self.kl_weight += self.alpha
            if self.kl_weight > 1:
                self.kl_weight = 1
                    
        return reconstructon_loss + KL_div * self.kl_weight, reconstructon_loss, KL_div, self.kl_weight
