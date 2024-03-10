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


"""
SECOND VERSION OF VAE LOSS WITH FREE BITS
"""


class VAELossV2(_Loss):
    def __init__(self, cat_loss_weight=1, features_loss_weight=1, reduction='sum', 
                 alpha=None, norm_constant=None, free_bits=None):
        """_summary_

        Args:
            reduction (str, optional): Reduction of loss. Defaults to 'sum'.
            alpha (_type_, optional): If None, vanilla VAE is applied, if < 1, Bowman method
            of slow increasing towards 1 is applied, if > 1, Beta VAE is applied. Defaults to None.
        """
        super(VAELossV2, self).__init__()
        self.cat_loss_weight = cat_loss_weight
        self.features_loss_weight = features_loss_weight
        
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
            
    def forward(self, y, y_true, mu, log_sigma):
        # RECONSTRUCTION LOSS
        mseloss = CustomReconLosslinear(reduction='none')
        features_loss, category_loss = mseloss(y, y_true)
        reconstruction_loss = (self.features_loss_weight * features_loss 
                              + self.cat_loss_weight * category_loss)
        
        # KL DIVERGENCE POSTERIOR || PRIOR
        # KLDIV = 1/2 (muT mu + tr(sigma) - k - log_det(sigma))
        # KLDIV = -1/2 (k + log_det(sigma) - muT mu - tr(sigma))
        KL_div = -0.5 * (mu.size(-1) + log_sigma.sum(-1, keepdim=True) - mu.pow(2).sum(-1, keepdim=True) 
                         - log_sigma.exp().sum(-1, keepdim=True))    
        
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
        elif self.reduction == 'mean':
            overall_loss = overall_loss.mean()
            reconstruction_loss = reconstruction_loss.mean()
            KL_div = KL_div.mean()
        else:
            raise ValueError(f'Illegal reduction {self.reduction}')  
                    
        return overall_loss, reconstruction_loss, KL_div, passage_kl_weight
    
    
"""
SECOND VERSION OF VAE LOSS FULL WITH FREE BITS
"""

class VAELossFullV2(_Loss):
    def __init__(self, reduction='sum', cat_loss_weight=1, features_loss_weight=1, 
                 alpha=None, norm_constant=None, free_bits=None):
        """_summary_

        Args:
            reduction (str, optional): Reduction of loss. Defaults to 'sum'.
            alpha (_type_, optional): If None, vanilla VAE is applied, if < 1, Bowman method
            of slow increasing towards 1 is applied, if > 1, Beta VAE is applied. Defaults to None.
            add_noise (bool, optional): Adds random noise to L diagonal. Defaults to False.
        """
        super(VAELossFullV2, self).__init__()
        self.cat_loss_weight=cat_loss_weight
        self.features_loss_weight=features_loss_weight
        self.reduction = reduction
        self.alpha = alpha
        self.norm_constant = norm_constant
        self.free_bits = free_bits
        if alpha is not None:
            self.kl_weight_frac = 0
        else:
            self.kl_weight_frac = 1
            
        if norm_constant is not None:
            self.kl_weight = norm_constant
        else:
            self.kl_weight = 1
    
    def forward(self, y, y_true, mu, L_diag):
        # RECONSTRUCTION LOSS
        mseloss = CustomReconLosslinear(reduction='none')
        features_loss, category_loss = mseloss(y, y_true)
        reconstruction_loss = (self.features_loss_weight * features_loss 
                              + self.cat_loss_weight * category_loss)
            
        muTmu = torch.sum(mu  * mu, dim=1, keepdim=True) #simplifies matmul
        #tr_sigma = sigma.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).unsqueeze(-1)
        tr_sigma = L_diag.pow(2).sum(dim=-1).sum(dim=-1).unsqueeze(-1)
        #tr_sigma =  sigma.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1)
        k = mu.size(1)
        # calculate determinant with the advantage of cholesky decomposition
        #log_det_sigma = L_diag.diagonal(offset=0, dim1=-1, dim2=-2).pow(2).prod(-1).log().unsqueeze(-1) 
        #log_det_sigma = 2 * L_diag.diagonal(offset=0, dim1=-1, dim2=-2).log().sum(-1).unsqueeze(-1)
        log_det_sigma = 2 * torch.log(L_diag.diagonal(dim1=-1, dim2=-2)).sum(-1).unsqueeze(-1)
        KL_div = 0.5*(muTmu + tr_sigma - k - log_det_sigma)
        
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
        elif self.reduction == 'mean':
            overall_loss = overall_loss.mean()
            reconstruction_loss = reconstruction_loss.mean()
            KL_div = KL_div.mean()
        else:
            raise ValueError(f'Illegal reduction {self.reduction}') 
                    
        return overall_loss, reconstruction_loss, KL_div, passage_kl_weight
