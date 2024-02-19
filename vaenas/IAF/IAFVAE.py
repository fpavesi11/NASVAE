import torch
from torch import nn
from cfg_nas.RecurrentTranslator import PickLast, KeepHidden
from directApproach.IAF.flows import AutoregressiveLinear, Highway, HighwayStandard
from directApproach.decoders.decoders import LinearDecoderTF, LinearDecoderV2


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dimension):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.latent_dimension = latent_dimension
        self.inner_structure = nn.Sequential(nn.LSTM(input_size=input_size,
                                                      hidden_size=256,
                                                      batch_first=True),
                                              KeepHidden(),
                                              PickLast(),
                                              nn.Linear(in_features=256,
                                                        out_features=512))
        self.estim_mu = nn.Linear(in_features=512,
                                  out_features=latent_dimension)
        
        self.estim_log_var = nn.Linear(in_features=512,
                                         out_features=latent_dimension)
        
        self.estim_h = nn.Linear(in_features=512,
                                 out_features=latent_dimension)

    def forward(self, x):
        x = self.inner_structure(x)
        mu = self.estim_mu(x)
        log_var = self.estim_log_var(x)
        h = self.estim_h(x)
        return mu, log_var, h
    

###
# IAF AS PROPOSED BY GERMAIN ET AL 2018 (MADE) with the little modification proposed in IAF
###

class IAFStandard(nn.Module):
    def __init__(self, latent_size):
        super(IAFStandard, self).__init__()

        self.latent_size = latent_size

        self.m = nn.ModuleList([
            HighwayStandard(self.latent_size),
            nn.ELU(),
            AutoregressiveLinear(self.latent_size, self.latent_size)
        ])

        self.s = nn.ModuleList([
            HighwayStandard(self.latent_size),
            nn.ELU(),
            AutoregressiveLinear(self.latent_size, self.latent_size)
        ])
    
    @staticmethod
    def passage(module, z, h):
        h_x = module[0](z, h)
        h_x = module[1](h_x)
        return module [2](h_x)

    def forward(self, z, h):
        """
        :param z: An float tensor with shape of [batch_size, z_size]
        :param h: An float tensor with shape of [batch_size, h_size]
        :return: An float tensor with shape of [batch_size, z_size] and log det value of the IAF mapping Jacobian
        """

        m = self.passage(self.m, z, h)
        s = self.passage(self.s, z, h)
        
        s = nn.Sigmoid()(s)

        z = s * z + (1 - s) * m

        log_det = s.log().sum(-1, keepdim=True)

        return z, log_det




###
# IAF AS IMPLEMENTED BY AUTHORS OF THE REPOSITORY (WITH JUST ONE LITTLE CHANGE)
###

class IAF(nn.Module):
    def __init__(self, latent_size, h_size):
        super(IAF, self).__init__()

        self.z_size = latent_size
        self.h_size = h_size

        self.h = Highway(self.h_size, 3, nn.ELU())

        self.m = nn.Sequential(
            AutoregressiveLinear(self.z_size + self.h_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size)
        )

        self.s = nn.Sequential(
            AutoregressiveLinear(self.z_size + self.h_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size),
            nn.ELU(),
            AutoregressiveLinear(self.z_size, self.z_size)
        )

    def forward(self, z, h):
        """
        :param z: An float tensor with shape of [batch_size, z_size]
        :param h: An float tensor with shape of [batch_size, h_size]
        :return: An float tensor with shape of [batch_size, z_size] and log det value of the IAF mapping Jacobian
        """

        h = self.h(h)

        input = torch.cat([z, h], 1)

        m = self.m(input)
        s = self.s(input)
        
        #ADDED BY ME
        s = nn.Sigmoid()(s)

        z = s * z + (1 - s) * m

        log_det = s.log().sum(-1, keepdim=True)

        return z, log_det


"""

IAF-VAE IMPLEMENTATION AS IN REPOSITORY

"""

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dimension, max_length, flow_depth=1, teacher_forcing=False,
                 use_predict=False):
        super(VAE, self).__init__()
        
        self.latent_dimension = latent_dimension

        self.encoder = Encoder(input_size=input_size, latent_dimension=latent_dimension)

        self.iaf = nn.ModuleList(IAF(latent_size=latent_dimension, h_size=latent_dimension) for _ in range(flow_depth))

        self.teacher_forcing = teacher_forcing
        if teacher_forcing:
            self.decoder = LinearDecoderTF(input_size=input_size,
                                           hidden_size=hidden_size,
                                           latent_dimension=latent_dimension,
                                           max_length=max_length,
                                           use_predict=use_predict)
        else:
            self.decoder = LinearDecoderV2(input_size=input_size,
                                           hidden_size=hidden_size,
                                           latent_dimension=latent_dimension,
                                           max_length=max_length)
        
    def reparametrization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z, eps
    
    def encode(self, input):
        mu, log_var, h = self.encoder(input)

        z, eps = self.reparametrization_trick(mu, log_var)

        z_T = z
        log_det = torch.zeros((input.size(0),1)).to(input.device)
        
        # IAFlow
        for iaf_layer in self.iaf:
            z_T, log_det_t = iaf_layer(z_T, h)
            log_det += log_det_t
            z_T = torch.flip(z_T, dims=[-1]) #reverse order of z
            
        return z_T


    def forward(self, input, x_true=None):
        
        mu, log_var, h = self.encoder(input)

        z, eps = self.reparametrization_trick(mu, log_var)

        z_T = z
        log_det = torch.zeros((input.size(0),1)).to(input.device)
        
        # IAFlow
        for iaf_layer in self.iaf:
            z_T = torch.flip(z_T, dims=[-1]) #reverse order of z
            z_T, log_det_t = iaf_layer(z_T, h)
            log_det += log_det_t

        if self.teacher_forcing:
            x_hat = self.decoder(z_T, x_true)
        else:
            x_hat = self.decoder(z_T)
        
        return x_hat, z_T, eps, mu, log_var, log_det

    def generate(self, n_instances):
        return self.decoder(torch.randn((n_instances, self.latent_dimension)))
    
    
    
###
# IAF-VAE IMPLEMENTATION FOLLOWING GERMAIN (MADE, 2018) AND KINGMA (IAF, 2019)
###

class VAEStandard(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dimension, max_length, flow_depth=1, teacher_forcing=False,
                 use_predict=False):
        super(VAEStandard, self).__init__()
        
        self.latent_dimension = latent_dimension

        self.encoder = Encoder(input_size=input_size, latent_dimension=latent_dimension)

        self.iaf = nn.ModuleList(IAFStandard(latent_size=latent_dimension) for _ in range(flow_depth))

        self.teacher_forcing = teacher_forcing
        if teacher_forcing:
            self.decoder = LinearDecoderTF(input_size=input_size,
                                           hidden_size=hidden_size,
                                           latent_dimension=latent_dimension,
                                           max_length=max_length,
                                           use_predict=use_predict)
        else:
            self.decoder = LinearDecoderV2(input_size=input_size,
                                           hidden_size=hidden_size,
                                           latent_dimension=latent_dimension,
                                           max_length=max_length)
        
    def reparametrization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z, eps
    
    def encode(self, input):
        mu, log_var, h = self.encoder(input)

        z, eps = self.reparametrization_trick(mu, log_var)

        z_T = z
        log_det = torch.zeros((input.size(0),1)).to(input.device)
        
        # IAFlow
        for iaf_layer in self.iaf:
            z_T, log_det_t = iaf_layer(z_T, h)
            log_det += log_det_t
            z_T = torch.flip(z_T, dims=[-1]) #reverse order of z
            
        return z_T


    def forward(self, input, x_true=None):
        
        mu, log_var, h = self.encoder(input)

        z, eps = self.reparametrization_trick(mu, log_var)

        z_T = z
        log_det = torch.zeros((input.size(0),1)).to(input.device)
        
        # IAFlow
        for iaf_layer in self.iaf:
            z_T = torch.flip(z_T, dims=[-1]) #reverse order of z
            z_T, log_det_t = iaf_layer(z_T, h)
            log_det += log_det_t

        if self.teacher_forcing:
            x_hat = self.decoder(z_T, x_true)
        else:
            x_hat = self.decoder(z_T)
        
        return x_hat, z_T, eps, mu, log_var, log_det

    def generate(self, n_instances):
        return self.decoder(torch.randn((n_instances, self.latent_dimension)))