import torch
from torch import nn
from cfg_nas.RecurrentTranslator import PickLast, KeepHidden
from vaenas.decoders.decoders import LinearDecoderV2, LinearDecoderTF

class Encoder(nn.Module):
    def __init__(self, input_size, latent_dimension):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.latent_dimension = latent_dimension
        self.inner_structure = nn.Sequential(nn.LSTM(input_size=input_size,
                                                      hidden_size=256,
                                                      num_layers=2,
                                                      batch_first=True),
                                              KeepHidden(),
                                              PickLast(),
                                              nn.Linear(in_features=256,
                                                        out_features=512),
                                              nn.LeakyReLU(0.2))
        self.estim_mu = nn.Sequential(nn.Linear(in_features=512,
                                                out_features = 512),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(in_features=512,
                                                out_features = latent_dimension))
        
        self.estim_log_sigma = nn.Sequential(nn.Linear(in_features=512,
                                                out_features = 512),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(in_features=512,
                                                out_features = latent_dimension))

    def forward(self, x):
        x = self.inner_structure(x)
        mu = self.estim_mu(x)
        log_sigma = self.estim_log_sigma(x)
        return mu, log_sigma
    
class ReparametrizationTrick(nn.Module):
    def __init__(self):
        super(ReparametrizationTrick, self).__init__()
    
    def forward(self, mu, log_sigma):
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_sigma)
        return z
    
    
class LinearVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dimension, max_length, teacher_forcing=False,
                 use_predict=False):
        super(LinearVAE, self).__init__()
        self.teacher_forcing = teacher_forcing
        self.encoder = Encoder(input_size=input_size,
                               latent_dimension=latent_dimension)
        self.reparametrization = ReparametrizationTrick()
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
        
    def forward(self, x, x_true=None):
        mu, log_sigma = self.encoder(x)
        z = self.reparametrization(mu, log_sigma)
        if self.teacher_forcing:
            out = self.decoder(z, x_true)
        else:
            out = self.decoder(z)
        return out, mu, log_sigma
    
    def latent_encoding(self, x):
        mu, log_sigma = self.encoder(x)
        z = self.reparametrization(mu, log_sigma)
        return z, mu, log_sigma
    
    def generate_network(self, sample):
        return self.decoder(sample)