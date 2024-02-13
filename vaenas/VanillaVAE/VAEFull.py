import torch
from torch import nn 
from torch.nn.modules.loss import _Loss
from cfg_nas.RecurrentTranslator import KeepHidden, PickLast
from torch import nn
from vaenas.decoders.decoders import LinearDecoderV2, LinearDecoderTF


class reshape_to_matrix(nn.Module):
    def __init__(self):
        super(reshape_to_matrix, self).__init__()

    def forward(self, x):
        mat_dim = int(torch.sqrt(torch.tensor(x.size(1))))
        return x.view(x.size(0), mat_dim, mat_dim)


class mask_out(nn.Module):
    def __init__(self):
        super(mask_out, self).__init__()

    def forward(self, x):
        mask = torch.tril(torch.ones(x.size(1), x.size(2)), diagonal=-1)
        return x * mask


class EncoderFull(nn.Module):
    def __init__(self, input_size, latent_dimension):
        super(EncoderFull, self).__init__()
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
                                  out_features = latent_dimension)

        self.estim_log_sigma = nn.Linear(in_features=512,
                                         out_features = latent_dimension)

        self.estim_L_mask = nn.Sequential(nn.Linear(in_features=512,
                                               out_features=latent_dimension**2),
                                     reshape_to_matrix(),
                                     mask_out())

    def forward(self, x):
        x = self.inner_structure(x)
        mu = self.estim_mu(x)
        log_sigma = self.estim_log_sigma(x)
        L_mask = self.estim_L_mask(x)
        return mu, log_sigma, L_mask


class ReparametrizationTrickFull(nn.Module):
    def __init__(self):
        super(ReparametrizationTrickFull, self).__init__()

    def forward(self, mu, log_sigma, L_mask):
        L_diag = L_mask + torch.diag_embed(torch.exp(log_sigma)) #notice here is sigma, while in standard vae is sigma2
        z = mu + torch.bmm(L_diag, torch.randn_like(mu).unsqueeze(-1)).squeeze(-1)
        return z, L_diag


class LinearVAEFull(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dimension, max_length, teacher_forcing=False,
                 use_predict=False):
        super(LinearVAEFull, self).__init__()
        self.teacher_forcing = teacher_forcing
        self.encoder = EncoderFull(input_size=input_size,
                                    latent_dimension=latent_dimension)
        self.reparametrization = ReparametrizationTrickFull()
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
        mu, log_sigma, L_mask = self.encoder(x)
        z, L_diag = self.reparametrization(mu, log_sigma, L_mask)
        if self.teacher_forcing:
            out = self.decoder(z, x_true)
        else:
            out = self.decoder(z)
        return out, mu, L_diag

    def latent_encoding(self, x):
        mu, log_sigma, L_mask = self.encoder(x)
        z, L_diag = self.reparametrization(mu, log_sigma, L_mask)
        return z, L_diag

    def generate_network(self, sample):
        return self.decoder(sample)
