import torch
from torch import nn
from cfg_nas.RecurrentTranslator import PickLast, KeepHidden
from vaenas.flowVAE.flows import Flow
from vaenas.decoders.decoders import LinearDecoderTF, LinearDecoderV2


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
    

class VAE(nn.Module):
    def __init__(self, in_dim, hidden_size, latent_dim, max_length, flow, flow_length, teacher_forcing=False,
                 use_predict=False):
   
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(in_dim, latent_dim)
        self.flow = Flow(latent_dim, flow, flow_length)
        self.teacher_forcing = teacher_forcing
        if teacher_forcing:
            self.decoder = LinearDecoderTF(input_size=in_dim,
                                        hidden_size=hidden_size,
                                        latent_dimension=latent_dim,
                                        max_length=max_length,
                                        use_predict=use_predict)
        else:
            self.decoder = LinearDecoderV2(input_size=in_dim,
                                        hidden_size=hidden_size,
                                        latent_dimension=latent_dim,
                                        max_length=max_length)
    def encode(self, x):
        """Encodes input.

        Args:
            x: input tensor (B x D).
        Returns:
            mean and log-variance of the gaussian approximate posterior.
        """
        mu, log_var = self.encoder(x)
        return mu, log_var

    def transform(self, mu, log_var):
        """Transforms approximate posterior.

        Args:
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
        Returns:
            transformed latent codes and the log-determinant of the Jacobian.
        """
        std = torch.exp(.5 * log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        return self.flow(z), eps

    def decode(self, z, x_true=None):
        """Decodes latent codes.

        Args:
            z: latent codes.
        Returns:
            reconstructed input.
        """
        if self.teacher_forcing:
            out = self.decoder(z, x_true)
        else:
            out = self.decoder(z)
        return out

    def sample(self, size):
        """Generates samples from the prior.

        Args:
            size: number of samples to generate.
        Returns:
            generated samples.
        """
        z = torch.randn(size, self.latent_dim)
        return self.decode(z) 

    def forward(self, x, x_true=None):
        """Forward pass.

        Args:
            x: input tensor (B x D).
        Returns:
            average loss over the minibatch.
        """
        mu, log_var = self.encode(x)
        (z, log_det), eps = self.transform(mu, log_var)
        x_hat = self.decode(z, x_true)
        
        return x_hat, z, eps, mu, log_var, log_det