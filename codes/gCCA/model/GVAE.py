import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
from torch.utils.data import Dataset

# Variational Autoencoder for Genomaps (gVAE)
# Current model only supports 40 by 40 genomaps, as the convolution layer size is hardcoded.
# We are actively working on a more flexible model that can handle different image sizes.
class gVAE(L.LightningModule):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        in_channels = 1 # genomap is a single channel image
        self.kld_weight = 0.001
        self.IMAGE_SIZE = 5 # after the convolution, the image size is 5*5
        modules = []
        hidden_dims = [8, 16, 32]
        
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.last_hidden_dim = hidden_dims[-1]
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] *  self.IMAGE_SIZE *  self.IMAGE_SIZE, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] *  self.IMAGE_SIZE *  self.IMAGE_SIZE, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] *  self.IMAGE_SIZE *  self.IMAGE_SIZE)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.Conv2d(hidden_dims[-1], out_channels= 1,
                                      kernel_size= 3, padding= 1),
                            )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.last_hidden_dim,  self.IMAGE_SIZE,  self.IMAGE_SIZE)
        h = self.decoder(h)
        x_hat = self.final_layer(h)
        return x_hat
    
    def encode(self, img):
        h = self.encoder(img)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


    def training_step(self, imgs):
        # training_step defines the train loop.
        h = self.encoder(imgs)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        recons_loss = nn.MSELoss()(x_hat, imgs)       
        kld_loss = self.kld_weight * torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss =  recons_loss +  kld_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("recon", recons_loss, prog_bar=True)
        self.log("kld", kld_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    def forward(self, imgs):
        # training_step defines the train loop.
        h = self.encoder(imgs)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, z
    
    def decode_z(self, z):
        x_hat = nn.ReLU()(self.decode(z))
        return x_hat 
    
class genoMapDataset(Dataset):
    def __init__(self, img):
        self.img = img

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        # Select sample
        image = self.img[index]
        X = torch.tensor(image).double()
        return X
