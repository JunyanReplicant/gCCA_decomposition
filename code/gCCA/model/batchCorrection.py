#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:45:56 2024

@author: jun
"""

import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
from torch.utils.data import Dataset

class AE(L.LightningModule):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.IMAGE_SIZE = 5
        in_channels = 1
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
        self.fc = nn.Linear(hidden_dims[-1] *  self.IMAGE_SIZE *  self.IMAGE_SIZE, latent_dim)

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



    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.last_hidden_dim,  self.IMAGE_SIZE,  self.IMAGE_SIZE)
        h = self.decoder(h)
        x_hat = self.final_layer(h)
        return x_hat
    
    def encode(self, img):
        h = self.encoder(img)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc(h)
        return mu


    def training_step(self, img):
        # training_step defines the train loop.
        h = self.encoder(img)
        h = torch.flatten(h, start_dim=1)
        latent = self.fc(h)
        x_hat = self.decode(latent)
        recons_loss = nn.MSELoss()(x_hat, img)       
        loss =  recons_loss 
        self.log("train_loss", loss, prog_bar=True)
        self.log("recon", recons_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def sample(self,
               num_samples
               , **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim).double()
        samples = self.decode(z)
        return samples.detach()


    def forward(self, img):
        # training_step defines the train loop.
        h = self.encoder(img)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc(h)
        x_hat = self.decode(mu)
        return x_hat, mu
    
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
        X = torch.tensor(image)
        return X


class AE2(L.LightningModule):
    def __init__(self, latent_dim, decoder):
        super().__init__()
        self.latent_dim = latent_dim
        in_channels = 1
        self.IMAGE_SIZE = 5
        modules = []
        hidden_dims = [8, 16, 4]
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
        self.fc = nn.Linear(hidden_dims[-1] *  self.IMAGE_SIZE *  self.IMAGE_SIZE, latent_dim)
        self.decoder = decoder
    
    def encode(self, img):
        h = self.encoder(img)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc(h)
        return mu


    def training_step(self, batch):
        img, real = batch
        # training_step defines the train loop.
        h = self.encoder(img)
        h = torch.flatten(h, start_dim=1)
        latent = self.fc(h)
        x_hat = self.decoder(latent)
        loss = nn.MSELoss()(x_hat, real)  
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def validation_step(self, batch):
        img, real = batch
        # training_step defines the train loop.
        h = self.encoder(img)
        h = torch.flatten(h, start_dim=1)
        latent = self.fc(h)
        x_hat = self.decoder(latent)
        loss = nn.MSELoss()(x_hat, real)       
        self.log("validation_loss", loss, prog_bar=True)


        return loss

    def forward(self, img):
        # training_step defines the train loop.
        h = self.encoder(img)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc(h)
        x_hat = self.decoder(mu)
        return x_hat, mu
    
    def decode_z(self, z):
        x_hat = nn.ReLU()(self.decoder(z))
        return x_hat 
    
class AEDataset(Dataset):
    def __init__(self, img, label):
        self.img = img
        self.label = label
    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        # Select sample
        image = self.img[index]
        X = torch.tensor(image)
        Y = self.label[index]
        return X,Y
