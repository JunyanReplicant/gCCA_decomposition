# -*- coding: utf-8 -*-
import torch
from torch import nn
import lightning as L
from torch.utils.data import Dataset

class Decompose(L.LightningModule):
    def __init__(self, celltypes , g_mean, g_std, vae):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(celltypes): 
            self.convs.append(nn.Conv2d(in_channels = 1, out_channels= 1, kernel_size= 3, stride= 1, bias = False))
        self.avgpool = nn.AvgPool2d(kernel_size= 3, stride= 1)
        self.celltypes = celltypes
        self.g_mean = g_mean.cuda()
        self.g_std = g_std.cuda()

        self.gaussians = []    
        for i in range(celltypes):
            self.gaussians.append(torch.distributions.multivariate_normal.MultivariateNormal(
                loc = self.g_mean[i], 
                covariance_matrix=torch.diag(self.g_std[i])
                )
            )
            
        self.latentVars= nn.Parameter(torch.tensor(self.g_mean[:,:]), requires_grad=True) # celltype by latent space
        self.vae = vae

        # Create a random patched mask for decomposition
        # This is to make uncertainty prediction
        mask = torch.zeros((40, 40))
        
        patch_size = 10
        num_patches_h = 40 // patch_size
        num_patches_w = 40 // patch_size
        total_patches = num_patches_h * num_patches_w
        # Randomly select patches
        selected_patches = torch.randperm(total_patches)[:8]
        
        for patch_idx in selected_patches:
            # Convert patch index to 2D coordinates
            patch_h = (patch_idx // num_patches_w) * patch_size
            patch_w = (patch_idx % num_patches_w) * patch_size
            
            # Set patch region to 1
            h_end = min(patch_h + patch_size, 40)
            w_end = min(patch_w + patch_size, 40)
            mask[patch_h:h_end, patch_w:w_end] = 1
        self.mask = mask.view(1, 1, 40, 40).cuda()


    def training_step(self, img):
        sigMats_conv  = torch.zeros((self.celltypes, 1, 38, 38))
        sigMats = torch.zeros((self.celltypes,1,40,40)).cuda()
        #Generate signatures back from latent space with VAE decoder
        for i in range(self.celltypes):
            sigMats[i,0,:,:] = self.vae[i].decode_z(self.latentVars[i].cuda())  #self.sigMats 
        #Compute the GMM probability of latent variables
        prob =  torch.ones(1).cuda()
        for i in range(self.celltypes):
            prob += - self.gaussians[i].log_prob(self.latentVars[i].cuda())/10
        # Compute the signature after convolution
        for i in range(self.celltypes):
            sigMats_conv[i:i+1,:,:,:] = nn.ReLU() (self.convs[i](sigMats[i:i+1].double()*self.mask)).cuda() 
        # Compute the reconstructed image
        recon = torch.sum(sigMats_conv, dim = 0 ,keepdim=True).double().cuda()
        img_conv = self.avgpool(img*self.mask).cuda()
        
        loss_MSE =  nn.MSELoss()(recon, img_conv)
        loss_Guassian = prob.cuda()
        # Final loss is the sum of the two losses
        loss = loss_MSE + 0.01 * loss_Guassian 
        self.log("recon", loss_MSE, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                if name == 'latentVars':
                    # Latent variables should be within a standard deviations of the mean, 
                    # to prevent the latent variables from being too far away to capture the outliers
                    param.data = param.data.clamp(min = self.g_mean.cuda() - 3.0 * self.g_std.cuda(), 
                                                  max = self.g_mean.cuda() + 3.0 * self.g_std.cuda())
                else:
                    # Cell proportion should be positive
                    param.data = torch.abs(param.data)

###### We reuse the dataset class from the previous gVAE code, so this is not needed.
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


        #######################################################################
        ###### Alternatve loss function with cosine similarity. Tested works but Performed not as good as MSELoss.
        ###### This is not used in the final model. But readers can try it out.
        # img1_flat = recon.view(recon.size(0), -1)
        # img2_flat = img_conv.view(img_conv.size(0), -1)
        
        # # Compute cosine embedding loss
        # # Target=1 means we want the vectors to be similar
        # cos_loss = torch.nn.CosineEmbeddingLoss()
        # target = torch.ones(img1_flat.size(0)).to(img1_flat.device)
        
        # loss3 = cos_loss(img1_flat, img2_flat, target)
        #####################################################################