import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import lightning as L
from gCCA.model.gCCA_core import Decompose, genoMapDataset



def train_gCCA(bulk_samples, gVAEs, c_mean, c_std, epochs = 600):

    paramsList = []
    paramsList_norm = []
    latentVar = []
    CELLTYPE = len(gVAEs)
    for currentSample in range(len(bulk_samples)):

        if len(bulk_samples) > 1:
            print('\033[96m **This is in Training CCA for sample',currentSample)
        sample = bulk_samples[currentSample:currentSample+1,:,:,:] 
        train_dataset = genoMapDataset(img = sample)
        train_dl = DataLoader(train_dataset, batch_size = 1, shuffle=False,drop_last=False)
    
        sc = Decompose(celltypes = CELLTYPE, g_mean = c_mean, g_std = c_std ,vae = gVAEs).double()

        trainer = L.Trainer(max_epochs = 800, enable_model_summary=False)
        trainer.fit(model=sc, train_dataloaders=train_dl)
        
        trainer = L.Trainer(max_epochs = epochs, enable_model_summary=False)
        trainer.fit(model=sc, train_dataloaders=train_dl)     

        params = []
        for name, param in sc.named_parameters():
            if param.requires_grad == True:
                if name != 'latentVars':
                    params.append(np.mean(np.squeeze(param.cpu().detach().numpy())))
                else:
                    latentVar.append(param.cpu().detach().numpy())
        
        params = np.array(params)
        params_norm = params/np.sum(params)
        paramsList.append(params)
        paramsList_norm.append(params_norm)

    paramsList = np.array(paramsList)
    paramsList_norm = np.array(paramsList_norm)
    latentVar = np.array(latentVar)
    # paramsList is the cell proportion (absolute value) for each cell type
    # paramsList_norm is the cell proportion (normalized) for each cell type, sum to 1
    # latentVar is the latent variables for each cell type and each sample (sample size * cell type size * latent dimension)
    return paramsList, paramsList_norm, latentVar