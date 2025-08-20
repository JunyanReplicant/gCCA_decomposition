import torch
from torch.utils.data import DataLoader
from gCCA.model.GVAE import gVAE, genoMapDataset
import numpy as np
import lightning as L
import scipy
import gCCA.utils.simulation as sim
import gCCA.utils.preprocessing as pre
from sklearn.mixture import GaussianMixture

# This is the training loop for gVAE

def train_gVAE(projMat, scRNA_ref, labels_main, labels_sub, num_epochs = 1500, latent_dim = 16):
    # As scRNA-seq data is sparse, we randomly sample and combine cells from each major cell type to form a pseudobulk.
    # We then train gVAE on the pseudobulk data.

    # gVAEs stores the trained gVAE models
    # pseudobulks stores the pseudobulk data
    if scipy.sparse.issparse(scRNA_ref.X):
        adata_numpy = np.array(scRNA_ref.X.todense())
    else:
        adata_numpy = np.array(scRNA_ref.X)
    gVAEs = []
    pseudobulks = []
    CELLTYPE = len(np.unique(labels_main)) #scRNA_ref.obs['cell_type']
    for i in range(CELLTYPE):
        print('\033[96m **This is in Training gVAE for cell type',i)
        cellsIdx = labels_main == i # select cells of the same cell type
        # some dataset has sub-labels, here we use the sub-labels to create the pseudobulk so each pseudobulk is well-mixed by various subtypes
        # For datasets without sub-labels, we can directly use the major labels to create the pseudobulk
        cellMix, _ = sim.createMix(adata_numpy[cellsIdx], labels_sub[cellsIdx], sampleNum = 2000)
        vae_learning = pre.convertGenomaps(cellMix, projMat, colNum=40, rowNum = 40)
        pseudobulks.append(cellMix)
        train_dataset = genoMapDataset(img = vae_learning)
        train_dl = DataLoader(train_dataset, batch_size = 1000, shuffle=True,drop_last=False)
        gVAE_model = gVAE(latent_dim = latent_dim).double()
        trainer = L.Trainer(max_epochs = num_epochs, enable_model_summary=False)
        trainer.fit(model=gVAE_model , train_dataloaders=train_dl)
        # Once the gVAE is trained, we freeze it for downstream decomposition
        gVAE_model.freeze()
        gVAE_model.to('cuda')
        gVAEs.append(gVAE_model)

    return gVAEs, pseudobulks


def GMM_fit(pseudobulks, projMat, gVAEs):
    # We fit the gaussian mixture model on the latent space of the gVAE
    # We save the GMM parameters for downstream decomposition, which is the mean and variance.
    c_mean = []
    c_std = []
    CELLTYPE = len(gVAEs)
    for i in range(CELLTYPE):
        # We project the pseudobulk data into the genomap space, obtain the latent space representation
        _ ,latent = gVAEs[i](torch.tensor(pre.convertGenomaps(pseudobulks[i], projMat)).cuda() )  #torch.tensor(X_train_VAE).cuda()
        latent = latent.cpu().detach().numpy()
        gm = GaussianMixture(n_components=1, random_state = 0, covariance_type = 'diag').fit(latent)
        c_mean.append(gm.means_)
        c_std.append(np.sqrt(gm.covariances_))
        
    c_mean = torch.tensor(np.squeeze(np.array(c_mean)))
    c_std = torch.tensor(np.squeeze(np.array(c_std)))
    
    return c_mean, c_std