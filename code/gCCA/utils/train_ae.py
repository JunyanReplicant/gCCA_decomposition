import numpy as np
import gCCA.utils.simulation as sim
import gCCA.utils.preprocessing as pre
import lightning as L
from gCCA.model.batchCorrection import AE, genoMapDataset
from torch.utils.data import DataLoader
import scipy

def train_ae(scRNA_ref, genomap_ref, field_name = 'majortype_num'):
    # 1. Train an autoencoder for pseudobulks from scRNA reference
    if scipy.sparse.issparse(scRNA_ref.X):
        mix_train, _ = sim.createMix(np.array(scRNA_ref.X.todense()), scRNA_ref.obs[field_name] , sampleNum = 1000)
    else:
        mix_train, _ = sim.createMix(np.array(scRNA_ref.X), scRNA_ref.obs[field_name] , sampleNum = 1000)
        
    mixMaps_train = pre.convertGenomaps(np.array(mix_train), genomap_ref['projMat'])

    # 2. Train the autoencoder
    train_dataset = genoMapDataset(img = mixMaps_train)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True,drop_last=False)

    # 3. Define the model and the trainer
    model = AE(latent_dim = 32).double()
    trainer = L.Trainer(max_epochs = 200, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, train_loader)

    model.freeze()
    model.to('cuda')

    return model
