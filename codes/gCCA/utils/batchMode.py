import numpy as np
from gCCA.model.batchCorrection import AE2, AEDataset
from torch.utils.data import DataLoader
import torch
import lightning as L
import gCCA.utils.preprocessing as pre 

def Train(genomap_ref, scRNA_ref, training_bulk, training_gt, ae1, field_name = 'majortype_num'):

    # 1. Create pseudobulks from scRNA-seq data according to the available ground truth
    CELLTYPE = len(np.unique(scRNA_ref.obs[field_name]))
    scRNA_avg = np.zeros((CELLTYPE, 1 ,40, 40))
    for i in range(CELLTYPE):
        scRNA_avg[i, 0, :, :] = np.mean(genomap_ref['genoMaps'][scRNA_ref.obs[field_name] == i ,:,:,:], axis = 0)
        
    scRNA_avg = scRNA_avg * genomap_ref['gStd']  + genomap_ref['gMean']
    training_gt = training_gt / training_gt.sum(axis=1)[:,None]

    pseudobulks = np.zeros((training_bulk.shape[0], 1, 40, 40))
    for j in range(training_bulk.shape[0]):
        for i in range(CELLTYPE):
            pseudobulks[j, 0, :, :] += training_gt[j, i] * scRNA_avg[i, 0, :, :]

    # 2. Add random noise to both the training bulk to make the model more robust
    train_bulk_noise = []
    pseudobulks_noise = []
    for i in range(len(training_bulk)):
        for _ in range(100):
            noised_signal = training_bulk[i,:,:,:] + np.random.normal(0, 1, 1600).reshape(40,40)
            noised_signal[noised_signal < 0] = 0
            train_bulk_noise.append(noised_signal )
            pseudobulks_noise.append(pseudobulks[i,:,:,:])
    train_bulk_noise = np.array(train_bulk_noise)
    pseudobulks_noise = np.array(pseudobulks_noise)

    # 3. Use the autoencoder to learn the map between the pseudobulks and the scRNA-seq data
    bulk_dataset = AEDataset(img = train_bulk_noise, label = pseudobulks_noise)
    bulk_loader = DataLoader(bulk_dataset, batch_size=100, shuffle=True,drop_last=False)

    ae2 = AE2( latent_dim=32, decoder=ae1.decode).double()
    trainer = L.Trainer(max_epochs=80, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(ae2, bulk_loader)
    ae2.freeze()
    return ae2

def Correct(batch_correction_model, batch_correct_autoencoder, testing_bulk):
    
    latent_bulk = batch_correction_model.encode(torch.tensor(testing_bulk))
    corrected_bulk = batch_correct_autoencoder.decode(latent_bulk.cuda())
    corrected_bulk[corrected_bulk < 0] = 0  
    return corrected_bulk

def Split(bulk_sample,projMat, groundTruth, idx):
    
    genoMap_bulk = pre.convertGenomaps(bulk_sample.X, projMat,colNum=40, rowNum = 40)
    sample_idx = np.arange(len(bulk_sample.X))
    training_idx = np.delete(sample_idx, idx)
    training_bulk = genoMap_bulk[training_idx, :]
    training_gt = np.delete(groundTruth, idx, axis=0)

    testing_idx = idx
    testing_bulk = genoMap_bulk[testing_idx:testing_idx+1, :]
    testing_gt = groundTruth[idx,:]

    return training_bulk, training_gt, testing_bulk, testing_gt
