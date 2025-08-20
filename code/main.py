
# This code can reproduce the results presented in the paper.
# At present, it supports 40x40 genomaps in keeping with the original implementation.
# This means user have to select 1600 genes exactly
# We are actively working on expanding and refining the code to make it more flexible.
# A complete, user-friendly package will be released in future.


import torch
import warnings
import numpy as np
# from examples.data_preprocessing.breast_cancer import preprocessing
import gCCA.utils.preprocessing as genomap
from gCCA.utils.train_gVAE import train_gVAE, GMM_fit
from gCCA.utils.train_gCCA import train_gCCA
from gCCA.utils.errorMetric import ComputeCorr
from gCCA.utils.saveResults import plot_genomaps
from gCCA.utils.train_ae import train_ae
import gCCA.utils.batchMode as bm
import os
######################################################################
######################################################################
############################################################################
# Start with breast non batch mode
save_dir = '../results/breast/non_batch'
# Create directory for saving results if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Check if GPU is available, or it may take a long time to run
gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")

# Load data
# scRNA_ref: reference scRNA-seq anndata (cells x genes)
# bulk_sample: bulk RNA-seq anndata (samples x genes)
# patient_ids: patient IDs
# groundTruth: ground truth proportions of cell types
# Both scRNA_ref and bulk_sample contain the same genes, and are normalized to the sum of 1e4.

# Because the original data is more than 2 GB, we only provide the preprocessed data in the examples folder. 
# If you want to run the code on the raw data, please refer to the preprocessing function in the data_preprocessing folder. (./data/data_preprocessing starts from the raw file)
# scRNA_ref, bulk_sample, groundTruth, annotations = preprocessing()
# When no bulk samples with known cell proportions are available, we can use the nonBatchMode for decomposition.
# Currently we only support 1600 genes (40 by 40 genomaps), as implemented in the paper. 
# We are improving to include more genes.

# Genes can be selected by 'rank_genes_groups', and 'highly_variable_genes'.
# For example:
        # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="wilcoxon")
        # deg = pd.DataFrame(adata.uns['wilcoxon']['names']).head(400)
        # deg_np = np.squeeze(deg.to_numpy().reshape(-1,1).astype(str))
# or
        # sc.pp.highly_variable_genes(adata, n_top_genes=1600)
        # adata = adata[:, temp.var['highly_variable']]

# To replicate the results of the paper, we fix the gene selection by directly loading genes.
warnings.filterwarnings('ignore')
seed = 21
import anndata as ad
# the scRNAref.h5ad exceed the upload size limit, so we provide the data in a compressed format.
# User can go to the below folder and unzip the file first.
scRNA_ref = ad.read_h5ad('../data/examples/data/Wu_etal_2021/scRNAref.h5ad')
bulk_sample = ad.read_h5ad('../data/examples/data/Wu_etal_2021/bulk.h5ad')
groundTruth = np.loadtxt('../data/examples/data/Wu_etal_2021/groundTruth.txt')
print('\033[93m Data loaded')

# Create genomaps from the scRNA-seq data
# Convert the scRNA-seq data to numpy array
genomap_ref = genomap.create(scRNA_ref, colNum=40, rowNum=40)
projMat = genomap_ref['projMat']
# save the genomap images
plot_genomaps(genomap_ref['genoMaps'], scRNA_ref.obs['cell_type'], save_dir=save_dir)
print('\033[93m Genomaps created')

# Because scRNA is usually very sparse, so we create pseudobulks to train the gVAE (VAE for genomaps)
# gVAEs stores the trained VAE model, while pseudobulks stores the pseudobulk created from scRNA-seq data.
# When cell_subtype is not available, we use cell_type as the sub label, 
# this only ensures that the subtypes are well-mixed in pseudobulks
gVAEs, pseudobulks = train_gVAE(projMat,
                               scRNA_ref,
                               labels_main=scRNA_ref.obs['cell_type'],
                               labels_sub=scRNA_ref.obs['cell_subtype'],
                               num_epochs = 1500
                               )
print('\033[93m gVAE training done')


# Train scikit-learn Gaussian Mixture model, save the mean and std for each cell type
torch.manual_seed(seed)
c_mean, c_std = GMM_fit(pseudobulks, projMat, gVAEs)

# Select a patient for decomposition
patient_id = 6
genoMap_bulk = genomap.convertGenomaps(bulk_sample.X, projMat,colNum=40, rowNum = 40)
# This is non batch mode, so the data are directly input into the gCCA model
# Here we only need the cell proportions, which is the second output of the train_gCCA function
_, paramsList_norm, _ = train_gCCA(genoMap_bulk[patient_id:patient_id+1], gVAEs, c_mean, c_std)

cell_proportions = np.array(paramsList_norm)

# Calculate correlations between the decomposed cell proportions and the ground truth
corr_gCCA = ComputeCorr(cell_proportions, groundTruth[patient_id:patient_id+1])
print('Correlations:', corr_gCCA)
print('Median correlation:', np.median(corr_gCCA))


with open(f'{save_dir}/results_nonbatch_mode.txt', 'w') as f:
    f.write(f'Patient ID: {patient_id}\n')
    f.write(f'Cell proportions correlation with ground truth:\n')
    f.write(f'{corr_gCCA}\n\n')
    f.write(f'Cell proportions:\n')
    f.write(f'{cell_proportions}\n')

######################################################################
######################################################################
######################################################################
# Next is the Breast batch mode
save_dir = '../results/breast/batch'
# Create directory for saving results if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Load data
# scRNA_ref: reference scRNA-seq anndata (cells x genes)
# bulk_sample: bulk RNA-seq anndata (samples x genes)
# patient_ids: patient IDs
# groundTruth: ground truth proportions of cell types
# Both scRNA_ref and bulk_sample contain the same genes, and are normalized to the sum of 1e4.

# save the genomap images
plot_genomaps(genomap_ref['genoMaps'], scRNA_ref.obs['cell_type'], save_dir=save_dir)
print('\033[93m Genomaps created')


torch.manual_seed(720)
c_mean, c_std = GMM_fit(pseudobulks, projMat, gVAEs)
# For batch mode, we assume all samples except the target sample are known,
# so we first split the data into known and target samples
batch_correct_autoencoder = train_ae(scRNA_ref, genomap_ref, field_name='cell_type')
print('\033[93m autoencoder training done')

# Select a patient for decomposition
patient_id = 5
# Leave one out: split training and testing
training_bulk, training_gt, testing_bulk, testing_gt = bm.Split(bulk_sample, projMat, groundTruth, patient_id)
# # For batch mode, we input the following: 
# # 1. scRNA reference (genomaps) to generate pseudobulks 
# # 2. scRNA labels (cell types)
# # 3. training_bulk: the genomaps of the known samples
# # 4. training_gt: the ground truth of the known samples
# # To avoid leaks of the testing set, this model needs to be trained separately for each testing sample
batch_correction_model = bm.Train(genomap_ref, scRNA_ref, training_bulk, training_gt, batch_correct_autoencoder, field_name = 'cell_type')
print('\033[93m batch correction model training done')
corrected_bulk = bm.Correct(batch_correction_model, batch_correct_autoencoder, testing_bulk)
# Here we only need the cell proportions, which is the second output of the train_gCCA function
print('\033[93m This is in training for sample',patient_id)
_, cell_proportions , _ = train_gCCA(corrected_bulk, gVAEs, c_mean, c_std)

# Calculate correlations between the decomposed cell proportions and the ground truth
corr_gCCA = ComputeCorr(cell_proportions, groundTruth[patient_id:patient_id+1])
print('\033[93m Correlations:', corr_gCCA)
print('\033[93m Median correlation:', np.median(corr_gCCA))


with open(f'{save_dir}/results_batch_mode.txt', 'w') as f:
    f.write(f'Patient ID: {patient_id}\n')
    f.write(f'Cell proportions correlation with ground truth:\n')
    f.write(f'{corr_gCCA}\n\n')
    f.write(f'Cell proportions:\n')
    f.write(f'{cell_proportions}\n')
######################################################################
####################################################################
######################################################################
# PBMC NON Batch
save_dir = '../results/pbmc/non_batch'
# Create directory for saving results if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Check if GPU is available, or it may take a long time to run
gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")

warnings.filterwarnings('ignore')
# load data
# scRNA_ref: reference scRNA-seq anndata (cells x genes)
# bulk_sample: bulk RNA-seq anndata (samples x genes)
# celltypeLabel: Raw data has string labels, we convert to numerical. This acts as a dictionary between cell type and number.
# Both scRNA_ref and bulk_sample contain the same genes, and are normalized to the sum of 1e4.
# scRNA_ref, bulk_sample, celltypeLabel = preprocessing()
import anndata as ad
scRNA_ref = ad.read_h5ad('../data/examples/data/pbmc_real/scRNAref_nb.h5ad')
bulk_sample = ad.read_h5ad('../data/examples/data/pbmc_real/bulk_nb.h5ad')

print('\033[93m Data loaded')
# calculate the ground truth, and select only the bulk samples that have ground truth
# groundTruth, patient_ids, bulk_sample = getGroundTruth(bulk_sample)

# Similar as breast cancer example, we directly load the genes to replicate the results in the paper
# These genes are selected by 'rank_genes_groups', and 'highly_variable_genes'.
# For example:
        # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="wilcoxon")
        # deg = pd.DataFrame(adata.uns['wilcoxon']['names']).head(400)
        # deg_np = np.squeeze(deg.to_numpy().reshape(-1,1).astype(str))
# or
        # sc.pp.highly_variable_genes(adata, n_top_genes=1600)
        # adata = adata[:, temp.var['highly_variable']]


# Create genomaps from the scRNA-seq data
# Convert the scRNA-seq data to numpy array
genomap_ref = genomap.create(scRNA_ref, colNum=40, rowNum=40)
projMat = genomap_ref['projMat']
plot_genomaps(genomap_ref['genoMaps'], scRNA_ref.obs['majortype_num'], save_dir=save_dir)
print('\033[93m genomaps created')

# Because scRNA is usually very sparse, so we create pseudobulks to train the gVAE (VAE for genomaps)
# gVAEs stores the trained VAE model, while pseudobulks stores the pseudobulk created from scRNA-seq data.
gVAEs, pseudobulks = train_gVAE(projMat,
                               scRNA_ref,
                               labels_main=scRNA_ref.obs['majortype_num'],
                               labels_sub=scRNA_ref.obs['majortype_num'],
                               num_epochs=1500
                               )
# when cell_subtype is not available, we use cell_type as the sub label, 
# this only ensures that the subtypes are well-mixed in pseudobulks
print('\033[93m gVAE training done')

# Train scikit-learn Gaussian Mixture model
torch.manual_seed(720)
c_mean, c_std = GMM_fit(pseudobulks, projMat, gVAEs)

# Select a patient for decomposition to save time
patient_id = 0
# Convert bulk samples to genomaps and train gCCA
genoMap_bulk = genomap.convertGenomaps(bulk_sample.X, projMat, colNum=40, rowNum=40)
# Here we only need the cell proportions, which is the second output of the train_gCCA function
_, cell_proportions, _ = train_gCCA(genoMap_bulk[patient_id:patient_id+1], gVAEs, c_mean, c_std)

# The 5th column of the cell proportions is the 'OTHER' cell type (as preprocessed), which is not known in the ground truth
# So we set it to 0 (in the ground truth we already pre-set it to 0)
cell_proportions = np.delete(cell_proportions, 4, axis=1)
groundTruth = np.loadtxt('../data/examples/data/pbmc_real/groundTruth.txt')
groundTruth = np.delete(groundTruth, 4, axis=1)
corr_gCCA = ComputeCorr(cell_proportions, groundTruth[patient_id:patient_id+1])
print('The median correlation is:', np.median(corr_gCCA))



with open(f'{save_dir}/results_pbmc_nonbatch.txt', 'w') as f:
    f.write(f'Patient ID: {patient_id}\n')
    f.write(f'Cell proportions correlation with ground truth:\n')
    f.write(f'{corr_gCCA}\n\n')
    f.write(f'Cell proportions:\n')
    f.write(f'{cell_proportions}\n')
######################################################################
######################################################################
####################################################################
# PBMC Batch

save_dir = '../results/pbmc/batch'
# Create directory for saving results if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Check if GPU is available, or it may take a long time to run
gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")

warnings.filterwarnings('ignore')
# load data
# scRNA_ref: reference scRNA-seq anndata (cells x genes)
# bulk_sample: bulk RNA-seq anndata (samples x genes)
# celltypeLabel: Raw data has string labels, we convert to numerical. This acts as a dictionary between cell type and number.
# Both scRNA_ref and bulk_sample contain the same genes, and are normalized to the sum of 1e4.
# scRNA_ref, bulk_sample, celltypeLabel = preprocessing()
import anndata as ad
scRNA_ref = ad.read_h5ad('../data/examples/data/pbmc_real/scRNAref_b.h5ad')
bulk_sample = ad.read_h5ad('../data/examples/data/pbmc_real/bulk_b.h5ad')
groundTruth = np.loadtxt('../data/examples/data/pbmc_real/groundTruth.txt')
print('\033[93m Data loaded')
# calculate the ground truth, and select only the bulk samples that have ground truth
# groundTruth, patient_ids, bulk_sample = getGroundTruth(bulk_sample)

# Similar as breast cancer example, we directly load the genes to replicate the results in the paper
# These genes are selected by 'rank_genes_groups', and 'highly_variable_genes'.
# For example:
        # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="wilcoxon")
        # deg = pd.DataFrame(adata.uns['wilcoxon']['names']).head(400)
        # deg_np = np.squeeze(deg.to_numpy().reshape(-1,1).astype(str))
# or
        # sc.pp.highly_variable_genes(adata, n_top_genes=1600)
        # adata = adata[:, temp.var['highly_variable']]


# Create genomaps from the scRNA-seq data
# Convert the scRNA-seq data to numpy array
genomap_ref = genomap.create(scRNA_ref, colNum=40, rowNum=40)
projMat = genomap_ref['projMat']
plot_genomaps(genomap_ref['genoMaps'], scRNA_ref.obs['majortype_num'], save_dir=save_dir)
print('\033[93m genomaps created')

# Because scRNA is usually very sparse, so we create pseudobulks to train the gVAE (VAE for genomaps)
# gVAEs stores the trained VAE model, while pseudobulks stores the pseudobulk created from scRNA-seq data.
gVAEs, pseudobulks = train_gVAE(projMat,
                               scRNA_ref,
                               labels_main=scRNA_ref.obs['majortype_num'],
                               labels_sub=scRNA_ref.obs['majortype_num'],
                               num_epochs=1500
                               )
# when cell_subtype is not available, we use cell_type as the sub label, 
# this only ensures that the subtypes are well-mixed in pseudobulks
print('\033[93m gVAE training done')

# Train scikit-learn Gaussian Mixture model
torch.manual_seed(720)
c_mean, c_std = GMM_fit(pseudobulks, projMat, gVAEs)

# Select a patient for decomposition to save time
patient_id = 2
# Convert bulk samples to genomaps and train gCCA
batch_correct_autoencoder = train_ae(scRNA_ref, genomap_ref,field_name = 'majortype_num')
training_bulk, training_gt, testing_bulk, testing_gt = bm.Split(bulk_sample, projMat, groundTruth, patient_id)
# # For batch mode, we input the following: 
# # 1. scRNA reference (genomaps) to generate pseudobulks 
# # 2. scRNA labels (cell types)
# # 3. training_bulk: the genomaps of the known samples
# # 4. training_gt: the ground truth of the known samples
# # To avoid leaks of the testing set, this model needs to be trained separately for each testing sample
batch_correction_model = bm.Train(genomap_ref, scRNA_ref, training_bulk, training_gt, batch_correct_autoencoder)
print('\033[93m batch correction model training done')
corrected_bulk = bm.Correct(batch_correction_model, batch_correct_autoencoder, testing_bulk)

# Here we need the cell proportions, which is the second output of the train_gCCA function
print('\033[93m This is in training for sample',patient_id)
_, cell_proportions, latentVars = train_gCCA(corrected_bulk, gVAEs, c_mean, c_std)


# The 5th column of the cell proportions is the 'OTHER' cell type (as preprocessed), which is not known in the ground truth
# So we set it to 0 (in the ground truth we already pre-set it to 0)
cell_proportions = np.delete(cell_proportions, 4, axis=1)
groundTruth = np.loadtxt('../data/examples/data/pbmc_real/groundTruth.txt')
groundTruth = np.delete(groundTruth, 4, axis=1)
corr_gCCA = ComputeCorr(cell_proportions, groundTruth[patient_id:patient_id+1])
print('The median correlation is:', np.median(corr_gCCA))



with open(f'{save_dir}/results_pbmc_batch.txt', 'w') as f:
    f.write(f'Patient ID: {patient_id}\n')
    f.write(f'Cell proportions correlation with ground truth:\n')
    f.write(f'{corr_gCCA}\n\n')
    f.write(f'Cell proportions:\n')
    f.write(f'{cell_proportions}\n')
######################################################################
######################################################################
######################################################################

# ROSMAP NON Batch

save_dir = '../results/rosmap/non_batch'
# Create directory for saving results if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Check if GPU is available, or it may take a long time to run
gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")

warnings.filterwarnings('ignore')
# load data
# scRNA_ref: reference scRNA-seq anndata (cells x genes)
# bulk_sample: bulk RNA-seq anndata (samples x genes)
# celltypeLabel: Raw data has string labels, we convert to numerical. This acts as a dictionary between cell type and number.
# Both scRNA_ref and bulk_sample contain the same genes, and are normalized to the sum of 1e4.
# scRNA_ref, bulk_sample, celltypeLabel = preprocessing()
import anndata as ad
scRNA_ref = ad.read_h5ad('../data/examples/data/rosmap/scRNAref_nb.h5ad')
bulk_sample = ad.read_h5ad('../data/examples/data/rosmap/bulk_nb.h5ad')
groundTruth = np.loadtxt('../data/examples/data/rosmap/groundTruth.txt')

print('\033[93m Data loaded')
# calculate the ground truth, and select only the bulk samples that have ground truth
# groundTruth, patient_ids, bulk_sample = getGroundTruth(bulk_sample)

# Similar as breast cancer example, we directly load the genes to replicate the results in the paper
# These genes are selected by 'rank_genes_groups', and 'highly_variable_genes'.
# For example:
        # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="wilcoxon")
        # deg = pd.DataFrame(adata.uns['wilcoxon']['names']).head(400)
        # deg_np = np.squeeze(deg.to_numpy().reshape(-1,1).astype(str))
# or
        # sc.pp.highly_variable_genes(adata, n_top_genes=1600)
        # adata = adata[:, temp.var['highly_variable']]


# Create genomaps from the scRNA-seq data
# Convert the scRNA-seq data to numpy array
genomap_ref = genomap.create(scRNA_ref, colNum=40, rowNum=40)
projMat = genomap_ref['projMat']
plot_genomaps(genomap_ref['genoMaps'], scRNA_ref.obs['celltype_num'], save_dir=save_dir)
print('\033[93m genomaps created')

# Because scRNA is usually very sparse, so we create pseudobulks to train the gVAE (VAE for genomaps)
# gVAEs stores the trained VAE model, while pseudobulks stores the pseudobulk created from scRNA-seq data.
gVAEs, pseudobulks = train_gVAE(projMat,
                               scRNA_ref,
                               labels_main=scRNA_ref.obs['celltype_num'],
                               labels_sub=scRNA_ref.obs['celltype_num'],
                               num_epochs=1500
                               )
# when cell_subtype is not available, we use cell_type as the sub label, 
# this only ensures that the subtypes are well-mixed in pseudobulks
print('\033[93m gVAE training done')

# Train scikit-learn Gaussian Mixture model
torch.manual_seed(42)
c_mean, c_std = GMM_fit(pseudobulks, projMat, gVAEs)

# Select a patient for decomposition to save time
patient_id = 1
# Convert bulk samples to genomaps and train gCCA
genoMap_bulk = genomap.convertGenomaps(bulk_sample.X, projMat, colNum=40, rowNum=40)
# Here we only need the cell proportions, which is the second output of the train_gCCA function
_, cell_proportions, _ = train_gCCA(genoMap_bulk[patient_id:patient_id+1], gVAEs, c_mean, c_std)

# The 5th column of the cell proportions is the 'OTHER' cell type (as preprocessed), which is not known in the ground truth
# So we set it to 0 (in the ground truth we already pre-set it to 0)
corr_gCCA = ComputeCorr(cell_proportions, groundTruth[patient_id:patient_id+1])
print('The median correlation is:', np.median(corr_gCCA))



with open(f'{save_dir}/results_rosmap_nonbatch.txt', 'w') as f:
    f.write(f'Patient ID: {patient_id}\n')
    f.write(f'Cell proportions correlation with ground truth:\n')
    f.write(f'{corr_gCCA}\n\n')
    f.write(f'Cell proportions:\n')
    f.write(f'{cell_proportions}\n')
######################################################################
#######################################################################
######################################################################
# Rosmap batch

save_dir = '../results/rosmap/batch'
# Create directory for saving results if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Check if GPU is available, or it may take a long time to run
gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")

warnings.filterwarnings('ignore')
# load data
# scRNA_ref: reference scRNA-seq anndata (cells x genes)
# bulk_sample: bulk RNA-seq anndata (samples x genes)
# celltypeLabel: Raw data has string labels, we convert to numerical. This acts as a dictionary between cell type and number.
# Both scRNA_ref and bulk_sample contain the same genes, and are normalized to the sum of 1e4.
# scRNA_ref, bulk_sample, celltypeLabel = preprocessing()
import anndata as ad
scRNA_ref = ad.read_h5ad('../data/examples/data/rosmap/scRNAref_b.h5ad')
bulk_sample = ad.read_h5ad('../data/examples/data/rosmap/bulk_b.h5ad')
groundTruth = np.loadtxt('../data/examples/data/rosmap/groundTruth.txt')

print('\033[93m Data loaded')
# calculate the ground truth, and select only the bulk samples that have ground truth
# groundTruth, patient_ids, bulk_sample = getGroundTruth(bulk_sample)

# Similar as breast cancer example, we directly load the genes to replicate the results in the paper
# These genes are selected by 'rank_genes_groups', and 'highly_variable_genes'.
# For example:
        # sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, key="wilcoxon")
        # deg = pd.DataFrame(adata.uns['wilcoxon']['names']).head(400)
        # deg_np = np.squeeze(deg.to_numpy().reshape(-1,1).astype(str))
# or
        # sc.pp.highly_variable_genes(adata, n_top_genes=1600)
        # adata = adata[:, temp.var['highly_variable']]


# Create genomaps from the scRNA-seq data
# Convert the scRNA-seq data to numpy array
genomap_ref = genomap.create(scRNA_ref, colNum=40, rowNum=40)
projMat = genomap_ref['projMat']
plot_genomaps(genomap_ref['genoMaps'], scRNA_ref.obs['celltype_num'], save_dir=save_dir)
print('\033[93m genomaps created')

# Because scRNA is usually very sparse, so we create pseudobulks to train the gVAE (VAE for genomaps)
# gVAEs stores the trained VAE model, while pseudobulks stores the pseudobulk created from scRNA-seq data.
gVAEs, pseudobulks = train_gVAE(projMat,
                               scRNA_ref,
                               labels_main=scRNA_ref.obs['celltype_num'],
                               labels_sub=scRNA_ref.obs['celltype_num'],
                               num_epochs=1500
                               )
# when cell_subtype is not available, we use cell_type as the sub label, 
# this only ensures that the subtypes are well-mixed in pseudobulks
print('\033[93m gVAE training done')

# Train scikit-learn Gaussian Mixture model
torch.manual_seed(42)
c_mean, c_std = GMM_fit(pseudobulks, projMat, gVAEs)

# Select a patient for decomposition to save time
patient_id = 3
batch_correct_autoencoder = train_ae(scRNA_ref, genomap_ref,field_name = 'celltype_num')

training_bulk, training_gt, testing_bulk, testing_gt = bm.Split(bulk_sample, projMat, groundTruth, patient_id)
# # For batch mode, we input the following: 
# # 1. scRNA reference (genomaps) to generate pseudobulks 
# # 2. scRNA labels (cell types)
# # 3. training_bulk: the genomaps of the known samples
# # 4. training_gt: the ground truth of the known samples
# # To avoid leaks of the testing set, this model needs to be trained separately for each testing sample
batch_correction_model = bm.Train(genomap_ref, scRNA_ref, training_bulk, training_gt, batch_correct_autoencoder, field_name = 'celltype_num')
print('\033[93m batch correction model training done')
corrected_bulk = bm.Correct(batch_correction_model, batch_correct_autoencoder, testing_bulk)
# Here we only need the cell proportions, which is the second output of the train_gCCA function
print('\033[93m This is in training for sample',patient_id)
_, cell_proportions , _ = train_gCCA(corrected_bulk, gVAEs, c_mean, c_std)

# The 5th column of the cell proportions is the 'OTHER' cell type (as preprocessed), which is not known in the ground truth
# So we set it to 0 (in the ground truth we already pre-set it to 0)
corr_gCCA = ComputeCorr(cell_proportions, groundTruth[patient_id:patient_id+1])
print('The median correlation is:', np.median(corr_gCCA))


with open(f'{save_dir}/results_rosmap_batch.txt', 'w') as f:
    f.write(f'Patient ID: {patient_id}\n')
    f.write(f'Cell proportions correlation with ground truth:\n')
    f.write(f'{corr_gCCA}\n\n')
    f.write(f'Cell proportions:\n')
    f.write(f'{cell_proportions}\n')
