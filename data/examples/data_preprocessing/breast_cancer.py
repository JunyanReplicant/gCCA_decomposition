import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import pandas as pd
import scanpy as sc
import numpy as np

def preprocessing():
    # load mtx files
    metadata = pd.read_csv('./examples/data/Wu_etal_2021/metadata.csv',index_col=0)
    adata = sc.read_mtx('./examples/data/Wu_etal_2021/matrix.mtx')
    adata_bc=pd.read_csv('./examples/data/Wu_etal_2021/barcodes.tsv',header=None)
    adata_features=pd.read_csv('./examples/data/Wu_etal_2021/features.tsv',header=None)
    adata= adata.T
    adata.obs['cell_id']= adata_bc[0].tolist()
    adata.obs.index= adata.obs['cell_id']
    adata.var['gene_name']= adata_features[0].tolist()
    adata.var.index= adata.var['gene_name']

    # organize cells in correct order
    adata = adata[metadata.index,:]
    adata.obs['cell_type'] = metadata['celltype_major']
    adata.obs['cell_subtype'] = metadata['celltype_minor']
    adata.obs['patient_id'] = metadata['orig.ident']
    adata.obs['cancer_type'] = metadata['subtype']

    # load bulk files
    bulk = pd.read_csv('./examples/data/Wu_etal_2021/GSE176078_Wu_etal_2021_bulkRNAseq_raw_counts.txt',sep='\t',index_col=0).transpose()
    bulk = bulk.dropna(axis = 1)
    adata_bulk = sc.AnnData(bulk,bulk.index.to_frame(),bulk.columns.to_frame())

    # normalization
    sc.pp.normalize_total(adata_bulk, target_sum = 1e4)
    sc.pp.normalize_total(adata, target_sum = 1e4)

    # The raw data do not match the ground truth files (with extra patients in the raw data). We need to re-organize the adatas.
    # only keep the common_patients between the raw data and the ground truth files
    getPatientId = adata.obs.loc[:,['patient_id','cancer_type']].drop_duplicates()
    getPatientId.index = getPatientId.index.str.split('_').str[0]
    getPatientId['patient_name'] = getPatientId.index
    getPatientId = getPatientId.set_index('patient_id')

    common_patients = np.intersect1d(adata_bulk.obs, getPatientId.index.to_numpy())
    adata = adata[adata.obs['patient_id'].isin(common_patients)]
    adata_bulk = adata_bulk[common_patients,:]
    
    # The raw data has categorical labels, we need to convert them to numbers for downstream analysis

    majortypes= pd.factorize(adata.obs['cell_type'])[1].to_numpy()
    adata.obs['cell_type'] = pd.Categorical(adata.obs['cell_type'])
    adata.obs['cell_type'] = pd.Categorical(pd.factorize(adata.obs['cell_type'])[0])

    minortypes= pd.factorize(adata.obs['cell_subtype'])[1].to_numpy()
    adata.obs['cell_subtype'] = pd.Categorical(adata.obs['cell_subtype'])
    adata.obs['cell_subtype'] = pd.Categorical(pd.factorize(adata.obs['cell_subtype'])[0])

    cancertypes= pd.factorize(adata.obs['cancer_type'])[1].to_numpy()
    adata.obs['cancer_type'] = pd.Categorical(adata.obs['cancer_type'])
    adata.obs['cancer_type'] = pd.Categorical(pd.factorize(adata.obs['cancer_type'])[0])

    # return the number of each label, so we could convert back to the original labels in the downstream analysis
    annotations = {
        'cell_type': majortypes,
        'cell_subtype': minortypes,
        'cancer_type': cancertypes,
        'patient_id': common_patients,
    }
    # only keep common genes between scRNA-seq and bulk RNA-seq
    commongenes = np.intersect1d(adata_bulk.var.index, adata.var.index)
    adata = adata[:,commongenes]
    adata_bulk = adata_bulk[:,commongenes]

    CELLTYPE = len(pd.factorize(adata.obs['cell_type'])[1].to_numpy())

    groundTruth = np.zeros((len(common_patients), CELLTYPE))
    for idx in range(len(common_patients)):
        cellSelected = adata[adata.obs['patient_id'] == common_patients[idx]]
        for j in range(CELLTYPE):
            subtypeSelected = cellSelected[cellSelected.obs['cell_type'] == j]
            groundTruth[idx,j] = len(subtypeSelected.obs) / len(cellSelected.obs)

    print('\033[93m Load data successfully')
    return adata, adata_bulk, groundTruth, annotations