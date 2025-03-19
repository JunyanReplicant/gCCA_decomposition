import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import pandas as pd
import scanpy as sc
import numpy as np

def preprocessing():
    Dar_ref =  pd.read_csv('./brainROSMAP/Darmanis_ref.csv',index_col = 0)
    Dar_label = pd.read_csv('./brainROSMAP/Darmanis_label.csv',index_col = 0)

    adata_ref = sc.AnnData(Dar_ref.to_numpy().transpose(),Dar_ref.columns.to_frame(),Dar_ref.index.to_frame())
    adata_ref = adata_ref[np.array(Dar_label.index),:]
    adata_ref.obs['celltype'] = np.array(Dar_label.to_numpy()).flatten().astype(str)
    adata_ref.obs['celltype_num'] = pd.factorize(adata_ref.obs['celltype'])[0]
    celltypeLabel = pd.factorize(adata_ref.obs['celltype'])[1]

    astro = pd.read_csv('./brainROSMAP/IHC.astro.txt', delim_whitespace=True).transpose()
    astro.columns = ['aligo']
    endo = pd.read_csv('./brainROSMAP/IHC.endo.txt', delim_whitespace=True).transpose()
    endo.columns = ['endo']
    microglia = pd.read_csv('./brainROSMAP/IHC.microglia.txt', delim_whitespace=True).transpose()
    microglia.columns = ['microglia']
    neuro = pd.read_csv('./brainROSMAP/IHC.neuro.txt', delim_whitespace=True).transpose()
    neuro.columns = ['neuro']
    oligo = pd.read_csv('./brainROSMAP/IHC.oligo.txt', delim_whitespace=True).transpose()
    oligo.columns = ['oligo']

    temp1 = astro.merge(endo, how='outer', left_index=True, right_index=True)
    temp1 = temp1.merge(microglia, how='outer', left_index=True, right_index=True)
    temp1 = temp1.merge(neuro, how='outer', left_index=True, right_index=True)
    groundTruth = temp1.merge(oligo, how='outer', left_index=True, right_index=True)
    groundTruth.dropna(axis = 0,inplace=True)
    SampleID = np.array(groundTruth.index)

    groundTruth = groundTruth.to_numpy()
    groundTruth = groundTruth/groundTruth.sum(axis=1)[:,None]

    bulk = pd.read_csv('./brainROSMAP/geneExpr.txt', delim_whitespace=True, index_col = 0)
    bulk_raw = pd.read_csv('./brainROSMAP/geneExprRaw.txt', delim_whitespace=True, index_col = 0)

    commonGenes = np.intersect1d(np.array(adata_ref.var).flatten(),np.array(bulk.index))
    commonGenes = np.intersect1d(commonGenes,np.array(bulk_raw.index))
    bulk_raw = bulk_raw.loc[commonGenes,SampleID]
    adata_bulk = sc.AnnData(bulk_raw.transpose(),bulk_raw.columns.to_frame(),bulk_raw.index.to_frame())
    adata_ref = adata_ref[:,commonGenes]
    adata_ref.obs.rename( columns={0:'CellNames'}, inplace=True )
    adata_ref.var.rename( columns={0:'GeneNames'}, inplace=True )
    adata_bulk.obs.rename( columns={0:'CellNames'}, inplace=True )
    adata_bulk.var.rename( columns={0:'GeneNames'}, inplace=True )
    return adata_ref, adata_bulk, groundTruth, celltypeLabel


