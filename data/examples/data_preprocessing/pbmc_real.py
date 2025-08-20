import sys
sys.path.insert(1, './preprocessing')
sys.path.insert(1, './genomap-core')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import biomart 

# If this file reports error, it is usually a bad connection to biomart
def preprocessing():
    # Load the scRNA-seq data and bulk samples
    # Two datasets contains slightly different cell types annotation
    # We need to merge them and select the common cell types
    pbmc3k = sc.read_10x_mtx('./examples/data/pbmc_real/pbmc3k_filtered_gene_bc_matrices/filtered_gene_bc_matrices/hg19')
    pbmc3kcluster = pd.read_csv('./examples/data/pbmc_real/pbmc3k_sctype.csv',index_col = 0)
    pbmc3k = pbmc3k[np.array(pbmc3kcluster.index).astype(str),:]
    pbmc3k.obs['subtype'] = pbmc3kcluster['sctype_classification']
    temp = pbmc3kcluster['sctype_classification'].copy()
    for i in range(len(temp)):
        if temp[i].startswith('CD8+ NKT'):
            temp[i] = 'Other'
        if temp[i].startswith('Platelets'):
            temp[i] = 'Other'
        if temp[i].startswith('Classical'):
            temp[i] = 'Mono'
        if temp[i].startswith('Memory'):
            temp[i] = 'CD4+T'
        if temp[i].startswith('Naive CD8+'):
            temp[i] = 'CD8+T'
        if temp[i].startswith('Non-classical'):
            temp[i] = 'Mono'
    pbmc3k.obs['majortype'] = temp
    pbmc3k.obs['majortype_num'] = pd.factorize(pbmc3k.obs['majortype'].to_numpy().astype('str'))[0]
    celltypeLabel = pd.factorize(pbmc3k.obs['majortype'].to_numpy().astype('str'))[1]
    
    bulk = pd.read_csv('./examples/data/pbmc_real/GSE107011_Processed_data_TPM.txt',sep='\t', index_col=0).transpose()
    adata_bulk = ad.AnnData( bulk, bulk.index.to_frame(), bulk.columns.to_frame())

    # The raw data is in Ensembl ID, we need to convert it to gene symbol
    server = biomart.BiomartServer('http://www.ensembl.org/biomart')         
    mart = server.datasets['hsapiens_gene_ensembl']     
    attributes = ['ensembl_transcript_id', 'hgnc_symbol',
                    'ensembl_gene_id']
    response = mart.search({'attributes': attributes})    

    data = response.raw.data.decode('ascii')   
    ensembl_to_genesymbol = {}                                                  
    for line in data.splitlines():                                              
            line = line.split('\t')                                                 
            # The entries are in the same order as in the `attributes` variable
            transcript_id = line[0]                                                 
            gene_symbol = line[1]                                                   
            ensembl_gene = line[2]                                                                                               
            if gene_symbol :                                                                       
                # Some of these keys may be an empty string. If you want, you can 
                # avoid having a '' key in your dict by ensuring the attributes
                # have a nonzero length before adding them to the dict
                ensembl_to_genesymbol[transcript_id] = gene_symbol                      
                ensembl_to_genesymbol[ensembl_gene] = gene_symbol

    eids = adata_bulk.var.index.map(lambda x: x.split('.')[0])
    gene_id_list = []
    for name in eids:
        try:
            gene_id_list.append(ensembl_to_genesymbol[name]) 
        except:
            gene_id_list.append(name) 
    gene_id_list = np.array(gene_id_list)
    adata_bulk.var.index = gene_id_list
    adata_bulk.var_names_make_unique()


    # Finally, we need to select the common genes between scRNA-seq and bulk data
    # And normalize the data to the same scale
    commonGenes = np.unique(np.intersect1d(pbmc3k.var.index, gene_id_list))
    adata_bulk = adata_bulk[:,commonGenes]
    pbmc3k = pbmc3k[:,commonGenes]
    sc.pp.normalize_total(pbmc3k, target_sum = 1e4)
    sc.pp.normalize_total(adata_bulk, target_sum = 1e4)



    return pbmc3k, adata_bulk, celltypeLabel

def getGroundTruth(adata_bulk):
    # celltypeLabel = pd.factorize(pbmc3k.obs['majortype'].to_numpy().astype('str'))[1]
    # Load the ground truth and select the common patients

    groundTruth = pd.read_excel('./examples/data/pbmc_real/s13groundtruth.xlsx', index_col=0)  

    commonPatients = np.intersect1d(np.array(groundTruth.index).astype(str),
                                    np.array(adata_bulk.obs.index).astype(str))
    adata_bulk = adata_bulk[commonPatients,:]
    groundTruth = groundTruth.loc[commonPatients]
    groundTruth = groundTruth.to_numpy()/100

    return groundTruth,commonPatients,adata_bulk
