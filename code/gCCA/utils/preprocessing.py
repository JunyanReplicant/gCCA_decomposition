import sys
sys.path.append('..')  # Go up one level to gCCA directory

from gCCA.genomap_core.genomap import construct_genomap
import pandas as pd
import scipy
import numpy as np
import scanpy as sc


# create genomaps from the scRNA-seq data, the data have been normalized and z-scored first.
# the input data is a numpy array with shape (n_sample, n_genes)
# Currently the code only supports 1600 genes (40 by 40 genomaps), we are actively working on the code to support more genes.
# This function outputs genoMaps, projection matrix, mean genomap, and standard deviation genomap.
def create(adata_scRNA, colNum=40, rowNum = 40):
    if scipy.sparse.issparse(adata_scRNA.X):
        adata_numpy = np.array(adata_scRNA.X.todense())
    else:
        adata_numpy = np.array(adata_scRNA.X)
    dataNorm, mean, std = zscore_modified(adata_numpy, axis = 0, ddof=1) 
    # Construction of genomaps
    dataNorm[np.isnan(dataNorm)] = 0.001 * np.random.randn(*dataNorm[np.isnan(dataNorm)].shape)
    genoMaps,projMat = construct_genomap(dataNorm,colNum,rowNum,epsilon=0.0,num_iter=200)

    genoMaps = np.transpose(genoMaps, (0, 3, 1, 2))
    meanProj = np.matmul(mean, projMat)
    genoMaps_mean = np.reshape(meanProj, (colNum, rowNum), order='F')
    genoMaps_mean = np.expand_dims(genoMaps_mean, axis = 0)
    genoMaps_mean = np.expand_dims(genoMaps_mean, axis = 0) # 1*1*colNum*rowNum for the neural network
    
    stdProj = np.matmul(std, projMat)
    genoMaps_std = np.reshape(stdProj, (colNum, rowNum), order='F')
    genoMaps_std = np.expand_dims(genoMaps_std, axis = 0)
    genoMaps_std = np.expand_dims(genoMaps_std, axis = 0) # 1*1*colNum*rowNum for the neural network

    return {
            "genoMaps":genoMaps,
            "projMat":projMat,
            "gMean":genoMaps_mean,
            "gStd":genoMaps_std
           }  


# Once we have the projection matrix, we can project the scRNA-seq data with geneNum = 1600 into the genomap space
def convertGenomaps(scRNA_numpy, projMat, colNum=40, rowNum = 40):
    NpData = np.matmul(scRNA_numpy, projMat)
    genoMapsNew = []
    for i in range(len(NpData)):
        genoMapsNew.append(np.reshape(NpData[i,:], (colNum, rowNum), order='F'))
    genoMapsNew = np.expand_dims(np.array( genoMapsNew), axis = 1) 
    return genoMapsNew




# overwrite scipy.stats.zscore to return the mean and standard deviation
def zscore_modified(a, axis=0, ddof=0, nan_policy='propagate'):
    return zmap_modified(a, a, axis=axis, ddof=ddof, nan_policy=nan_policy)

def zmap_modified(scores, compare, axis=0, ddof=0, nan_policy='propagate'):
    a = np.asanyarray(compare)
    if a.size == 0:
        return np.empty(a.shape)

    contains_nan, nan_policy = scipy._lib._util._contains_nan(a, nan_policy)
    if contains_nan and nan_policy == 'omit':
        if axis is None:
            mn = scipy.stats._stats_py._quiet_nanmean(a.ravel())
            std = scipy.stats._stats_py._quiet_nanstd(a.ravel(), ddof=ddof)
            isconst = scipy.stats._stats_py._isconst(a.ravel())
        else:
            mn = np.apply_along_axis(scipy.stats._stats_py._quiet_nanmean, axis, a)
            std = np.apply_along_axis(scipy.stats._stats_py._quiet_nanstd, axis, a, ddof=ddof)
            isconst = np.apply_along_axis(scipy.stats._stats_py._isconst, axis, a)
    else:
        mn = a.mean(axis=axis, keepdims=True)
        std = a.std(axis=axis, ddof=ddof, keepdims=True)
        if axis is None:
            isconst = (a.item(0) == a).all()
        else:
            isconst = (scipy.stats._stats_py._first(a, axis) == a).all(axis=axis, keepdims=True)

    # Set std deviations that are 0 to 1 to avoid division by 0.
    std[isconst] = 1.0
    z = (scores - mn) / std
    # Set the outputs associated with a constant input to nan.
    z[np.broadcast_to(isconst, z.shape)] = np.nan
    return z,mn,std
