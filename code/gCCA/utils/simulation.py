
import numpy as np
# For simulation, we need to create pseudobulk (random mixture of cell types) from the reference data
# Create Mix dataset
# The input is the reference scRNA-seq data (numpy array) and the cell type labels (annotations), how many samples to create
# The output will be an numpy array with sampleNum x geneNum, and the cell type proportions ground truth


def createMix(scRNA_numpy, celltype, sampleNum = 100):
    totalCellNum = 200
    typeNum = len(np.unique(celltype)) - np.sum(np.unique(celltype) < 0)
    proportions = []
    mixtures = []
    uniqueCellType = np.unique(celltype)
    for _ in range(sampleNum):
        comb = np.atleast_1d(np.squeeze(np.random.dirichlet(np.ones(typeNum),size=1)))
        if typeNum > 1:
            comb = randomZeroComb(comb)
        bulk = np.zeros((1,scRNA_numpy.shape[1]))
        for i in range(typeNum):
            groups = np.where(np.array(celltype) == uniqueCellType[i])[0]
            groupsSampleNum = int(np.floor(totalCellNum*comb[i]))
            groupSelected = np.random.choice(groups, groupsSampleNum)
            bulk += np.array(np.sum(scRNA_numpy[groupSelected,:],axis = 0))
        mixtures.append(bulk/totalCellNum) # obtain the average profiles of the mixture
        proportions.append(comb)

    
    mixtures = np.squeeze(np.array(mixtures))
    proportions = np.array(proportions)
    ######### original version is to return an anndata object #########
    # geneNum = scRNA_numpy.shape[1]
    # obs_names = pd.DataFrame([f"Peudo_{i:d}" for i in range(sampleNum)], columns=["features"]).set_index('features', drop=False)
    # var_names = pd.DataFrame([f"Gene_{i:d}" for i in range(geneNum)], columns=["features"]).set_index('features', drop=False)
    # adata_temp =  sc.AnnData(mixtures,obs_names,var_names)
    return mixtures, proportions

# To mimic the real data where one or more cell types are not present in the mixture, we randomly set some cell types to 0
def randomZeroComb(comb):
    NumOfZeros = np.random.randint(1, comb.size)
    inds = np.random.choice(comb.size, size= NumOfZeros, replace = False)
    temp = comb.copy()
    temp[inds] = 0
    temp = temp / temp.sum()
    
    return temp