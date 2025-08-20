### "Unveiling Tissue Heterogeneity through Genomic Interaction-Encoded Image Representation of RNA Sequencing Data"

Detailed usage instructions are provided in **tutorial.ipynb**. 

Before running the tutorial example, users must first navigate to the **data/examples/Wu_etal_2021** folder and unzip the compressed files (This file is compressed and split due to file size limitations). 

To ensure compatibility, we recommend creating a virtual environment using the package versions specified in requirements.txt, as newer package versions may cause compatibility issues.

### Creating a Virtual Environment

```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows:
myenv\Scripts\activate

# On macOS/Linux:
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

The tutorial examples in **tutorial.ipynb** corresponds to the Breast cancer data, PBMC.s13 data, and Rosmap data in the paper.

The results of the tutorial are saved in the **/results** folder.

The general idea is to first use:
```python
genomap.create(scRNA_reference, colNum=40, rowNum=40)
```
to obtain the genomap images and the projection matrix. Then use:
```python
gVAEs, pseudobulks = train_gVAE(projection_matrix,
                               scRNA_reference,
                               labels_main=cell_labels,
                               labels_sub=cell_labels,
                               num_epochs=1500
                               )
c_mean, c_std = GMM_fit(pseudobulks, projection_matrix, gVAEs)
```
to train the VAE and GMM. Eventually, use the VAE and GMM to fit for the cell proportions
```python
genoMap_bulk = genomap.convertGenomaps(bulk_sample, projection_matrix, colNum=40, rowNum=40)
_, cell_proportions, _ = train_gCCA(genoMap_bulk, gVAEs, c_mean, c_std)
```