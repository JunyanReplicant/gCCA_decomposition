This is the code for "Unveiling Tissue Heterogeneity through Genomic Interaction-Encoded Image Representation of RNA Sequencing Data"

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

The results are saved in the **/results** folder.