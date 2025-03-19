import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def ComputeCorr(a,b):
     if isinstance(a, (pd.DataFrame)):
          a = a.to_numpy()
     if isinstance(b, (pd.DataFrame)):
          b = b.to_numpy()
     correfficients = []
     for i in range(len(a)):
          pcc = np.corrcoef(a[i,:],b[i,:])[0][1]
          correfficients.append(pcc)
     
     return np.array(correfficients)

def ComputeCorr1D(a,b):
     if isinstance(a, (pd.DataFrame)):
          a = a.to_numpy()
     if isinstance(b, (pd.DataFrame)):
          b = b.to_numpy()
     correfficients = []
     for i in range(len(a)):
          pcc = np.corrcoef(a,b)[0][1]
          correfficients.append(pcc)
     return np.array(correfficients)

def CompareCorr(corvariables):
    # Create a list to store the data for the boxplot
    data_to_plot = []
    labels = []
    for var in corvariables:
        data_to_plot.append(locals()[var])
        # Create a more readable label by removing 'corr_' prefix and replacing underscores
        label = var.title()
        labels.append(label)

    # Create the boxplot
    plt.figure(figsize=(12, 6))
    box = plt.boxplot(data_to_plot, patch_artist=True, labels=labels, showfliers=False)

    # Customize the boxplot colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(corvariables)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Add scatter points on top of the boxplot
    for i, d in enumerate(data_to_plot):
        # Spread out the points horizontally
        x = np.random.normal(i+1, 0.04, size=len(d))
        plt.scatter(x, d, alpha=0.4, color='black', s=5)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add labels and title
    plt.ylabel('Correlation')
    plt.title('Correlation Comparison Across Different Models')

    # Adjust layout to prevent clipping of tick-labels
    plt.tight_layout()