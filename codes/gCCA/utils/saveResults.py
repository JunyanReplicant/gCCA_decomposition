import numpy as np
import matplotlib.pyplot as plt
import pickle
def save_result(obj, filename, save_dir):
    """Helper function to save results"""
    with open(f'{save_dir}/{filename}.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_result(filename, save_dir='results'):
    """Helper function to load results"""
    with open(f'{save_dir}/{filename}.pkl', 'rb') as f:
        return pickle.load(f)
    
def plot_genomaps(genomaps, cell_types, save_dir='results'):
    """
    Plot and save the average genomap for each cell type
    
    Args:
        genomaps: numpy array of genomaps with shape (n_cells, height, width)
        cell_types: array of cell type labels corresponding to each genomap
        save_dir: directory to save the plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    unique_types = np.unique(cell_types)
    
    for cell_type in unique_types:
        # Get genomaps for this cell type
        type_maps = genomaps[cell_types == cell_type]
        
        # Calculate average
        avg_map = np.squeeze(np.mean(type_maps, axis=0))
        
        # Create figure
        plt.figure(figsize=(8, 8))
        plt.title(f'Average Genomap - Cell Type {cell_type}')
        plt.imshow(avg_map, cmap='viridis')
        
        # Save figure
        plt.savefig(os.path.join(save_dir, f'avg_genomap_celltype_{cell_type}.png'), dpi=500)
        plt.close()
