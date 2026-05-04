import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import BoundaryNorm


def create_board(n_rows, n_cols, final_sequence, sequence_length, feat_sel):
    board = np.zeros((n_rows, n_cols))
    
    for i in range(n_cols):
        start_index = int(final_sequence[i]) 
        end_index = int(final_sequence[i])  + int(sequence_length[i])
        if feat_sel[i] != 0:
            board[start_index:end_index, i] = 1
    
    return board


def plot_board(board, column_names,show):
    plt.figure(figsize=(25, 12))
    
    
    n_bins = 2  # adjust as needed
    bounds = np.linspace(0, 1, n_bins + 1)  # covers your vmin=-1 to vmax=1
    cmap = plt.get_cmap("pink_r", n_bins)  # discrete version of seismic
    norm = BoundaryNorm(bounds, cmap.N)

    
    
    img = plt.imshow(board, cmap=cmap, origin='lower', aspect='auto')  # <---
    cbar = plt.colorbar(img, fraction=0.02, pad=0.04, ticks=[0.25, 0.75])
    cbar.set_ticklabels(["False", "True"])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Feature selected", fontsize=25)
    
    plt.xticks(np.arange(len(column_names)), column_names, rotation=90,fontsize=11)
    plt.yticks(np.arange(0,board.shape[0],1)-0.5,np.arange(0,board.shape[0],1))
    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    
    plt.xticks(np.arange(len(column_names)), column_names, rotation=90, fontsize=20)
    plt.yticks(np.arange(0,26,4) - 0.5, range(0,26,4),fontsize=20)

    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.gca().xaxis.grid(which='minor', color='gray', linewidth=1)
    plt.gca().yaxis.grid(which='major', color='gray', linewidth=1)

    plt.ylabel('Weeks to initialisation', fontsize=25)
    plt.ylim([-0.5,28])
    #plt.xlim([-0.5,59.5])

    for i in range(0,int(len(column_names)/5),1):
        plt.axvline(x=4.5+(5*i),color='black',lw='4')
