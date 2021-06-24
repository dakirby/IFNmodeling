from ifnclass.ifndata import IfnData
from ifnclass.ifnmodel import IfnModel
from ifnclass.ifnplot import Trajectory, TimecoursePlot, DoseresponsePlot
from numpy import linspace, logspace, log10, nan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os


if __name__ == '__main__':
    H_detailed = pickle.load(open('results/Sloppiness/Detailed_Model_Hessian.pkl','rb'))
    H_simple = pickle.load(open('results/Sloppiness/Simplified_Model_Hessian.pkl','rb'))
    # Compute eigenspectrum and normalize by largest eigenvalue
    #   - small eigenvalues correspond to sloppy parameters while large eigenvalues give stiff parameters
    #   - strictly speaking the eigenvalues correspond to the sloppiness along principal axes (eigenvectors) of H
    #   - the width of the error ellipse along the corresponding axis is 1/sqrt(eval)
    evals_d = np.linalg.eigvals(H_detailed)
    evals_d = np.nan_to_num(evals_d)
    normalized_evals_d = np.divide(evals_d, max(evals_d))
    log_normalized_evals_d = log10(np.absolute(normalized_evals_d))

    # removes degenerate eigenvalues arising from unused parameters in model
    log_normalized_evals_d = log_normalized_evals_d[log_normalized_evals_d > -50]

    evals_s = np.linalg.eigvals(H_simple)
    evals_s = np.nan_to_num(evals_s)
    normalized_evals_s = np.divide(evals_s, max(evals_s))
    log_normalized_evals_s = log10(np.absolute(normalized_evals_s))

    # removes degenerate eigenvalues arising from unused parameters in model
    log_normalized_evals_s = log_normalized_evals_s[log_normalized_evals_s > -50]

    # Plot spectra on log scale
    fig, ax = plt.subplots(ncols=2)
    # bins_list = list(range(-10,1)) # integer spaced bins
    sns.distplot(log_normalized_evals_d, kde=False, rug=True, vertical=True, ax=ax[0])  # , hist_kws={"bins": bins_list})
    sns.distplot(log_normalized_evals_s, kde=False, rug=True, vertical=True, ax=ax[1])  # , hist_kws={"bins": bins_list})
    
    ax[0].set_xlabel(r'log($\lambda / \lambda_{max}$)')
    ax[0].set_ylabel('bin count')
    ax[1].set_xlabel(r'log($\lambda / \lambda_{max}$)')
    ax[0].set_title('Detailed Model')
    ax[1].set_title('Simplified Model')
    ax[0].set_ylim(bottom=-9, top=0)
    ax[1].set_ylim(bottom=-9, top=0)

    plt.savefig(os.path.join(os.getcwd(), 'results', 'Sloppiness', 'compare_spectra.pdf'))
    plt.show()



