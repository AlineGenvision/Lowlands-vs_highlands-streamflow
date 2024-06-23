import numpy as np
from fitter import Fitter
import scipy.stats as sc
import matplotlib.pyplot as plt

from scipy.special import gamma as gamma_function
from apollo import metrics as me


def plot_distribution(y_train, u_of_y, psi, kappa):

    # Check ALIGNMENT
    combined_matrix = np.column_stack((y_train.squeeze(), u_of_y.squeeze(), psi.squeeze()))
    data = combined_matrix[np.argsort(combined_matrix[:, 0])]

    # Create a plot
    plt.figure(figsize=(8, 6))
    plt.plot(data[:, 0], data[:, 1], marker='x', linestyle='-', color='b', label='U(y)')
    plt.plot(data[:, 0], data[:, 2], marker='x', linestyle='-', color='r', label='Psi(y)')
    plt.plot(kappa[0], kappa[1], marker='o', markersize=10, color='r', label='kappa')
    plt.xlabel('Flow (x)')
    plt.ylabel('U_of_y (y)')
    #plt.xlim(0,10)
    plt.grid(True)
    plt.legend()
    plt.show()


def psi_distribution(y_train, mode=None, alpha=1, beta=1, plot=False):

    if mode is None:
        f = Fitter(y_train, distributions=['lognorm', 'gamma'])
        f.fit()
        best_fit = f.get_best(method='sumsquare_error')
        mode = list(best_fit.keys())[0]

    if mode == 'lognorm':

        shape, loc, scale = sc.lognorm.fit(y_train, loc=0)

        # KAPPA (max U(y)) calculation for LOGNORM
        kappa_max_x = loc + scale * np.exp(- shape ** 2)
        kappa_max_y = sc.lognorm.pdf(kappa_max_x, s=shape, scale=scale, loc=loc)
        # ALTERNATIVE
        # kappa_direct = np.exp(shape**2/2)/(shape* scale*np.sqrt(2*np.pi))
        u_of_y = sc.lognorm.pdf(y_train, s=shape, scale=scale, loc=kappa_max_y)

    elif mode == 'gamma':

        shape, loc, scale = sc.gamma.fit(y_train)

        # KAPPA (max U(y)) calculation for GAMMA
        kappa_max_x = loc + (shape - 1) * scale / gamma_function(shape)
        kappa_max_y = sc.gamma.pdf(kappa_max_x, a=shape, scale=scale, loc=loc)
        u_of_y = sc.gamma.pdf(y_train, a=shape, scale=scale, loc=loc)

    psi = me.RELossWeight(u_of_y, alpha=alpha, beta=beta, kappa=kappa_max_y)

    if plot is True:
        plot_distribution(y_train, u_of_y, psi, (kappa_max_x, kappa_max_y))

    return psi