from typing import Callable

from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
from scipy import sparse
import scipy as sp
import numpy as np


def arpls(y, lam, ratio, itermax) -> np.ndarray:
    r"""
    Baseline correction using asymmetrically
    reweighed penalized least squares smoothing
    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015)

    Abstract

    Baseline correction methods based on penalized least squares are successfully
    applied to various spectral analyses. The methods change the weights iteratively
    by estimating a baseline. If a signal is below a previously fitted baseline,
    large weight is given. On the other hand, no weight or small weight is given
    when a signal is above a fitted baseline as it could be assumed to be a part
    of the peak. As noise is distributed above the baseline as well as below the
    baseline, however, it is desirable to give the same or similar weights in
    either case. For the purpose, we propose a new weighting scheme based on the
    generalized logistic function. The proposed method estimates the noise level
    iteratively and adjusts the weights correspondingly. According to the
    experimental results with simulated spectra and measured Raman spectra, the
    proposed method outperforms the existing methods for baseline correction and
    peak height estimation.

    :param y: input data (i.e. chromatogram of spectrum)
    :param lam: parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
    :param ratio: weighting deviations: 0 < ratio < 1, smaller values allow less negative values
    :param itermax: number of iterations to perform
    :return: the fitted background vector

    """
    assert itermax > 0, f"itermax parameter must be greater than 0, but is {itermax}"

    z = None

    N = len(y)
    D = sp.sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]

    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sp.sparse.diags(w, 0, shape=(N, N))
        WH = sp.sparse.csc_matrix(W + H)
        cholesky_matrix = cholesky(WH.todense())
        C = sparse.csc_matrix(cholesky_matrix)
        fsolve = sparse.linalg.spsolve(C.T, w * y.astype(np.float64))
        z = sparse.linalg.spsolve(C, fsolve)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z


def als(y, lam, ratio, itermax) -> np.ndarray:
    r"""
    edit
    Implements an Asymmetric Least Squares Smoothing
    baseline correction algorithm (P. Eilers, H. Boelens 2005)

    Baseline Correction with Asymmetric Least Squares Smoothing
    based on https://github.com/vicngtor/BaySpecPlots

    Baseline Correction with Asymmetric Least Squares Smoothing
    Paul H. C. Eilers and Hans F.M. Boelens
    October 21, 2005

    Description from the original documentation:

    Most baseline problems in instrumental methods are characterized by a smooth
    baseline and a superimposed signal that carries the analytical information: a series
    of peaks that are either all positive or all negative. We combine a smoother
    with asymmetric weighting of deviations from the (smooth) trend get an effective
    baseline estimator. It is easy to use, fast and keeps the analytical peak signal intact.
    No prior information about peak shapes or baseline (polynomial) is needed
    by the method. The performance is illustrated by simulation and applications to
    real data.

    :param y: input data (i.e. chromatogram of spectrum)
    :param lam: parameter that can be adjusted by user. The larger lambda is,
                the smoother the resulting background, z
    :param ratio: weighting deviations: 0 < ratio < 1, smaller values allow less negative values
    :param itermax: number of iterations to perform
    :return: the fitted background vector

    """

    z = None

    L = len(y)
    # D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    D = sparse.eye(L, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]
    D = D.T
    w = np.ones(L)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w * y.astype(np.float64))
        w = ratio * (y > z) + (1 - ratio) * (y < z)
    return z


def algorithm_by_index(index: int) -> Callable:
    return [arpls, als][index]


def correct(data: np.ndarray, fkt: callable, **params):
    baseline = fkt(data, **params)

    return data - baseline, baseline


if __name__ == '__main__':
    print(algorithm_by_index(1))
