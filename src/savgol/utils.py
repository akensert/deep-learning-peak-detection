import numpy as np
import scipy.signal
import scipy.linalg
import scipy.sparse


def rescale_deriv(x_deriv):
    '''
    Rescales inputted array by dividing arr with max(abs(array))
    '''
    if not isinstance(x_deriv, np.ndarray):
        x_deriv = np.array(x_deriv)

    return x_deriv / np.max(np.abs(x_deriv))

def find_peaks(x_deriv, height=0.05, distance=None, width=[16, 100]):
    '''
    Local minima search with scipy's find_peaks
    '''
    # use scipy's find_peaks to find local minima. Parameters probably
    # need some optimization.
    loc = scipy.signal.find_peaks(
        -x_deriv, height=height, distance=distance, width=width, rel_height=0.5)[0]
    # remove local minima which is above 0 (perhaps this has to be modified)
    keep_idx = np.where(x_deriv[loc] <= 0)[0]
    return loc[keep_idx]

def savgol_filter(x, window_length=33, polyorder=2, deriv=0):
    '''
    Savitzky-Golay smoothing
    '''
    x_smooth = scipy.signal.savgol_filter(x, window_length, polyorder, 0)
    if deriv > 0:
        # if derivative > 0 is passed to savgol_fitler, the derivative is returned
        x_deriv = scipy.signal.savgol_filter(x, window_length, polyorder, deriv)
        return x_smooth, x_deriv
    return x_smooth

def durbin_watson_criterion(x_orig, x_smooth):
    '''
    The Durbin-Watson value closest to 2 is considered to be the optimal
    '''
    distance_i = x_orig[1:] - x_smooth[1:]
    distance_j = x_orig[:1] - x_smooth[:1]
    oscillation = (distance_i - distance_j) ** 2
    return np.sum(oscillation) / np.sum(distance_i ** 2) * (len(x_orig) / (len(x_orig) - 1))

def asymmetric_least_squares(x, lam=1e9, p=1e-4, n_iter=3):
    '''
    Asymmetric Least-Squares baseline fitting.

    Reference:
        Asymmetric Least Squares Smoothing by P. Eilers and H. Boelens in 2005.

    Notice:
        D = sparse.csc_matrix(np.diff(np.eye(L), 2)) could be used instead.
        However, this dense matrix diff computation could bring about memory issues.
    '''
    L = len(x)
    D = scipy.sparse.diags([1,-2,1],[0,-1,-2], shape=(L, L-2))
    w = np.ones(L)
    for i in range(n_iter):
        W = scipy.sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = scipy.sparse.linalg.spsolve(Z, w*x)
        w = p * (x > z) + (1-p) * (x < z)
    return z
