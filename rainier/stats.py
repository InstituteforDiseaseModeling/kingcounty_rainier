""" stats.py

Statistics tools used throughout the project. """
import sys

## Standard imports 
import numpy as np


#### WLS
##############################################################################
def WeightedLeastSquares(X, Y, weights=None, verbose=False, standardize=True):
    """ Weighted LS, reduces to OLS when weights is None. This implementation computes
    the estimator and covariance matrix based on sample variance. For TSIR however, the

    NB: x is assumed to be an array with shape = (num_data points, num_features) """

    ## Get the dimensions
    num_data_points = X.shape[0]
    try:
        num_features = X.shape[1]
    except:
        num_features = 1
        X = X.reshape((num_data_points, 1))

    ## Initialize weight matrix
    if weights is None:
        W = np.eye(num_data_points)
    else:
        W = np.diag(weights)

    ## Standardize the inputs and outputs to help with
    ## stability of the matrix inversion. This is needed because
    ## cumulative cases and births both get very large.
    if standardize:
        muY = Y.mean()
        sigY = Y.std()
        muX = X.mean(axis=0)
        sigX = X.std(axis=0)
        X = (X - muX) / sigX
        Y = (Y - muY) / sigY

    ## Compute the required matrix inversion
    ## i.e. inv(x.T*w*x), which comes from minimizing
    ## the residual sum of squares (RSS) and solving for
    ## the optimum coefficients. See eq. 3.6 in EST
    xTwx_inv = np.linalg.inv(np.dot(X.T, np.dot(W, X)))

    ## Now use that matrix to compute the optimum coefficients
    ## and their uncertainty.
    beta_hat = np.dot(xTwx_inv, np.dot(X.T, np.dot(W, Y)))

    ## Compute the estimated variance in the data points
    residual = Y - np.dot(X, beta_hat)
    RSS = (residual) ** 2
    var = RSS.sum(axis=0) / (num_data_points - num_features)

    ## Then the uncertainty (covariance matrix) is simply a
    ## reapplication of the inv(x.T*x):
    if weights is None:
        beta_var = var * xTwx_inv
    else:
        beta_var = xTwx_inv

    ## Reshape the outputs
    beta_hat = beta_hat.reshape((num_features,))

    ## Rescale back to old values
    if standardize:
        X = sigX * X + muX
        Y = sigY * Y + muY
        beta_hat = beta_hat * (sigY / sigX)
        sig = np.diag(sigY / sigX)
        beta_var = np.dot(sig, np.dot(beta_var, sig))
        residual = sigY * residual + muY - np.dot(muX, beta_hat)

    ## Print summary if needed
    if verbose:
        for i in range(num_features):
            output = (i, beta_hat[i], 2. * np.sqrt(beta_var[i, i]))
            print("Feature %i: coeff = %.4f +/- %.3f." % output)

    return beta_hat, beta_var, residual


def ConstantWLS(x, var):
    """ Avg and var are the avg and variance of samples. Weights are computed with the
    variance, and WLS is used to estimate a fit to a constant. """

    ## Compute weights
    weights = 1. / var / np.sum(1. / var)

    ## Create model based estimates with weighted LS on a
    ## constant.
    weighted_mean = np.sum(x * weights)
    weighted_var = np.sum(weights * ((x - weighted_mean) ** 2))

    return weighted_mean, weighted_var
