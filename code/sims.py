"""Summary
"""
import numpy as np
from sklearn import linear_model
import pandas as pd
import scipy


def estimator_1(X_small, y_small, beta, lam):
    reg = linear_model.Ridge(alpha=lam)
    reg.fit(X_small, y_small)
    beta_hat = reg.coef_
    return np.linalg.norm(beta - beta_hat)**2

def estimator_2(X_small, y_small, X_large, y_large, beta, lam):
    X = np.vstack([X_small, X_large])
    y = np.hstack([y_small, y_large])
    reg = linear_model.Ridge(alpha=lam)
    reg.fit(X, y)
    beta_hat = reg.coef_
    return np.linalg.norm(beta - beta_hat)**2

def map_estimator(X_small, y_small, X_large, y_large, cov, sigma):
    m, _ = X_large.shape
    cov_tilde = X_large@cov@X_large.T + sigma**2*np.identity(m)
    chol = np.linalg.cholesky(cov_tilde)
    inv_ct = np.linalg.inv(chol)

    X = np.vstack([1/sigma*X_small, inv_ct@X_large])
    y = np.hstack([1/sigma*y_small, inv_ct@y_large])
    reg = linear_model.RidgeCV()
    reg.fit(X, y)
    return reg.coef_



def generate_models(X_small, X_large, cov, lam):
    """Summary

    Args:
        X_small (array): an nxp numpy array containing the smaller dataset
        X_large (array): an mxp numpy array containing the larger dataset
        cov (array): a pxp numpy array containing the covariance of the bias term
        lam (float): the regularization parameter

    Returns:
        list: a list of the risks associated with three different estimators
    """
    # generate new samples
    n,p = X_small.shape
    m,p = X_large.shape
    beta = np.random.normal(size=p)
    delta = np.random.multivariate_normal(np.zeros(p), cov)
    y_small = X_small @ beta
    y_large = X_large @ (beta + delta)

    # estimate beta
    reg = linear_model.Ridge(alpha=lam)

    beta_1 = reg.fit(X_small, y_small).coef_
    beta_2 = reg.fit(np.vstack([X_small, X_large]), np.hstack([y_small, y_large])).coef_

    # MAP estimator
    cov_tilde = X_large@cov@X_large.T + sigma_sq*np.identity(m)
    chol = np.linalg.cholesky(cov_tilde)
    inv_ct = np.linalg.inv(chol)

    beta_3 = reg.fit(np.vstack([X_small, inv_ct@X_large]), np.hstack([y_small, inv_ct@y_large])).coef_

    # compute errors
    r1 = np.linalg.norm(beta - beta_1)
    r2 = np.linalg.norm(beta-beta_2)
    r3 = np.linalg.norm(beta-beta_3)
    return [r1, r2, r3]

def build_cov(a, m):
    """
    args:
        a (float): the value you want on the off-diagonal in your diagonal block matrix
        m (int): the dimension of the larger dataset (should be even)

    returns:
        array: an mxm covariance matrix
    """
    block = np.array([[1, a], [a, 1]])
    cov = np.kron(np.eye(int(m/2),dtype=int),block)
    return cov


def generate_data(m,n, a, theta):
    # generate data according to banarjee's paper
    cov = build_cov(a, m)
    chol = np.linalg.cholesky(cov)
    p = theta.shape[0]
    X = np.random.normal(size=(m,p))
    eta = np.random.normal(size=m)
    y = X@theta + chol@eta
    for i in range(n-1):
        X_new = np.random.normal(size=(m,p))
        eta = np.random.normal(size=m)
        y_new = X_new@theta + chol@eta

        X = np.vstack([X, X_new])
        y = np.append(y, y_new)
    return X,y


def altmin(X, y, m, n, init='good', niter=10):
    mod = linear_model.RidgeCV()
    mod.fit(X,y)
    if init == 'good':
        theta_new = mod.coef_
    elif type(init)!=str:
        theta_new = init
    else:
        theta_new = np.random.normal(size=X.shape[1])
    rig = linear_model.Ridge(alpha=mod.alpha_)
    for i in range(niter):
        theta_hat = theta_new
        cov_hat = sum([np.outer((y[j:j+m] - X[j:j+m]@theta_hat),(y[j:j+m] - X[j:j+m]@theta_hat) ) for j in np.arange(0, n*m, m)])/n # sample covariance of residuals
        # MAP estimator using cov_hat
        #chol = np.linalg.cholesky(cov_hat)
        chol = np.linalg.cholesky(cov_hat + np.identity(m)*1e-6) #perturb so pd
        inv_ct = np.linalg.inv(chol)
        inv_ct_diag = np.kron(np.eye(int(n),dtype=int),inv_ct)
        theta_new = mod.fit(inv_ct_diag@X, inv_ct_diag@y).coef_
        if np.linalg.norm(theta_hat - theta_new) <= 1e-3:
            return theta_new, cov_hat
    #print('did not converge, error is: ', np.linalg.norm(theta_hat - theta_new))
    return theta_hat, cov_hat


def generate_simobs(nobs, nsim, a, theta, sigma=1):
    p = theta.shape[0]
    # obs
    X_obs = np.random.normal(size=(nobs, p))
    y_obs = X_obs@theta + np.random.normal(size=nobs, scale=sigma)

    # sim
    X_sim = np.random.normal(size=(nsim, p))
    cov = build_cov(a, p)*sigma
    delta = np.random.multivariate_normal(mean=np.zeros(p), cov=cov)
    y_sim = X_sim@(theta + delta) + np.random.normal(size=nsim, scale=sigma)
    return X_obs, y_obs, X_sim, y_sim, cov


def stack_data(X1, X2, y1, y2, how='naive'):
    if how == 'naive':
        XX = np.vstack([X1, X2])
        Y = np.append(y1, y2)
    else:
        nobs = X1.shape[0]
        k = int(X2.shape[0]/nobs)
        XX = np.vstack([X1[0], X2[:k]])
        Y = np.append(y1[0], y2[:k])
        for i in np.arange(1, nobs):
            XX = np.vstack([XX, np.vstack([X1[i], X2[k*i:k*(i+1)]])])
            Y = np.append(Y, np.append(y1[i], y2[k*i:k*(i+1)]))
    return XX, Y
