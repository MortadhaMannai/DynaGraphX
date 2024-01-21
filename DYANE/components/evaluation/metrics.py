import warnings  # 2-dim. KS test
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, cdist  # 2-dim. KS test
from scipy.special import rel_entr
from scipy.stats import genextreme  # 2-dim. KS test
from scipy.stats import kstwobign, pearsonr  # 2-dim. KS test
from sklearn.metrics import accuracy_score


def eval_accuracy(y_true, y_pred, model_name=None):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def eval_spearmans(y, y_hat, model_name=None):
    rho, pval = stats.spearmanr(y, y_hat, axis=None)
    return rho, pval


def eval_kl_div(y, y_hat, model_name=None):
    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html
    kl = rel_entr(y, y_hat)  # avoid extra terms

    return np.mean(kl)


def eval_ks_test(y, y_hat, model_name=None):
    y = np.array(y)
    y_hat = np.array(y_hat)
    num_preds = len(y)

    ks = []
    pvals = []
    for n in range(num_preds):
        ks_n, pval_n = stats.ks_2samp(y[n], y_hat[n])
        ks.append(ks_n)
        pvals.append(pval_n)
    return np.mean(ks), np.mean(pvals)


def eval_ks_test_2dim(true_probs, pred_probs, model_name=None):
    true_probs = np.array(true_probs)
    pred_probs = np.array(pred_probs)
    num_preds = len(true_probs)

    y1 = [np.percentile(true_probs[n], 25, interpolation='midpoint') for n in range(num_preds)]
    y2 = [np.percentile(true_probs[n], 75, interpolation='midpoint') for n in range(num_preds)]

    x1 = [np.percentile(pred_probs[n], 25, interpolation='midpoint') for n in range(num_preds)]
    x2 = [np.percentile(pred_probs[n], 75, interpolation='midpoint') for n in range(num_preds)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pval, ks = ks2d2s(x1, y1, x2, y2, extra=True)
        return ks, pval


def ks2d2s(x1, y1, x2, y2, nboot=None, extra=False):
    '''Two-dimensional Kolmogorov-Smirnov test on two samples.
    Parameters
    ----------
    x1, y1 : ndarray, shape (n1, )
        Data of sample 1.
    x2, y2 : ndarray, shape (n2, )
        Data of sample 2. Size of two samples can be different.
    extra: bool, optional
        If True, KS statistic is also returned. Default is False.
    Returns
    -------
    p : float
        Two-tailed p-value.
    D : float, optional
        KS statistic. Returned if keyword `extra` is True.
    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. Note that the p-value is only an approximation as the analytic distribution is unkonwn. The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. When p-value > 0.20, the value may not be accurate, but it certainly implies that the two samples are not significantly different. (cf. Press 2007)
    References
    ----------
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8
    '''
    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    D = avgmaxdist(x1, y1, x2, y2)

    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]  # correlation in axis=0
        r2 = pearsonr(x2, y2)[0]  # correlation in axis=0
        r = np.sqrt(1 - 0.5 * (r1 ** 2 + r2 ** 2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = np.random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n2:]
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot
    if extra:
        return p, D
    else:
        return p


def avgmaxdist(x1, y1, x2, y2):
    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


def maxdist(x1, y1, x2, y2):
    n1 = len(x1)
    D1 = np.empty((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

    # re-assign the point to maximize difference,
    # the discrepancy is significant for N < ~50
    D1[:, 0] -= 1 / n1

    dmin, dmax = -D1.min(), D1.max() + 1 / n1
    return max(dmin, dmax)


def quadct(x, y, xx, yy):
    n = len(xx)
    ix1, ix2 = xx <= x, yy <= y
    a = np.sum(ix1 & ix2) / n
    b = np.sum(ix1 & ~ix2) / n
    c = np.sum(~ix1 & ix2) / n
    d = 1 - a - b - c
    return a, b, c, d


def estat2d(x1, y1, x2, y2, **kwds):
    return estat(np.c_[x1, y1], np.c_[x2, y2], **kwds)


def estat(x, y, nboot=1000, replace=False, method='log', fitting=False):
    '''
    Energy distance statistics test.
    Reference
    ---------
    Aslan, B, Zech, G (2005) Statistical energy as a tool for binning-free
      multivariate goodness-of-fit tests, two-sample comparison and unfolding.
      Nuc Instr and Meth in Phys Res A 537: 626-636
    Szekely, G, Rizzo, M (2014) Energy statistics: A class of statistics
      based on distances. J Stat Planning & Infer 143: 1249-1272
    Brian Lau, multdist, https://github.com/brian-lau/multdist
    '''
    n, N = len(x), len(x) + len(y)
    stack = np.vstack([x, y])
    stack = (stack - stack.mean(0)) / stack.std(0)
    if replace:
        rand = lambda x: np.random.randint(x, size=x)
    else:
        rand = np.random.permutation

    en = energy(stack[:n], stack[n:], method)
    en_boot = np.zeros(nboot, 'f')
    for i in range(nboot):
        idx = rand(N)
        en_boot[i] = energy(stack[idx[:n]], stack[idx[n:]], method)

    if fitting:
        param = genextreme.fit(en_boot)
        p = genextreme.sf(en, *param)
        return p, en, param
    else:
        p = (en_boot >= en).sum() / nboot
        return p, en, en_boot


def energy(x, y, method='log'):
    dx, dy, dxy = pdist(x), pdist(y), cdist(x, y)
    n, m = len(x), len(y)
    if method == 'log':
        dx, dy, dxy = np.log(dx), np.log(dy), np.log(dxy)
    elif method == 'gaussian':
        raise NotImplementedError
    elif method == 'linear':
        pass
    else:
        raise ValueError
    z = dxy.sum() / (n * m) - dx.sum() / n ** 2 - dy.sum() / m ** 2
    return z


# Code from https://github.com/tongjiyiming/TGGAN
def MMD(X, Y, sigma=-1, SelectSigma=2):
    '''Performs the relative MMD test which returns a test statistic for whether Y is closer to X or than Z.
    See http://arxiv.org/pdf/1511.04581.pdf
    The bandwith heuristic is based on the median heuristic (see Smola,Gretton).
    '''
    Kyy = grbf(Y, Y, sigma)
    Kxy = grbf(X, Y, sigma)
    Kyynd = Kyy - np.diag(np.diagonal(Kyy))
    m = Kxy.shape[0]
    n = Kyy.shape[0]
    u_yy = np.sum(Kyynd) * (1. / (n * (n - 1)))
    u_xy = np.sum(Kxy) / (m * n)

    Kxx = grbf(X, X, sigma)
    Kxxnd = Kxx - np.diag(np.diagonal(Kxx))
    u_xx = np.sum(Kxxnd) * (1. / (m * (m - 1)))
    MMDXY = u_xx + u_yy - 2. * u_xy
    return MMDXY


def grbf(x1, x2, sigma):
    '''Calculates the Gaussian radial base function kernel'''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape

    k1 = np.sum((x1 * x1), 1)
    q = np.tile(k1, (m, 1)).transpose()
    del k1

    k2 = np.sum((x2 * x2), 1)
    r = np.tile(k2.T, (n, 1))
    del k2

    h = q + r
    del q, r

    # The norm
    h = h - 2 * np.dot(x1, x2.transpose())
    h = np.array(h, dtype=float)

    return np.exp(-1. * h / (2. * pow(sigma, 2)))
