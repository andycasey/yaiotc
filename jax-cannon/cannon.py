import jax.numpy as np
import jaxopt as opt
from functools import cached_property, partial
from jax import jit, vmap, lax
from time import time
from scipy import linalg
from joblib import Parallel, delayed
from itertools import cycle
from os import cpu_count

class TheCannon:

    def __init__(self, degree=2, **kwargs):
        self.degree = degree
        self.shifts = None
        self.scales = None
        self.coeffs = None
        featurizers = {
            1: _featurize_degree_1,
            2: _featurize_degree_2
        }
        tester = {
            1: _test_degree_1,
            2: _test_degree_2
        }      
        try:
            self.featurizer = featurizers[self.degree]
            self._test = tester[self.degree]
        except KeyError:
            raise ValueError(f"Degree must be one of {tuple(featurizers.keys())}")
        return None


    def train(self, X, W, Y):
        self.X, self.W, self.Y, self.N, self.L, self.K = _check_train_inputs(X, W, Y)

        self.shifts, self.scales = _shifts_and_scales(Y)
        Y_norm = self._normalize(Y)
        
        # Faster than using np.array([...]) and supplying triu_indices.
        self.features = vmap(self.featurizer)(Y_norm) # ~0.03 sec
        
        self.sqrtW = np.sqrt(W)
        self.coeffs = np.array([
            linalg.lstsq(self.features * sqrtW_[:, None], X_ * sqrtW_)[0]
            for X_, sqrtW_ in zip(self.X.T, self.sqrtW.T)
        ]).T # ~ 0.28 sec

        L2 = (self.X - self._predict(Y_norm))**2
        self.s2 = np.clip(np.nanmean(L2 - 1/W, axis=0), 0, np.inf) # ~6.7e-4 sec
        return None


    def test(self, X, W, Y_init=None, scale_bound=10, n_jobs=-1):
        n_jobs = _evaluate_n_jobs(n_jobs)
        if Y_init is None:
            Y_init = cycle([np.zeros(self.K)])

        result = Parallel(n_jobs, prefer="threads")(
            delayed(self._test)(*args, self.coeffs, scale_bound) for args in zip(X, W, Y_init)
        ) # ~1.17 sec
        Y_norm, states = zip(*result)
        Y_norm = np.array(Y_norm)
        return self._denormalize(Y_norm)

    def predict(self, Y):
        Y = np.atleast_2d(Y)
        return self._predict(self._normalize(Y))
    
    def _predict(self, Y_norm):
        return vmap(self.featurizer)(Y_norm) @ self.coeffs

    def _normalize(self, Y):
        return (Y - self.shifts[None, :]) / self.scales[None, :]
    
    def _denormalize(self, Y_norm):
        return Y_norm * self.scales[None, :] + self.shifts[None, :]

def _evaluate_n_jobs(given_n_jobs):
    n_jobs = given_n_jobs or 1
    if n_jobs < 0:
        n_jobs = cpu_count()
    return n_jobs

def _op_l_bfgs_b_bounded(cost, bounds, init):
    # opt.LBFGS(cost).run(Y_init) returned extreme values that were very bad (like +/-1e9)
    # Instead we will bound within a multiple of the scale
    return opt.ScipyBoundedMinimize(fun=cost, method="l-bfgs-b").run(init, bounds=bounds)

# Serializable test functions (because stars are embarrassingly parallel)
def _test_degree_2(X, W, Y_init, coeffs, scale_bound):
    # Note this function *only* deals in normalized-Y, even though the variables are
    # not explicitly defined with _norm suffixes
    sqrtW = np.sqrt(W)    
    def cost(Y):
        chi = (X - _featurize_degree_2(Y) @ coeffs) * sqrtW
        return chi @ chi
    
    bounds = (
        -scale_bound * np.ones_like(Y_init),
        +scale_bound * np.ones_like(Y_init)
    )
    return _op_l_bfgs_b_bounded(cost, bounds, Y_init)

def _test_degree_1(X, W, Y_init, coeffs, scale_bound):
    # Note this function *only* deals in normalized-Y, even though the variables are
    # not explicitly defined with _norm suffixes
    sqrtW = np.sqrt(W)    
    def cost(Y):
        chi = (X - _featurize_degree_1(Y) @ coeffs) * sqrtW
        return chi @ chi
        
    bounds = (
        -scale_bound * np.ones_like(Y_init),
        +scale_bound * np.ones_like(Y_init)
    )
    return _op_l_bfgs_b_bounded(cost, bounds, Y_init)

@jit
def _featurize_degree_1(Y):
    return np.concatenate((np.array([1.0]), Y))

@jit
def _featurize_degree_2(Y):
    ii, jj = np.triu_indices(len(Y))
    Y2 = (Y[:, None] * Y[None, :])[ii, jj]
    return np.concatenate((np.array([1.0]), Y, Y2))


def _check_train_inputs(X, W, Y):
    X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    Y = np.atleast_2d(Y)

    N, L = X.shape
    expect_N, K = Y.shape
    if W.shape != X.shape:
        raise ValueError(f"W shape does not match X shape ({W.shape} != {X.shape})")
    if N != expect_N:
        raise ValueError(f"Y axis 0 length does not match X axis 0 length ({expect_N} != {N})")
    if not np.all(W >= 0):
        raise ValueError("All weights (inverse variances) W must be non-negative (W >= 0)")
    return (X, W, Y, N, L, K)


def _shifts_and_scales(Y, axis=0):
    shifts = np.mean(Y, axis=axis)
    scales = np.std(Y, axis=axis)
    return (shifts, scales)



if __name__ == "__main__":

    import numpy.random as rand
    rng = rand.default_rng(17)    
    
    N, L, K = 42, 1001, 3
    Ybig = rng.normal(size=(N, K))
    beta = rng.normal(size=(L, K))
    Xbig = Ybig @ beta.T + 1. + 0.01 * rng.normal(size=(N, L))
    Wbig = np.zeros_like(Xbig) + 1.e4
    Ybig[:, 0] *= 500.
    Ybig[:, 0] += 4000.
    X = Xbig[:31]
    W = Wbig[:31]
    Y = Ybig[:31]
    Xstar = Xbig[31:]
    Wstar = Wbig[31:]
    Ystar = Ybig[31:]
    print(X.shape, Y.shape, Xstar.shape, Ystar.shape)    

    t_init = time()
    model = TheCannon()
    model.train(X, W, Y)
    t_train = time() - t_init
    print(f"Time to train: {t_train:.1f} seconds")

    t_test = time()
    Ystar_test = model.test(Xstar, Wstar)
    t_test = time() - t_test
    print(f"Time to test: {t_test:.1f} seconds")

    '''
    t_test = time()
    Ystar_test_curve_fit = model.test_curve_fit(Xstar, Wstar)
    t_test = time() - t_test
    print(f"Time to test with curve fit: {t_test:.1f} seconds")
    '''

    from other_cannon import CannonModel


    t_init = time()
    other_model = CannonModel(Y, X, W, "abcdef"[:K])
    other_model.train()
    t_train = time() - t_init
    print(f"Time to train other model: {t_train:.1f} seconds")


    t_init = time()
    other_model.fit_spectrum(Xstar, Wstar)
    t_test = time() - t_init
    print(f"Time to test other model: {t_test:.1f} seconds")

    from thecannon.model import CannonModel as OriginalCannonModel
    from thecannon.vectorizer import PolynomialVectorizer

    vectorizer = PolynomialVectorizer("abcdef"[:K], 2)

    original_threads = 16

    t_init = time()
    original_model = OriginalCannonModel(Y, X, W, vectorizer)
    original_model.train(threads=original_threads)
    t_train = time() - t_init
    print(f"Time to train original model: {t_train:.1f} seconds")

    t_init = time()
    original_model.test(Xstar, Wstar, threads=original_threads)
    t_test = time() - t_init
    print(f"Time to test original model: {t_test:.1f} seconds")



    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(K)
    for k, ax in enumerate(axes):
        ax.scatter(Ystar[:, k], Ystar_test[:, k])
        #ax.scatter(Ystar[:, k], Ystar_test_curve_fit[:, k])

        lims = np.hstack([ax.get_xlim(), ax.get_ylim()]).flatten()
        lims = (np.min(lims), np.max(lims))
        ax.plot(lims, lims, c="#666666", ls=":", lw=0.5, zorder=-1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
