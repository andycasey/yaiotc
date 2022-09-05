import os
import numpy as np
import pickle
import warnings
from itertools import cycle
from functools import cached_property
from scipy import optimize as op
from sklearn.linear_model import Lasso, LinearRegression
from joblib import Parallel, delayed
from time import time
from tqdm import tqdm

from sklearn.exceptions import ConvergenceWarning

#from astra.utils import expand_path


class CannonModel:

    """
    A second-order polynomial Cannon model.

    The generative model for two labels (teff, logg) might look something like:

        f(\theta) = \theta_0
                  + (\theta_1 * teff)
                  + (\theta_2 * teff^2)
                  + (\theta_3 * logg)
                  + (\theta_4 * logg * teff)
                  + (\theta_5 * logg^2)
    """

    def __init__(
        self,
        training_labels,
        training_flux,
        training_ivar,
        label_names,
        dispersion=None,
        regularization=0,
        **kwargs,
    ) -> None:
        (
            self.label_names,
            self.training_labels,
            self.training_flux,
            self.training_ivar,
            self.offsets,
            self.scales,
        ) = _check_inputs(
            label_names, training_labels, training_flux, training_ivar, **kwargs
        )
        self.dispersion = dispersion
        self.regularization = regularization

        # If we are loading from a pre-trained model.
        self.theta = kwargs.get("theta", None)
        self.s2 = kwargs.get("s2", None)
        self.meta = kwargs.get("meta", {})
        return None

    @property
    def design_matrix(self):
        return _design_matrix(
            _normalize(self.training_labels, self.offsets, self.scales),
            self._design_matrix_indices,
        )

    @cached_property
    def _design_matrix_indices(self):
        return _design_matrix_indices(len(self.label_names))

    def train(
        self,
        hide_warnings=True,
        tqdm_kwds=None,
        n_threads=-1,
        prefer="processes",
        **kwargs,
    ):
        """
        Train the model.

        :param hide_warnings: [optional]
            Hide convergence warnings (default: True). Any convergence warnings will be recorded in
            `model.meta['warnings']`, which can be accessed after training.

        :param tqdm_kwds: [optional]
            Keyword arguments to pass to `tqdm` (default: None).
        """

        # Calculate design matrix without bias term, using normalized labels
        X = self.design_matrix[:, 1:]
        flux, ivar = self.training_flux, self.training_ivar
        N, L = X.shape
        N, P = flux.shape

        _tqdm_kwds = dict(total=P, desc="Training")
        _tqdm_kwds.update(tqdm_kwds or {})

        n_threads = _evaluate_n_threads(n_threads)
        args = (X, self.regularization, hide_warnings)
                

        t_init = time()
        results = Parallel(n_threads, prefer=prefer)(
            delayed(_fit_pixel)(p, Y, W, *args, **kwargs)
            for p, (Y, W) in tqdm(enumerate(zip(flux.T, ivar.T)), **_tqdm_kwds)
        )
        t_train = time() - t_init
        
        self.theta = np.zeros((1 + L, P))
        self.meta.update(
            t_train=t_train,
            train_warning=np.zeros(P, dtype=bool),
            n_iter=np.zeros(P, dtype=int),
            dual_gap=np.zeros(P, dtype=float),
        )
        for index, pixel_theta, meta in results:
            self.theta[:, index] = pixel_theta
            self.meta["train_warning"][index] = meta.get("warning", False)
            self.meta["n_iter"][index] = meta.get("n_iter", -1)
            self.meta["dual_gap"][index] = meta.get("dual_gap", np.nan)

        # Calculate the model variance given the trained coefficients.
        self.s2 = self._calculate_s2()
        return self

    def _calculate_s2(self, SMALL=1e-12):
        """Calculate the model variance (s^2)."""

        L2 = (self.training_flux - self.predict(self.training_labels)) ** 2
        mask = self.training_ivar > 0
        inv_W = np.zeros_like(self.training_ivar)
        inv_W[mask] = 1 / self.training_ivar[mask]
        inv_W[~mask] = SMALL
        # You want the mean, not the median.
        return np.clip(np.mean(L2 - inv_W, axis=0), 0, np.inf)

    @property
    def trained(self):
        """Boolean property defining whether the model is trained."""
        return self.theta is not None and self.s2 is not None

    def write(self, path, save_training_set=False, overwrite=False):
        """
        Write the model to disk.

        :param path:
            The path to write the model to.

        :param save_training_set: [optional]
            Include the training set in the saved model (default: False).
        """
        full_path = expand_path(path)
        if os.path.exists(full_path) and not overwrite:
            raise FileExistsError(f"File {full_path} already exists.")
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        if not save_training_set and not self.trained:
            raise ValueError(
                "Nothing to save: model not trained and save_training_set is False"
            )

        keys = [
            "theta",
            "s2",
            "meta",
            "dispersion",
            "regularization",
            "label_names",
            "offsets",
            "scales",
        ]
        if save_training_set:
            keys += ["training_labels", "training_flux", "training_ivar"]

        state = {k: getattr(self, k) for k in keys}
        with open(path, "wb") as fp:
            pickle.dump(state, fp)
        return True

    @classmethod
    def read(cls, path):
        """Read a model from disk."""

        full_path = expand_path(path)
        with open(full_path, "rb") as fp:
            state = pickle.load(fp)
        # if there's no training data, just give Nones
        for k in ("training_labels", "training_flux", "training_ivar"):
            state.setdefault(k, None)
        return cls(**state)

    @property
    def term_descriptions(self):
        """Return descriptions for all the terms in the design matrix."""
        js, ks = _design_matrix_indices(len(self.label_names))
        terms = []
        for j, k in zip(js, ks):
            if j == 0 and k == 0:
                terms.append(1)
            else:
                term = []
                if j > 0:
                    term.append(self.label_names[j - 1])
                if k > 0:
                    term.append(self.label_names[k - 1])
                terms.append(tuple(term))
        return terms

    @property
    def term_type_indices(self):
        """
        Returns a three-length tuple that contains:
        - indices of linear terms in the design matrix
        - indices of quadratic terms in the design matrix
        - indices of cross-terms in the design matrix
        """
        js, ks = _design_matrix_indices(len(self.label_names))
        indices = [[], [], []]
        for i, (j, k) in enumerate(zip(js, ks)):
            if j == 0 and k == 0:
                continue

            if min(j, k) == 0 and max(j, k) > 0:
                # linear term
                indices[0].append(i)
            elif j > 0 and j == k:
                # quadratic term
                indices[1].append(i)
            else:
                # cross-term
                indices[2].append(i)
        return indices

    def predict(self, labels):
        """
        Predict spectra, given some labels.
        """
        try:
            N, L = labels.shape
        except:
            labels = np.atleast_2d(labels)
        L = _normalize(labels, self.offsets, self.scales)
        return _design_matrix(L, self._design_matrix_indices) @ self.theta

    def chi_sq(self, labels, flux, ivar, aggregate=np.sum):
        """
        Return the total \chi^2 difference of the expected flux given the labels, and the observed
        flux. The total inverse variance (model and observed) is used to weight the \chi^2 value.

        :param labels:
            An array of stellar labels with shape `(n_spectra, n_labels)`.

        :param flux:
            An array of observed flux values with shape `(n_spectra, n_pixels)`.

        :param ivar:
            An array containing the inverse variance of the observed flux, with shape `(n_spectra, n_pixels)`.
        """
        adjusted_ivar = ivar / (1.0 + ivar * self.s2)
        return aggregate(adjusted_ivar * (self.predict(labels) - flux) ** 2)

    def reduced_chi_sq(self, labels, flux, ivar, aggregate=np.sum):
        nu = aggregate(ivar > 0) - labels.size
        return self.chi_sq(labels, flux, ivar, aggregate) / nu

    def fit_spectrum(
        self,
        flux,
        ivar,
        x0=None,
        frozen=None,
        continuum_order=-1,
        n_threads=None,
        prefer="processes",
        tqdm_kwds=None,
    ):
        """
        Return the stellar labels given the observed flux and inverse variance.

        :param flux:
            An array of observed flux values with shape `(n_spectra, n_pixels)`.

        :param ivar:
            An array containing the inverse variance of the observed flux, with shape `(n_spectra, n_pixels)`.

        :param x0: [optional]
            An array of initial values for the stellar labels with shape `(n_spectra, n_labels)`. If `None`
            is given (default) then the initial guess will be estimated by linear algebra.

        :param frozen: [optional]
            A dictionary with labels as keys and values of arrays that indicate the value to be frozen for each
            spectrum. For example, if `frozen = {"Teff": [0.5, 1.5]}` then the Teff label will be fixed to 0.5
            for the first spectrum and 1.5 for the second spectrum. If you supply `frozen`, then it is assumed
            that you will freeze the given variables for every spectrum. In other words, don't freeze a label
            for 9 spectra and hope to leave one free. Instead, just make a second call to this function.

        :param tqdm_kwds: [optional]
            Keyword arguments to pass to `tqdm` (default: None).
        """
        P = self.s2.size
        try:
            N, P = flux.shape
        except:
            try:
                N = int(flux.size / P)
            except:
                N = len(flux)

        L = len(self.label_names)

        _tqdm_kwds = dict(total=N, desc="Fitting")
        _tqdm_kwds.update(tqdm_kwds or {})

        if x0 is None:
            x0 = cycle([None])

        # Freeze values as necessary.
        frozen_values = np.nan * np.ones((N, L))
        if frozen is not None:
            for label_name, values in frozen.items():
                index = self.label_names.index(label_name)
                frozen_values[:, index] = values

        args = (
            self.theta,
            self._design_matrix_indices,
            self.s2,
            self.offsets,
            self.scales,
            continuum_order,
        )

        iterable = tqdm(zip(flux, ivar, x0, frozen_values), **_tqdm_kwds)

        n_threads = _evaluate_n_threads(n_threads)
        if N == 1 or n_threads in (0, 1, None):
            results = [_fit_spectrum(*data, *args) for data in iterable]
        else:
            results = Parallel(n_threads, prefer=prefer)(
                delayed(_fit_spectrum)(*data, *args) for data in iterable
            )

        # Aggregate nicely.
        K = L
        if continuum_order > -1:
            K += 1 + continuum_order

        all_labels = np.empty((N, K))
        all_cov = np.empty((N, K, K))
        all_meta = []
        for i, (labels, cov, meta) in enumerate(results):
            all_labels[i, :] = labels
            all_cov[i, :] = cov
            all_meta.append(meta)

        return (all_labels, all_cov, all_meta)

    def initial_estimate(self, flux, only_labels=None, clip_sigma=None):
        """
        Return an initial guess of the labels given a spectrum.

        :param flux:
            A (N, P) shape array of N spectra with P pixels.

        :param only_labels: [optional]
            A tuple containing label names to estimate labels for. If given, estimates will only be
            made for these label names.

            The resulting array will be ordered in the same order given by this tuple.
            For example, if your model has label names in the order ('teff', 'logg', 'fe_h')
            and you provide N spectra and give `only_labels=('fe_h', 'logg')` then you will
            get a Nx2 array of labels, where the first column is 'fe_h' and the second is 'logg'.
        """
        P = self.s2.size
        B = (flux - self.theta[0]).T
        # The 1: index here is to ignore the bias term.
        linear_term_indices = np.where((self._design_matrix_indices[1] == 0)[1:])[0]
        offsets = np.copy(self.offsets)
        scales = np.copy(self.scales)
        if only_labels is not None:
            theta_mask = np.array(
                [
                    (term != 1) and all([t in only_labels for t in term])
                    for term in self.term_descriptions
                ]
            )
            A = self.theta[theta_mask].T
            idx_label = np.array([self.label_names.index(ln) for ln in only_labels])
            offsets, scales = (offsets[idx_label], scales[idx_label])
            linear_term_indices = linear_term_indices[idx_label]
        else:
            A = self.theta[1:].T

        return __initial_estimate(
            A, B, linear_term_indices, offsets, scales, clip_sigma, normalize=False
        )


def _design_matrix_indices(L):
    return np.tril_indices(1 + L)


def _design_matrix(labels, idx):
    N, L = labels.shape
    # idx = _design_matrix_indices(L)
    iterable = np.hstack([np.ones((N, 1)), labels])[:, np.newaxis]
    return np.vstack([l.T.dot(l)[idx] for l in iterable])


def _initial_guess(flux, theta, idx, offsets, scales, **kwargs):
    B = (flux - theta[0]).T
    A = theta[1:].T
    linear_term_indices = np.where((idx[1] == 0)[1:])[0]
    return __initial_estimate(A, B, linear_term_indices, offsets, scales, **kwargs)


def __initial_estimate(
    A, B, linear_term_indices, offsets, scales, clip_sigma=None, normalize=False
):
    try:
        X, residuals, rank, singular = np.linalg.lstsq(A, B, rcond=-1)
    except np.linalg.LinAlgError:
        warnings.warn("Unable to make initial label estimate.")
        return np.zeros_like(offsets) if normalize else offsets
    else:
        x0 = X[linear_term_indices].T  # offset by 1 to skip missing bias term
        if clip_sigma is not None:
            x0 = np.clip(x0, -clip_sigma, +clip_sigma)
        return x0 if normalize else _denormalize(x0, offsets, scales)


def _get_continuum_x(F):
    S = -int(F / 2)
    return np.arange(S, S + F)


def _predict_flux(
    x,
    parameters,
    continuum_order,
    theta,
    idx,
    normalized_frozen_values,
    is_frozen,
    any_frozen,
):
    if continuum_order >= 0:
        thawed_labels = parameters[: -(continuum_order + 1)]
        continuum = np.polyval(parameters[-(continuum_order + 1) :], x)
    else:
        thawed_labels = parameters
        continuum = 1

    if any_frozen:
        labels = np.copy(normalized_frozen_values)
        labels[~is_frozen] = parameters[: sum(~is_frozen)]
    else:
        labels = thawed_labels

    l = np.atleast_2d(np.hstack([1, labels]))
    A = l.T.dot(l)[idx][np.newaxis]
    return continuum * (A @ theta)[0]


def _fit_spectrum(
    flux, ivar, x0, frozen_values, theta, idx, s2, offsets, scales, continuum_order=-1
):

    # NOTE: Here the design matrix is calculated with *DIFFERENT CODE* than what is used
    #       to construct the design matrix during training. The result should be exactly
    #       the same, but here we are taking advantage of not having to call np.tril_indices
    #       with every log likelihood evaluation.
    K = continuum_order + 1
    x = _get_continuum_x(flux.size)
    L = offsets.size

    is_frozen = np.isfinite(frozen_values)
    normalized_frozen_values = _normalize(frozen_values, offsets, scales)
    any_frozen = np.any(is_frozen)

    args = (
        continuum_order,
        theta,
        idx,
        normalized_frozen_values,
        is_frozen,
        any_frozen,
    )
    sigma = (ivar / (1.0 + ivar * s2)) ** -0.5

    meta = dict()
    if x0 is None:
        # no continuum for initial values
        # use combinations_with_replacement to sample (-1, 0, 1)?
        x0_normalized_trials = [
            np.zeros(L),
            +np.ones(L),
            -np.ones(L),
            _initial_guess(flux, theta, idx, offsets, scales, normalize=True),
        ]
        chi_sqs = []
        for x0_trial in x0_normalized_trials:
            x0_ = x0_trial[~is_frozen]
            chi_sqs.append(
                np.sum(((_predict_flux(x, x0_, -1, *args[1:]) - flux) / sigma) ** 2)
            )
        x0_normalized = x0_normalized_trials[np.argmin(chi_sqs)]
        x0 = _denormalize(x0_normalized, offsets, scales)
        meta["trial_x0"] = np.array(x0_normalized_trials) * scales + offsets
        meta["trial_chisq"] = np.array(chi_sqs)
    else:
        x0_normalized = _normalize(x0, offsets, scales)

    meta["x0"] = np.copy(frozen_values)
    meta["x0"][~is_frozen] = x0[~is_frozen]

    p0 = x0_normalized[~is_frozen]

    if continuum_order >= 0:
        p0 = np.hstack([p0, np.zeros(1 + continuum_order)])
        p0[-1] = 1

    try:
        p_opt_all, cov_norm = op.curve_fit(
            lambda x, *parameters: _predict_flux(x, parameters, *args),
            x,
            flux,
            p0=p0,
            sigma=sigma,
            absolute_sigma=True,
            maxfev=10_000,
        )
        model_flux = _predict_flux(x, p_opt_all, *args)
    except:
        N, P = np.atleast_2d(flux).shape
        p_opt = np.nan * np.ones(L)
        cov = np.nan * np.ones((L, L))
        meta.update(
            chi_sq=np.nan, reduced_chi_sq=np.nan, model_flux=np.nan * np.ones(P)
        )
    else:
        if continuum_order >= 0:
            p_opt_norm = p_opt_all[:-K]
        else:
            p_opt_norm = p_opt_all
        p_opt = np.copy(frozen_values)
        p_opt[~is_frozen] = _denormalize(
            p_opt_norm, offsets[~is_frozen], scales[~is_frozen]
        )

        if any_frozen:
            cov = np.nan * np.ones((L, L))  # TODO: deal with freezing
        else:
            cov = (
                cov_norm[:L, :L] * scales**2
            )  # TODO: define this with _normalize somehow

        # WARNING: Here we are calculating chi-sq with *DIFFERENT CODE* than what is used elsewhere.
        chi_sq = np.sum(((flux - model_flux) / sigma) ** 2)
        nu = np.sum(np.isfinite(sigma)) - L
        reduced_chi_sq = chi_sq / nu
        meta.update(
            chi_sq=chi_sq,
            reduced_chi_sq=reduced_chi_sq,
            p_opt_norm=p_opt_norm,
            p_opt_cont=p_opt_all[-K:],
            model_flux=model_flux,
        )
    finally:
        return (p_opt, cov, meta)


def _fit_pixel(index, Y, W, X, alpha, hide_warnings=True, **kwargs):
    N, T = X.shape
    if np.allclose(W, np.zeros_like(W)):
        return (index, np.zeros(1 + T), 0, {})

    if alpha == 0:
        kwds = dict(**kwargs)  # defaults
        lm = LinearRegression(**kwds)
    else:
        # defaults:
        kwds = dict(max_iter=20_000, tol=1e-10, precompute=True)
        kwds.update(**kwargs)  # defaults
        lm = Lasso(alpha=alpha, **kwds)

    args = (X, Y, W)
    t_init = time()
    if hide_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            lm.fit(*args)
    else:
        lm.fit(*args)
    t_fit = time() - t_init

    theta = np.hstack([lm.intercept_, lm.coef_])
    meta = dict(t_fit=t_fit)
    for attribute in ("n_iter", "dual_gap"):
        try:
            meta[attribute] = getattr(lm, f"{attribute}_")
        except:
            continue

    if "n_iter" in meta:
        meta["warning"] = meta["n_iter"] >= lm.max_iter

    return (index, theta, meta)


def _check_inputs(label_names, labels, flux, ivar, offsets=None, scales=None, **kwargs):
    label_names = list(label_names)
    if len(label_names) > len(set(label_names)):
        raise ValueError(f"Label names must be unique!")

    if labels is None and flux is None and ivar is None:
        # Try to get offsets and scales from kwargs
        # offsets, scales = (.get(k, None) for k in ("offsets", "scales"))
        if offsets is None or scales is None:
            print(f"No training set labels given, and no offsets or scales provided!")
            offsets, scales = (0, 1)
        return (label_names, labels, flux, ivar, offsets, scales)
    L = len(label_names)
    labels = np.atleast_2d(labels)
    flux = np.atleast_2d(flux)
    ivar = np.atleast_2d(ivar)

    N_0, L_0 = labels.shape
    N_1, P_1 = flux.shape
    N_2, P_2 = ivar.shape

    if L_0 != L:
        raise ValueError(
            f"{L} label names given but input labels has shape {labels.shape} and should be (n_spectra, n_labels)"
        )

    if N_0 != N_1:
        raise ValueError(
            f"labels should have shape (n_spectra, n_labels) and flux should have shape (n_spectra, n_pixels) "
            f"but labels has shape {labels.shape} and flux has shape {flux.shape}"
        )
    if N_1 != N_2 or P_1 != P_2:
        raise ValueError(
            f"flux and ivar should have shape (n_spectra, n_pixels) "
            f"but flux has shape {flux.shape} and ivar has shape {ivar.shape}"
        )

    if L_0 > N_0:
        raise ValueError(f"I don't believe that you have more labels than spectra")

    # Restrict to things that are fully sampled.
    # good = np.all(ivar > 0, axis=0)
    # ivar = np.copy(ivar)
    # ivar[:, ~good] = 0

    # Calculate offsets and scales.
    offsets, scales = _offsets_and_scales(labels)
    if not np.all(np.isfinite(offsets)):
        raise ValueError(f"offsets are not all finite: {offsets}")
    if len(offsets) != L:
        raise ValueError(f"{len(offsets)} offsets given but {L} are needed")

    if not np.all(np.isfinite(scales)):
        raise ValueError(f"scales are not all finite: {scales}")
    if len(scales) != L:
        raise ValueError(f"{len(scales)} scales given but {L} are needed")

    return (label_names, labels, flux, ivar, offsets, scales)


def _offsets_and_scales(labels):
    return (np.mean(labels, axis=0), np.std(labels, axis=0))


def _normalize(labels, offsets, scales):
    return (labels - offsets) / scales


def _denormalize(labels, offsets, scales):
    return labels * scales + offsets


def _evaluate_n_threads(given_n_threads):
    n_threads = given_n_threads or 1
    if n_threads < 0:
        n_threads = os.cpu_count()
    return n_threads