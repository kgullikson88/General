"""
A set of functions for various types of fitting.
"""

from scipy.optimize import fmin
import logging

import statsmodels.api as sm
from statsmodels.robust.norms import TukeyBiweight
import numpy as np
import DataStructures
from lmfit import Model, Parameters
import FittingUtilities


try:
    import emcee

    emcee_import = True
except ImportError:
    logging.warn("emcee module not loaded! BayesFit and bayesian_total_least_squares are unavailable!")
    emcee_import = False


def RobustFit(x, y, fitorder=3, weight_fcn=TukeyBiweight()):
    """
    Performs a robust fit (less sensitive to outliers) to x and y
    :param x: A numpy.ndarray with the x-coordinates of the function to fit
    :param y: A numpy.ndarray with the y-coordinates of the function to fit
    :param fitorder: The order of the fit
    :return:
    """
    # Re-scale x for stability
    x = (x - x.mean()) / x.std()
    X = np.ones(x.size)
    for i in range(1, fitorder + 1):
        X = np.column_stack((X, x ** i))
    fitter = sm.RLM(y, X, M=weight_fcn)
    results = fitter.fit()
    return results.predict(X)


if emcee_import:
    def BayesFit(data, model_fcn, priors, limits=None, burn_in=100, nwalkers=100, nsamples=100, nthreads=1,
                 full_output=False, a=2):
        """
        This function will do a Bayesian fit to the model. Warning! I don't think it quite works yet!

        Parameter description:
          data:         A DataStructures.xypoint instance containing the data
          model_fcn:    A function that takes an x-array and parameters,
                           and returns a y-array. The number of parameters
                           should be the same as the length of the 'priors'
                           parameter
          priors:       Either a 2d np array or a list of lists. Each index
                           should contain the expected value and the uncertainty
                           in that value (assumes all Gaussian priors!).
          limits:       If given, it should be a list of the same shape as
                           'priors', giving the limits of each parameter
          burn_in:      The burn-in period for the MCMC before you start counting
          nwalkers:     The number of emcee 'walkers' to use.
          nsamples:     The number of samples to use in the MCMC sampling. Note that
                            the actual number of samples is nsamples * nwalkers
          nthreads:     The number of processing threads to use (parallelization)
                            This probably needs MPI installed to work. Not sure though...
          full_ouput:   Return the full sample chain instead of just the mean and
                            standard deviation of each parameter.
          a:            See emcee.EnsembleSampler. Basically, it controls the step size
        """

        # Priors needs to be a np array later, so convert to that first
        priors = np.array(priors)

        # Define the likelihood, prior, and posterior probability functions
        likelihood = lambda pars, data, model_fcn: np.sum(
            -(data.y - model_fcn(data.x, *pars)) ** 2 / (2.0 * data.err ** 2))
        if limits == None:
            prior = lambda pars, priors: np.sum(-(pars - priors[:, 0]) ** 2 / (2.0 * priors[:, 1] ** 2))
            posterior = lambda pars, data, model_fcn, priors: likelihood(pars, data, model_fcn) + prior(pars, priors)
        else:
            limits = np.array(limits)
            prior = lambda pars, priors, limits: -9e19 if any(
                np.logical_or(pars < limits[:, 0], pars > limits[:, 1])) else np.sum(
                -(pars - priors[:, 0]) ** 2 / (2.0 * priors[:, 1] ** 2))
            posterior = lambda pars, data, model_fcn, priors, limits: likelihood(pars, data, model_fcn) + prior(pars,
                                                                                                                priors,
                                                                                                                limits)


        # Set up the MCMC sampler
        ndim = priors.shape[0]
        if limits == None:
            p0 = [np.random.normal(loc=priors[:, 0], scale=priors[:, 1]) for i in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, threads=nthreads, args=(data, model_fcn, priors),
                                            a=4)
        else:
            ranges = np.array([l[1] - l[0] for l in limits])
            p0 = [np.random.rand(ndim) * ranges + limits[:, 0] for i in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior, threads=nthreads,
                                            args=(data, model_fcn, priors, limits), a=a)

        # Burn-in the sampler
        pos, prob, state = sampler.run_mcmc(p0, burn_in)

        # Reset the chain to remove the burn-in samples.
        sampler.reset()

        # Run the sampler
        pos, prob, state = sampler.run_mcmc(pos, nsamples, rstate0=state)

        print "Acceptance fraction = %f" % np.mean(sampler.acceptance_fraction)
        maxprob_indice = np.argmax(prob)
        priors[:, 0] = pos[maxprob_indice]
        # Get the parameter estimates
        chain = sampler.flatchain
        for i in range(ndim):
            priors[i][1] = np.std(chain[:, i])

        if full_output:
            return priors, sampler
        return priors


class ListModel(Model):
    """
    Subclass of lmfit's Model, which can take a list of xypoints.
    The fit method reforms the list into a single array, and then
    passes off to the lmfit method.

    This is very bare bones now (Sep 25, 2014). Will probably need to add more later.
    """

    def __init__(self, func, independent_vars=None, param_names=None,
                 missing='none', prefix='', name=None, **kws):
        Model.__init__(self, func, independent_vars=independent_vars, param_names=param_names,
                       missing=missing, prefix=prefix, name=name, **kws)


    def fit(self, data, fitcont=True, fit_kws=None, **kws):
        x = np.hstack([d.x for d in data])
        y = np.hstack([d.y for d in data])
        w = np.hstack([1.0 / d.err for d in data])
        self.order_lengths = [d.size() for d in data]
        kws['x'] = x
        self.fitcont = fitcont
        output = Model.fit(self, y, weights=w, fit_kws=fit_kws, **kws)

        # Need to re-shape the best-fit
        best_fit = []
        length = 0
        for i in range(len(data)):
            best_fit.append(output.best_fit[length:length + data[i].size()])
            length += data[i].size()
        output.best_fit = best_fit
        return output


    def _residual(self, params, data, weights=None, **kwargs):
        "default residual:  (data-model)*weights"
        # Make sure the parameters are in the right format
        if not isinstance(params, Parameters):
            if 'names' in kwargs:
                parnames = kwargs['names']
            else:
                raise KeyError("Must give the parameter names if the params are just list instances!")
            d = {name: value for name, value in zip(parnames, params)}
            params = self.make_params(**d)
        # print params

        model = Model.eval(self, params, **kwargs)
        length = 0
        loglikelihood = []
        for i, l in enumerate(self.order_lengths):
            x = kwargs['x'][length:length + l]
            y = data[length:length + l]
            m = model[length:length + l]
            if self.fitcont:
                ratio = y / m
                cont = FittingUtilities.Continuum(x, ratio, fitorder=5, lowreject=2, highreject=2)
            else:
                cont = np.ones(x.size)
            loglikelihood.append((y - cont * m))

            length += l

        loglikelihood = np.hstack(loglikelihood)
        if weights is not None:
            loglikelihood *= weights
        return loglikelihood

    def MCMC_fit(self, data, priors, names, prior_type='flat', fitcont=True, model_getter=None, nthreads=1):
        """
        Do a fit using emcee

        :param data: list of xypoints
        :param priors: list of priors (each value must be a 2-D list)
        :param names: The names of the variables, in the same order as the priors list
        :keyword prior_type: The type of prior. Choices are 'flat' or 'gaussian'
        :keyword fitcont: Should we fit the continuum in each step?
        :param nthreads: The number of threads to spawn (for parallelization)
        :return:
        """
        x = np.hstack([d.x for d in data])
        y = np.hstack([d.y for d in data])
        c = np.hstack([d.cont for d in data])
        e = np.hstack([d.err for d in data])
        fulldata = DataStructures.xypoint(x=x, y=y, err=e, cont=c)
        weights = 1.0 / e
        self.order_lengths = [d.size() for d in data]
        self.fitcont = fitcont

        # Define the prior functions
        priors = np.array(priors)
        if prior_type.lower() == 'gauss':
            lnprior = lambda pars, prior_vals: np.sum(-(pars - prior_vals[:, 0]) ** 2 / (2.0 * prior_vals[:, 1] ** 2))
            guess = [p[0] for p in priors]
            scale = [p[1] / 10.0 for p in priors]
        elif prior_type.lower() == 'flat':
            def lnprior(pars, prior_vals):
                tmp = [prior_vals[i][0] < pars[i] < prior_vals[i][1] for i in range(len(pars))]
                return 0.0 if all(tmp) else -np.inf

            guess = [(p[0] + p[1]) / 2.0 for p in priors]
            scale = [(p[1] - p[0]) / 100.0 for p in priors]
        else:
            raise ValueError("prior_type must be one of 'gauss' or 'flat'")

        # Define the full probability functions
        def lnprob(pars, priors, data, weights, **kwargs):
            lp = lnprior(pars, priors)
            if not np.isfinite(lp):
                return -np.inf
            return lp + np.sum(self._residual(pars, data, weights, **kwargs) ** 2)


        # Set up the emcee sampler
        ndim = len(priors)
        nwalkers = 100
        pars = np.array(guess)
        pos = [pars + scale * np.random.randn(ndim) for i in range(nwalkers)]
        if model_getter is None:
            model_getter = self.opts['model_getter']
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(priors, fulldata.y, weights),
                                        kwargs={'model_getter': model_getter, 'names': names, 'x': x},
                                        threads=nthreads)

        return sampler, pos


# #################################################################
#    A set of functions for bayesian total least squares         #
##################################################################

def tls_model(p, x):
    """
    Parameterized model for y = f(x). For now, just a polynomial
    """
    return np.poly1d(p)(x)


def tls_lnlike(pars, x, y, xerr, yerr, model_fcn):
    """
    Likelihood function.
    pars:  the parameters. the first several are the 'true' x values
    x: the measured x values
    y: the measured y values
    xerr: the uncertainty in the measured x values
    yerr: the uncertainty in the measured y values
    model_fcn: A function that gives the model y = f(x)
               The model function should take the rest of the pars as the first argument.
    """
    xtrue = pars[:x.size]
    y_pred = model_fcn(pars[x.size:], xtrue)  # Predict the y value

    # Make the log-likelihood
    return np.sum(-(x - xtrue) ** 2 / xerr ** 2 - (y - y_pred) ** 2 / yerr * 2)


def tls_lnprior(pars):
    return 0.0


def tls_lnprob(pars, x, y, xerr, yerr, model_fcn):
    lp = tls_lnprior(pars)
    return lp + tls_lnlike(pars, x, y, xerr, yerr, model_fcn) if np.isfinite(lp) else -np.inf


def tls_get_initial_fit(x, y, yerr, model_fcn, fitorder=3):
    pars = np.zeros(fitorder + 1)
    pars[-2] = 1.0
    min_func = lambda p, xi, yi, yerri: np.sum((yi - model_fcn(p, xi)) ** 2 / yerri ** 2)

    best_pars = fmin(min_func, x0=pars, args=(x, y, yerr))
    return best_pars


if emcee_import:
    def bayesian_total_least_squares(x, y, xerr, yerr, fitorder=1, nwalkers=100, n_burn=200, n_prod=1000):
        """
        Perform a bayesian total least squares fit to data with errors in both the x- and y-axes.
        :param x:  A numpy ndarray with the independent variable
        :param y:  A numpy ndarray with the dependent variable
        :param xerr:  A numpy ndarray with the uncertainty in the independent variable
        :param yerr:  A numpy ndarray with the uncertainty in the dependent variable
        :param fitorder:  The polynomial fit order. Default = 1 (linear)
        :param nwalkers:  The number of walkers to use in the MCMC sampler
        :param n_burn:   The number of samples to discard for the burn-in portion
        :param n_prod:   The number of MCMC samples to take in the final production sampling
        :return: nwalker*nprod samples of the polynomial coefficients.
                 Returned as a numpy ndarray of shape (nwalker*nprod, fitorder)
        """
        # Perform the initial fit to get good guesses
        initial_pars = tls_get_initial_fit(x, y, xerr, tls_model, fitorder=fitorder)
        logging.info('Initial pars: ', initial_pars)

        # Set up the MCMC sampler
        pars = np.hstack((x, initial_pars))
        ndim = pars.size
        p0 = emcee.utils.sample_ball(pars, std=[1e-6] * ndim, size=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, tls_lnprob, args=(x, y, xerr, yerr, tls_model))

        # Burn-in
        print 'Running burn-in'
        p1, lnp, _ = sampler.run_mcmc(p0, n_burn)
        sampler.reset()

        print 'Running production'
        sampler.run_mcmc(p1, n_prod)

        return sampler.flatchain[:, x.size:]

