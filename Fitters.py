"""
A set of functions for various types of fitting.
"""

import logging
import FittingUtilities
from george import kernels

from scipy.optimize import fmin
from scipy.interpolate import interp1d
import numpy as np
from lmfit import Model, Parameters
from skmonaco import mcimport
import matplotlib.pyplot as plt
import george
import statsmodels.api as sm
from statsmodels.robust.norms import TukeyBiweight

import DataStructures
from HelperFunctions import IsListlike, ExtrapolatingUnivariateSpline
from FittingUtilities import bound
from astropy import units as u, constants


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
# Bayesian total least squares regression               #
# #################################################################

if emcee_import:
    class Bayesian_TLS(object):
        def __init__(self, x, y, xerr, yerr):
            """
            Class to perform a bayesian total least squares fit to data with errors in both the x- and y-axes.

            :param x:  A numpy ndarray with the independent variable
            :param y:  A numpy ndarray with the dependent variable
            :param xerr:  A numpy ndarray with the uncertainty in the independent variable
            :param yerr:  A numpy ndarray with the uncertainty in the dependent variable
            """
            self.x = x
            self.y = y
            self.xerr = xerr
            self.yerr = yerr

            # Default values for a bunch of stuff
            self.nwalkers = 100
            self.n_burn = 200
            self.n_prod = 1000
            self.sampler = None


        def model(self, p, x):
            """
            A parameteric model to fit y = f(x, p)
            This can be overridden in a class that inherits from this one to make a new model
            """
            return np.poly1d(p)(x)

        def _partial_likelihood(self, x, pars):
            """
            The part of the likelihood function that just compares the y values to the model prediction.
            :param pars:
            :return:
            """
            y_pred = self.model(pars, x)
            P = np.product(np.exp(-(self.y - y_pred) ** 2 / self.yerr ** 2))
            return P * (2 * np.pi) ** (self.x.size / 2.) * np.product(self.xerr)


        def _sampling_distribution(self, size=1, loc=0, scale=1):
            if IsListlike(loc):
                return np.array([np.random.normal(loc=l, scale=s, size=size) for l, s in zip(loc, scale)])
            return np.random.normal(loc=loc, scale=scale, size=size)


        def _lnlike(self, pars):
            """
            likelihood function. This uses the class variables for x,y,xerr, and yerr, as well as the 'model' instance.
            Uses Monte Carlo integration to remove the nuisance parameters (the true x locations of each point)
            """
            P, err = mcimport(self._partial_likelihood,
                              npoints=1000000, args=(pars,),
                              distribution=self._sampling_distribution,
                              dist_kwargs={'loc': self.x, 'scale': self.xerr / np.sqrt(2)},
                              nprocs=2)

            print('Relative error in integral: {}'.format(err / P))

            return np.log(P)

            """
            xtrue = pars[:self.x.size]
            y_pred = self.model(pars[self.x.size:], xtrue)  # Predict the y value

            # Make the log-likelihood
            return np.sum(-(self.x - xtrue) ** 2 / self.xerr ** 2 - (self.y - y_pred) ** 2 / self.yerr * 2)
            """


        def lnprior(self, pars):
            """
            Log of the prior for the parameters. This can be overridden to make custom priors
            """
            return 0.0


        def _lnprob(self, pars):
            """
            Log of the posterior probability of pars given the data.
            """
            lp = self.lnprior(pars)
            return lp + self._lnlike(pars) if np.isfinite(lp) else -np.inf


        def guess_fit_parameters(self, fitorder=1):
            """
            Do a normal fit to the data, ignoring the the uncertainty on the dependent variables.
            The result will be saved for use as initial guess parameters in the full MCMC fit.
            If you use a custom model, you will probably have to override this method as well.
            """

            pars = np.zeros(fitorder + 1)
            pars[-2] = 1.0
            min_func = lambda p, xi, yi, yerri: np.sum((yi - self.model(p, xi)) ** 2 / yerri ** 2)

            best_pars = fmin(min_func, x0=pars, args=(self.x, self.y, self.yerr))
            self.guess_pars = best_pars
            return best_pars


        def fit(self, nwalkers=None, n_burn=None, n_prod=None, guess=True, initial_pars=None, **guess_kws):
            """
            Perform the full MCMC fit.

            :param nwalkers:  The number of walkers to use in the MCMC sampler
            :param n_burn:   The number of samples to discard for the burn-in portion
            :param n_prod:   The number of MCMC samples to take in the final production sampling
            :param guess:    Flag for whether the data should be fit in a normal way first, to get decent starting parameters.
                             If true, it uses self.guess_fit_parameters and passes guess_kws to the function.
                             If false, it uses initial_pars. You MUST give initial_pars if guess=False!
            """
            nwalkers = self.nwalkers if nwalkers is None else nwalkers
            n_burn = self.n_burn if n_burn is None else n_burn
            n_prod = self.n_prod if n_prod is None else n_prod

            if guess:
                initial_pars = self.guess_fit_parameters(**guess_kws)
            elif initial_pars is None:
                raise ValueError('Must give initial pars if guess = False!')

            # Set up the MCMC sampler
            pars = np.hstack((self.x, initial_pars))
            ndim = pars.size
            p0 = emcee.utils.sample_ball(pars, std=[1e-6] * ndim, size=nwalkers)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob)

            # Burn-in
            print 'Running burn-in'
            p1, lnp, _ = sampler.run_mcmc(p0, n_burn)
            sampler.reset()

            print 'Running production'
            sampler.run_mcmc(p1, n_prod)

            # Save the sampler instance as a class variable
            self.sampler = sampler
            return


        def predict(self, x, N=100):
            """
            predict the y value for the given x values. Use the N most probable MCMC chains
            """
            if self.sampler is None:
                logging.warn('Need to run the fit method before predict!')
                return

            # Find the N best walkers
            if N == 'all':
                N = self.sampler.flatchain.shape[0]
            else:
                N = min(N, self.sampler.flatchain.shape[0])
            indices = np.argsort(self.sampler.lnprobability.flatten())[:N]
            pars = self.sampler.flatchain[indices, self.x.size:]

            y = np.array([self.model(p, x) for p in pars])
            return y



    class Bayesian_LS(object):
        def __init__(self, x=1, y=1, yerr=1):
            """
            Class to perform a bayesian least squares fit to data with errors in only the y-axis.

            :param x:  A numpy ndarray with the independent variable
            :param y:  A numpy ndarray with the dependent variable
            :param yerr:  A numpy ndarray with the uncertainty in the dependent variable
            """
            self.x = x
            self.y = y
            self.yerr = yerr

            # Default values for a bunch of stuff
            self.nwalkers = 100
            self.n_burn = 200
            self.n_prod = 1000
            self.sampler = None


        def model(self, p, x):
            """
            A parameteric model to fit y = f(x, p)
            This can be overridden in a class that inherits from this one to make a new model
            """
            return np.poly1d(p)(x)


        def _lnlike(self, pars):
            """
            likelihood function. This uses the class variables for x,y,xerr, and yerr, as well as the 'model' instance.
            """
            y_pred = self.model(pars, self.x)  # Predict the y value

            # Make the log-likelihood
            return -0.5 * np.sum((self.y - y_pred) ** 2 / self.yerr * 2)


        def lnprior(self, pars):
            """
            Log of the prior for the parameters. This can be overridden to make custom priors
            """
            return 0.0


        def _lnprob(self, pars):
            """
            Log of the posterior probability of pars given the data.
            """
            lp = self.lnprior(pars)
            return lp + self._lnlike(pars) if np.isfinite(lp) else -np.inf


        def guess_fit_parameters(self, fitorder=1):
            """
            Do a normal (non-bayesian) fit to the data.
            The result will be saved for use as initial guess parameters in the full MCMC fit.
            If you use a custom model, you will probably have to override this method as well.
            """

            pars = np.zeros(fitorder + 1)
            pars[-2] = 1.0
            min_func = lambda p, xi, yi, yerri: np.sum((yi - self.model(p, xi)) ** 2 / yerri ** 2)

            best_pars = fmin(min_func, x0=pars, args=(self.x, self.y, self.yerr))
            self.guess_pars = best_pars
            return best_pars


        def fit(self, nwalkers=None, n_burn=None, n_prod=None, guess=True, initial_pars=None, **guess_kws):
            """
            Perform the full MCMC fit.

            :param nwalkers:  The number of walkers to use in the MCMC sampler
            :param n_burn:   The number of samples to discard for the burn-in portion
            :param n_prod:   The number of MCMC samples to take in the final production sampling
            :param guess:    Flag for whether the data should be fit in a normal way first, to get decent starting parameters.
                             If true, it uses self.guess_fit_parameters and passes guess_kws to the function.
                             If false, it uses initial_pars. You MUST give initial_pars if guess=False!
            """
            nwalkers = self.nwalkers if nwalkers is None else nwalkers
            n_burn = self.n_burn if n_burn is None else n_burn
            n_prod = self.n_prod if n_prod is None else n_prod

            if guess:
                initial_pars = self.guess_fit_parameters(**guess_kws)
            elif initial_pars is None:
                raise ValueError('Must give initial pars if guess = False!')

            # Set up the MCMC sampler
            pars = np.array(initial_pars)
            ndim = pars.size
            p0 = emcee.utils.sample_ball(pars, std=[1e-6] * ndim, size=nwalkers)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob)

            # Burn-in
            print 'Running burn-in'
            p1, lnp, _ = sampler.run_mcmc(p0, n_burn)
            sampler.reset()

            print 'Running production'
            sampler.run_mcmc(p1, n_prod)

            # Save the sampler instance as a class variable
            self.sampler = sampler
            return


        def predict(self, x, N=100, highest=False):
            """
            predict the y value for the given x values. Use the N most probable MCMC chains if highest=False,
            otherwise use the first N chains.
            """
            if self.sampler is None:
                logging.warn('Need to run the fit method before predict!')
                return

            # Find the N best walkers
            if N == 'all':
                N = self.sampler.flatchain.shape[0]
            else:
                N = min(N, self.sampler.flatchain.shape[0])

            if highest:
                indices = np.argsort(self.sampler.flatlnprobability)[:N]
                pars = self.sampler.flatchain[indices]
            else:
                pars = self.sampler.flatchain[:N]

            y = np.array([self.model(p, x) for p in pars])
            return y


        def plot_samples(self, x, N=100, ax=None, *plot_args, **plot_kws):
            """
            Plot N best-fit curves at x-values x, on axis ax (if given)
            :param x:
            :param N:
            :param ax:
            :return: matplotlib axis object, with which to plot other stuff, label, etc
            """

            y = self.predict(x, N=N)
            if ax is None:
                ax = plt.gca()

            for i in range(N):
                ax.plot(x, y[i], *plot_args, **plot_kws)

            return ax

        def spoof_sampler(self, flatchain, flatlnprobability, force=False):
            """
            Create a sampler object with the flatchain and lnprobability attributes so self.predict will work.
            This is useful for predicting values from pre-tabulated MCMC parameter fits
            :param flatchain: The original sampler.flatchain property
            :param lnprobability: The original sampler.lnprobabiliity property
            :keyword force: Force creation of a sampler object, even if one already exists.
            :return: None
            """
            if self.sampler is not None and not force:
                logging.warn('sampler instance already exists! Use force=True to overwrite.')
                return

            self.sampler = MCSampler_Spoof(flatchain, flatlnprobability)
            return



    class GPFitter(Bayesian_LS):
        """
        A Subclass of Bayesian_LS that fits a guassian process on top of a model fit.
        """
        def __init__(self, x=1, y=1, yerr=1, solver=None):
            self.solver = george.BasicSolver if solver is None else solver
            super(GPFitter, self).__init__(x=x, y=y, yerr=yerr)
        
        def _lnlike(self, pars):
            """
            likelihood function. This uses the class variables for x,y,xerr, and yerr, as well as the 'model' instance.
            """
            #y_pred = self.x
            y_pred = self.model(pars[2:], self.x)
 
            a, tau = np.exp(pars[:2])
            gp = george.GP(a * kernels.ExpSquaredKernel(tau), solver=self.solver)
            gp.compute(self.x, self.yerr)
            return gp.lnlikelihood(self.y - y_pred)

        def lnprior(self, pars):
            """
            Prior. You may want to set a prior on the model parameters.
            """
            lna, lntau = pars[:2]
            modelpars = pars[2:]
            if -20 < lna < 30 and 0 < lntau < 30:
                return 0.0
            return -np.inf

        def guess_fit_parameters(self, fitorder=1):
            """
            Do a normal (non-bayesian and non-GP) fit to the data.
            The result will be saved for use as initial guess parameters in the full MCMC fit.
            If you use a custom model, you will probably have to override this method as well.
            """

            pars = np.zeros(fitorder + 1)
            pars[-2] = 1.0
            min_func = lambda p, xi, yi, yerri: np.sum((yi - self.model(p, xi)) ** 2 / yerri ** 2)

            best_pars = fmin(min_func, x0=pars, args=(self.x, self.y, self.yerr))
            self.guess_pars = [0, 10]
            self.guess_pars.extend(best_pars)
            return self.guess_pars

        def predict(self, x, N=100, highest=False):
            """
            Predict the y value for the given x values.
            """
            if self.sampler is None:
                logging.warn('Need to run the fit method before predict!')
                return

            # Find the N best walkers
            if N == 'all':
                N = self.sampler.flatchain.shape[0]
            else:
                N = min(N, self.sampler.flatchain.shape[0])
            
            if highest:
                indices = np.argsort(self.sampler.flatlnprobability)[:N]
                pars = self.sampler.flatchain[indices]
            else:
                pars = self.sampler.flatchain[:N]

            yvals = []
            for i, p in enumerate(pars):
                logging.info('Generating GP samples for iteration {}/{}'.format(i+1, len(pars)))
                a, tau = np.exp(p[:2])
                ypred_data = self.model(p[2:], self.x)
                ypred = self.model(p[2:], x)
                gp = george.GP(a * kernels.ExpSquaredKernel(tau), solver=self.solver)
                gp.compute(self.x, self.yerr)
                s = gp.sample_conditional(self.y - ypred_data, x) + ypred
                yvals.append(s)

            return np.array(yvals)


class Differential_RV(object):
    """
    This class performs a differential RV analysis on two observations of the same star.
    """
    def __init__(self, observation, reference, continuum_fit_order=2):
        """
        Initialize the class.
        :param observation: A list of xypoint objects for the observation spectrum
        :param reference: A list of xypoint objects for the reference spectrum
        :param continuum_fit_order: The polynomial order with which to fit the difference in continuum between the stars.
        """
        
        # Error checking
        assert len(observation) == len(reference)

        # The continuum shape should be the same for both, so we will just make it flat
        for i, order in enumerate(observation):
            observation[i].cont = np.ones(order.size()) * np.median(order.y)
        for i, order in enumerate(reference):
            reference[i].cont = np.ones(order.size()) * np.median(order.y)

        self.observation = [ExtrapolatingUnivariateSpline(o.x, o.y/o.cont) for o in observation]
        self.reference = [r.copy() for r in reference]
        self.x_arr = [r.x for r in reference]
        e = [ExtrapolatingUnivariateSpline(o.x, o.err, fill_value=0.0) for o in observation]
        self.err = [np.sqrt(e(r.x)**2 + r.err**2) for e, r in zip(e, reference)]
        self.continuum_fit_order = continuum_fit_order


    def model(self, x, RV, *args, **kwargs):
        """
        Return the observation array, interpolated on x and shifted by RV km/s.
        x should be a list of x-axes (take from the reference star)
        This method should be overridden for more complicated models (such as for fitting absolute RVs)
        """
        clight = constants.c.cgs.to(u.km/u.s).value
        output = []
        for xi, obs, ref in zip(x, self.observation, self.reference):
            data = obs(xi*(1+RV/clight))
            idx = ~np.isnan(data)
            cont = np.poly1d(np.polyfit(xi[idx], data[idx]/(ref.y[idx]/ref.cont[idx]), self.continuum_fit_order))(xi)
            output.append(data/cont)

        return output


    def lnlike(self, pars):
        """
        likelihood function. Uses class variables for model, and the two lists with 
        the observation and reference spectrum
        """
        model_orders = self.model(self.x_arr, *pars)

        lnlike = 0.0
        for ref_order, obs_order, err in zip(self.reference, model_orders, self.err):
            idx = ~np.isnan(obs_order)
            lnlike += -0.5 * np.sum((ref_order.y[idx] - obs_order[idx]*ref_order.cont[idx])**2 / (err[idx]**2))
        return lnlike


    def lnprior(self, pars):
        """
        Prior probability distribution for all the parameters.
        Override this if you add more parameters.
        """
        RV = pars[0]
        if -100 < RV < 100:
            return 0.0

        return -np.inf


    def lnprob(self, pars):
        """
        Log of the posterior probability of pars given the data.
        """
        lp = self.lnprior(pars)
        return lp + self.lnlike(pars) if np.isfinite(lp) else -np.inf


    def guess_fit_parameters(self, guess_pars=None):
        """
        Do a normal (non-bayesian) fit to the data. 
        :param guess_pars: Initial guess parameters. If not given, it guesses RV=0km/s
        """

        if guess_pars is None:
            guess_pars = [0]

        lnlike = lambda pars: -self.lnlike(pars) + 100.0*bound([-50, 50], pars[0])
        best_pars = fmin(lnlike, guess_pars)
        return best_pars


    def fit(self, nwalkers=100, n_burn=100, n_prod=500, guess=True, initial_pars=None, **guess_kws):
        if guess:
            initial_pars = self.guess_fit_parameters(**guess_kws)

        pars = np.array(initial_pars)
        ndim = pars.size
        p0 = emcee.utils.sample_ball(pars, std=[1e-6] * ndim, size=nwalkers)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)

        # Burn-in
        print 'Running burn-in'
        p1, lnp, _ = sampler.run_mcmc(p0, n_burn)
        sampler.reset()

        print 'Running production'
        sampler.run_mcmc(p1, n_prod)

        # Save the sampler instance as a class variable
        self.sampler = sampler
        return







class MCSampler_Spoof(object):
    def __init__(self, flatchain, flatlnprobability):
        self.flatchain = flatchain
        self.flatlnprobability = flatlnprobability
        return


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
