""" Module to generate empirical priors for the stock-and-flow model
for bednet distribution
"""

from numpy import *
from pymc import *

import simplejson as json
import os

import settings
import data
import graphics


def llin_discard_rate(recompute=False):
    """ Return the empirical priors for the llin discard rate Beta stoch,
    calculating them if necessary.

    Parameters
    ----------
    recompute : bool, optional
      pass recompute=True to force recomputation of empirical priors, even if json file exists

    Results
    -------
    returns a dict suitable for using to instantiate a Beta stoch
    """
    # load and return, if applicable
    fname = 'discard_prior.json'
    if fname in os.listdir(settings.PATH) and not recompute:
        f = open(settings.PATH + fname)
        return json.load(f)
        
    ### setup (hyper)-prior stochs
    pi = Beta('Pr[net is lost]', 1, 2)
    sigma = InverseGamma('standard error', 11, 1)

    vars = [pi, sigma]

    ### data likelihood from net retention studies
    retention_obs = []
    for d in data.retention:
        @observed
        @stochastic(name='retention_%s_%s' % (d['Name'], d['Year']))
        def obs(value=d['Retention_Rate'],
                T_i=d['Follow_up_Time'],
                pi=pi, sigma=sigma):
            return normal_like(value, (1. - pi) ** T_i, 1. / sigma**2)
        retention_obs.append(obs)

        vars += [retention_obs]

    # find model with MCMC
    mc = MCMC(vars, verbose=1)
    iter = 10000
    thin = 20
    burn = 20000
    mc.sample(iter*thin+burn, burn, thin)

    # save fit values for empirical prior
    x = pi.stats()['mean']
    v = pi.stats()['standard deviation']**2

    emp_prior_dict = dict(mu=x, var=v, tau=1/v,
                          alpha=x*(x*(1-x)/v-1), beta=(1-x)*(x*(1-x)/v-1))
    f = open(settings.PATH + fname, 'w')
    json.dump(emp_prior_dict, f)

    graphics.plot_discard_prior(pi, emp_prior_dict)
    
    return pi, sigma, emp_prior_dict


def admin_err_and_bias(recompute=False):
    """ Return the empirical priors for the admin error and bias stochs,
    calculating them if necessary.

    Parameters
    ----------
    recompute : bool, optional
      pass recompute=True to force recomputation of empirical priors,
      even if json file exists

    Results
    -------
    returns a dict suitable for using to instantiate a Beta stoch
    """
    # load and return, if applicable
    fname = 'admin_err_and_bias_prior.json'
    if fname in os.listdir(settings.PATH) and not recompute:
        f = open(settings.PATH + fname)
        return json.load(f)

    mu_pi = llin_discard_rate()['mu']

    # setup hyper-prior stochs
    sigma = Gamma('error in admin dist data', 1., 1.)
    eps = Normal('bias in admin dist data', 0., 1.)
    vars = [sigma, eps]

    ### setup data likelihood stochs
    data_dict = {}
    # store admin data for each country-year
    for d in data.admin_llin:
        key = (d['Country'], d['Year'])
        if not data_dict.has_key(key):
            data_dict[key] = {}
        data_dict[key]['obs'] = d['Program_LLINs']

    # store household data for each country-year
    for d in data.hh_llin_flow:
        key = (d['Country'], d['Year'])
        if not data_dict.has_key(key):
            data_dict[key] = {}
        data_dict[key]['time'] =  d['mean_survey_date'] - (d['Year'] + .5)
        data_dict[key]['truth'] = d['Total_LLINs'] / (1-mu_pi)**data_dict[key]['time']
        data_dict[key]['se'] = d['Total_st']
        
    # keep only country-years with both admin and survey data
    for key in data_dict.keys():
        if len(data_dict[key]) != 4:
            data_dict.pop(key)

    # create the observed stochs
    for d in data_dict.values():
        @observed
        @stochastic
        def obs(value=log(d['obs']), log_truth=log(d['truth']),
                log_v=1.1*d['se']**2/d['truth']**2,
                eps=eps, sigma=sigma):
            return normal_like(value, log_truth + eps,
                               1. / (log_v + sigma**2))
        vars.append(obs)

    # sample from empirical prior distribution via MCMC
    mc = MCMC(vars, verbose=1)
    iter = 10000
    thin = 20
    burn = 20000

    mc.sample(iter*thin+burn, burn, thin)

    # output information on empirical prior distribution
    emp_prior_dict = dict(
        sigma=dict(mu=sigma.stats()['mean'],
                   std=sigma.stats()['standard deviation'],
                   tau=sigma.stats()['standard deviation']**-2),
        eps=dict(mu=eps.stats()['mean'],
                 std=eps.stats()['standard deviation'],
                 tau=eps.stats()['standard deviation']**-2))

    f = open(settings.PATH + fname, 'w')
    json.dump(emp_prior_dict, f)

    graphics.plot_admin_priors(eps, sigma, emp_prior_dict, data_dict)

    return eps, sigma, emp_prior_dict, data_dict


def cov_and_zif(recompute=False):
    """ Return the empirical priors for the coverage factor and
    zero-inflation factor, calculating them if necessary.

    Parameters
    ----------
    recompute : bool, optional
      pass recompute=True to force recomputation of empirical priors,
      even if json file exists

    Results
    -------
    returns a dict suitable for using to instantiate normal and beta stochs
    """
    # load and return, if applicable
    fname = 'cov_and_zif_prior.json'
    if fname in os.listdir(settings.PATH) and not recompute:
        f = open(settings.PATH + fname)
        return json.load(f)

    # setup hyper-prior stochs
    e = Normal('coverage factor', 5., 5.)
    z = Beta('zero inflation factor', 1., 10.)
    vars = [e, z]

    ### setup data likelihood stochs
    data_dict = {}

    # store population data for each country-year
    for d in data.population:
        key = (d['Country'], d['Year'])
        data_dict[key] = {}
        data_dict[key]['pop'] =  d['Pop']*1000

    # store stock data for each country-year
    for d in data.hh_llin_stock:
        key = (d['Country'], d['Survey_Year2'])
        data_dict[key]['stock'] = d['SvyIndex_LLINstotal'] / data_dict[key]['pop']
        data_dict[key]['stock_se'] = d['SvyIndex_st'] / data_dict[key]['pop']

    # store coverage data for each country-year
    for d in data.llin_coverage:
        key = (d['Country'], d['Survey_Year2'])
        data_dict[key]['uncovered'] =  d['Per_0LLINs']
        data_dict[key]['se'] = d['LLINs0_SE']
        
    # keep only country-years with both stock and coverage
    for key in data_dict.keys():
        if len(data_dict[key]) != 5:
            data_dict.pop(key)

    # create stochs from stock and coverage data
    for k, d in data_dict.items():
        stock = Normal('stock_%s_%s' % k, mu=d['stock'], tau=d['stock_se']**-2)
        
        @observed
        @stochastic
        def obs(value=d['uncovered'], stock=stock, tau=d['se']**-2,
                e=e, z=z):
            return normal_like(value, z + (1-z) * exp(-e * stock), tau)
        vars += [stock, obs]


    # sample from empirical prior distribution via MCMC
    mc = MCMC(vars, verbose=1)
    iter = 10000
    thin = 20
    burn = 20000

    mc.sample(iter*thin+burn, burn, thin)


    # output information on empirical prior distribution
    x = z.stats()['mean']
    v = z.stats()['standard deviation']**2

    emp_prior_dict = dict(
        eta=dict(mu=e.stats()['mean'],
                 std=e.stats()['standard deviation'],
                 tau=e.stats()['standard deviation']**-2),
        zeta=dict(mu=x, var=v, tau=1/v,
                  alpha=x*(x*(1-x)/v-1), beta=(1-x)*(x*(1-x)/v-1))
        )

    f = open(settings.PATH + fname, 'w')
    json.dump(emp_prior_dict, f)

    graphics.plot_cov_and_zif_priors(e, z, emp_prior_dict, data_dict)

    return e, z, emp_prior_dict, data_dict


def survey_design(recompute=False):
    """ Return the empirical prior for the survey design factor

    Results
    -------
    returns a dict suitable for using to instantiate normal and beta stochs
    """
    return dict(mu=2., tau=.25**-2)
    
if __name__ == '__main__':
    import optparse
    
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args) != 0:
        parser.error('incorrect number of arguments')

    llin_discard_rate(recompute=True)
    admin_err_and_bias(recompute=True)
    cov_and_zif(recompute=True)
    survey_design(recompute=True)
