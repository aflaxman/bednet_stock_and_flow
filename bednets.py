"""  Module to fit stock-and-flow compartmental model of bednet distribution
>>> for i in range(50):  bednets.main(i)
"""

import settings

from pylab import *
from pymc import *

import copy
import time
import optparse
import random

from data import Data
data = Data()

import emp_priors
import graphics

def main(country_id):
    from settings import year_start, year_end

    c = sorted(data.countries)[country_id]
    print c

    # get population data for this country, to calculate LLINs per capita
    pop = data.population_for(c, year_start, year_end)

    ### setup the model variables
    vars = []

       #######################
      ### compartmental model
     ###
    #######################

    # Empirical Bayesian priors
    prior = emp_priors.llin_discard_rate()
    pi = Beta('Pr[net is lost]', prior['alpha'], prior['beta'])
    vars += [pi]

    prior = emp_priors.admin_err_and_bias()
    e_d = Normal('bias in admin dist data',
                 prior['eps']['mu'], prior['eps']['tau'])
    s_d = Normal('error in admin dist data',
                 prior['sigma']['mu'], prior['sigma']['tau'])
    beta = Normal('relative weights of next year to current year in admin dist data',
                  prior['beta']['mu'], prior['beta']['tau'])
    vars += [s_d, e_d, beta]

    prior = emp_priors.neg_binom()
    eta = Normal('coverage parameter', prior['eta']['mu'], prior['eta']['tau'], value=prior['eta']['mu'])
    alpha = Gamma('dispersion parameter', prior['alpha']['alpha'], prior['alpha']['beta'], value=prior['alpha']['mu'])
    vars += [eta, alpha]

    prior = emp_priors.survey_design()
    gamma = Normal('survey design factor for coverage data', prior['mu'], prior['tau'])
    vars += [gamma]

    # Fully Bayesian priors
    s_m = Lognormal('error_in_llin_ship', log(.05), .5**-2, value=.05)
    vars += [s_m]

    s_rb = Lognormal('recall bias factor', log(.05), .5**-2, value=.05)
    vars += [s_rb]

    mu_N = .001 * pop
    std_N = where(arange(year_start, year_end) <= 2003, .2, 2.)

    log_delta = Normal('log(llins distributed)', mu=log(mu_N), tau=std_N**-2, value=log(mu_N))
    delta = Lambda('llins distributed', lambda x=log_delta: exp(x))

    log_mu = Normal('log(llins shipped)', mu=log(mu_N), tau=std_N**-2, value=log(mu_N))
    mu = Lambda('llins shipped', lambda x=log_mu: exp(x))

    log_Omega = Normal('log(non-llin household net stock)',
                        mu=log(mu_N), tau=2.**-2, value=log(mu_N))
    Omega = Lambda('non-llin household net stock', lambda x=log_Omega: exp(x))

    vars += [log_delta, delta, log_mu, mu, log_Omega, Omega]

    @deterministic(name='llin warehouse net stock')
    def Psi(mu=mu, delta=delta):
        Psi = zeros(year_end-year_start)
        for t in range(year_end - year_start - 1):
            Psi[t+1] = Psi[t] + mu[t] - delta[t]
        return Psi

    @deterministic(name='1-year-old household llin stock')
    def Theta1(delta=delta):
        Theta1 = zeros(year_end-year_start)
        Theta1[1:] = delta[:-1]
        return Theta1

    @deterministic(name='2-year-old household llin stock')
    def Theta2(Theta1=Theta1, pi=pi):
        Theta2 = zeros(year_end-year_start)
        Theta2[1:] = Theta1[:-1] * (1 - pi) ** .5
        return Theta2

    @deterministic(name='3-year-old household llin stock')
    def Theta3(Theta2=Theta2, pi=pi):
        Theta3 = zeros(year_end-year_start)
        Theta3[1:] = Theta2[:-1] * (1  - pi)
        return Theta3

    @deterministic(name='household llin stock')
    def Theta(Theta1=Theta1, Theta2=Theta2, Theta3=Theta3):
        return Theta1 + Theta2 + Theta3

    @deterministic(name='household itn stock')
    def itns_owned(Theta=Theta, Omega=Omega):
        return Theta + Omega

    @deterministic(name='llin coverage')
    def llin_coverage(Theta=Theta, pop=pop,
                      eta=eta, alpha=alpha):
        return 1. - (alpha / (eta*Theta/pop + alpha))**alpha

    @deterministic(name='itn coverage')
    def itn_coverage(llin=Theta, non_llin=Omega, pop=pop,
                     eta=eta, alpha=alpha):
        return 1. - (alpha / (eta*(llin + non_llin)/pop + alpha))**alpha

    vars += [Psi, Theta, Theta1, Theta2, Theta3, itns_owned, llin_coverage, itn_coverage]

    # set initial conditions on nets manufactured to have no stockouts
    if min(Psi.value) < 0:
        log_mu.value = log(maximum(1., mu.value - 2*min(Psi.value)))

       #####################
      ### additional priors
     ###
    #####################
    @potential
    def positive_stocks(Theta=Theta, Psi=Psi, Omega=Omega):
        if any(Psi < 0) or any(Theta < 0) or any(Omega < 0):
            return sum(minimum(Psi,0)) + sum(minimum(Theta, 0)) + sum(minimum(Omega, 0))
        else:
            return 0.
    vars += [positive_stocks]

    proven_capacity_std = .5
    @potential
    def proven_capacity(delta=delta, Omega=Omega, tau=proven_capacity_std**-2):
        total_dist = delta[:-1] + .5*(Omega[1:] + Omega[:-1])
        max_log_d = log(maximum(1.,[max(total_dist[:(i+1)]) for i in range(len(total_dist))]))
        amt_below_cap = minimum(log(maximum(total_dist,1.)) - max_log_d, 0.)
        return normal_like(amt_below_cap, 0., tau)
    vars += [proven_capacity]

    itn_composition_std = .5
    @potential
    def itn_composition(llin=Theta, non_llin=Omega, tau=itn_composition_std**-2):
        frac_llin = llin / (llin + non_llin)
        return normal_like(frac_llin[[0,1,2,6,7,8,9,10,11]],
                           [0., 0., 0., 1., 1., 1., 1., 1., 1.], tau)
    vars += [itn_composition]

    smooth_std = .5
    @potential
    def smooth_coverage(itn_coverage=itn_coverage, tau=smooth_std**-2):
        return normal_like(diff(log(itn_coverage)), 0., tau)
    vars += [smooth_coverage]


       #####################
      ### statistical model
     ###
    #####################


    ### nets shipped to country (reported by manufacturers)

    manufacturing_obs = []
    for d in data.llin_manu:
        if d['country'] != c:
            continue

        @observed
        @stochastic(name='manufactured_%s_%s' % (d['country'], d['year']))
        def obs(value=max(1., float(d['manu_itns'])), year=int(d['year']), mu=mu, s_m=s_m):
            return normal_like(log(value),  log(max(1., mu[year - year_start])), 1. / s_m**2)
        manufacturing_obs.append(obs)

        # also take this opportinuty to set better initial values for the MCMC
        cur_val = copy.copy(mu.value)
        cur_val[int(d['year']) - year_start] = min(d['manu_itns'], 10.)
        log_mu.value = log(maximum(1., cur_val))

    vars += [manufacturing_obs]



    ### nets distributed in country (reported by NMCP)

    # store admin data for this country for each year
    data_dict = {}
    for d in data.admin_llin:
        if d['country'] != c:
            continue
        data_dict[d['year']] = d['program_llins']
        
    admin_distribution_obs = []
    for year, d in data_dict.items():

        @observed
        @stochastic(name='administrative_distribution_%s' % year)
        def obs(value=log(d), year=year,
                delta=delta, s_d=s_d, e_d=e_d, beta=beta):
            pred = log(max(1., delta[year - year_start] + beta*delta[year+1 - year_start])) + e_d
            return normal_like(value, pred, 1. / s_d**2)
        admin_distribution_obs.append(obs)

        # also take this opportinuty to set better initial values for the MCMC
        cur_val = copy.copy(delta.value)
        cur_val[year - year_start] = d
        log_delta.value = log(cur_val)

    vars += [admin_distribution_obs]


    ### nets distributed in country (observed in household survey)

    household_distribution_obs = []
    for d in data.hh_llin_flow:
        if d['country'] != c:
            continue

        d2_i = d['total_llins']
        estimate_year = int(d['year'])
        
        mean_survey_date = time.strptime(d['mean_svydate'], '%d-%b-%y')
        survey_year = mean_survey_date[0] + mean_survey_date[1]/12.

        s_d2_i = float(d['total_st'])

        @observed
        @stochastic(name='household_distribution_%s_%s' % (d['country'], d['year']))
        def obs(value=d2_i,
                estimate_year=estimate_year,
                survey_year=survey_year,
                survey_err=s_d2_i,
                delta=delta, pi=pi, s_rb=s_rb):
            return normal_like(
                value,
                delta[estimate_year - year_start] * (1 - pi) ** (survey_year - estimate_year - .5),
                1./ (survey_err*(1+s_rb))**2)
        household_distribution_obs.append(obs)

        # also take this opportinuty to set better initial values for the MCMC
        cur_val = copy.copy(delta.value)
        cur_val[estimate_year - year_start] = d2_i / (1 - pi.value)**(survey_year - estimate_year - .5)
        log_delta.value = log(cur_val)

    vars += [household_distribution_obs]


    ### net stock in households (from survey)
    household_stock_obs = []
    for d in data.hh_llin_stock:
        if d['country'] != c:
            continue
        mean_survey_date = time.strptime(d['mean_svydate'], '%d-%b-%y')
        d['year'] = mean_survey_date[0] + mean_survey_date[1]/12.

        @observed
        @stochastic(name='LLIN_HH_Stock_%s_%s' % (d['country'], d['survey_year2']))
        def obs(value=d['svyindex_llins'],
                year=d['year'],
                std_err=d['svyindexllins_se'],
                Theta=Theta):
            year_part = year-floor(year)
            Theta_i = (1-year_part) * Theta[floor(year)-year_start] + year_part * Theta[ceil(year)-year_start]
            return normal_like(value, Theta_i, 1. / std_err**2)
        household_stock_obs.append(obs)

    vars += [household_stock_obs]


    ### llin and itn coverage (from survey and survey reports)
    coverage_obs = []
    for d in data.llin_coverage:
        if d['country'] != c:
            continue

        if d['llins0_se']: # data from survey, includes standard error
            d['coverage'] = 1. - float(d['per_0llins'])
            d['coverage_se'] = float(d['llins0_se'])
            mean_survey_date = time.strptime(d['mean_svydate'], '%d-%b-%y')
            d['year'] = mean_survey_date[0] + mean_survey_date[1]/12.
            
            @observed
            @stochastic(name='LLIN_Coverage_%s_%s' % (d['country'], d['survey_year2']))
            def obs(value=d['coverage'],
                    year=d['survey_year2'],
                    std_err=d['coverage_se'],
                    coverage=llin_coverage):
                year_part = year-floor(year)
                coverage_i = (1-year_part) * coverage[floor(year)-year_start] + year_part * coverage[ceil(year)-year_start]
                return normal_like(value, coverage_i, 1. / std_err**2)
        else: # data is imputed from under 5 usage, so estimate standard error
            d['coverage'] = 1. - float(d['per_0llins'])
            N = d['sample_size'] or 1000
            d['sampling_error'] = d['coverage']*(1-d['coverage'])/sqrt(N)
            d['coverage_se'] = d['sampling_error']*gamma.value

            mean_survey_date = time.strptime(d['mean_svydate'], '%d-%b-%y')
            d['year'] = mean_survey_date[0] + mean_survey_date[1]/12.

            @observed
            @stochastic(name='LLIN_Coverage_Imputation_%s_%s' % (d['country'], d['year']))
            def obs(value=d['coverage'],
                    year=d['year'],
                    sampling_error=d['sampling_error'],
                    design_factor=gamma,
                    coverage=llin_coverage):
                year_part = year-floor(year)
                coverage_i = (1-year_part) * coverage[floor(year)-year_start] + year_part * coverage[ceil(year)-year_start]
                return normal_like(value, coverage_i, 1. / (design_factor * sampling_error)**2)
        coverage_obs.append(obs)
            

    for d in data.itn_coverage:
        if d['country'] != c:
            continue

        d['coverage'] = 1. - float(d['per_0itns'])

        if d['itns0_se']: # data from survey, includes standard error
            d['coverage_se'] = d['itns0_se']

            mean_survey_date = time.strptime(d['mean_svydate'], '%d-%b-%y')
            d['year'] = mean_survey_date[0] + mean_survey_date[1]/12.

            @observed
            @stochastic(name='ITN_Coverage_%s_%s' % (d['country'], d['year']))
            def obs(value=d['coverage'],
                    year=d['year'],
                    std_err=d['coverage_se'],
                    coverage=itn_coverage):
                year_part = year-floor(year)
                coverage_i = (1-year_part) * coverage[floor(year)-year_start] + year_part * coverage[ceil(year)-year_start]
                return normal_like(value, coverage_i, 1. / std_err**2)

        else: # data from survey report, must calculate standard error
            mean_survey_date = time.strptime(d['mean_svydate'], '%d-%b-%y')
            d['year'] = mean_survey_date[0] + mean_survey_date[1]/12.

            N = d['sample_size'] or 1000
            d['sampling_error'] = d['coverage']*(1-d['coverage'])/sqrt(N)
            d['coverage_se'] = d['sampling_error']*gamma.value

            @observed
            @stochastic(name='ITN_Coverage_Report_%s_%s' % (d['country'], d['year']))
            def obs(value=d['coverage'],
                    year=d['year'],
                    sampling_error=d['sampling_error'],
                    design_factor=gamma,
                    coverage=itn_coverage):
                year_part = year-floor(year)
                coverage_i = (1-year_part) * coverage[floor(year)-year_start] + year_part * coverage[ceil(year)-year_start]
                return normal_like(value, coverage_i, 1. / (design_factor * sampling_error)**2)

        coverage_obs.append(obs)


        # also take this opportinuty to set better initial values for the MCMC
        t = floor(d['year'])-year_start
        cur_val = copy.copy(Omega.value)
        cur_val[t] = max(.0001*pop[t], log(1-d['coverage']) * pop[t] / eta.value - Theta.value[t])
        log_Omega.value = log(cur_val)

    vars += [coverage_obs]


       #################
      ### fit the model
     ###
    #################
    print 'running fit for net model in %s...' % c

    if settings.TESTING:
        map = MAP(vars)
        map.fit(method='fmin', iterlim=100, verbose=1)
    else:
        # just optimize some variables, to get reasonable initial conditions
        map = MAP([log_mu,
                   positive_stocks,
                   manufacturing_obs])
        map.fit(method='fmin_powell', verbose=1)

        map = MAP([log_delta,
                   positive_stocks,
                   admin_distribution_obs, household_distribution_obs,
                   household_stock_obs])
        map.fit(method='fmin_powell', verbose=1)

        map = MAP([log_mu, log_delta, log_Omega,
                   positive_stocks, #itn_composition,
                   coverage_obs])
        map.fit(method='fmin_powell', verbose=1)

        for stoch in [s_m, s_d, e_d, pi, eta, alpha]:
            print '%s: %s' % (str(stoch), str(stoch.value))

    if settings.METHOD == 'MCMC':
        mc = MCMC(vars, verbose=1, db='pickle', dbname=settings.PATH + 'bednet_model_%s_%d_%s.pickle' % (c, country_id, time.strftime('%Y_%m_%d_%H_%M')))
        mc.use_step_method(Metropolis, s_m, proposal_sd=.001)
        mc.use_step_method(Metropolis, eta, proposal_sd=.001)

        try:
            if settings.TESTING:
                iter = 100
                thin = 1
                burn = 0
            else:
                iter = settings.NUM_SAMPLES
                thin = settings.THIN
                burn = settings.BURN
            mc.sample(iter*thin+burn, burn, thin)
        except KeyError:
            pass
        mc.db.commit()

    elif settings.METHOD == 'NormApprox':
        na = NormApprox(vars)
        na.fit(method='fmin_powell', tol=.00001, verbose=1)
        for stoch in [s_m, s_d, e_d, pi]:
            print '%s: %s' % (str(stoch), str(stoch.value))
        na.sample(1000)

    else:
        assert 0, 'Unknown estimation method'

    # save results in output file
    col_headings = [
        'Country', 'Year', 'Population',
        'LLINs Shipped (Thousands)', 'LLINs Shipped Lower CI', 'LLINs Shipped Upper CI',
        'LLINs Distributed (Thousands)', 'LLINs Distributed Lower CI', 'LLINs Distributed Upper CI',
        'LLINs Not Owned Warehouse (Thousands)', 'LLINs Not Owned Lower CI', 'LLINs Not Owned Upper CI',
        'LLINs Owned (Thousands)', 'LLINs Owned Lower CI', 'LLINs Owned Upper CI',
        'non-LLIN ITNs Owned (Thousands)', 'non-LLIN ITNs Owned Lower CI', 'non-LLIN ITNs Owned Upper CI',
        'ITNs Owned (Thousands)', 'ITNs Owned Lower CI', 'ITNs Owned Upper CI',
        'LLIN Coverage (Percent)', 'LLIN Coverage Lower CI', 'LLIN Coverage Upper CI',
        'ITN Coverage (Percent)', 'ITN Coverage Lower CI', 'ITN Coverage Upper CI',
        ]

    try:  # sleep for a random time interval to avoid collisions when writing results
        print 'sleeping...'
        #time.sleep(random.random()*30)
        print '...woke up'
    except:  # but let user cancel with cntl-C if there is a rush
        print '...work up early'

    if not settings.CSV_NAME in os.listdir(settings.PATH):
        f = open(settings.PATH + settings.CSV_NAME, 'a')
        f.write('%s\n' % ','.join(col_headings))
    else:
        f = open(settings.PATH + settings.CSV_NAME, 'a')

    for t in range(year_end - year_start):
        f.write('%s,%d,%d,' % (c,year_start + t,pop[t]))
        if t == year_end - year_start - 1:
            val = [-99, -99, -99]
            val += [-99, -99, -99]
        else:
            val = [mu.stats()['mean'][t]/1000] + list(mu.stats()['95% HPD interval'][t]/1000)
            val += [delta.stats()['mean'][t]/1000] + list(delta.stats()['95% HPD interval'][t]/1000)
        val += [Psi.stats()['mean'][t]/1000] + list(Psi.stats()['95% HPD interval'][t]/1000)
        val += [Theta.stats()['mean'][t]/1000] + list(Theta.stats()['95% HPD interval'][t]/1000)
        val += [Omega.stats()['mean'][t]/1000] + list(Omega.stats()['95% HPD interval'][t]/1000)
        val += [itns_owned.stats()['mean'][t]/1000] + list(itns_owned.stats()['95% HPD interval'][t]/1000)
        val += [100*llin_coverage.stats()['mean'][t]] + list(100*llin_coverage.stats()['95% HPD interval'][t])
        val += [100*itn_coverage.stats()['mean'][t]] + list(100*itn_coverage.stats()['95% HPD interval'][t])
        f.write(','.join(['%.2f']*(len(col_headings)-3)) % tuple(val))
        f.write('\n')
    f.close()
    
    f = open(settings.PATH + 'traces/itn_coverage_%s_%d_%s.csv' % (c, country_id, time.strftime('%Y_%m_%d_%H_%M')), 'w')
    for row in itn_coverage.trace():
        f.write(','.join(['%.4f' % cell for cell in row]))
        f.write('\n')
    f.close()
    
    f = open(settings.PATH + 'traces/itn_stock_%s_%d_%s.csv' % (c, country_id, time.strftime('%Y_%m_%d_%H_%M')), 'w')
    for row in itns_owned.trace():
        f.write(','.join(['%.4f' % cell for cell in row]))
        f.write('\n')
    f.close()

    graphics.plot_posterior(country_id, c, pop,
                            s_m, s_d, e_d, pi, mu, delta, Psi, Theta, Omega, gamma, eta, alpha, s_rb,
                            manufacturing_obs, admin_distribution_obs, household_distribution_obs,
                            itn_coverage, llin_coverage, itns_owned, data
                            )

if __name__ == '__main__':
    usage = 'usage: %prog [options] country_id'
    parser = optparse.OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error('incorrect number of arguments')
    elif args[0] == 'summarize':
        import explore
        explore.summarize_fits()
    else:
        try:
            country_id = int(args[0])
        except ValueError:
            parser.error('country_id must be an integer (or summarize to generate summary tables)')

        main(country_id)
