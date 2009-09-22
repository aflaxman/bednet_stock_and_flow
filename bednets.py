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

import data
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
    vars += [s_d, e_d]

    prior = emp_priors.cov_and_zif()
    eta = Normal('coverage factor', prior['eta']['mu'], prior['eta']['tau'])
    zeta = Beta('zero inflation factor', prior['zeta']['alpha'], prior['zeta']['beta'])
    vars += [eta, zeta]

    prior = emp_priors.survey_design()
    s_r_c = Normal('survey design factor for coverage data', prior['mu'], prior['tau'])
    vars += [s_r_c]

    # Fully Bayesian priors
    s_m = Lognormal('error in llin ship', log(.1), .5**-2, value=.1)
    vars += [s_m]

    s_rb = Lognormal('recall bias factor', log(.1), .5**-2, value=.01)
    vars += [s_rb]

    mu_nd = .001 * pop
    log_nd = Normal('log(llins distributed)', mu=log(mu_nd), tau=3.**-2, value=log(mu_nd))
    nd = Lambda('llins distributed', lambda x=log_nd: exp(x))

    mu_nm = where(arange(year_start, year_end) <= 2003, .00005, .001) * pop
    log_nm = Normal('log(llins shipped)', mu=log(mu_nm), tau=3.**-2, value=log(mu_nm)) 
    nm = Lambda('llins shipped', lambda x=log_nm: exp(x))

    mu_h_prime = .001 * pop
    log_Hprime = Normal('log(non-llin household net stock)',
                        mu=log(mu_h_prime), tau=3.**-2, value=log(mu_h_prime))
    Hprime = Lambda('non-llin household net stock', lambda x=log_Hprime: exp(x))

    vars += [log_nd, nd, log_nm, nm, log_Hprime, Hprime]

    @deterministic(name='llin warehouse net stock')
    def W(nm=nm, nd=nd):
        W = zeros(year_end-year_start)
        for t in range(year_end - year_start - 1):
            W[t+1] = W[t] + nm[t] - nd[t]
        return W

    @deterministic(name='1-year-old household llin stock')
    def H1(nd=nd):
        H1 = zeros(year_end-year_start)
        H1[1:] = nd[:-1]
        return H1

    @deterministic(name='2-year-old household llin stock')
    def H2(H1=H1, pi=pi):
        H2 = zeros(year_end-year_start)
        H2[1:] = H1[:-1] * (1 - pi) ** .5
        return H2

    @deterministic(name='3-year-old household llin stock')
    def H3(H2=H2, pi=pi):
        H3 = zeros(year_end-year_start)
        H3[1:] = H2[:-1] * (1  - pi)
        return H3

    @deterministic(name='4-year-old household llin stock')
    def H4(H3=H3, pi=pi):
        H4 = zeros(year_end-year_start)
        H4[1:] = H3[:-1] * (1 - pi)
        return H4

    @deterministic(name='household llin stock')
    def H(H1=H1, H2=H2, H3=H3, H4=H4):
        return H1 + H2 + H3 + H4

    @deterministic(name='household itn stock')
    def hh_itn(H=H, Hprime=Hprime):
        return H + Hprime

    @deterministic(name='llin coverage')
    def llin_coverage(H=H, pop=pop,
                      eta=eta, zeta=zeta):
        return 1. - zeta - (1-zeta)*exp(-eta * H / pop)

    @deterministic(name='itn coverage')
    def itn_coverage(H_llin=H, H_non_llin=Hprime, pop=pop,
                     eta=eta, zeta=zeta):
        return 1. - zeta - (1-zeta)*exp(-eta * (H_llin + H_non_llin) / pop)

    vars += [W, H, H1, H2, H3, H4, hh_itn, llin_coverage, itn_coverage]

    # set initial conditions on nets manufactured to have no stockouts
    if min(W.value) < 0:
        log_nm.value = log(nm.value - 2*min(W.value))

       #####################
      ### additional priors
     ###
    #####################

    @potential
    def smooth_stocks(W=W, H=H, Hprime=Hprime):
        return normal_like(diff(log(maximum(W,1))), 0., 1.**-2) + normal_like(diff(log(maximum(H,1))), 0., 1.**-2) + normal_like(diff(log(maximum(Hprime,1))), 0., 1.**-2)

    @potential
    def positive_stocks(H=H, W=W, Hprime=Hprime):
        if any(W < 0) or any(H < 0) or any(Hprime < 0):
            return sum(minimum(W,0)) + sum(minimum(H, 0)) + sum(minimum(Hprime, 0))
        else:
            return 0.
    vars += [smooth_stocks, positive_stocks]

    @potential
    def proven_capacity(nd=nd):
        max_log_nd = log(maximum(1.,[max(nd[:(i+1)]) for i in range(len(nd))]))
        amt_below_cap = minimum(log(maximum(nd,1.)) - max_log_nd, 0.)
        return normal_like(amt_below_cap, 0., .25**-2)
    vars += [proven_capacity]


       #####################
      ### statistical model
     ###
    #####################


    ### observed nets manufactured

    manufacturing_obs = []
    for d in data.llin_manu:
        if d['Country'] != c:
            continue

        @observed
        @stochastic(name='manufactured_%s_%s' % (d['Country'], d['Year']))
        def obs(value=float(d['Manu_Itns']), year=int(d['Year']), nm=nm, s_m=s_m):
            return normal_like(log(value),  log(max(1., nm[year - year_start])), 1. / s_m**2)
        manufacturing_obs.append(obs)

        # also take this opportinuty to set better initial values for the MCMC
        cur_val = copy.copy(nm.value)
        cur_val[int(d['Year']) - year_start] = min(d['Manu_Itns'], 10.)
        log_nm.value = log(cur_val)

    vars += [manufacturing_obs]



    ### observed nets distributed

    admin_distribution_obs = []
    for d in data.admin_llin:
        if d['Country'] != c:
            continue

        @observed
        @stochastic(name='administrative_distribution_%s_%s' % (d['Country'], d['Year']))
        def obs(value=d['Program_LLINs'], year=int(d['Year']),
                nd=nd, s_d=s_d, e_d=e_d):
            return normal_like(log(value), e_d + log(max(1., nd[year - year_start])), 1. / s_d**2)
        admin_distribution_obs.append(obs)

        # also take this opportinuty to set better initial values for the MCMC
        cur_val = copy.copy(nd.value)
        cur_val[int(d['Year']) - year_start] = d['Program_LLINs']
        log_nd.value = log(cur_val)

    vars += [admin_distribution_obs]


    household_distribution_obs = []
    for d in data.hh_llin_flow:
        if d['Country'] != c:
            continue

        d2_i = float(d['Total_LLINs'])
        estimate_year = int(d['Year'])
        survey_year = int(d['Survey_Year2'])
        s_d2_i = float(d['Total_st'])
        @observed
        @stochastic(name='household_distribution_%s_%s' % (d['Country'], d['Year']))
        def obs(value=d2_i,
                estimate_year=estimate_year,
                survey_year=survey_year,
                survey_err=s_d2_i,
                nd=nd, pi=pi, s_rb=s_rb):
            return normal_like(
                value,
                nd[estimate_year - year_start] * (1 - pi) ** (survey_year - estimate_year - .5),
                1./ (survey_err*(1+s_rb))**2)
        household_distribution_obs.append(obs)

        # also take this opportinuty to set better initial values for the MCMC
        cur_val = copy.copy(nd.value)
        cur_val[estimate_year - year_start] = d2_i / (1 - pi.value)**(survey_year - estimate_year - .5)
        log_nd.value = log(cur_val)

    vars += [household_distribution_obs]


    ### observed household stocks (from survey)
    household_stock_obs = []
    for d in data.hh_llin_stock:
        if d['Country'] != c:
            continue
        mean_survey_date = time.strptime(d['Mean_SvyDate'], '%d-%b-%y')
        d['Year'] = mean_survey_date[0] + mean_survey_date[1]/12.

        @observed
        @stochastic(name='LLIN_HH_Stock_%s_%s' % (d['Country'], d['Survey_Year2']))
        def obs(value=d['SvyIndex_LLINstotal'],
                year=d['Year'],
                std_err=d['SvyIndex_st'],
                H=H):
            year_part = year-floor(year)
            H_i = (1-year_part) * H[floor(year)-year_start] + year_part * H[ceil(year)-year_start]
            return normal_like(value, H_i, 1. / std_err**2)
        household_stock_obs.append(obs)

    vars += [household_stock_obs]


    ### observed coverage 
    coverage_obs = []
    for d in data.llin_coverage:
        if d['Country'] != c:
            continue

        if d['LLINs0_SE']: # data from survey, includes standard error
            d['coverage'] = 1. - float(d['Per_0LLINs'])
            d['coverage_se'] = float(d['LLINs0_SE'])
            mean_survey_date = time.strptime(d['Mean_SvyDate'], '%d-%b-%y')
            d['Year'] = mean_survey_date[0] + mean_survey_date[1]/12.
            
            @observed
            @stochastic(name='LLIN_Coverage_%s_%s' % (d['Country'], d['Survey_Year2']))
            def obs(value=d['coverage'],
                    year=d['Survey_Year2'],
                    std_err=d['coverage_se'],
                    coverage=llin_coverage):
                year_part = year-floor(year)
                coverage_i = (1-year_part) * coverage[floor(year)-year_start] + year_part * coverage[ceil(year)-year_start]
                return normal_like(value, coverage_i, 1. / std_err**2)
        else: # data is imputed from under 5 usage, so estimate standard error
            d['coverage'] = 1. - float(d['Per_0LLINs'])
            N = d['Total_HH'] or 1000
            d['sampling_error'] = d['coverage']*(1-d['coverage'])/sqrt(N)
            d['coverage_se'] = d['sampling_error']*s_r_c.value
            d['Year'] = d['Survey_Year1'] + .5
            @observed
            @stochastic(name='LLIN_Coverage_Imputation_%s_%s' % (d['Country'], d['Year']))
            def obs(value=d['coverage'],
                    year=d['Year'],
                    sampling_error=d['sampling_error'],
                    design_factor=s_r_c,
                    coverage=llin_coverage):
                year_part = year-floor(year)
                coverage_i = (1-year_part) * coverage[floor(year)-year_start] + year_part * coverage[ceil(year)-year_start]
                return normal_like(value, coverage_i, 1. / (design_factor * sampling_error)**2)
        coverage_obs.append(obs)
            

    for d in data.itn_coverage:
        if d['Country'] != c:
            continue

        d['coverage'] = 1. - float(d['Per_0ITNs'])

        if d['ITNs0_SE']: # data from survey, includes standard error
            d['coverage_se'] = d['ITNs0_SE']
            mean_survey_date = time.strptime(d['Mean_SvyDate'], '%d-%b-%y')
            d['Year'] = mean_survey_date[0] + mean_survey_date[1]/12.
            @observed
            @stochastic(name='ITN_Coverage_%s_%s' % (d['Country'], d['Year']))
            def obs(value=d['coverage'],
                    year=d['Year'],
                    std_err=d['coverage_se'],
                    coverage=itn_coverage):
                year_part = year-floor(year)
                coverage_i = (1-year_part) * coverage[floor(year)-year_start] + year_part * coverage[ceil(year)-year_start]
                return normal_like(value, coverage_i, 1. / std_err**2)

        else: # data from survey report, must calculate standard error
            d['Year'] = d['Survey_Year1'] + .5
            N = d['Total_HH'] or 1000
            d['sampling_error'] = d['coverage']*(1-d['coverage'])/sqrt(N)
            d['coverage_se'] = d['sampling_error']*s_r_c.value
            @observed
            @stochastic(name='ITN_Coverage_Report_%s_%s' % (d['Country'], d['Year']))
            def obs(value=d['coverage'],
                    year=d['Year'],
                    sampling_error=d['sampling_error'],
                    design_factor=s_r_c,
                    coverage=itn_coverage):
                year_part = year-floor(year)
                coverage_i = (1-year_part) * coverage[floor(year)-year_start] + year_part * coverage[ceil(year)-year_start]
                return normal_like(value, coverage_i, 1. / (design_factor * sampling_error)**2)

        coverage_obs.append(obs)


        # also take this opportinuty to set better initial values for the MCMC
        t = floor(d['Year'])-year_start
        cur_val = copy.copy(Hprime.value)
        cur_val[t] = max(.0001*pop[t], log(1-d['coverage']) * pop[t] / eta.value - H.value[t])
        log_Hprime.value = log(cur_val)

    vars += [coverage_obs]



       #################
      ### fit the model
     ###
    #################
    print 'running fit for net model in %s...' % c

    if settings.METHOD == 'MCMC':
        if settings.TESTING:
            map = MAP(vars)
            map.fit(method='fmin', iterlim=100, verbose=1)
        else:
            # just optimize some variables, to get reasonable initial conditions
            map = MAP([log_nm,
                       smooth_stocks, positive_stocks,
                       manufacturing_obs])
            map.fit(method='fmin_powell', verbose=1)

            map = MAP([log_nd,
                       smooth_stocks, positive_stocks,
                       admin_distribution_obs, household_distribution_obs,
                       household_stock_obs])
            map.fit(method='fmin_powell', verbose=1)

            map = MAP([log_nm, log_nd, log_Hprime,
                       smooth_stocks, positive_stocks, coverage_obs])
            map.fit(method='fmin_powell', verbose=1)

        for stoch in [s_m, s_d, e_d, pi, eta, zeta]:
            print '%s: %s' % (str(stoch), str(stoch.value))

        mc = MCMC(vars, verbose=1)
        mc.use_step_method(AdaptiveMetropolis, [eta, zeta])

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

    elif settings.METHOD == 'NormApprox':
        na = NormApprox(vars)
        na.fit(method='fmin_powell', tol=.00001, verbose=1)
        for stoch in [s_m, s_d, e_d, pi]:
            print '%s: %s' % (str(stoch), str(stoch.value))
        na.sample(1000)

    else:
        assert 0, 'Unknown estimation method'

    # save results in output file
    col_headings = ['Country', 'Year', 'Population',
                    'LLINs Shipped (Thousands)', 'LLINs Shipped Lower CI', 'LLINs Shipped Upper CI',
                    'LLINs Distributed (Thousands)', 'LLINs Distributed Lower CI', 'LLINs Distributed Upper CI',
                    'LLINs in Warehouse (Thousands)', 'LLINs in Warehouse Lower CI', 'LLINs in Warehouse Upper CI',
                    'LLINs Owned (Thousands)', 'LLINs Owned Lower CI', 'LLINs Owned Upper CI',
                    'non-LLIN ITNs Owned (Thousands)', 'non-LLIN ITNs Owned Lower CI', 'non-LLIN ITNs Owned Upper CI',
                    'LLIN Coverage (Percent)', 'LLIN Coverage Lower CI', 'LLIN Coverage Upper CI',
                    'ITN Coverage (Percent)', 'ITN Coverage Lower CI', 'ITN Coverage Upper CI']

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
            val = [nm.stats()['mean'][t]/1000] + list(nm.stats()['95% HPD interval'][t]/1000)
            val += [nd.stats()['mean'][t]/1000] + list(nd.stats()['95% HPD interval'][t]/1000)
        val += [W.stats()['mean'][t]/1000] + list(W.stats()['95% HPD interval'][t]/1000)
        val += [H.stats()['mean'][t]/1000] + list(H.stats()['95% HPD interval'][t]/1000)
        val += [Hprime.stats()['mean'][t]/1000] + list(Hprime.stats()['95% HPD interval'][t]/1000)
        val += [100*llin_coverage.stats()['mean'][t]] + list(100*llin_coverage.stats()['95% HPD interval'][t])
        val += [100*itn_coverage.stats()['mean'][t]] + list(100*itn_coverage.stats()['95% HPD interval'][t])
        f.write(','.join(['%.2f']*(len(col_headings)-3)) % tuple(val))
        f.write('\n')
    f.close()

    graphics.plot_posterior(country_id, c, pop,
                            s_m, s_d, e_d, pi, nm, nd, W, H, Hprime, s_r_c, eta, zeta, s_rb,
                            manufacturing_obs, admin_distribution_obs, household_distribution_obs,
                            itn_coverage, llin_coverage, hh_itn)

if __name__ == '__main__':
    usage = 'usage: %prog [options] country_id'
    parser = optparse.OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error('incorrect number of arguments')
    else:
        try:
            country_id = int(args[0])
        except ValueError:
            parser.error('country_id must be an integer')

        main(country_id)
