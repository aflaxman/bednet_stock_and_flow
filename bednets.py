"""  Script to fit stock-and-flow compartmental model of bednet distribution
"""

from pylab import *
from pymc import *

import copy

def load_csv(fname):
    """ Quick function to load each row of a csv file as a dict
    Parameters
    ----------
    fname : str
      name of the .csv file to load
    
    Results
    -------
    returns a list of dicts, one list item for each row of the csv
    (keyed by the first row)

    Notes
    -----
    every value in the dict will be a string.  remember to convert
    numbers to floats before doing any math with them.
    """
    import csv
    f = open(fname)
    csv_f = csv.DictReader(f)
    data = [d for d in csv_f]
    f.close()

    return data


### load all data from csv files
manufacturing_data = load_csv('manuitns_forabie07072009.csv')
administrative_distribution_data = load_csv('programitns_forabie07072009.csv')
household_stock_data = load_csv('stock_surveyitns_forabie07072009.csv')
household_distribution_data = load_csv('surveyitns_forabie7072009.csv')
retention_data = load_csv('retention07072009.csv')

### setup the canvas for our plots
figure(figsize=(88, 68), dpi=75)

### pick the country of interest
country_set = set([d['Country'] for d in manufacturing_data])
print 'fitting models for %d countries...' % len(country_set)

### set years for estimation
year_start = 1999
year_end = 2010

for c in sorted(country_set):
    ### find some descriptive statistics to use as priors
    nd_all = [float(d['Program_Itns']) for d in administrative_distribution_data \
                  if d['Country'] == c] \
                  + [float(d['Survey_Itns']) for d in household_distribution_data \
                         if d['Name'] == c and d['Year'] == d['Survey_Year']]
    # if there is no distribution data, make some up
    if len(nd_all) == 0:
        nd_all = [ 1000. ]

    nd_min = min(nd_all)
    nd_avg = mean(nd_all)
    nd_ste = std(nd_all)

    nm_all = [float(d['Manu_Itns']) for d in manufacturing_data if d['Country'] == c]
    # if there is no manufacturing data, make some up
    if len(nm_all) == 0:
        nm_all = [ 1000. ]
    nm_min = min(nm_all)
    nm_avg = mean(nm_all)


    ### setup the model variables
    vars = []
       #######################
      ### compartmental model
     ###
    #######################

    logit_p_l = Normal('logit(Pr[net is lost])', mu=logit(.05), tau=10.)
    p_l = InvLogit('Pr[net is lost]', logit_p_l, verbose=1)

    vars += [logit_p_l, p_l]

    
    s_r = Gamma('error in retention data', 20., 20./.05, value=.05)
    s_m = Gamma('error in manufacturing data', 20., 20./.05, value=.05)
    s_d = Gamma('error in admin dist data', 20., 20./.05, value=.05)

    vars += [s_r, s_m, s_d]

    
    nd = Lognormal('nets distributed', mu=log(nd_min) * ones(year_end-year_start-1), tau=1.)
    nm = Lognormal('nets manufactured', mu=log(nm_min) * ones(year_end-year_start-1), tau=1.)

    W_0 = Lognormal('initial warehouse net stock', mu=log(1000), tau=10., value=1000)
    H_0 = Lognormal('initial household net stock', mu=log(1000), tau=10., value=1000)

    @deterministic(name='warehouse net stock')
    def W(W_0=W_0, nm=nm, nd=nd):
        W = zeros(year_end-year_start)
        W[0] = W_0
        for t in range(year_end - year_start - 1):
            W[t+1] = W[t] + nm[t] - nd[t]
        return W

    @deterministic(name='household net stock')
    def H(H_0=H_0, nd=nd, p_l=p_l):
        H = zeros(year_end-year_start)
        H[0] = H_0
        for t in range(year_end - year_start - 1):
            H[t+1] = H[t] * (1 - p_l) + nd[t]
        return H

    vars += [nd, nm, W_0, H_0, W, H]

    
    # set initial condition on W_0 to have no stockouts
    if min(W.value) < 0:
        W_0.value = W_0.value - 2*min(W.value)

       #####################
      ### additional priors
     ###
    #####################

    @potential
    def smooth_W(W=W):
        return normal_like(diff(log(maximum(W,10000))), 0., 1. / (1.)**2)

    @potential
    def smooth_H(H=H):
        return normal_like(diff(log(maximum(H,10000))), 0., 1. / (1.)**2)

    @potential
    def positive_stocks(H=H, W=W):
        return -1000 * (dot(H**2, H < 0) + dot(W**2, W < 0))

    vars += [smooth_H, smooth_W, positive_stocks]


       #####################
      ### statistical model
     ###
    #####################


    ### observed nets manufactured

    manufacturing_obs = []
    for d in manufacturing_data:
        if d['Country'] != c:
            continue

        @observed
        @stochastic(name='manufactured_%s_%s' % (d['Country'], d['Year']))
        def obs(value=float(d['Manu_Itns']), year=int(d['Year']), nm=nm, s_m=s_m):
            return normal_like(value / nm[year - year_start], 1., 1. / s_m**2)
        manufacturing_obs.append(obs)

        # also take this opportinuty to set better initial values for the MCMC
        cur_val = copy.copy(nm.value)
        cur_val[int(d['Year']) - year_start] = float(d['Manu_Itns'])
        nm.value = cur_val

    vars += [manufacturing_obs]



    ### observed nets distributed

    admin_distribution_obs = []
    for d in administrative_distribution_data:
        if d['Country'] != c:
            continue

        @observed
        @stochastic(name='administrative_distribution_%s_%s' % (d['Country'], d['Year']))
        def obs(value=float(d['Program_Itns']), year=int(d['Year']), nd=nd, s_d=s_d):
            return normal_like(value / nd[year-year_start], 1., 1. / s_d**2)
        admin_distribution_obs.append(obs)

        # also take this opportinuty to set better initial values for the MCMC
        cur_val = copy.copy(nd.value)
        cur_val[int(d['Year']) - year_start] = float(d['Program_Itns'])
        nd.value = cur_val

    vars += [admin_distribution_obs]


    household_distribution_obs = []
    for d in household_distribution_data:
        if d['Name'] != c:
            continue

        d2_i = float(d['Survey_Itns'])
        estimate_year = int(d['Year'])
        survey_year = int(d['Survey_Year'])
        s_d2_i = float(d['Ste_Survey_Itns'])
        @observed
        @stochastic(name='household_distribution_%s_%s' % (d['Name'], d['Year']))
        def obs(value=d2_i,
                estimate_year=estimate_year,
                survey_year=survey_year,
                survey_err=s_d2_i,
                retention_err=s_r,
                nd=nd, p_l=p_l):
            return normal_like(
                value,
                nd[estimate_year - year_start] * (1 - p_l) ** (survey_year - estimate_year),
                1./ (survey_err * (1 + (survey_year - estimate_year) * retention_err))**2)
        household_distribution_obs.append(obs)

        # also take this opportinuty to set better initial values for the MCMC
        cur_val = copy.copy(nd.value)
        cur_val[estimate_year - year_start] = d2_i / (1 - p_l.value)**(survey_year - estimate_year)
        nd.value = cur_val

    vars += [household_distribution_obs]



    ### observed household net stocks
    household_stock_obs = []
    for d in household_stock_data:
        if d['Name'] != c:
            continue

        @observed
        @stochastic(name='household_stock_%s_%s' % (d['Name'], d['Year']))
        def obs(value=float(d['Survey_Itns']),
                year=int(d['Year']),
                std_err=float(d['Ste_Survey_Itns']),
                H=H):
            return normal_like(value, H[year-year_start], 1. / std_err ** 2)
        household_stock_obs.append(obs)

    vars += [household_stock_obs]


    ### observed net retention 

    retention_obs = []
    for d in retention_data:
        @observed
        @stochastic(name='retention_%s_%s' % (d['Name'], d['Year']))
        def obs(value=float(d['Retention_Rate']),
                T_i=float(d['Follow_up_Time']),
                p_l=p_l, s_r=s_r):
            return normal_like(value, (1. - p_l) ** T_i, 1. / s_r**2)
        retention_obs.append(obs)

    vars += [retention_obs]



       #################
      ### fit the model
     ###
    #################
    print 'running fit for net model in %s...' % c

    method = 'MCMC'
    #method = 'NormApprox'

    if method == 'MCMC':
        map = MAP(vars)
        map.fit(method='fmin_powell', verbose=1)
        for stoch in [s_m, s_d, s_r, p_l]:
            print '%s: %f' % (stoch, stoch.value)

        mc = MCMC(vars, verbose=1)
        #mc.use_step_method(AdaptiveMetropolis, [nd, nm, W_0, H_0], verbose=0)
        #mc.use_step_method(AdaptiveMetropolis, nd, verbose=0)
        #mc.use_step_method(AdaptiveMetropolis, nm, verbose=0)

        try:
            iter = 100
            thin = 500
            burn = 20000
            mc.sample(iter*thin+burn, burn, thin)
        except:
            pass

    elif method == 'NormApprox':
        na = NormApprox(vars)
        na.fit(method='fmin_powell', tol=.00001, verbose=1)
        for stoch in [s_m, s_d, s_r, p_l]:
            print '%s: %f' % (stoch, stoch.value)
        na.sample(1000)


       ######################
      ### plot the model fit
     ###
    ######################
    fontsize = 12
    small_fontsize = 10
    tiny_fontsize = 7

    def plot_fit(f, scale=1.e6):
        """ Plot the posterior mean and 95% UI
        """
        plot(year_start + arange(len(f.value)),
             f.stats()['mean']/scale, 'k-', linewidth=2, label='Est Mean')

        x = np.concatenate((year_start + arange(len(f.value)),
                            year_start + arange(len(f.value))[::-1]))
        y = np.concatenate((f.stats()['quantiles'][2.5]/scale,
                            f.stats()['quantiles'][97.5][::-1]/scale))
        fill(x, y, alpha=.95, label='Est 95% UI', facecolor='.8', alpha=.5)

    def scatter_data(data_list, country, country_key, data_key,
                     error_key=None, error_val=None,  p_l=None, s_r=None,
                     fmt='go', scale=1.e6, label=''):
        """ This convenience function is a little bit of a mess, but it
        avoids duplicating code for scatter-plotting various types of
        data, with various types of error bars
        """

        if p_l == None:
            data_val = array([float(d[data_key]) for d in data_list if d[country_key] == c])
        else:
            # account for the nets lost prior to survey
            data_val = array([
                    float(d[data_key]) / (1-p_l)**(int(d['Survey_Year']) - int(d['Year']))
                    for d in data_list if d[country_key] == c])

        if error_key:
            if s_r == None:
                error_val = array([1.96*float(d[error_key]) \
                                       for d in data_list if d[country_key] == c])
            else:
                error_val = array([1.96*float(d[error_key])
                                   * (1 + (int(d['Survey_Year']) - int(d['Year'])) * s_r) \
                                       for d in data_list if d[country_key] == c])

        elif error_val:
            error_val = 1.96 * error_val * data_val
        errorbar([float(d['Year']) for d in data_list if d[country_key] == c],
                 data_val/scale,
                 error_val/scale, fmt=fmt, alpha=.95, label=label)


    def decorate_figure():
        """ Set the axis, etc."""
        l,r,b,t = axis()
        vlines(range(year_start,year_end), 0, t, color=(0,0,0), alpha=.3)
        axis([year_start, year_end-1, 0, t])
        ylabel('# of Nets (Millions)', fontsize=fontsize)
        xticks([2001, 2003, 2005, 2007], ['2001', '2003', '2005', '2007'], fontsize=fontsize)

    def my_hist(stoch):
        """ Plot a histogram of the posterior distribution of a stoch"""
        hist(stoch.trace(), normed=True, log=False)
        l,r,b,t = axis()
        vlines(ravel(stoch.stats()['quantiles'].values()), b, t,
               linewidth=2, alpha=.75, linestyle='dashed',
               color=['k', 'k', 'r', 'k', 'k'])
        yticks([])
        a,l = xticks()
        l = [int(x*100) for x in a]
        l[0] = str(l[0]) + '%'
        xticks(floor(a*100.)/100., l, fontsize=fontsize)
        title(str(stoch), fontsize=fontsize)
        ylabel('probability density')
        

    def my_acorr(stoch):
        """ Plot the autocorrelation of the a stoch trace"""
        vals = copy.copy(stoch.trace())

        if shape(vals)[-1] == 1:
            vals = ravel(vals)

        if len(shape(vals)) > 1:
            vals = vals[5]

        vals -= mean(vals)
        acorr(vals, normed=True, maxlags=min(8, len(vals)))
        hlines([0],-8,8, linewidth=2, alpha=.7, linestyle='dotted')
        xticks([])
        ylabel(str(stoch).replace('error in ', '').replace('data','err'),
               fontsize=tiny_fontsize)
        yticks([0,1], fontsize=tiny_fontsize)
        title('mcmc autocorrelation', fontsize=small_fontsize)


    ### actual plotting code start here
    clf()

    figtext(.055, .5, 'a' + ' '*10 + c + ' '*10 + 'a', rotation=270, fontsize=100,
             bbox={'facecolor': 'black', 'alpha': 1},
              color='white', verticalalignment='center', horizontalalignment='right')
    cols = 4

    for ii, stoch in enumerate([p_l, s_r, s_m, s_d, nm, nd, W, H]):
        subplot(8, cols*2, 2*cols - 1 + ii*2*cols)
        try:
            plot(stoch.trace(), linewidth=2, alpha=.5)
        except Exception, e:
            print 'Error: ', e

        xticks([])
        yticks([])
        title('mcmc trace', fontsize=small_fontsize)
        ylabel(str(stoch).replace('error in ', '').replace('data','err'),
               fontsize=tiny_fontsize)

        subplot(8, cols*2, 2*cols + ii*2*cols)
        try:
            my_acorr(stoch)
        except Exception, e:
            print 'Error: ', e

        try:
            if stoch in [p_l, s_r, s_m, s_d]:
                subplot(4, cols, ii*cols + 3)
                my_hist(stoch)
        except Exception, e:
            print 'Error: ', e
    
    subplot(4, cols/2, 1)
    title('nets manufactured', fontsize=fontsize)
    plot_fit(nm)
    if len(manufacturing_obs) > 0:
        try:
            scatter_data(manufacturing_data, c, 'Country', 'Manu_Itns',
                         error_val=1.96 * s_m.stats()['mean'])
        except Exception, e:
            print 'Error: ', e
            scatter_data(manufacturing_data, c, 'Country', 'Manu_Itns',
                         error_val=1.96 * s_m.value)
    decorate_figure()


    subplot(4, cols/2, 2*(cols/2)+1)
    title('nets distributed', fontsize=fontsize)
    plot_fit(nd)
    if len(admin_distribution_obs) > 0:
        label = 'Administrative Data'
        try:
            scatter_data(administrative_distribution_data, c, 'Country', 'Program_Itns',
                         error_val=1.96 * s_d.stats()['mean'], label=label)
        except Exception, e:
            print 'Error: ', e
            scatter_data(administrative_distribution_data, c, 'Country', 'Program_Itns',
                         error_val=1.96 * s_m.value, label=label)
    if len(household_distribution_obs) > 0:
        label = 'Survey Data'
        try:
            scatter_data(household_distribution_data, c, 'Name', 'Survey_Itns',
                         error_key='Ste_Survey_Itns', fmt='bs',
                         p_l=p_l.stats()['mean'][0], s_r=s_r.stats()['mean'],
                         label=label)
        except Exception, e:
            print 'Error: ', e
            scatter_data(household_distribution_data, c, 'Name', 'Survey_Itns',
                         error_key='Ste_Survey_Itns', fmt='bs',
                         p_l=p_l.value, s_r=s_r.value,
                         label=label)
    legend(loc='upper left')
    decorate_figure()


    subplot(4, cols/2, (cols/2)+1)
    title('nets in warehouse', fontsize=fontsize)
    plot_fit(W)
    decorate_figure()


    subplot(4, cols/2, 3*(cols/2)+1)
    title('nets in households', fontsize=fontsize)
    plot_fit(H)
    if len(household_stock_obs) > 0:
        scatter_data(household_stock_data, c, 'Name', 'Survey_Itns',
                     error_key='Ste_Survey_Itns', fmt='bs')
    decorate_figure()

    savefig('bednets_%s.png' % c)
