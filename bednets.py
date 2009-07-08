"""  Script to fit stock-and-flow compartmental model of bednet distribution
"""

from pylab import *
from pymc import *

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

### pick the country of interest
c = 'Zambia'

### find some averages to use as priors
nd_all = [float(d['Program_Itns']) for d in administrative_distribution_data \
             if d['Country'] == c]
nd_avg = mean(nd_all)
nd_ste = std(nd_all)

vars = []

   #######################
  ### compartmental model
 ###
#######################

p_l = Beta('Pr[net is lost]', alpha=2., beta=10.)

# TODO: consider choosing better priors
nd = Lognormal('nets distributed', mu=log(1000) * ones(10), tau=1.)
nm = Lognormal('nets manufactured', mu=log(1000) * ones(10), tau=1.)

# TODO: consider choosing better priors
W_0 = Lognormal('initial warehouse net stock', mu=log(1000), tau=10.)
H_0 = Lognormal('initial household net stock', mu=log(1000), tau=10.)

@deterministic(name='warehouse net stock')
def W(W_0=W_0, nm=nm, nd=nd):
    W = zeros(10)
    W[0] = W_0
    for t in range(9):
        W[t+1] = W[t] + nm[t] - nd[t]
    return W

@deterministic(name='household net stock')
def H(H_0=H_0, nd=nd, p_l=p_l):
    H = zeros(10)
    H[0] = H_0
    for t in range(9):
        H[t+1] = H[t] * (1 - p_l) + nd[t]
    return H

vars += [p_l, nd, nm, W_0, H_0, W, H]


   #####################
  ### additional priors
 ###
#####################

# TODO: consider choosing better priors
s_r = Gamma('standard error in retention rate', 10., 10./.01)
s_m = Gamma('error in manufacturing data', 100., 100./.01, value=.001)
s_d = Gamma('error in administrative distribution data', 100., 100./.01, value=.001)

vars += [s_r, s_m, s_d]

@potential
def smooth_H(H=H):
    return normal_like(H[:-1] / H[1:], 1., 1.)

@potential
def smooth_W(W=W):
    return normal_like(W[:-1] / W[1:], 1., 1.)

@potential
def positive_stocks(H=H, W=W):
    return -1000 * (dot(H**2, H < 0) + dot(W**2, W < 0))

vars += [smooth_H, smooth_W, positive_stocks]


   #####################
  ### statistical model
 ###
#####################

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


### observed nets manufactured

manufacturing_obs = []
for d in manufacturing_data:
    if d['Country'] != c:
        continue
    
    @observed
    @stochastic(name='manufactured_%s_%s' % (d['Country'], d['Year']))
    def obs(value=float(d['Manu_Itns']), year=int(d['Year']), nm=nm, s_m=s_m):
        return normal_like(value / nm[year-2000], 1., 1. / s_m**2)
    manufacturing_obs.append(obs)

vars += [manufacturing_obs]


### observed nets distributed

admin_distribution_obs = []
for d in administrative_distribution_data:
    if d['Country'] != c:
        continue

    @observed
    @stochastic(name='administrative_distribution_%s_%s' % (d['Country'], d['Year']))
    def obs(value=float(d['Program_Itns']), year=int(d['Year']), nd=nd, s_d=s_d):
        return normal_like(value / nd[year-2000], 1., 1./ s_d**2)
    admin_distribution_obs.append(obs)

vars += [admin_distribution_obs]


household_distribution_obs = []
for d in household_distribution_data:
    if d['Name'] != c:
        continue

    @observed
    @stochastic(name='household_distribution_%s_%s' % (d['Name'], d['Year']))
    def obs(value=float(d['Survey_Itns']),
            estimate_year=int(d['Year']),
            survey_year=int(d['Survey_Year']),
            std_err=float(d['Ste_Survey_Itns']),
            nd=nd, p_l=p_l):
        return normal_like(value,
                           nd[estimate_year - 2000] * \
                               (1 - p_l) ** (survey_year - estimate_year),
                           1./ std_err**2)
    household_distribution_obs.append(obs)

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
        return normal_like(value, H[year-2000], 1./ std_err**2)
    household_stock_obs.append(obs)

vars += [household_stock_obs]


   #################
  ### fit the model
 ###
#################
print 'running MCMC...'
mc = MCMC(vars)
mc.sample(20000,10000,20)



   ######################
  ### plot the model fit
 ###
######################
def plot_fit(f, scale=1.e6):
    plot(range(2000,2010), f.stats()['mean']/scale, 'k', linewidth=2, alpha=.9)
    plot(range(2000,2010), f.stats()['quantiles'][2.5]/scale, 'k:', linewidth=2, alpha=.5)
    plot(range(2000,2010), f.stats()['quantiles'][97.5]/scale, 'k:', linewidth=2, alpha=.5)

def scatter_data(data_list, country, country_key, data_key,
                 error_key=None, error_val=None, fmt='gs', scale=1.e6):
    data_val = array([float(d[data_key]) for d in data_list if d[country_key] == c])
    
    if error_key:
        error_val = array([1.96*float(d[error_key]) \
                               for d in data_list if d[country_key] == c])
    elif error_val:
        error_val = 1.96 * error_val * data_val
    errorbar([float(d['Year']) for d in data_list if d[country_key] == c],
             data_val/scale,
             error_val/scale, fmt=fmt, alpha=.5)

def decorate_figure():
    axis([2000,2010,0,4])
    xticks([2000, 2005, 2010])

def my_hist(stoch):
    hist(stoch.trace(), normed=True, log=False)
    l,r,b,t = axis()
    vlines(stoch.stats()['quantiles'].values(), b, t,
           linewidth=2, alpha=.5, linestyle='dotted',
           color=['k', 'k', 'r', 'k', 'k'])
    yticks([])

clf()

subplot(2,3,1)
title('nets manufactured')
plot_fit(nm)
scatter_data(manufacturing_data, c, 'Country', 'Manu_Itns',
             error_val=1.96 * s_m.stats()['mean'])
decorate_figure()


subplot(2,3,2)
title('nets distributed')
plot_fit(nd)
scatter_data(administrative_distribution_data, c, 'Country', 'Program_Itns',
             error_val=1.96 * s_d.stats()['mean'])
scatter_data(household_distribution_data, c, 'Name', 'Survey_Itns',
             error_key='Ste_Survey_Itns', fmt='bs')
decorate_figure()


subplot(2,3,4)
title('nets in warehouse')
plot_fit(W)
decorate_figure()


subplot(2,3,5)
title('nets in households')
plot_fit(H)
scatter_data(household_stock_data, c, 'Name', 'Survey_Itns',
             error_key='Ste_Survey_Itns', fmt='bs')
decorate_figure()

subplot(2,3,3)
title(str(p_l))
vlines(p_l.stats()['quantiles'].values(), 1, 1000,
       linewidth=2, alpha=.5, linestyle='dashed',
       color=['k', 'b', 'r', 'b', 'k'])
hist(p_l.trace(), normed=True, log=True)

for ii, stoch in enumerate([s_r, s_m, s_d]):
    subplot(8, 3, (6+ii)*3)
    my_hist(stoch)
    title(str(stoch), fontsize=8)
    
