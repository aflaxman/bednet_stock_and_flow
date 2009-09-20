""" Module to generate graphics for the stock-and-flow model for
bednet distribution
"""

from pylab import *
from pymc import *

import data
import settings

def plot_discard_prior(pi, discard_prior):
    """ Generate a plot of the hyper-prior and empirical prior discard
    rate

    Parameters
    ----------
    pi : the hyper-prior stoch, after sampling from it's posterior
      distribution via MCMC
    discard_prior : the dict of empirical prior parameters for the
      discard rate

    Results
    -------
    Generates and saves graphics file 'discard_prior.png'
    """
    figure(figsize=(6,4), dpi=settings.DPI)
    
    # plot hyper-prior
    p_vals = arange(0.001,1,.001)
    map = MAP([pi])
    plot(p_vals, exp([-map.func(p) for p in p_vals]),
         linewidth=2, alpha=.75, color='green', linestyle='dashed',
         label='hyper-prior')

    # plot posterior
    hist(pi.trace(), 20, normed=True,
         edgecolor='grey', facecolor='cyan', alpha=.75,
         label='posterior')

    # plot empirical prior
    alpha, beta = discard_prior['alpha'], discard_prior['beta']
    plot(p_vals, exp([beta_like(p, alpha, beta)  for p in p_vals]),
         linewidth=2, alpha=.75, color='blue', linestyle='solid',
         label='empirical prior')

    # find the plot bounds
    l, r, b, t = axis()
    
    # plot data
    data_vals = []
    for d in data.retention:
        data_vals.append(1. - d['Retention_Rate'] ** (1 / d['Follow_up_Time']))

    vlines(data_vals, 0, t+1,
           linewidth=2, alpha=.75, color='black',
           linestyle='solid', label='data')

    # decorate the figure
    axis([0, .2, 0, t])
    #legend()  # this doesnt work
    title('Annual Risk of LLIN Loss')

    savefig('discard_prior.png')

def plot_admin_priors(eps, sigma, admin_priors, data_dict):
    figure(figsize=(8.5,4), dpi=settings.DPI)

    ## plot prior for eps
    subplot(1,2,1)

    # plot hyper-prior
    p_vals = arange(-2.,2,.001)
    map = MAP([eps])
    plot(p_vals, exp([-map.func(p) for p in p_vals]),
         linewidth=2, alpha=.75, color='green', linestyle='dashed',
         label='hyper-prior')

    # plot posterior
    hist(eps.trace(), 20, normed=True,
         edgecolor='grey', facecolor='cyan', alpha=.75,
         label='posterior')

    # plot empirical prior
    mu, tau = admin_priors['eps']['mu'], admin_priors['eps']['tau']
    plot(p_vals, exp([normal_like(p, mu, tau)  for p in p_vals]),
         linewidth=2, alpha=.75, color='blue', linestyle='solid',
         label='empirical prior')

    # decorate the figure
    title('Bias in Admin LLIN flow')

    ## plot prior for sigma
    subplot(1,2,2)

    # plot hyper-prior
    p_vals = arange(.001,4.,.001)
    map = MAP([sigma])
    plot(p_vals, exp([-map.func(p) for p in p_vals]),
         linewidth=2, alpha=.75, color='green', linestyle='dashed',
         label='hyper-prior')

    # plot posterior
    hist(sigma.trace(), 20, normed=True,
         edgecolor='grey', facecolor='cyan', alpha=.75,
         label='posterior')

    # plot empirical prior
    mu, tau = admin_priors['sigma']['mu'], admin_priors['sigma']['tau']
    plot(p_vals, exp([normal_like(p, mu, tau)  for p in p_vals]),
         linewidth=2, alpha=.75, color='blue', linestyle='solid',
         label='empirical prior')


    # decorate the figure
    title('Error in Admin LLIN flow')

    savefig('admin_priors.png')

    figure(figsize=(8.5,8.5), dpi=settings.DPI)

    y = array([data_dict[k]['obs'] for k in sorted(data_dict.keys())])
    x = array([data_dict[k]['truth'] for k in sorted(data_dict.keys())])
    y_e = array([data_dict[k]['se'] for k in sorted(data_dict.keys())])
    plot(x, y, 'o', alpha=.9)
    #errorbar(x, y, 1.96*y_e, fmt=',', alpha=.9, linewidth=1.5)
    
    loglog([1000,5e6], [1000,5e6], 'k--', alpha=.5, linewidth=2, label='y=x')

    #x_predicted = arange(1000, 5e6, 1000)
    #y_predicted = exp(admin_priors['eps']['mu'] + log(x_predicted))
    #plot(x_predicted, y_predicted, 'r:', alpha=.75, linewidth=2, label='predicted value')

    #x = np.concatenate((x_predicted, x_predicted[::-1]))
    #y = np.concatenate((
    #    exp(admin_priors['eps']['mu'] + 1.96*admin_priors['eps']['std'] + log(x_predicted)),
    #    exp(admin_priors['eps']['mu'] - 1.96*admin_priors['eps']['std'] + log(x_predicted))[::-1]))
    #fill(x, y, label='Sys Err 95% UI', facecolor=(.8,.4,.4), alpha=.5)

#     x = np.concatenate(((1 + mean_e_d - 1.96*prior_e_d.stats()['standard deviation']) * y_predicted * (1 - 1.96*prior_s_d.stats()['mean']),
#                        ((1 + mean_e_d + 1.96*prior_e_d.stats()['standard deviation']) * y_predicted * (1 + 1.96*prior_s_d.stats()['mean']))[::-1]))
#     x = maximum(10, x)
#     fill(x, y, alpha=.95, label='Total Err 95% UI', facecolor='.8', alpha=.5)

    axis([1e4,5e6,1e4,5e6])
    #legend(loc='lower right')
    xlabel('LLINs distributed according to household survey')
    ylabel('LLINs distributed according to administrative data')
    #for k in data_dict:
    #    d = data_dict[k]
    #    text(d['truth'], d['obs'], ' %s, %s' % k, fontsize=12, alpha=.5, verticalalignment='center')



def plot_cov_and_zif_priors(eta, zeta, factor_priors, data_dict):
    figure(figsize=(8.5,4), dpi=settings.DPI)

    ## plot prior for eta
    subplot(1,2,1)

    # plot hyper-prior
    p_vals = arange(0.,10.,.001)
    map = MAP([eta])
    plot(p_vals, exp([-map.func(p) for p in p_vals]),
         linewidth=2, alpha=.75, color='green', linestyle='dashed',
         label='hyper-prior')

    # plot posterior
    hist(eta.trace(), 20, normed=True,
         edgecolor='grey', facecolor='cyan', alpha=.75,
         label='posterior')

    # plot empirical prior
    mu, tau = factor_priors['eta']['mu'], factor_priors['eta']['tau']
    plot(p_vals, exp([normal_like(p, mu, tau)  for p in p_vals]),
         linewidth=2, alpha=.75, color='blue', linestyle='solid',
         label='empirical prior')

    # decorate the figure
    title('Coverage Factor ($\eta_c$)')

    ## plot prior for sigma
    subplot(1,2,2)

    # plot hyper-prior
    p_vals = arange(.001,1.,.001)
    map = MAP([zeta])
    plot(p_vals, exp([-map.func(p) for p in p_vals]),
         linewidth=2, alpha=.75, color='green', linestyle='dashed',
         label='hyper-prior')

    # plot posterior
    hist(zeta.trace(), 20, normed=True,
         edgecolor='grey', facecolor='cyan', alpha=.75,
         label='posterior')

    # plot empirical prior
    alpha, beta = factor_priors['zeta']['alpha'], factor_priors['zeta']['beta']
    plot(p_vals, exp([beta_like(p, alpha, beta)  for p in p_vals]),
         linewidth=2, alpha=.75, color='blue', linestyle='solid',
         label='empirical prior')


    # decorate the figure
    title('Zero-Inflation Factor ($\zeta_c$)')

    savefig('cov_and_zif_priors.png')
