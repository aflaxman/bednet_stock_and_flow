""" Module to generate graphics for the stock-and-flow model for
bednet distribution
"""

import settings

from pylab import *
from pymc import *
import time
import copy

from data import Data
data = Data()

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
    xlabel('Annual Risk of LLIN Loss')
    ylabel('Probability Density')

    savefig('discard_prior.png')
    savefig('discard_prior.eps')

def plot_survey_design_prior(design_prior, data_vals):
    """ Generate a plot of the empirical prior for survey design effect

    Parameters
    ----------
    design_prior : the dict of empirical prior parameters for the
      discard rate

    Results
    -------
    Generates and saves graphics file 'survey_design_effect_prior.png'
    """
    figure(figsize=(6,4), dpi=settings.DPI)

    p_vals = arange(1., 3., .001)
    
    # plot empirical prior
    mu, tau = design_prior['mu'], design_prior['tau']
    plot(p_vals, exp([normal_like(p, mu, tau)  for p in p_vals]),
         linewidth=2, alpha=.75, color='blue', linestyle='solid',
         label='empirical prior')

    # plot data
    hist(data_vals, 20, normed=True,
         edgecolor='grey', facecolor='cyan', alpha=.75,
         label='posterior')

    xlabel('Survey Design Effect')
    ylabel('Probability Density')

    savefig('survey_design_effect_prior.png')
    savefig('survey_design_effect_prior.eps')

    
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
    xlabel('Bias in Admin LLIN flow')
    ylabel('Probability Density')

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
    xlabel('Error in Admin LLIN flow')
    ylabel('Probability Density')

    savefig('admin_priors.png')
    savefig('admin_priors.eps')

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

    savefig('admin_scatter.png')
    savefig('admin_scatter.eps')


def plot_neg_binom_priors(eta, alpha, factor_priors, data_dict):
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
    xlabel('Coverage Parameter')
    ylabel('probability density')

    ## plot prior for sigma
    subplot(1,2,2)

    # plot hyper-prior
    p_vals = arange(.001,6.,.001)
    map = MAP([alpha])
    plot(p_vals, exp([-map.func(p) for p in p_vals]),
         linewidth=2, alpha=.75, color='green', linestyle='dashed',
         label='hyper-prior')

    # plot posterior
    hist(alpha.trace(), 20, normed=True,
         edgecolor='grey', facecolor='cyan', alpha=.75,
         label='posterior')

    # plot empirical prior
    a, b = factor_priors['alpha']['alpha'], factor_priors['alpha']['beta']
    plot(p_vals, exp([gamma_like(p, a, b)  for p in p_vals]),
         linewidth=2, alpha=.75, color='blue', linestyle='solid',
         label='empirical prior')


    # decorate the figure
    xlabel('Dispersion Parameter')
    ylabel('probability density')
    
    savefig('neg_binom_priors.png')
    savefig('neg_binom_priors.eps')

def plot_neg_binom_fits():
    """ Generate figure that demonstrates how suitable the
    negative-binomial distribution is for fitting LLIN coverage"""
    import scipy.optimize

    figure(figsize=(8.5,8.5), dpi=settings.DPI)
    error_list = []
    
    for ii, d in enumerate(data.llin_num):
        subplot(5, 4, ii+1)

        # plot data
        x = arange(4)
        y = array([d['Per_%dLLINs' % i] for i in x])
        yerr = array([d['LLINs%d_SE' % i] for i in x])
        bar(x, y*100, yerr=yerr*100,
            width=.5, alpha=.5, color='blue',
            log=True)

        # fit data
        def rmse(theta):
            y_est = exp([negative_binomial_like(i, theta[0], theta[1])
                         for i in range(4)])
            return sqrt(mean((y - y_est) ** 2))
        mu, alpha = scipy.optimize.fmin(rmse, [5., 1.])
        print mu, alpha

        # plot fit
        y_est = exp([negative_binomial_like(i, mu, alpha) for i in x])
        bar(x+.4, y_est*100,
            width=.5, alpha=.5, color='green')

        text(4, 50, '%s, %d\nRMSE=%.2f' %
             (d['Country'], d['Survey_Year1'], 100*rmse([mu,alpha])) + '%',
             fontsize=10, horizontalalignment='right',verticalalignment='top')

        print 'errors: ', rmse([mu,alpha]), sqrt(sum(yerr**2))
        error_list.append(rmse([mu,alpha]))

        if ii >= 12:
            xticks(x+.5,[0,1,2,3], fontsize=8)
            xlabel('# of LLINs', fontsize=8)
        else:
            xticks([])

        if ii % 4 == 0:
            yticks([.1, 1, 10, 100], [.1, 1, 10, '100%'], fontsize=8)
            ylabel('% of Households', fontsize=8)
        else:
            yticks([])
            
        axis([0, 4, .1, 100])
    savefig('neg_binom_fits.png')
    savefig('neg_binom_fits.eps')
    print mean(error_list)
    
def plot_posterior(c_id, c, pop,
                   s_m, s_d, e_d, pi, nm, nd, W, H, Hprime, s_r_c, eta, alpha, s_rb,
                   manufacturing_obs, admin_distribution_obs, household_distribution_obs,
                   itn_coverage, llin_coverage, hh_itn, data, u5_use):
    from settings import year_start, year_end
    
    ### setup the canvas for our plots
    figure(**settings.FIGURE_OPTIONS)

    fontsize = 14
    small_fontsize = 12
    tiny_fontsize = 10

    def plot_fit(f, scale=1.e6, style='lines'):
        """ Plot the posterior mean and 95% UI
        """
        if style=='lines' or style=='alt lines':
            x = year_start + arange(len(f.value))
            y = f.stats()['mean']/scale
            lb = f.stats()['quantiles'][2.5]/scale
            ub = f.stats()['quantiles'][97.5]/scale
        elif style=='steps':
            x = []
            for ii in range(len(f.value)):
                x.append(ii)
                x.append(ii)

            y = (f.stats()['mean']/scale)[x]
            lb = (f.stats()['quantiles'][2.5]/scale)[x]
            ub = (f.stats()['quantiles'][97.5]/scale)[x]
            x = array(x[1:] + [ii+1]) + year_start
        else:
            raise ValueError, 'unrecognized style option: %s' % str(style)

        if style=='alt lines':
            plot(x, y, 'b:', alpha=.75)
            plot(x, lb, 'b:', alpha=.75)
            plot(x, ub, 'b:', alpha=.75)
        else:
            plot(x, y, 'k-', linewidth=2, label='Est Mean')

            x = np.concatenate((x, x[::-1]))
            y = np.concatenate((lb, ub[::-1]))
            fill(x, y, alpha=.95, label='Est 95% UI', facecolor='.8')

    def scatter_data(data_list, country, country_key, data_key,
                     error_key=None, error_val=0.,  pi=None, s_r=None,
                     fmt='go', scale=1.e6, label='', offset=0.):
        """ This convenience function is a little bit of a mess, but it
        avoids duplicating code for scatter-plotting various types of
        data, with various types of error bars
        """

        data_val = array([float(d[data_key]) for d in data_list if d[country_key] == c])
        if len(data_val) == 0:
            return

        if error_key:
            error_val = array([1.96*float(d[error_key]) \
                                   for d in data_list if d[country_key] == c])

        elif error_val:
            error_val = 1.96 * error_val * data_val
        x = array([float(d['Year']) for d in data_list if d[country_key] == c])
        errorbar(x + offset,
                 data_val/scale,
                 error_val/scale, fmt=fmt, alpha=.75, label=label,
                 markersize=20)

    def stoch_max(stoch):
        return max(stoch.stats()['95% HPD interval'][:,1])

    def decorate_figure(ystr='# of Nets (Per Capita)', ymax=False):
        """ Set the axis, etc."""
        l,r,b,t = axis()
        if ymax:
            t = ymax*1.2
        vlines(range(year_start,year_end), 0, t, color=(0,0,0), alpha=.3)
        axis([year_start, 2009, 0, t])
        ylabel(ystr, fontsize=fontsize)
        xticks([1999.5, 2001.5, 2003.5, 2005.5, 2007.5], ['1999', '2001', '2003', '2005', '2007'], fontsize=fontsize)

    def my_hist(stoch):
        """ Plot a histogram of the posterior distribution of a stoch"""
        hist(stoch.trace(), normed=True, log=False, label=str(stoch), alpha=.5)
        #l,r,b,t = axis()
        #vlines(ravel(stoch.stats()['quantiles'].values()), b, t,
        #       linewidth=2, alpha=.75, linestyle='dashed',
        #       color=['black', 'black', 'red', 'black', 'black'])
        yticks([])

        if str(stoch).find('distribution waiting time') == -1:
            a,l = xticks()
            l = [int(floor(x*100.)) for x in a]
            l[0] = str(l[0]) + '%'
            xticks([])
            xticks(floor(array(a)*100.)/100., l, fontsize=small_fontsize)
        #title(str(stoch), fontsize=small_fontsize)
        ylabel('probability density')

        leg = legend(loc='upper left')
        # the matplotlib.patches.Rectangle instance surrounding the legend
        frame = leg.get_frame()  
        frame.set_alpha(0.)    # set the frame face color to light gray
        frame.set_edgecolor('white')    # set the frame face color to light gray

        # matplotlib.text.Text instances
        for t in leg.get_texts():
            t.set_fontsize('small')    # the legend text fontsize


    def my_acorr(stoch):
        """ Plot the autocorrelation of the a stoch trace"""
        vals = copy.copy(stoch.trace())
        if shape(vals)[-1] == 1:
            vals = ravel(vals)

        if len(shape(vals)) > 1:
            vals = array(vals)[:,5]

        vals -= mean(vals, 0)
        acorr(vals, normed=True, maxlags=min(8, len(vals)))
        hlines([0],-8,8, linewidth=2, alpha=.7, linestyle='dotted')
        xticks([])
        #ylabel(str(stoch).replace('error in ', '').replace('data','err'),
        #       fontsize=tiny_fontsize)
        yticks([0,1], fontsize=tiny_fontsize)
        #title('mcmc autocorrelation', fontsize=small_fontsize)


    ### actual plotting code start here
    clf()

    figtext(.055, .5, 'a' + ' '*20 + c + ' '*20 + 'a', rotation=270, fontsize=100,
             bbox={'facecolor': 'black', 'alpha': 1},
              color='white', verticalalignment='center', horizontalalignment='right')

    stochs_to_plot = [s_m, s_d, e_d, pi, nm, nd, W, H, Hprime, s_r_c, eta, alpha, s_rb]

    cols = 4
    rows = len(stochs_to_plot)

    figtext(6.05/8., .925, 'mcmc trace', horizontalalignment='center', verticalalignment='top', fontsize=small_fontsize)
    figtext(6.85/8., .925, 'mcmc autocorrelation', horizontalalignment='center', verticalalignment='top', fontsize=small_fontsize)

    for ii, stoch in enumerate(stochs_to_plot):
        figtext(6.45/8., .097 + .814*(1-(ii+.0)/rows), str(stoch), horizontalalignment='center', verticalalignment='top', fontsize=small_fontsize)
        subplot(rows, cols*2, 2*cols - 1 + ii*2*cols)
        try:
            plot(stoch.trace(), linewidth=2, alpha=.5)
        except Exception, e:
            print 'Error: ', e

        xticks([])
        yticks([])

        subplot(rows, cols*2, 2*cols + ii*2*cols)
        try:
            my_acorr(stoch)
        except Exception, e:
            print 'Error: ', e

    subplot(5, cols, 0*cols + 3)
    my_hist(s_m)
    xticks([0., .05, .1, .15, .20, .25], ['0%', '5', '10', '15', '20', '25'], fontsize=small_fontsize)


    subplot(5, cols, 1*cols + 3)
    my_hist(s_d)
    my_hist(e_d)
    xticks([0., .5, 1., 1.5, 2.], ['0%', '50', '100', '150', '200'], fontsize=small_fontsize)

    subplot(5, cols, 2*cols + 3)
    my_hist(pi)
    xticks([0., .1, .2, .3, .4], ['0%', '10', '20', '30', '40'], fontsize=small_fontsize)

    subplot(5, cols, 3*cols + 3)
    my_hist(eta)
    my_hist(alpha)
    my_hist(s_r_c)
    xticks([])
    xticks([1, 3, 5, 7, 9], [1, 3, 5, 7, 9], fontsize=small_fontsize)

    subplot(5, cols, 4*cols + 3)
    my_hist(s_rb)
    xticks([0., .05, .1, .15, .20, .25], ['0%', '5', '10', '15', '20', '25'], fontsize=small_fontsize)


    rows = 5
    subplot(rows, cols/2, 0*(cols/2)+1)
    title('LLINs shipped (per capita)', fontsize=fontsize)
    plot_fit(nm, scale=pop, style='steps')
    if len(manufacturing_obs) > 0:
        scatter_data(data.llin_manu, c, 'Country', 'Manu_Itns', scale=mean(pop),
                     error_val=1.96 * s_m.stats()['mean'], offset=.5)
    decorate_figure(ymax=.3)

    subplot(rows, cols/2, 1*(cols/2)+1)
    title('LLINs in country, not in households (per capita)', fontsize=fontsize)
    plot_fit(W, scale=pop)
    decorate_figure(ymax=.3)

    subplot(rows, cols/2, 2*(cols/2)+1)
    title('LLINs distributed (per capita)', fontsize=fontsize)
    plot_fit(nd, style='steps', scale=pop)
    if len(admin_distribution_obs) > 0:
        label = 'Administrative Data'
        scatter_data(data.admin_llin, c, 'Country', 'Program_LLINs', scale=mean(pop),
                     error_val=1.96 * s_d.stats()['mean'], label=label, offset=.5)
    if len(household_distribution_obs) > 0:
        label = 'Survey Data'
        scatter_data(data.hh_llin_flow, c, 'Country', 'Total_LLINs', scale=mean(pop),
                     error_key='Total_st', fmt='bs',
                     pi=pi.stats()['mean'],
                     label=label, offset=.5)
    legend(loc='upper left')
    decorate_figure(ymax=.3)

    subplot(rows, cols/2, 4*(cols/2)+1)
    title('ITN and LLIN coverage', fontsize=fontsize)
    plot_fit(itn_coverage, scale=.01)
    plot_fit(llin_coverage, scale=.01, style='alt lines')
    plot_fit(u5_use, scale=.01)
    if max(itn_coverage.stats()['mean']) > .1:
        hlines([80], 1999, 2009, linestyle='dotted', color='blue', alpha=.5)

    scatter_data(data.llin_coverage, c, 'Country', 'coverage', 'coverage_se',
                 fmt='bs', scale=.01)
    scatter_data(data.itn_coverage, c, 'Country', 'coverage', 'coverage_se',
                 fmt='r^', scale=.01)
    decorate_figure(ystr='At least one net (%)', ymax=80)

    subplot(rows, cols/2, 3*(cols/2)+1)
    title('ITNs and LLINs in households (per capita)', fontsize=fontsize)
    plot_fit(hh_itn, scale=pop)
    plot_fit(H, scale=pop, style='alt lines')
    for d in data.hh_llin_stock:
        mean_survey_date = time.strptime(d['Mean_SvyDate'], '%d-%b-%y')
        d['Year'] = mean_survey_date[0] + mean_survey_date[1]/12.
    scatter_data(data.hh_llin_stock, c, 'Country', 'SvyIndex_LLINs', scale=mean(pop),
                 error_key='SvyIndexLLINs_SE', fmt='bs')
    decorate_figure(ymax=.3)

    savefig('bednets_%s_%d_%s.png' % (c, c_id, time.strftime('%Y_%m_%d_%H_%M')))
    
if __name__ == '__main__':
    plot_neg_binom_fits()
