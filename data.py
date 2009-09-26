""" Module to handle loading the data from the stock-and-flow model
for bednet distribution
"""

import csv
import time
from numpy import zeros, mean

import settings

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
    f = open(settings.PATH + fname)
    csv_f = csv.DictReader(f)
    data = [d for d in csv_f]
    f.close()

    # make sure all floats are floats
    for d in data:
        for k in d.keys():
            try:
                d[k] = float(d[k].replace(',',''))
            except ValueError:
                pass

    return data

class Data:
    def __init__(self):
        ### load all data from csv files
        self.retention = load_csv('reten.csv')
        self.design = load_csv('design.csv')

        self.llin_manu = load_csv('manuitns.csv')
        self.admin_llin = load_csv('adminllins_itns.csv')

        self.hh_llin_stock = load_csv('stock_llins.csv')
        self.hh_llin_flow = load_csv('flow_llins.csv')

        # add mean survey date to hh_llin data
        for d in self.hh_llin_stock + self.hh_llin_flow:
            mean_survey_date = time.strptime(d['Mean_SvyDate'], '%d-%b-%y')
            d['mean_survey_date'] = mean_survey_date[0] + mean_survey_date[1]/12.

        self.llin_coverage = load_csv('llincc.csv')
        self.itn_coverage = load_csv('itncc.csv')
        self.holdout_itn_coverage = []

        self.llin_num = load_csv('numllins.csv')

        self.population = load_csv('pop.csv')

        self.countries = set([d['Country'] for d in self.population])

    def population_for(self, c, year_start, year_end):
        pop_vec = zeros(year_end - year_start)
        for d in self.population:
            if d['Country'] == c:
                pop_vec[int(d['Year']) - year_start] = d['Pop']*1000

        # since we might be predicting into the future, fill in population with last existing value
        for ii in range(1, year_end-year_start):
            if pop_vec[ii] == 0.:
                pop_vec[ii] = pop_vec[ii-1]

        return pop_vec

    def holdout(self, holdout_frac=.2):
        """ choose a pseudo-random subset of itn coverage data to use for
        measuring predictive validity"""
        from numpy import sqrt
        import random
        random.seed(12345)

        self.holdout_itn_coverage = random.sample(self.itn_coverage,
                                                  int(holdout_frac * len(self.itn_coverage)))
        self.itn_coverage = [d for d in self.itn_coverage if not d in self.holdout_itn_coverage]

        for d in self.holdout_itn_coverage:
            try:
                mean_survey_date = time.strptime(d['Mean_SvyDate'], '%d-%b-%y')
                d['Year'] = mean_survey_date[0] + mean_survey_date[1]/12.
            except ValueError:
                d['Year'] = d['Survey_Year1'] + .5
            d['coverage'] = 1. - float(d['Per_0ITNs'])
            N = d['Total_HH'] or 1000
            d['sampling_error'] = d['coverage']*(1-d['coverage'])/sqrt(N)
            d['coverage_se'] = d['sampling_error']*2.

    def add_synthetic_data(self):
        # find population by country-year
        pop = {}
        for d in self.population:
            pop[(d['Country'], d['Year'])] = d['Pop'] * 1000.

        # find mean LLINs manu and dist per capita by year
        manu = [[] for y in range(settings.year_start, settings.year_end)]
        for d in self.llin_manu:
            if pop.has_key((d['Country'], d['Year'])):
                manu[int(d['Year']-settings.year_start)].append(
                    d['Manu_Itns'] / pop[(d['Country'], d['Year'])])

        mu_manu = zeros(len(manu))
        for ii in range(settings.year_end - settings.year_start):
            if manu[ii]:
                mu_manu[ii] = mean(manu[ii])
        mu_manu += .005

        dist = [[] for y in range(settings.year_start, settings.year_end)]
        for d in self.hh_llin_flow:
            if pop.has_key((d['Country'], d['Year'])):
                dist[int(d['Year']-settings.year_start)].append(
                    d['Total_LLINs'] / pop[(d['Country'], d['Year'])])

        mu_dist = zeros(len(dist))
        for ii in range(settings.year_end - settings.year_start):
            if dist[ii]:
                mu_dist[ii] = mean(dist[ii])


        mu = mu_manu * 1000000
        delta = mu_dist * 1000000

        Psi = zeros(len(mu))
        Th1 = zeros(len(mu))
        Th2 = zeros(len(mu))
        Th3 = zeros(len(mu))
        Th4 = zeros(len(mu))

        for ii in range(0,len(mu)-1):
            Psi[ii+1] = Psi[ii] + mu[ii] - delta[ii]
            Th1[ii+1] = delta[ii]
            Th2[ii+1] = Th1[ii]
            Th3[ii+1] = Th2[ii]
            Th4[ii+1] = Th3[ii]

        return mu, delta, Psi, Th1, Th2, Th3, Th4, Th1+Th2+Th3+Th4
