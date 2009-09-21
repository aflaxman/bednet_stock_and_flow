""" Module to handle loading the data from the stock-and-flow model
for bednet distribution
"""

import csv
import time
from numpy import zeros

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

### load all data from csv files
retention = load_csv('reten.csv')
design = load_csv('design.csv')

llin_manu = load_csv('manuitns.csv')
admin_llin = load_csv('adminllins_itns.csv')

hh_llin_stock = load_csv('stock_llins.csv')
hh_llin_flow = load_csv('flow_llins.csv')

# add mean survey date to hh_llin data
for d in hh_llin_stock + hh_llin_flow:
    mean_survey_date = time.strptime(d['Mean_SvyDate'], '%d-%b-%y')
    d['mean_survey_date'] = mean_survey_date[0] + mean_survey_date[1]/12.

llin_coverage = load_csv('llincc.csv')
itn_coverage = load_csv('itncc.csv')

population = load_csv('pop.csv')

countries = set([d['Country'] for d in population])

def population_for(c, year_start, year_end):
    pop_vec = zeros(year_end - year_start)
    for d in population:
        if d['Country'] == c:
            pop_vec[int(d['Year']) - year_start] = d['Pop']*1000

    # since we might be predicting into the future, fill in population with last existing value
    for ii in range(1, year_end-year_start):
        if pop_vec[ii] == 0.:
            pop_vec[ii] = pop_vec[ii-1]

    return pop_vec
