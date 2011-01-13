"""
notes on how to do this:
1a. create a new folder for data,
1b. create a subfolder for traces
2.  modify local_settings.py to point to the new data folder
3.  copy all the input csv files to the new data folder
4.  generate the initial empirical priors::

    $ python emp_priors.py

5. adjust empirical priors based on expert priors, if desired
6. generate posteriors

    $ python run_all.py

"""

import optparse
import os
import subprocess

import settings

def run_all(fit_empirical_priors=True):
    """ Enqueues all jobs necessary to fit model

    Example
    -------
    >>> import run_all
    >>> run_all.run_all()
    """

    if fit_empirical_priors:
        # fit empirical priors (by pooling data from all regions)
        import emp_priors, graphics

        emp_priors.admin_err_and_bias(recompute=True)
        emp_priors.llin_discard_rate(recompute=True)
        emp_priors.neg_binom(recompute=True)
        emp_priors.survey_design(recompute=True)
    
        graphics.plot_neg_binom_fits()
        

    #fit each region individually for this model
    from data import Data
    data = Data()
    post_names = []
    dir = settings.PATH
    for ii, r in enumerate(sorted(data.countries)):
        o = '%s/%s-stdout.txt' % (dir, r[0:3])
        e = '%s/%s-stderr.txt' % (dir, r[0:3])
        name_str = 'ITN%s-%d' % (r[0:3].strip(), ii)
        post_names.append(name_str)
        call_str = 'qsub -cwd -o %s -e %s ' % (o,e) \
                   + '-N %s ' % name_str \
                   + 'fit.sh %d' % ii
        subprocess.call(call_str, shell=True)
        
    # TODO: after all posteriors have finished running, notify me via email
    hold_str = '-hold_jid %s ' % ','.join(post_names)
    o = '%s/summarize.stdout' % dir
    e = '%s/summarize.stderr' % dir
    call_str = 'qsub -cwd -o %s -e %s ' % (o,e) \
               + hold_str \
               + '-N netsdone ' \
               + 'fit.sh summarize'
    subprocess.call(call_str, shell=True)

def main():
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args) != 0:
        parser.error('incorrect number of arguments')

    # make a directory for the output traces
    try:
        #settings.PATH = args[0]

        # TODO: create new dir if necessary
        #os.mkdir(settings.PATH)
        #os.mkdir(settings.PATH + 'traces')

        # TODO: somehow copy csv files into new dir
        #for fname in 'pop.csv reten.csv design.csv manuitns.csv adminllins_itns.csv stock_llins.csv flow_llins.csv llincc.csv itncc.csv numllins.csv'.split(' '):
        #cp pop.csv reten.csv design.csv manuitns.csv adminllins_itns.csv stock_llins.csv flow_llins.csv llincc.csv itncc.csv numllins.csv ../2010_07_26
        #cp *.json ../2010_07_26
        pass
        
    except IOError, e:
        parser.error('failed to create data/output directory: %s' % e)

    run_all()


if __name__ == '__main__':
    main()
