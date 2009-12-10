""" Script for exploratory analysis of the bednet model estimates"""

import re
import pylab as pl
import pymc

def load_pickles(path='.'):
    import os, sys
    file_list = os.listdir(path)

    db = {}
    for f in file_list:
        match = re.match('^bednet_model.*pickle$', f)
        if match:
            print 'loading', f, '...',
            sys.stdout.flush()
            k=match.group()
            db[k] = pymc.database.pickle.load(f)
            print 'finished.'

    return db

def scatter_stats(db, s1, s2, f1=None, f2=None, **kwargs):
    if f1 == None:
        f1 = lambda x: x # constant function

    if f2 == None:
        f2 = f1
    
    x = []
    xerr = []

    y = []
    yerr = []
    
    for k in db:
        x_k = [f1(x_ki) for x_ki in db[k].__getattribute__(s1).gettrace()]
        y_k = [f2(y_ki) for y_ki in db[k].__getattribute__(s2).gettrace()]
        
        x.append(pl.mean(x_k))
        xerr.append(pl.std(x_k))

        y.append(pl.mean(y_k))
        yerr.append(pl.std(y_k))

        pl.text(x[-1], y[-1], ' %s' % k, fontsize=8, alpha=.4, zorder=-1)

    default_args = {'fmt': 'o', 'ms': 10}
    default_args.update(kwargs)
    pl.errorbar(x, y, xerr=xerr, yerr=yerr, **default_args)
    pl.xlabel(s1)
    pl.ylabel(s2)
    
def compare_models(db, stoch='itn coverage', stat_func=None, plot_type='', **kwargs):
    if stat_func == None:
        stat_func = lambda x: x

    X = {}
    for k in sorted(db.keys()):
        c = k.split('_')[2]
        X[c] = []

    for k in sorted(db.keys()):
        c = k.split('_')[2]
        X[c].append(
            [stat_func(x_ki) for x_ki in
             db[k].__getattribute__(stoch).gettrace()]
            )

    x = pl.array([pl.mean(xc[0]) for xc in X.values()])
    xerr = pl.array([pl.std(xc[0]) for xc in X.values()])
    y = pl.array([pl.mean(xc[1]) for xc in X.values()])
    yerr = pl.array([pl.std(xc[1]) for xc in X.values()])
        
    if plot_type == 'scatter':
        default_args = {'fmt': 'o', 'ms': 10}
        default_args.update(kwargs)
        for c in X.keys():
            pl.text(pl.mean(X[c][0]),
                    pl.mean(X[c][1]),
                    ' %s' % c, fontsize=8, alpha=.4, zorder=-1)
        pl.errorbar(x, y, xerr=xerr, yerr=yerr, **default_args)
        pl.xlabel('First Model')
        pl.ylabel('Second Model')
        pl.plot([0,1], [0,1], alpha=.5, linestyle='--', color='k', linewidth=2)

    elif plot_type == 'rel_diff':
        d1 = sorted(100*(x-y)/x)
        d2 = sorted(100*(xerr-yerr)/xerr)
        pl.subplot(2,1,1)
        pl.title('Percent Model 2 deviates from Model 1')

        pl.plot(d1, 'o')
        pl.xlabel('Countries sorted by deviation in mean')
        pl.ylabel('deviation in mean (%)')

        pl.subplot(2,1,2)
        pl.plot(d2 ,'o')
        pl.xlabel('Countries sorted by deviation in std err')
        pl.ylabel('deviation in std err (%)')
    elif plot_type == 'abs_diff':
        d1 = sorted(x-y)
        d2 = sorted(xerr-yerr)
        pl.subplot(2,1,1)
        pl.title('Percent Model 2 deviates from Model 1')

        pl.plot(d1, 'o')
        pl.xlabel('Countries sorted by deviation in mean')
        pl.ylabel('deviation in mean')

        pl.subplot(2,1,2)
        pl.plot(d2 ,'o')
        pl.xlabel('Countries sorted by deviation in std err')
        pl.ylabel('deviation in std err')
    else:
        assert 0, 'plot_type must be abs_diff, rel_diff, or scatter'

    return pl.array([x,y,xerr,yerr])
