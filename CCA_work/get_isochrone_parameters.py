import matplotlib.pyplot as plt

from astropy import *
import astropy.units as u
from astropy.io import fits
from astropy.table import Table,vstack,hstack
from astropy.io import ascii
import astropy.units as u
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
from astropy.time import Time
from astropy.visualization.units import quantity_support
import subprocess

import numpy as np
import pandas as pd
import random
import warnings
# import seaborn as sns
import glob
import subprocess

from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import kstest

from isochrones.mist import MISTIsochroneGrid
from isochrones.priors import *
from isochrones import get_ichrone
from isochrones import get_ichrone, SingleStarModel, BinaryStarModel
from astroquery.vizier import Vizier


path = '/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/fits/'
plt.style.use(path +'paper_style.mplstyle')
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['text.usetex'] = True

def get_distance(dmnn, av):
    return 10.0**(0.2*(dmnn + 5 - av))

def get_obs_params(star, parameters):
    c_params = parameters[parameters['Cluster']==star['cluster']]

    feh = float(c_params['[Fe/H]50'][0])
    e_feh = float(c_params['[Fe/H]50'][0] - c_params['[Fe/H]16'][0])
    
    teff = float(star['teff'])
    e_teff = float(star['e_teff'])
    
    logg = float(star['logg'])
    e_logg = float(star['e_logg'])

    j = float(star['j_m'])
    e_j = float(star['j_msigcom'])
    h = float(star['h_m'])
    e_h = float(star['h_msigcom'])
    k = float(star['ks_m'])
    e_k = float(star['ks_msigcom'])

    filters = ['J','H','K']
    
    params = {'Teff': (teff, e_teff), 'logg': (logg, 0.1), 'feh': (feh, e_feh), 'J': (j, e_j), 'H': (h, e_h), 'K': (k, e_k)}

    return params,filters

from isochrones.priors import *
def run_star(star,parameters,params,filters):
    subprocess.call(['rm','-rf','chains/'])

    #####
    
    c_params = parameters[parameters['Cluster']==star['cluster']] 

    #####
    
    mist = get_ichrone('mist', bands=filters)
    mod = SingleStarModel(mist, **params)

    age_center = c_params['logAge50'][0]
    age_disp = abs(c_params['logAge50'][0] - c_params['logAge16'][0]) 

    dist_center = c_params['dist50'][0]
    dist_disp = abs(c_params['dist50'][0] - c_params['dist16'][0]) 

    av_center = c_params['AV50'][0]
    av_disp = abs(c_params['AV50'][0] - c_params['AV16'][0])


    feh_center = c_params['[Fe/H]50'][0]

    mod._priors['age'] = GaussianPrior(age_center,age_disp)
    mod._priors['AV'] = GaussianPrior(av_center,av_disp)
    mod._priors['distance'] = GaussianPrior(dist_center,dist_disp)
    mod._priors['feh'] = GaussianPrior(feh_center,0.1)

    mod._bounds['AV'] = (0,6)
    mod._bounds['feh'] = (-2,1)
    
    mod.fit()

    plots_path = '/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/cca_work/iso_fits/iso_plots/'

    plt.figure()
    mod.corner_observed()
    fname = 'galah_{}_corner.jpg'.format(star['sobject_id'])
    plt.savefig(plots_path + fname)
    plt.close()

    results_path = '/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/cca_work/iso_fits/iso_results/'
    fname = 'galah_{}_derived_parameters.csv'.format(star['sobject_id'])
    results = mod.derived_samples.describe()
    results.to_csv(results_path + fname)

    fname = 'galah_{}_summary.csv'.format(star['sobject_id'])
    results = mod.samples.describe()
    results.to_csv(results_path + fname)


#### 
def __main__():
    path = '/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/fits/'
    galah = Table.read(path + 'galah_dr4_allstar.fits')
    members = Table.read('galah_candidate_cluster_members.fits')

    hunt_parameters = Table.read(path + 'asu.fit')

    from astroquery.vizier import Vizier
    Vizier.ROW_LIMIT = -1

    catalog_list = Vizier.find_catalogs('J/AJ/167/12')
    catalogs = Vizier.get_catalogs(catalog_list.keys())
    cantello_parameters = catalogs[0]

    good_cantello_parameters = cantello_parameters[cantello_parameters['Quality'] < 1]
    good_cantello_parameters = good_cantello_parameters[good_cantello_parameters['kind']=='o']


    good_cantello_parameters['dist16'] = get_distance(good_cantello_parameters['dMod16'], good_cantello_parameters['AV16'])
    good_cantello_parameters['dist50'] = get_distance(good_cantello_parameters['dMod50'], good_cantello_parameters['AV50'])
    good_cantello_parameters['dist84'] = get_distance(good_cantello_parameters['dMod84'], good_cantello_parameters['AV84'])

    for i in members:
        params,filters = get_obs_params(i, good_cantello_parameters)
        run_star(i,good_cantello_parameters,params,filters)

__main__()

    