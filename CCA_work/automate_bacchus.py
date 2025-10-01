import matplotlib.pyplot as plt
from astropy import *
import astropy.units as u
from astropy.io import fits
import numpy as np
import glob as glob
import pandas as pd
from astropy.table import Table,vstack,hstack

import subprocess
import sys
import os
import logging
import numpy as np

from scipy.interpolate import interp1d
import time
from astropy.stats import sigma_clip
import scipy
from scipy.interpolate import interp1d
from scipy import interpolate

import sys

import PyBACCHUS
from PyBACCHUS.bacchus import BACCHUS
from PyBACCHUS.star import Star

os.chdir('/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/')
x = input('Which folder (b_x): ')

bacchus_path = "/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/b_{}/".format(x)
os.chdir(bacchus_path)
print(bacchus_path)
print('Current Directory: {}'.format(x))

tools_path = '/uufs/astro.utah.edu/common/home/u1363702/notebooks/CCA_work/'
sys.path.append(tools_path)
import bacchus_tools_MK2 as b
print()

def update_init(teff,logg,met,vmicro,conv):
    with open(bacchus_path + "init.com") as file:
        init = file.readlines()
        print(init[19]) #SPH
        print(init[59]) #alllines_list
        print(init[60]) #Teff
        print(init[61]) #logg
        print(init[62]) #met
        print(init[63]) #vmicro
        print(init[64]) #conv
        
        print("updating init.com...")
        if logg < 3.0:
            init[19] = 'set SPH = T\n'
        else:
            init[19] = 'set SPH = F\n'

        init[60] = 'set TEFFunknown = "{}"\n'.format(teff)
        init[61] = 'set LOGGunknown = "{}"\n'.format(logg)
        init[62] = 'set METALLICunknown = "{}"\n'.format(met)
        init[63] = 'set TURBVELunknown = "{}"\n'.format(vmicro)
        init[64] = 'set CONVOLunknown = "{}"\n'.format(conv)
        
        print(init[19]) #SPH
        print(init[59]) #alllines_list
        print(init[60]) #Teff
        print(init[61]) #logg
        print(init[62]) #met
        print(init[63]) #vmicro
        print(init[64]) #conv
    file.close()
    
    with open(bacchus_path + "init.com", "w") as file:
        file.writelines(init)
    file.close()



def __main__():
    bpath = '/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/'
    b = input('Which BACCHUS directory (b_x): ')
    bacchus = BACCHUS(bpath + 'b_{}/'.format(b))
    path = '/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/fits/'

    csv_path = path + 'galah_{}.csv'.format(b)
    stars = pd.read_csv(csv_path)

    for i in stars['star_name']:
        star = Star(i)
        bacchus.load_parameters(star)
        update_init(0,0,1,0,1)
        
        bacchus.abund(star)
        bacchus.eqw(star)

        update_init(0,0,0,0,0)

        elements = ['C','Mg', 'Si', 'Ca', 'Ti','Na', 'Al', 'K',
                   'Sc', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Zn',
                   'Rb', 'Sr', 'Y', 'Zr', 'Mo', 'Ba', 'La', 'Ce', 'Nd',
                   'Ru', 'Sm', 'Eu']
        for j in elements:
            bacchus.abund(star, j)

        filepath = bacchus_path + '/' + i
        subprocess.run(["cp",'-R',filepath,'/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/done_stars/'])
__main__()