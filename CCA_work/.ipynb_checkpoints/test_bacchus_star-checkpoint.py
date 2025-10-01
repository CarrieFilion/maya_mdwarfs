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

import PyBACCHUS
from PyBACCHUS.bacchus import BACCHUS
from PyBACCHUS.star import Star

def __main__():
    bpath = '/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/b_0/'
    bacchus = BACCHUS(bpath)
    star = Star('galah_id_140111003900031')
    bacchus.eqw(star, 'Fe')


__main__()