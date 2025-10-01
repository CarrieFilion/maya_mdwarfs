import subprocess
import sys
import os
import logging
from astropy.table import Table
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
#import report_bacchus_abunds as r
from astropy.stats import sigma_clip
import scipy
from scipy.interpolate import interp1d
from scipy import interpolate

x = input('Which Directory: ')
BACCHUS_DIR = "/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/b_{}/".format(x)
# if x == 0:
#     BACCHUS_DIR = "/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/BACCHUS_abundances"
# else:
#     BACCHUS_DIR = "/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/BACCHUS_uncertainties"


# BACCHUS_DIR = "/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/done_stars/"
# BACCHUS_DIR = "/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/BACCHUS_abundances/"
def retrieve_star_params(star):
    par = Table.read(BACCHUS_DIR+"{}/best_parameters.tab".format(star), format='ascii')
    par2 = Table.read(BACCHUS_DIR+"{a}/{b}.par".format(a=star, b=star), format='ascii', data_start=0)
    try:
        line = np.where(par['conv'] == 1)[0][0]-1
        teff = par['Teff'][line]
        eteff = par['err'][line+1]
        logg = par['logg'][line]
        elogg = par['err_1'][line+1]
        met = par['initmet'][line]
        emet = par['err_2'][line+1]
        mic = par['initxit'][line]
        emic = par['err_3'][line+1]
        convol = par2[7][3]
        spec = par2[6][3]
    except Exception as e:
        print(e)
        print(star, "params didn't actually converge!!!")
    try:
        return teff, eteff, logg, elogg, met, emet, mic, emic, convol, spec
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def load_parameters(star):
    os.chdir(BACCHUS_DIR)
    subprocess.call(['load_parameters.com',star])

def get_convolution(star):
    os.chdir(BACCHUS_DIR)
    subprocess.call(['bacchus.eqw',star,'Si'])
    # subprocess.call(['bacchus.eqw',star,'O'])

def get_eqw(star,el):
    os.chdir(BACCHUS_DIR)
    subprocess.call(['bacchus.eqw',star,el])

def get_abundances(star,el):
    os.chdir(BACCHUS_DIR)
    subprocess.call(['bacchus.abund',star,el])

def get_bounded_abundances(star,el,bounds):
    os.chdir(BACCHUS_DIR)
    subprocess.call(['bacchus.abund',star,el,bounds])  

def get_star_param(star,el):
    print("First changing to bacchus directory...")
    os.chdir(BACCHUS_DIR)
    subprocess.call(['bacchus.param',star,el], cwd='./')
    return 0

def faux_best_parameters(ref,teff_err=0.0,logg_err=0.0,m_h_err=0.0,vmicro_err=0.0):
    with open(BACCHUS_DIR+'/'+ref+'/'+ref+'.par') as pars:
        lines = pars.readlines()
    pars.close()
    i = 0
    k = 0
    for j, row in enumerate(lines):
        if 'MODEL' in row:
            i = j
        if 'TURBVEL' in row:
            k = j
    mod = lines[i]
    teff = int(mod.split('=')[1].split('g')[0])
    logg = np.round(float(mod.split('g')[1].split('z')[0].split('m')[0]), 2)
    mh = np.round(float(mod.split('z')[1].split('_')[0]), 2)
    turbvel = float(lines[k].split("'")[1])
    if os.path.isfile(BACCHUS_DIR+'/'+ref+'/'+'best_parameters.tab'):
        print("best parameters already exists!")
    else:
        with open(BACCHUS_DIR+'/'+ref+'/'+'best_parameters.tab', "w") as file:
            file.write("#conv Teff +/- err initTeff conv_Teff  logg +/- err initlogg conv_logg met +/- err  initmet xit +/- err initxit conv_xit\n")
        with open(BACCHUS_DIR+'/'+ref+'/'+'best_parameters.tab', "a") as file:
            file.write("1  {a} +/- {e_a} {a} 1    {b} +/- {e_b} {b} 1    {c} +/- {e_c}  {c}     {d} +/- {e_d} {d} 1\n".format(a=teff,e_a=teff_err, b=logg,e_b=logg_err, c=mh,e_c=m_h_err, d=turbvel,e_d=vmicro_err))
            file.write("1  {a} +/- {e_a} {a} 1    {b} +/- {e_b} {b} 1    {c} +/- {e_c}  {c}     {d} +/- {e_d} {d} 1".format(a=teff,e_a=teff_err, b=logg,e_b=logg_err, c=mh,e_c=m_h_err, d=turbvel,e_d=vmicro_err))
    with open(BACCHUS_DIR+'/'+ref+'/'+'best_parameters.tab', "r") as file:
        lines = file.readlines()
    print(lines)
    
# def faux_best_parameters(ref):
#     with open(BACCHUS_DIR+'/'+ref+'/'+ref+'.par') as pars:
#         lines = pars.readlines()
#     pars.close()
#     i = 0
#     k = 0
#     for j, row in enumerate(lines):
#         if 'MODEL' in row:
#             i = j
#         if 'TURBVEL' in row:
#             k = j
#     mod = lines[i]
#     teff = int(mod.split('=')[1].split('g')[0])
#     logg = np.round(float(mod.split('g')[1].split('z')[0].split('m')[0]), 2)
#     mh = np.round(float(mod.split('z')[1].split('_')[0]), 2)
#     turbvel = float(lines[k].split("'")[1])
#     if os.path.isfile(BACCHUS_DIR+'/'+ref+'/'+'best_parameters.tab'):
#         print("best parameters already exists!")
#     else:
#         with open(BACCHUS_DIR+'/'+ref+'/'+'best_parameters.tab', "w") as file:
#             file.write("#conv Teff +/- err initTeff conv_Teff  logg +/- err initlogg conv_logg met +/- err  initmet xit +/- err initxit conv_xit\n")
#         with open(BACCHUS_DIR+'/'+ref+'/'+'best_parameters.tab', "a") as file:
#             file.write("1  {a} +/- 0 {a} 1    {b} +/- 0 {b} 1    {c} +/- 0  {c}     {d} +/- 0 {d} 1\n".format(a=teff, b=logg, c=mh, d=turbvel))
#             file.write("1  {a} +/- 0 {a} 1    {b} +/- 0 {b} 1    {c} +/- 0  {c}     {d} +/- 0 {d} 1".format(a=teff, b=logg, c=mh, d=turbvel))
#     with open(BACCHUS_DIR+'/'+ref+'/'+'best_parameters.tab', "r") as file:
#         lines = file.readlines()
#     print(lines)

def run_star_diff(star, ref):
    ref_params = retrieve_star_params(ref)
    if np.nan in ref_params:
        raise Exception("Reference star parameters did not converge")
        return 0
    z = 7.45 + ref_params[4]

    # answer = input('Ref star metallicity {}.  Do you want to continue?:'.format(ref_params[4]))
    # if answer.lower().startswith("y"):
    #     print("ok, carry on then")
        
    # elif answer.lower().startswith("n"):
    #     print("goodbye!")
    #     sys.exit()

    if z == np.nan:
        raise Exception("reference did not converge!")
    else:
        with open("init.com") as file:
            init = file.readlines()
        print(init[21:24])
        print("turning init.com into differential format...")
        init[21] = "set diff_star = '{}'\n".format(ref)
        init[22] = "#set diff_star = 'dummy'\n"
        init[23] = "set diff_star_offset = '{}'\n".format(str(np.round(z, 2)))
        print(init[21:24])
        with open("init.com", "w") as file:
            file.writelines(init)
        print("---------")
        get_star_param(star)
    
        with open("init.com") as file:
            init = file.readlines()
        print(init[21:24])
        print("returning init.com to normal format...")
        init[21] = "#set diff_star = '{}'\n".format(ref)
        init[22] = "set diff_star = 'dummy'\n"
        init[23] = "set diff_star_offset = '{}'\n".format(str(np.round(z, 2)))
        print(init[21:24])
        with open("init.com", "w") as file:
            file.writelines(init)
        return None


def coadd(fnames, new_fname, exps = None):
    wmin = 0
    wmax = 100000
    finterps = []
    for f in fnames:
        t = Table.read(f, format='ascii')
        if np.nanmin(t['waveobs']) > wmin:
            wmin = np.nanmin(t['waveobs'])
        if np.nanmax(t['waveobs']) < wmax:
            wmax = np.nanmax(t['waveobs'])

        finterp = interp1d(t['waveobs'][np.argsort(t['waveobs'])], t['flux'][np.argsort(t['waveobs'])])
        finterps.append(finterp)
    wgrid = t['waveobs'][np.argsort(t['waveobs'])][np.where((t['waveobs'][np.argsort(t['waveobs'])] < wmax) &
                                                            (t['waveobs'][np.argsort(t['waveobs'])] > wmin))]#np.arange(wmin, wmax, 0.005)

    for ind, fi in enumerate(finterps):
        print(len(exps))
        if len(exps) <= 1:
            if ind == 0:
                ff = fi(wgrid)
            else:
                ff = np.vstack([ff, fi(wgrid)])
        else:
            if ind == 0:
                ff = fi(wgrid)*exps[ind]/np.nansum(exps)
                print(exps[ind]/np.nansum(exps))
            else:
                ff = np.vstack([ff, fi(wgrid)*exps[ind]/np.nansum(exps)])
                print(exps[ind]/np.nansum(exps))
    if len(exps) <= 1:
        print("Assuming even exposures...")
        f_final = np.nanmedian(ff, axis=0)
    else:
        f_final = np.nansum(ff, axis=0)
    print(f_final)
    Table([wgrid, f_final, np.zeros(len(wgrid))], names=("waveobs", "flux", "err")).write(new_fname, format='ascii', overwrite=True)
    print("Wrote to ", new_fname)

def update_star_param_file(star, spec_loc = '', teff = None, logg = None, mh = None, microturb = None, convol = None, rv = None):
    star_line = "{star}\t{spec_loc}\t{teff}\t{logg}\t{mh}\t{microturb}\t{convol}\t{rv}\n".format(star=star, spec_loc = spec_loc, teff = teff, logg = logg, mh = mh, microturb = microturb, convol=convol, rv=rv)
    print(star_line)
    with open(BACCHUS_DIR+"stellar_parameters.tab", "r+") as f:
        lines = f.readlines()
    f.close()
    objects = []
    spot = -10
    for i, line in enumerate(lines):
        objects.append(line.split('\t')[0].strip(' '))
    print('OBJECTS ', objects)
    print(star in objects)
    if star in objects:
        spot = np.where(np.array(objects) == star)[0][0]
        print(spot)
        lines[spot] = star_line
        with open(BACCHUS_DIR+"stellar_parameters.tab", "w+") as f:
            for line in lines:
                f.write(line)
        f.close()
    else:
        with open(BACCHUS_DIR+"stellar_parameters.tab", "a") as f:
            f.write(star_line+'\n')
        f.close()

def get_last_runs_params(star):
    with open(BACCHUS_DIR+"{a}/{b}.par".format(a=star, b=star), "r+") as f:
        lines = f.readlines()
    f.close()
    spec_loc = lines[6].split("=")[-1][:-1]
    teff = float(lines[2].split("=")[-1].split("g")[0])
    logg = float(lines[2].split("=")[-1].split("g")[1][:4])
    mh = float(lines[9].split("=")[-1][1:-1].strip("'`"))
    microturb = float(lines[10].split("=")[-1][1:-1].strip("'`"))
    convol = float(lines[7].split("=")[-1][:-1].strip("'`"))
    rv = float(lines[8].split("=")[-1][:-1])
    return spec_loc, teff, logg, mh, microturb, convol, rv

def redo_param(star):
    print("Retrieving results of previous run...")
    spec_loc, teff, logg, mh, microturb, convol, rv = get_last_runs_params(star)
    print("Updating stellar_parameters.tab with last run's results...")
    update_star_param_file(star, spec_loc, teff, logg, mh, microturb, convol, rv)
    print("Changing to bacchus directory...")
    os.chdir(BACCHUS_DIR)
    print("Removing ", star)
    for f in glob.glob(BACCHUS_DIR+star+"/*"):
        try:
            print("Removing ", f)
            os.remove(f)
        except:
            pass
    for f in glob.glob(BACCHUS_DIR+star+"/models/*"):
        try:
            print("Removing ", f)
            os.remove(f)
        except:
            pass
    os.rmdir(str(star)+"/models")
    os.rmdir(str(star))
    
    print("Redoing ", star)
    get_star_param(star)
    return 0
    

def redo_if_necessary(star):
    redo = False
    try:
        with open("{a}/{b}_done".format(a=star, b=star), "r+") as f:
            lines = f.readlines()
        f.close()
        for line in lines:
            if "Maybe" in line:
                print("redoing params for ", star)
                redo = True
        par = Table.read(BACCHUS_DIR+"{}/best_parameters.tab".format(star), format='ascii')
        par2 = Table.read(BACCHUS_DIR+"{a}/{b}.par".format(a=star, b=star), format='ascii', data_start=0)
        line = np.where(par['conv'] == 1)
        if len(line[0]) == 0:
            redo = True
    except:
        print("redoing params for ", star)
        redo = True
    if redo == False:
        print(star, " already has robust parameters. No need to redo.")
    else:
        redo_param(star)

def redo_diff_param(star, ref):
    print("Retrieving results of previous run...")
    spec_loc, teff, logg, mh, microturb, convol, rv = get_last_runs_params(star)
    print("Updating stellar_parameters.tab with last run's results...")
    print(spec_loc, teff, logg, mh, microturb, convol, rv)
    update_star_param_file(star, spec_loc, teff, logg, mh, microturb, convol, rv)
    print("Changing to bacchus directory...")
    os.chdir(BACCHUS_DIR)
    print("Removing ", star)
    for f in glob.glob(BACCHUS_DIR+star+"/*"):
        try:
            
            os.remove(f)
        except Exception as e:
            print(e)
    for f in glob.glob(BACCHUS_DIR+star+"/models/*"):
        try:
            print("Removing ", f)
            os.remove(f)
        except:
            pass
    os.rmdir(str(star)+"/models")
    os.rmdir(str(star))
    
    print("Redoing ", star, "differentially to reference star ", ref)
    run_star_diff(star, ref)
    return 0

def redo_diff_if_necessary(star, ref):
    redo = False
    try:
        with open("{a}/{b}_done".format(a=star, b=star), "r+") as f:
            lines = f.readlines()
        f.close()
        for line in lines:
            if "Maybe" in line:
                print("redoing params for ", star)
                redo = True
        par = Table.read(BACCHUS_DIR+"{}/best_parameters.tab".format(star), format='ascii')
        par2 = Table.read(BACCHUS_DIR+"{a}/{b}.par".format(a=star, b=star), format='ascii', data_start=0)
        line = np.where(par['conv'] == 1)
        if len(line[0]) == 0:
            redo = True
    except:
        print("redoing params for ", star)
        redo = True
    if redo == False:
        print(star, " already has robust parameters. No need to redo.")
    else:
        redo_diff_param(star, ref)

def run_program_stars_param(targ_table, niter=2):
    for row in targ_table:
        star, spec_loc, teff, logg, mh, microturb, convol, rv = row[targ_table.colnames]
        update_star_param_file(star, spec_loc, teff, logg, mh, microturb, convol, rv)
        time.sleep(3)
        if niter == 1:
            get_star_param(star)
        else:
            get_star_param(star)
            sleep(3)
            try:
                for i in range(niter-1):
                    redo_param(star)
                    sleep(3)
            except:
                try:
                    get_star_param(star)
                except:
                    pass
    return 0

def redo_program_stars_param(targ_table, niter=2):
    for row in targ_table:
        star, spec_loc, teff, logg, mh, microturb, convol, rv = row[targ_table.colnames]
        try:
            for i in range(niter):
                redo_param(star)
        except:
            try:
                get_star_param(star)
            except:
                pass
    return 0

def get_abund(star, elements = ['Li', 'C', 'N', 'O', 'Mg', 'Si', 'Ca', 'Na', 'Al', 'Sc', 'Ti', 'V', 'Cr', 'Mn',  'Co', 'Ni', 'Cu','Zn', 'Zr', 'Sr', 'Y',  'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Eu', 'Fe']):
    print(elements)
    redo = False
    try:
        par = Table.read(BACCHUS_DIR+"{}/best_parameters.tab".format(star), format='ascii')
        par2 = Table.read(BACCHUS_DIR+"{a}/{b}.par".format(a=star, b=star), format='ascii', data_start=0)
        line = np.where(par['conv'] == 1)
        if len(line[0]) == 0:
            redo = True
    except:
        print("bad parameters for ", star)
        redo = True
    if redo == False:
        print(star, " has robust parameters. getting abundances now.")
        print("Working on ", star)
        for el in elements:
            subprocess.call(['bacchus.abund',star,el])
    else:
        print("parameters did not converge. not doing bacchus.abund")
    # elements = ['Li', 'C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Sr', 'Y', 'Zr', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Eu', 'Yb']
    

def build_abund_sensitivity_stars(ref):
    teff, eteff, logg, elogg, met, emet, mic, emic, convol, spec = retrieve_star_params(ref)
    if elogg > .1:
        elogg = .1
    update_star_param_file(ref, spec, teff, logg, met, mic, convol, 0)
    upteff = ref+"_teffp"+str(eteff)
    update_star_param_file(upteff, spec, teff+eteff, logg, met, mic, convol, 0)
    downteff = ref+"_teff-"+str(eteff)
    update_star_param_file(downteff, spec, teff-eteff, logg, met, mic, convol, 0)
    upg = ref+"_loggp"+str(elogg)
    update_star_param_file(upg, spec, teff, logg+elogg, met, mic, convol, 0)
    downg = ref+"_logg-"+str(elogg)
    update_star_param_file(downg, spec, teff, logg-elogg, met, mic, convol, 0)
    upm = ref+"_mhp"+str(emet)
    update_star_param_file(upm, spec, teff, logg, met+emet, mic, convol, 0)
    downm = ref+"_mh-"+str(emet)
    update_star_param_file(downm, spec, teff, logg, met-emet, mic, convol, 0)
    upvmic = ref+"_vmicp"+str(emic)
    update_star_param_file(upvmic, spec, teff, logg, met, mic+emic, convol, 0)
    downvmic = ref+"_vmic-"+str(emic)
    update_star_param_file(downvmic, spec, teff, logg, met, mic-emic, convol, 0)
    for x in [upteff, downteff, upg, downg, upm, downm, upvmic, downvmic]:
        subprocess.call(['load_parameters.com',x])
    for star in [upteff, downteff, upg, downg, upm, downm, upvmic, downvmic]:
        faux_best_parameters(star)
        get_abund(star)

def extract_abunds(object, sub=None):
    print("Producing abundance summary for ", object)
    all_els = np.array([])
    all_lines = np.array([])
    all_abunds = np.array([])
    all_flags = np.array([])
    all_syn_flags = np.array([])
    all_eqw_flags = np.array([])
    all_int_flags = np.array([])
    if sub == None:
        all_abund_files = glob.glob(BACCHUS_DIR+object+"/*-{}.abu".format(object))
    else:
        all_abund_files = glob.glob(BACCHUS_DIR+object+"/*-{}.abu".format(object.split('_')[0]+'_'+sub))
    for af in all_abund_files:
        try:
            t = Table.read(af, format='ascii')
            element = af.split('/')[-1].split('-')[0]
            els = np.repeat(element, len(t))
            lines = np.array(t['col1'])
            A_abunds = np.array(t['col9'])
            A_flags = np.array(t['col10'])
            synflag = np.array(t['col4'])
            eqflag = np.array(t['col6'])
            intflag = np.array(t['col8'])

            all_els = np.append(all_els, els)
            all_lines = np.append(all_lines, lines)
            all_abunds = np.append(all_abunds, A_abunds)
            all_flags = np.append(all_flags, A_flags)
            all_syn_flags = np.append(all_syn_flags, synflag)
            all_eqw_flags = np.append(all_eqw_flags, eqflag)
            all_int_flags = np.append(all_int_flags, intflag)
        except:
            print("Couldn't open file ", af)
            pass
    big_t = Table([all_els, all_lines, all_abunds, all_flags, all_syn_flags, all_eqw_flags, all_int_flags], names=('Element', 'Line', 'A', 'Flag', 'SynFlag', 'EqFlag', 'IntFlag'))
    big_t.write(BACCHUS_DIR+object+'/all_A_abunds.txt', format='ascii', overwrite=True)
    return big_t

def get_diff_abund(ref, star, nodir=False):
    if nodir:
        ref_t = Table.read(ref+'/all_A_abunds.txt', format='ascii')
        star_t = Table.read(star+'/all_A_abunds.txt', format='ascii')
    else:
        ref_t = Table.read(BACCHUS_DIR+ref+'/all_A_abunds.txt', format='ascii')
        star_t = Table.read(BACCHUS_DIR+star+'/all_A_abunds.txt', format='ascii')
    elements = ['Li', 'C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Sr', 'Y', 'Zr', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Eu', 'Yb']
    comp_t = Table(names=("Element", "Line", "Delta_A"), dtype=('S2', 'f4', 'f4'))
    for el in elements:
        spot = np.where(ref_t['Element'] == el)[0]

        if spot.size>0:
            for row in ref_t[spot]:
                line = row['Line']
                refA = row['A']
                if (row['Flag'] == 1) & (row['SynFlag'] == 1) & (row['EqFlag'] == 1):
                    spot_star = np.where((star_t['Element'] == el) & (star_t['Line'] == line))[0]
                    if spot_star.size > 0:
                        rowstar = star_t[spot_star]
                        if (rowstar['Flag'] == 1) & (rowstar['SynFlag'] == 1) & (rowstar['EqFlag'] == 1):

                            starA = rowstar['A']
                            comp_t.add_row([el, line, starA - refA])
    if nodir:
        comp_t.write(ref+'/{a}_{b}_diff_abunds.txt'.format(a=star.split('/')[-1], b=ref.split('/')[-1]), format='ascii', overwrite=True)
        comp_t.write(star+'/{a}_{b}_diff_abunds.txt'.format(a=star.split('/')[-1], b=ref.split('/')[-1]), format='ascii', overwrite=True)
    else:
        comp_t.write(BACCHUS_DIR+ref+'/{a}_{b}_diff_abunds.txt'.format(a=star.split('/')[-1], b=ref.split('/')[-1]), format='ascii', overwrite=True)
        comp_t.write(BACCHUS_DIR+star+'/{a}_{b}_diff_abunds.txt'.format(a=star.split('/')[-1], b=ref.split('/')[-1]), format='ascii', overwrite=True)
   
def return_diff(ref, star):
    sid_ref = float(ref.split('_')[-1])
    sid_star = float(star.split('_')[-1])
    comp_t = Table.read(BACCHUS_DIR+ref+'/{a}_{b}_diff_abunds.txt'.format(a=star.split('/')[-1], b=ref.split('/')[-1]), format='ascii')
    elements = ['Li', 'C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Sr', 'Y', 'Zr', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Eu', 'Yb']
    rang = len(elements)
    els = []
    stds = []
    means = []
    nlines = []
    for el in elements:
        spot = np.where(comp_t['Element'] == el)[0]
        if spot.size>0:
            mean_delta_A = np.nanmedian(comp_t['Delta_A'][spot].astype(np.float64))
            std_delta_A = np.nanstd(comp_t['Delta_A'][spot].astype(np.float64))
            els.append(el)
            stds.append(std_delta_A/len(comp_t['Delta_A'][spot]))
            means.append(mean_delta_A)
            nlines.append(spot.size)
        else:
            els.append(el)
            stds.append(np.nan)
            means.append(np.nan)
            nlines.append(0)
    return means, stds, nlines


def get_shared_lines_for_reference_star(ref, path_ref):
    ref_file = path_ref+"{a}/{b}_all_abunds.txt".format(a=ref, b=ref)
    reftab = Table.read(ref_file, format='ascii')
    good_lines = reftab['Line'][np.where(reftab['Used?'] == 'True')]
    for star in ids['ID']:
        star=star+str(2)
        try:
            star_file = path_star+"{a}/{b}_all_abunds.txt".format(a=star, b=star)
            startab = Table.read(star_file, format='ascii')
            good_lines = np.intersect1d(good_lines, np.array(startab['Line'][np.where(startab['Used?'] == 'True')]))
        except Exception as e:
            print(e)
            pass
    return good_lines

def get_diff(ref, path_ref, star, path_star, linelist = None, exceptions = None):
    ref_file = path_ref+"{a}/{b}_all_abunds.txt".format(a=ref, b=ref)
    reftab = Table.read(ref_file, format='ascii')
    good_lines = reftab['Line'][np.where(reftab['Used?'] == 'True')]
    star_file = path_star+"{a}/{b}_all_abunds.txt".format(a=star, b=star)
    startab = Table.read(star_file, format='ascii')
    good_lines = np.intersect1d(good_lines, np.array(startab['Line'][np.where(startab['Used?'] == 'True')]))
    if linelist is not None:
        print("Using input linelist")
        good_lines = linelist
    if exceptions is not None:
        for el in exceptions:
            extra_lines = reftab['Line'][np.where((reftab['Flag'] == 0) & (reftab['SynFlag'] == 0) & (reftab['EqFlag'] == 0) & (reftab['Element'] == el))]
            extra_lines = np.intersect1d(extra_lines, np.array(startab['Line'][np.where((startab['Flag'] == 0) & (startab['SynFlag'] == 0) & (startab['EqFlag'] == 0) & (startab['Element'] == el))]))
            good_lines = np.append(good_lines, extra_lines)
            
    print("Number of good shared lines: ", len(good_lines))
    diff_abunds = []
    els = []
    for line in good_lines:
        refspot = np.where(reftab['Line'] == line)[0][0]
        starspot = np.where(startab['Line'] == line)[0][0]
        els.append(startab[starspot]['Element'])
        diff_abund = startab['[X/H]'][starspot] - reftab['[X/H]'][refspot]
        diff_abunds.append(diff_abund)
    if not os.path.exists(BACCHUS_DIR+"/diff_against_{}".format(ref)):
        os.makedirs(BACCHUS_DIR+"/diff_against_{}".format(ref))
    dt = Table([els, good_lines, diff_abunds], names=('Element', 'Line', 'Delta_Abund'))
    dt.write(BACCHUS_DIR+"/diff_against_{}".format(ref)+'/'+star+'_diff_against_'+ref+'.txt', format='ascii', overwrite=True)

def get_shared_lines_for_reference_star(ref, path_ref, path_star, stars, delta_teff = None, ref_teff = None):
    ref_file = path_ref+"{a}/{b}_all_abunds.txt".format(a=ref, b=ref)
    reftab = Table.read(ref_file, format='ascii')
    good_lines = reftab['Line'][np.where(reftab['Used?'] == 'True')]
    if 1 == 1:
        for star in stars:
            try:
                star_file = path_star+"{a}/{b}_all_abunds.txt".format(a=star, b=star)
                startab = Table.read(star_file, format='ascii')
                good_lines = np.intersect1d(good_lines, np.array(startab['Line'][np.where(startab['Used?'] == 'True')]))
            except Exception as e:
                print(e)
                pass
    return good_lines

def get_diff(ref, path_ref, star, path_star, linelist = None):
    ref_file = path_ref+"{a}/{b}_all_abunds.txt".format(a=ref, b=ref)
    reftab = Table.read(ref_file, format='ascii')
    good_lines = reftab['Line'][np.where(reftab['Used?'] == 'True')]
    star_file = path_star+"{a}/{b}_all_abunds.txt".format(a=star, b=star)
    startab = Table.read(star_file, format='ascii')
    good_lines = np.intersect1d(good_lines, np.array(startab['Line'][np.where(startab['Used?'] == 'True')]))
    if linelist is not None:
        print("Using input linelist")
        good_lines = linelist
    print("Number of good shared lines: ", len(good_lines))
    diff_abunds = []
    els = []
    for line in good_lines:
        refspot = np.where(reftab['Line'] == line)[0][0]
        starspot = np.where(startab['Line'] == line)[0][0]
        els.append(startab[starspot]['Element'])
        diff_abund = startab['[X/H]'][starspot] - reftab['[X/H]'][refspot]
        diff_abunds.append(diff_abund)
    if not os.path.exists(BACCHUS_DIR+"/diff_against_{}".format(ref)):
        os.makedirs(BACCHUS_DIR+"/diff_against_{}".format(ref))
    dt = Table([els, good_lines, diff_abunds], names=('Element', 'Line', 'Delta_Abund'))
    dt.write(BACCHUS_DIR+"/diff_against_{}".format(ref)+'/'+star+'_diff_against_'+ref+'.txt', format='ascii', overwrite=True)


def extract_abunds(object, path=BACCHUS_DIR+'/'):
    #print("Producing abundance summary for ", object)
    all_els = np.array([])
    all_lines = np.array([])
    all_abunds = np.array([])
    all_eqw_abunds = np.array([])
    all_int_abunds = np.array([])
    all_flags = np.array([])
    all_syn_flags = np.array([])
    all_eqw_flags = np.array([])
    all_int_flags = np.array([])
    all_abund_files = glob.glob(path+object+"/*-{}.abu".format(object))
    #print(all_abund_files)
    for af in all_abund_files:
        try:
            t = Table.read(af, format='ascii')
            element = af.split('/')[-1].split('-')[0]
            els = np.repeat(element, len(t))
            lines = np.array(t['col1'])
            A_abunds = np.array(t['col9'])
            A_flags = np.array(t['col10'])
            int_ab = np.array(t['col7'])
            synflag = np.array(t['col4'])
            eqw_ab = np.array(t['col5'])
            eqflag = np.array(t['col6'])
            intflag = np.array(t['col8'])

            all_els = np.append(all_els, els)
            all_lines = np.append(all_lines, lines)
            all_abunds = np.append(all_abunds, A_abunds)
            all_flags = np.append(all_flags, A_flags)
            all_syn_flags = np.append(all_syn_flags, synflag)
            all_eqw_flags = np.append(all_eqw_flags, eqflag)
            all_int_flags = np.append(all_int_flags, intflag)
            all_eqw_abunds = np.append(all_eqw_abunds, eqw_ab)
            all_int_abunds = np.append(all_int_abunds, int_ab)
        except:
            print("Couldn't open file ", af)
            pass
    big_t = Table([all_els, all_lines, all_abunds, all_flags, all_syn_flags, all_eqw_flags, all_int_flags, all_int_abunds, all_eqw_abunds], names=('Element', 'Line', 'A', 'Flag', 'SynFlag', 'EqFlag', 'IntFlag', 'IntAbund', 'EqwAbund'))
    big_t.write(path+object+'/all_A_abunds.txt', format='ascii', overwrite=True)
    return big_t

def make_bracket_abund_table(object, lineselection = None, path=BACCHUS_DIR+'/'):
    if lineselection is not None:
        lineselect = Table.read(lineselection, format='ascii')
        #print(lineselect)
    big_t = extract_abunds(object, path)
    big_t['x_h'] = 10**(big_t['A'] - 12)
    s = Table.read("/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/BACCHUS_abundances/solabu.dat", format='ascii')
    keys = list(np.array(s['col1']))
    values = list(np.array(s['col2']))
    sol_abund_dict = dict(zip(keys, values))

    big_t['sol_A'] = np.zeros(len(big_t))
    for i in range(len(big_t)):
        big_t['sol_A'][i] = sol_abund_dict[np.array(big_t['Element'])[i]]
    big_t['sol_x_h'] = 10**(big_t['sol_A'] - 12)
    big_t['[X/H]'] = np.log10(big_t['x_h']/big_t['sol_x_h'])
    big_t['Used?'] = np.repeat(False, len(big_t))
    good_fe = np.where((big_t['Element'] == 'Fe') & (big_t['Flag'] == 1) & (big_t['SynFlag'] == 1) & (big_t['EqFlag'] == 1) & (big_t['IntFlag'] == 1))
    # good_fe = np.where((big_t['Element'] == 'Fe') & (big_t['Flag'] == 1))
    big_t['Used?'][good_fe] = True
    #big_t['[X/Fe]'] = big_t['[X/H]'] - mean_fe_h
    for el in set(big_t['Element']):
        good = np.where((big_t['Element'] == el) & (big_t['Flag'] == 1) & (big_t['EqFlag'] == 1) & (big_t['SynFlag'] == 1) & (big_t['IntFlag'] == 1))
        # good = np.where((big_t['Element'] == el) & (big_t['Flag'] == 1))
        if lineselection is not None:
            for g in good[0]:
                if big_t['Line'][g] in lineselect['col2']:
                    big_t['Used?'][g] = True
        else:
            big_t['Used?'][good] = True
    #print(big_t.colnames)
    big_t.write(path+object+'/{}_all_abunds.txt'.format(object), format='ascii', overwrite=True)
    #print('wrote to ', path+object+'/{}_all_abunds.txt'.format(object))

def get_bracket_abunds(object, lineselection = None, trust_flags = True, path=BACCHUS_DIR+'/'):
    if trust_flags == True:
        make_bracket_abund_table(object, lineselection, path)
    t_sum = Table(names=('Abund', 'Mean', 'Std Dev', 'Std Err'), dtype=(str, 'f4', 'f4', 'f4'))
    big_t = Table.read(path+object+'/{}_all_abunds.txt'.format(object), format='ascii')
    good_fe = np.where((big_t['Used?'] == 'True') & (big_t['Element'] == 'Fe'))
    mean_fe_h = np.nanmedian(big_t['[X/H]'][good_fe])
    std_fe_h = np.nanstd(big_t['[X/H]'][good_fe])
    num_lines_fe = np.sqrt(len(big_t['[X/H]'][good_fe]))
    t_sum.add_row(['[Fe/H]', mean_fe_h, std_fe_h, std_fe_h/num_lines_fe])
    #print("[Fe/H] = ", mean_fe_h, ' +/- ', std_fe_h/num_lines_fe)
    big_t['[X/Fe]'] = big_t['[X/H]'] - mean_fe_h
    for el in set(big_t['Element']):
        good = np.where((big_t['Used?'] == 'True') & (big_t['Element'] == el))
        mean_x_fe = np.nanmedian(big_t['[X/Fe]'][good])
        mean_x_h = np.nanmedian(big_t['[X/H]'][good])
        std_x_fe = np.nanstd(big_t['[X/Fe]'][good])
        num_lines_x_fe = np.sqrt(len(big_t['[X/Fe]'][good]))
        #print("[{}/Fe] = ".format(el), mean_x_fe, ' +/- ', std_x_fe/num_lines_x_fe)
        if el != 'Fe':
            t_sum.add_row(["[{}/Fe]".format(el), mean_x_fe, std_x_fe, std_x_fe/num_lines_x_fe])
            t_sum.add_row(["[{}/H]".format(el), mean_x_h, 0, 0])

    s = Table.read("/uufs/chpc.utah.edu/common/home/astro/zasowski/sinha/bacchus_files/BACCHUS_abundances/solabu.dat", format='ascii')
    #print(s)
    keys = list(np.array(s['col1']))
    values = list(np.array(s['col2']))
    sol_abund_dict = dict(zip(keys, values))
    #sol_abund_dict = {'C':8.43, 'O':8.69, 'Na':6.24, 'Mg': 7.60, 'Si':7.51, 'Ca':6.34, 'Fe':7.50, 'Ba':2.18, 'Y':2.21, 'Al':6.45,
    #'Cr':5.64, 'Ti':4.95}
    goodmg = np.where((big_t['Used?'] == 'True') & (big_t['Element'] == 'Mg'))
    goodsi = np.where((big_t['Used?'] == 'True') & (big_t['Element'] == 'Si'))
    mean_nobrack_mg = np.nanmedian(big_t['x_h'][goodmg])
    mean_nobrack_si = np.nanmedian(big_t['x_h'][goodsi])

    sol_mg_h = (sol_abund_dict['Mg'] - 12)
    sol_si_h = (sol_abund_dict['Si'] - 12)
    sol_c_h = (sol_abund_dict['C'] - 12)
    sol_o_h = (sol_abund_dict['O'] - 12)
    mg_si_temp = mean_nobrack_mg/mean_nobrack_si
    mg_si = (10**(t_sum['Mean'][np.where(t_sum['Abund']=='[Mg/H]')] + sol_mg_h))/(10**(t_sum['Mean'][np.where(t_sum['Abund']=='[Si/H]')] + sol_si_h))
    c_o = (10**(t_sum['Mean'][np.where(t_sum['Abund']=='[C/H]')] + sol_c_h))/(10**(t_sum['Mean'][np.where(t_sum['Abund']=='[O/H]')] + sol_o_h))
    #t_sum.add_row(["Mg/Si", mg_si, 0, 0])
    #t_sum.add_row(["C/O", c_o, 0, 0])
    par = Table.read(path+"{}/best_parameters.tab".format(object), format='ascii')
    try:
        line = np.where(par['conv'] == 1)[0][0]-1
        teff = par['Teff'][line]
        eteff = par['err'][line]
        logg = par['logg'][line]
        elogg = par['err_1'][line]
        met = par['met'][line+1]
        emet = par['err_2'][line+1]
        t_sum.add_row(["Teff", teff, 0, eteff])
        t_sum.add_row(["logg", logg, 0, elogg])
        t_sum.add_row(["z", met, 0, emet])
    except:
        pass#print(object, "params didn't actually converge?")
    #print(t_sum)
    t_sum.write(path+object+'/{}_bracket_abunds_summary.txt'.format(object), format='ascii', overwrite=True)
    #print('wrote to ', path+object+'/{}_bracket_abunds_summary.txt'.format(object))
    big_t.write(path+object+'/{}_all_abunds.txt'.format(object), format='ascii', overwrite=True)
