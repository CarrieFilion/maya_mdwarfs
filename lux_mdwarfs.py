#%%
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.constraints import real
import astropy.table as at
import scipy.interpolate as interp
from gather_data import load_spectra

import pollux as plx
from pollux.models.transforms import LinearTransform

jax.config.update("jax_enable_x64", True)
%matplotlib inline
#%%
galah = at.Table.read('/mnt/home/cfilion/ceph/abundances/galah_dr4_allstar_240705.fits')
cluster_labels = at.Table.read('/mnt/home/cfilion/abundances/maya_mdwarfs/CCA_work/galah_members_isochrone_parameters_feh_fitted.fits', format='fits')
#%%
#%%
#ok so I want to manually give the m-dwarfs in the clusters the same abundances as their warmer friends?
common_cluster_lab_subset = ['fe_h', 'mg_fe', 'v_fe','iso_logg', 'iso_teff']
galah_subset_names = ['fe_h', 'mg_fe', 'v_fe','logg', 'teff']
common_cluster_lab_subset_names = ['[Fe/H]', '[Mg/Fe]', '[V/Fe]', 'logg', 'Teff']
cluster_mdwarfs = cluster_labels[(cluster_labels['iso_teff']<4100)&(cluster_labels['iso_logg']>3.5)]
print(len(cluster_mdwarfs), ' cluster mdwarf members with good SNR')
#%%
#%%
#oh my god galah is so rank, the wavelengths are just straight up not the same for different stars
#and they're different lengths?! gonna just trim to common space I think?
wide_bins = at.Table.read('/mnt/home/cfilion/abundances/maya_mdwarfs/CCA_work/m_dwarf_g_dwarf_wide_binaries.fits')
wide_bins_mdwarfs = wide_bins[(wide_bins['teff']<4100)] #adding two where we know the answers
additional_mdwarfs = galah[(galah['logg']>4)&(galah['teff']<4100)&(galah['snr_px_ccd3']>100)]
subset_of_idds = additional_mdwarfs['sobject_id'][0:98].value
subset_of_idds = np.append(subset_of_idds, wide_bins_mdwarfs['sobject_id'].value)
all_mdwarfs_considered = np.append(subset_of_idds, cluster_mdwarfs['sobject_id'].value)
min_wav = np.array([])
max_wav = np.array([])
alt_max = np.array([])
len_wav = np.array([])
break_1 = np.array([])
break_2 = np.array([])
break_3 = np.array([])
start_1 = np.array([])
start_2 = np.array([])
start_3 = np.array([])
null_sobjectid = np.array([])
for sobjectid in all_mdwarfs_considered:
    try:
        dat = load_spectra(sobjectid)
        min_wav = np.append(min_wav, np.min(dat[1].data['wave'][dat[1].data['mob']]))
        max_wav = np.append(max_wav, np.max(dat[1].data['wave'][dat[1].data['mob']]))
        alt_max = np.append(alt_max, dat[1].data['wave'][dat[1].data['mob']][-1])
        len_wav = np.append(len_wav, len(dat[1].data['wave'][dat[1].data['mob']]))
        idds = np.where(np.diff(dat[1].data['wave'])>.1)[0]
        break_1 = np.append(break_1, dat[1].data['wave'][idds[0]])
        break_2 = np.append(break_2, dat[1].data['wave'][idds[1]])
        break_3 = np.append(break_3, dat[1].data['wave'][idds[2]])
        start_1 = np.append(start_1, dat[1].data['wave'][idds[0]+1])
        start_2 = np.append(start_2, dat[1].data['wave'][idds[1]+1])
        start_3 = np.append(start_3, dat[1].data['wave'][idds[2]+1])
    except:
        print(sobjectid, ' doesnt work ')
        null_sobjectid = np.append(null_sobjectid, sobjectid)
        pass
#%%#%%
subset_of_idds = subset_of_idds[(subset_of_idds!=null_sobjectid[0])&\
                                      (subset_of_idds!=null_sobjectid[1])&\
                                    (subset_of_idds!=null_sobjectid[2])]
#%%
#going to just do straight up normal interpolation to a common wavelength grid here
#oh shoot - how is this handling the fact that we have big ass gaps
#why is max wave not working right?!


dloglam = np.log(1+1/28000)
np.max(min_wav), np.min(break_1)
wave_mins = [np.log(np.max(min_wav)), np.log(np.max(start_1)), np.log(np.max(start_2)), np.log(np.max(start_3))]
wave_maxes = [np.log(np.min(break_1)), np.log(np.min(break_2)), np.log(np.min(break_3)), np.log(np.min(max_wav))]
first_stint = np.arange(wave_mins[0], wave_maxes[0], dloglam)
second_stint = np.arange(wave_mins[1], wave_maxes[1], dloglam)
third_stint = np.arange(wave_mins[2], wave_maxes[2], dloglam)
fourth_stint = np.arange(wave_mins[3], wave_maxes[3]-dloglam, dloglam) ### hacky fix to make interpolator shut up
stints = [first_stint, second_stint, third_stint, fourth_stint]
wav_grid = np.hstack([first_stint, second_stint, third_stint, fourth_stint])
test_spec = load_spectra(cluster_mdwarfs['sobject_id'][0])
spectra_arr = np.ones((len(cluster_mdwarfs), len(wav_grid)))
spec_err_arr = np.ones((len(cluster_mdwarfs), len(wav_grid)))
label_arr = np.ones((len(cluster_mdwarfs), len(common_cluster_lab_subset)))
label_err_arr = np.ones((len(cluster_mdwarfs), len(common_cluster_lab_subset)))
interp_spec = np.array([])
interp_err = np.array([])
ii = -1
for sobjectid in cluster_mdwarfs['sobject_id']:
    ii += 1
    dat = load_spectra(sobjectid)
    #chip gaps are super annoying here
    idds = np.append(-1, np.append(np.where(np.diff(dat[1].data['wave'])>.1)[0], -2)) #want to start at start
    for bn in range(len(idds)-1):
        #fill spec array, where masked or flux = 0, replace with 1
        wave = np.log(dat[1].data['wave'])[idds[bn]+1:idds[bn+1]+1][dat[1].data['mob'][idds[bn]+1:idds[bn+1]+1]]
        obs_spec = dat[1].data['sob'][idds[bn]+1:idds[bn+1]+1][dat[1].data['mob'][idds[bn]+1:idds[bn+1]+1]]
        obs_err = dat[1].data['uob'][idds[bn]+1:idds[bn+1]+1][dat[1].data['mob'][idds[bn]+1:idds[bn+1]+1]]

        interp_spec = np.append(interp_spec, interp.interp1d(wave, 
                                    obs_spec, kind='cubic')(stints[bn]))
        interp_err = np.append(interp_err, interp.interp1d(wave, 
                                    obs_err, kind='cubic')(stints[bn]))
    spectra_arr[ii,:] = np.where(interp_spec==0, 1, interp_spec) 
    #fill err array, masked or flux = 0 pixels have err = 9999
    spec_err_arr[ii,:] = np.where(interp_spec==0, 9999, interp_err) 
    interp_spec = np.array([])
    interp_err = np.array([])
    mdwarf_home = cluster_mdwarfs[cluster_mdwarfs['sobject_id']==sobjectid]['cluster']
    cluster_parameters = cluster_labels[(cluster_labels['cluster']==mdwarf_home)&(cluster_labels['iso_teff']>=4100)]
    #fill label array
    jj = -1
    for element in common_cluster_lab_subset:
        jj += 1
        if element == 'iso_logg' or element=='iso_teff':
            measurement = cluster_mdwarfs[cluster_mdwarfs['sobject_id']==sobjectid][element].value
            measurement_err = .1
        else:
            measurement = np.nanmean(cluster_parameters[element].value)
            measurement_err = np.nanstd(cluster_parameters[element].value)

        label_arr[ii, jj] = measurement
        label_err_arr[ii, jj] = measurement_err
#%%
### getting test set up and running
test_spectra_arr = np.ones((len(subset_of_idds), len(wav_grid)))
test_spec_err_arr = np.ones((len(subset_of_idds), len(wav_grid)))
test_label_arr = np.ones((len(subset_of_idds), len(galah_subset_names)))
test_label_err_arr = np.ones((len(subset_of_idds), len(galah_subset_names)))
interp_spec = np.array([])
interp_err = np.array([])
ii = -1
for sobjectid in subset_of_idds:
    ii += 1
    dat = load_spectra(sobjectid)
    #chip gaps are super annoying here
    idds = np.append(-1, np.append(np.where(np.diff(dat[1].data['wave'])>.1)[0], -2)) #want to start at start
    for bn in range(len(idds)-1):
        #fill spec array, where masked or flux = 0, replace with 1
        wave = np.log(dat[1].data['wave'])[idds[bn]+1:idds[bn+1]+1][dat[1].data['mob'][idds[bn]+1:idds[bn+1]+1]]
        obs_spec = dat[1].data['sob'][idds[bn]+1:idds[bn+1]+1][dat[1].data['mob'][idds[bn]+1:idds[bn+1]+1]]
        obs_err = dat[1].data['uob'][idds[bn]+1:idds[bn+1]+1][dat[1].data['mob'][idds[bn]+1:idds[bn+1]+1]]

        interp_spec = np.append(interp_spec, interp.interp1d(wave, 
                                    obs_spec, kind='cubic')(stints[bn]))
        interp_err = np.append(interp_err, interp.interp1d(wave, 
                                    obs_err, kind='cubic')(stints[bn]))
    test_spectra_arr[ii,:] = np.where(interp_spec==0, 1, interp_spec) 
    test_spectra_arr[ii,:] = np.where(interp_err<0, 1, interp_spec) #oh fun, sometimes error is negative?
    #fill err array, masked or flux = 0 pixels have err = 9999
    test_spec_err_arr[ii,:] = np.where(interp_spec==0, 9999, interp_err) 
    test_spec_err_arr[ii,:] = np.where(interp_err<0, 9999, interp_err)
    interp_spec = np.array([])
    interp_err = np.array([])
        #fill label array
    jj = -1
    for element in galah_subset_names:
        jj += 1
        measurement = galah[galah['sobject_id']==sobjectid][element]
        measurement_err = galah[galah['sobject_id']==sobjectid]['e_'+element]
        if measurement.mask == True or measurement_err.mask == True:
            test_label_arr[ii, jj] = np.mean(galah[element])
            test_label_err_arr[ii, jj] = 9999
        else:
            test_label_arr[ii, jj] = measurement.value
            test_label_err_arr[ii, jj] = measurement_err.value
#%%
label_err_arr.shape
#%%
flux_dict = {}
flux_dict['flux'] = spectra_arr
flux_dict['flux_err'] = spec_err_arr
label_dict = {}
label_dict['label'] = label_arr
label_dict['label_err'] = label_err_arr
training_data_raw = plx.data.PolluxData(
    flux=plx.data.OutputData(
        data=flux_dict['flux'],
        err=flux_dict['flux_err'],
        preprocessor=plx.data.ShiftScalePreprocessor.from_data(flux_dict['flux']),
    ),
    label=plx.data.OutputData(
        data=label_dict['label'],
        err=label_dict['label_err'],
        preprocessor=plx.data.ShiftScalePreprocessor.from_data(label_dict['label']),
    ),
)

flux_dict2 = {}
flux_dict2['flux'] = test_spectra_arr
flux_dict2['flux_err'] = test_spec_err_arr
label_dict2 = {}
label_dict2['label'] = test_label_arr
label_dict2['label_err'] = test_label_err_arr
test_data_raw = plx.data.PolluxData(
    flux=plx.data.OutputData(
        data=flux_dict2['flux'],
        err=flux_dict2['flux_err'],
        preprocessor=plx.data.ShiftScalePreprocessor.from_data(flux_dict['flux']),
    ),
    label=plx.data.OutputData(
        data=label_dict2['label'],
        err=label_dict2['label_err'],
        preprocessor=plx.data.ShiftScalePreprocessor.from_data(label_dict['label']),
    ),)

train_data = training_data_raw.preprocess()
test_data = test_data_raw.preprocess()
#%%
len(train_data), len(test_data)
#%%
model = plx.LuxModel(latent_size=500)
#%%
print(train_data.keys()) 
model.register_output("label", LinearTransform(output_size=label_arr.shape[1]))
model.register_output("flux", LinearTransform(output_size=spectra_arr.shape[1]))
#%%

opt_params, svi_results = model.optimize(
    train_data,
    rng_key=jax.random.PRNGKey(112358),
    optimizer=numpyro.optim.Adam(1e-3),
    num_steps=100_000,
    svi_run_kwargs={"progress_bar": True},
)
#%%
opt_params
#%%
plt.plot(svi_results.losses[-1000:])
#%%
predict_train_values = model.predict_outputs(opt_params["latents"], opt_params)
predict_train_values['label'].shape
#%%
#%%
fixed_params = {
    "label": {"A": opt_params["label"]["A"]},
    "flux": {"A": opt_params["flux"]["A"]},
}

test_opt_params, _ = model.optimize(
    test_data,
    rng_key=jax.random.PRNGKey(12345),
    optimizer=numpyro.optim.Adam(1e-3),
    num_steps=100_000,
    fixed_params=fixed_params,
    svi_run_kwargs={"progress_bar": False},
)
#%%
test_data['flux'].data.flatten()[(test_data['flux'].err.flatten()<0)]
test_spec_err_arr.flatten()[test_spec_err_arr.flatten()<0]
##### most of the way thru this - need to figure out transform I think but woio!!!
#%%
predict_test_values = model.predict_outputs(test_opt_params["latents"], fixed_params)
#%%
predict_test_values_unprocessed = test_data.unprocess(predict_test_values)
#%%
fig, (ax) = plt.subplots(1,test_data_raw['label'].data.shape[1], figsize=(4*test_data_raw['label'].data.shape[1], 4))
for i in range(test_data_raw['label'].data.shape[1]):
    ax[i].scatter(test_data_raw['label'].data[:,i],
            predict_test_values_unprocessed['label'].data[:,i], c='plum')
    ax[i].plot(np.linspace(np.min(test_data_raw['label'].data[:,i]),
                           np.max(test_data_raw['label'].data[:,i]), 100),
                np.linspace(np.min(test_data_raw['label'].data[:,i]),
                           np.max(test_data_raw['label'].data[:,i]), 100), c='k')
    ax[i].set(xlabel=str(common_cluster_lab_subset_names[i])+' GALAH value',
              ylabel=str(common_cluster_lab_subset_names[i])+' Predicted value')
plt.subplots_adjust(wspace=.3)
plt.savefig('test_mdwarf.jpeg')
#%%
wide_bins[(wide_bins['teff']>4100)]

fig, (ax) = plt.subplots(1,3, figsize=(4*3, 4))
fig.suptitle('Wide Binaries')
for i in range(3):
    print(galah_subset_names[i], ' for binary ', wide_bins[(wide_bins['teff']>4100)][galah_subset_names[i]])
    print('predicted: ',predict_test_values_unprocessed['label'].data[-2:,i])
    ax[i].scatter(wide_bins[(wide_bins['teff']>4100)][galah_subset_names[i]],
            predict_test_values_unprocessed['label'].data[-2:,i], c='Pink')
    ax[i].set(xlabel=str(common_cluster_lab_subset_names[i])+' GALAH More Massive Companion',
              ylabel=str(common_cluster_lab_subset_names[i])+' Predicted value')
plt.subplots_adjust(wspace=.3)
plt.savefig('test_mdwarf_bin.jpeg') 
#oh this does not look good
#%%
#%%

#%%
np.linspace(np.min(test_data_raw['label'].data[:,1]),
                           np.max(test_data_raw['label'].data[:,1]), 100)
# %%
