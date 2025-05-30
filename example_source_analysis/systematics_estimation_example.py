import sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.maps import WcsGeom, MapAxis, Map, WcsNDMap
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.data import DataStore
from gammapy.datasets import Datasets, FluxPointsDataset, MapDataset
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel, LogParabolaSpectralModel, PointSpatialModel, PowerLawSpectralModel 
from regions import CircleSkyRegion, RectangleSkyRegion
from gammapy.estimators import (FluxPoints, FluxPointsEstimator, ExcessMapEstimator)
from gammapy.modeling.models import FoVBackgroundModel 

import sys  
sys.path.insert(1, '/home/hpc/caph/mppi103h/Documents/On-OFF-matching-woody')
import common_utils
from common_utils import get_excluded_regions



config = 'std_ImPACT_fullEnclosure_updated'


#load all the auxiliary information
auxiliary = '/home/wecapstor1/caph/mppi103h/On-Off-matching'
DB_general = pd.read_csv('{}/db_data.csv'.format(auxiliary), header=0)
correction_factor = pd.read_csv('{}/zenith_correction_factors_clean.csv'.format(auxiliary), sep='\t')
#hardware phases of the HESS telescopes (example set)
muon_phases2 = [20000, 40000, 60000, 120000, 180000]


#load the observationlists
bgmodel_version = 'v08c'
data_path = '/home/wecapstor1/caph/mppi103h/On-Off-matching/runlists/final/onoff_list_systematics_set1.txt'
runs = np.loadtxt(data_path, usecols=(0,), dtype=int)
off_runs = np.loadtxt(data_path, usecols=(1,), dtype=int)
deviations = np.loadtxt(data_path, usecols=(2,), dtype=str)

obs_list = ds2.get_observations(runs, required_irf='all-optional') 
obs_list_off = ds.get_observations(off_runs) 




entries_sculp = pd.DataFrame(columns=['on run', 'off run', 'deviation', 'bkg_dev', 'bkg_dev error', 'mu_off', 'mu_fov'])

for i in range(0,len(runs)):
    #choose the correct run and read in the correct fits file:
    run = runs[i]
    off_run = off_runs[i]
    deviation = deviations[i]

    #load the ON and OFF observations
    basedir = '/path/to/data/hess1/{}'.format(config)
    ds = DataStore.from_dir(basedir,
                               'hdu-index.fits.gz'.format(bgmodel_version),
                               'obs-index.fits.gz'.format(bgmodel_version))
    obs_list = ds.get_observations([run]) 
    obs_list_off = ds.get_observations([off_run])

    ds_fov = DataStore.from_dir(basedir,
                               'hdu-index.fits.gz'.format(bgmodel_version),
                               'obs-index.fits.gz'.format(bgmodel_version))
    obs_list_fov = ds_fov.get_observations([run]) 
    
    
    #define the geometry and dataset you want to use:
    ra_obj = 329.7208
    dec_obj = -30.2217
    name_obj = 'Crab'
    target = SkyCoord(ra_obj, dec_obj, frame='icrs', unit='deg')

    e_reco = np.logspace(-1, 2, 25) * u.TeV 
    e_true = np.logspace(-1, 2, 49) * u.TeV 

    energy_axis = MapAxis.from_edges(e_reco, unit='TeV', name='energy', interp='log')
    energy_axis_true = MapAxis.from_edges(e_true, unit='TeV', name="energy_true", interp='log')

    geom_match = WcsGeom.create(skydir=(ra_obj, dec_obj), binsz=0.02, width=(7, 7), frame="icrs", proj="CAR", axes=[energy_axis],)
    geom_fov = WcsGeom.create(skydir=(ra_obj, dec_obj), binsz=0.02, width=(7, 7), frame="icrs", proj="CAR", axes=[energy_axis],)
    
    
    #######################################################################################################################
    ############################## OFF run matching #######################################################################
    #######################################################################################################################

    #define the geometry of the ON run
    stacked_off = MapDataset.create(geom=geom_match, name="OFF",  energy_axis_true= energy_axis_true,)

    offset_max = 2.0 * u.deg
    maker = MapDatasetMaker()
    maker_safe_mask2 = SafeMaskMaker(methods=["offset-max", 'aeff-default', 'aeff-max', 'edisp-bias', 'bkg-peak'], offset_max=offset_max, bias_percent=10)
    maker_fov = FoVBackgroundMaker(method="fit")
    
    on_run = obs_list[0]
    cutout_on = stacked_off.cutout(on_run.pointing.fixed_icrs, width=2 * offset_max, name=f"obs-{on_run.obs_id}")
    dataset_on = maker.run(cutout_on, on_run)
    dataset_on = maker_safe_mask2.run(dataset_on, on_run) 
    
    number_on = 0
    for k in range(0,24): 
        for l in range(0,200):
            for m in range(0,200):
                if dataset_on.mask_safe.data[k][l][m] == True:
                    number_on = k 
                    break
        if number_on != 0:
            break


    
    #define the geometry for the OFF run
    off_run = obs_list_off[obs_list.index(on_run)]
    geom_off = WcsGeom.create(skydir=off_run.pointing.fixed_icrs, binsz=0.02, width=(4.0, 4.0), frame="icrs", proj="CAR", axes=[energy_axis],)
    cutout_off = MapDataset.create(geom=geom_off, name=f"obs-{on_run.obs_id}", energy_axis_true= energy_axis_true,)


    
    #create a exclusion mask around the target of the OFF run
    hap_exclusion_regions = get_excluded_regions(ra_obj, dec_obj, 5)
    excl_regions = []
    for source in hap_exclusion_regions:
        center = SkyCoord(source.ra, source.dec, unit='deg', frame='icrs')
        region = CircleSkyRegion(center=center, radius=source.radius*u.deg)
        excl_regions.append(region)
    ra = off_run.pointing.fixed_icrs.ra.value - DB_general[DB_general['Run']==off_run.obs_id]['Offset_x'].iloc[0]
    dec = off_run.pointing.fixed_icrs.dec.value - DB_general[DB_general['Run']==off_run.obs_id]['Offset_y'].iloc[0]
    excl_regions.append(CircleSkyRegion(center=SkyCoord(ra*u.deg, dec*u.deg), radius=0.5*u.deg))
    data2 = geom_off.region_mask(regions=excl_regions, inside=False)
    maker_fov_off = FoVBackgroundMaker(method="fit", exclusion_mask=data2)
    ex = maker_fov_off.exclusion_mask.cutout(off_run.pointing.fixed_icrs, width=2 * offset_max)



    
    #create the dataset and fit the background outside of the defined FOV mask, adjust the energy threshold of both
    dataset_off = maker.run(cutout_off, off_run)
    dataset_off = maker_safe_mask2.run(dataset_off, off_run)  
    
    number_off = 0
    for k in range(0,24): 
        for l in range(0,200):
            for m in range(0,200):
                if dataset_off.mask_safe.data[k][l][m] == True:
                    number_off = k 
                    break
        if number_off != 0:
            break
            
    if number_on > number_off:
        number_off = number_on

    if number_off > number_on:
        number_on = number_off

    bkg_array = np.zeros_like(dataset_on.background.data)
    for a in range(number_off, bkg_array.shape[0]):
        for b in range(0,200):
            for c in range(0,200):
                bkg_array[a][b][c] = dataset_on.background.data[a][b][c]
    
    back_on = WcsNDMap(geom=dataset_on.background.geom, data=np.array(bkg_array))
    dataset_on.background = back_on

    
    counts_array = np.zeros_like(dataset_on.counts.data)
    for a in range(number_off, counts_array.shape[0]):
        for b in range(0,200):
            for c in range(0,200):
                counts_array[a][b][c] = dataset_on.counts.data[a][b][c]
    
    counts_on = WcsNDMap(geom=dataset_on.counts.geom, data=np.array(counts_array))
    dataset_on.counts = counts_on

    bkg_array_off = np.zeros_like(dataset_off.background.data)
    for a in range(number_off, bkg_array_off.shape[0]):
        for b in range(0,200):
            for c in range(0,200):
                bkg_array_off[a][b][c] = dataset_off.background.data[a][b][c]
    
    back_off = WcsNDMap(geom=dataset_off.background.geom, data=np.array(bkg_array_off))
    dataset_off.background = back_off
    
    counts_array_off = np.zeros_like(dataset_off.counts.data)
    for a in range(number_off, counts_array_off.shape[0]):
        for b in range(0,200):
            for c in range(0,200):
                counts_array_off[a][b][c] = dataset_off.counts.data[a][b][c]
    
    counts_off = WcsNDMap(geom=dataset_off.counts.geom, data=np.array(counts_array_off))
    dataset_off.counts = counts_off
    

    bkg_model = FoVBackgroundModel(dataset_name=dataset_off.name)
    dataset_off.models=bkg_model
    dataset_off.background_model.spectral_model.tilt.frozen = False
    dataset_off = maker_fov.run(dataset_off)
    

    #map the background to the ON run
    bkg_model = FoVBackgroundModel(dataset_name=dataset_on.name)
    dataset_on.models=bkg_model
    dataset_on.background_model.spectral_model.norm.value = dataset_off.background_model.spectral_model.norm.value
    dataset_on.background_model.spectral_model.tilt.value = dataset_off.background_model.spectral_model.tilt.value
    dataset_on.background_model.spectral_model.norm.frozen = True
    dataset_on.background_model.spectral_model.tilt.frozen = True


    #Apply the corrections:
    livetime_dev = off_run.observation_live_time_duration.value - on_run.observation_live_time_duration.value
    counts_per_sec = dataset_on.background.data/off_run.observation_live_time_duration.value
    #Livetime correction:
    factors = counts_per_sec*livetime_dev
    bkg = [x + y for x, y in zip(dataset_on.background.data, factors)]

    for phase in range(0,len(muon_phases2)-1):
        if on_run.obs_id > muon_phases2[phase] and on_run.obs_id < muon_phases2[phase+1]:
            factor1 = correction_factor.x_1.iloc[phase]
            factor2 = correction_factor.x_2.iloc[phase]


    bkg = np.array(bkg)*np.cos((np.deg2rad(ds.obs_table[ds.obs_table['OBS_ID']==off_run.obs_id]["ZEN_PNT"])-np.deg2rad(ds.obs_table[ds.obs_table['OBS_ID']==on_run.obs_id]["ZEN_PNT"])))**factor2
    
    
    #now we do the on-off matching
    bkg = WcsNDMap(geom=dataset_on.counts.geom, data=np.array(bkg))
    dataset_on.background = bkg

    stacked_off.stack(dataset_on)
    
    
    
    ########################################################################################################################
    ############################################## FOV background ##########################################################
    ########################################################################################################################
    
    
    stacked_fov = MapDataset.create(geom=geom_fov, name="FOV",  energy_axis_true= energy_axis_true,)

    offset_max = 2.0 * u.deg
    maker = MapDatasetMaker()
    maker_safe_mask = SafeMaskMaker(methods=["offset-max", 'aeff-default', 'aeff-max', 'edisp-bias', "bkg-peak"], offset_max=offset_max, bias_percent=10)

    #get the exclusion regions 
    hap_exclusion_regions = get_excluded_regions(ra_obj, dec_obj, 5)
    excl_regions = []
    excl_regions_without_source = []
    for source in hap_exclusion_regions:
        center = SkyCoord(source.ra, source.dec, unit='deg', frame='icrs')
        region = CircleSkyRegion(center=center, radius=source.radius*u.deg)
        excl_regions.append(region)
        if not source.name == 'HESSB1055-52':
            excl_regions_without_source.append(region)
    for source in hap_exclusion_regions:
        print(source.name, source.ra, source.dec, source.radius)

    center=SkyCoord(ra_obj* u.deg, dec_obj* u.deg)
    excl_region = CircleSkyRegion(center, radius=0.3*u.deg)
    excl_regions.append(excl_region)


    data = geom_fov.region_mask(regions=excl_regions, inside=False)
    maker_fov = FoVBackgroundMaker(method="fit", exclusion_mask=data)
    
    
    obs = obs_list_fov[0]
    # First a cutout of the target map is produced
    cutout = stacked_fov.cutout(
        obs.pointing.fixed_icrs, width=2 * offset_max, name=f"obs-{obs.obs_id}")
    ex = maker_fov.exclusion_mask.cutout(obs.pointing.fixed_icrs, width=2 * offset_max)

    # A MapDataset is filled in this cutout geometry
    dataset = maker.run(cutout, obs)
    maker_safe_mask.make_mask_energy_bkg_peak(dataset)
    dataset = maker_safe_mask.run(dataset, obs)
    
    if number_off > number_on:
        for k in range(0,number_off): 
            for l in range(0,200):
                for m in range(0,200):
                    dataset_on.mask_safe.data[k][l][m] == False

    # fit background model
    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models=bkg_model
    dataset.background_model.spectral_model.tilt.frozen = False
    dataset = maker_fov.run(dataset)

    stacked_fov.stack(dataset)
    

    
    #####################################################################################################################
    ###################################### calculation of the background rate ###########################################
    #####################################################################################################################
    reg = CircleSkyRegion(center=SkyCoord(ra_obj * u.deg, dec_obj * u.deg), radius=0.5* u.deg)
    #bkg_off = stacked_off.background.sum_over_axes().data.sum()
    #bkg_fov = stacked_fov.background.sum_over_axes().data.sum()
    off = stacked_off.background * stacked_off.background.geom.region_mask(reg, inside=False)
    fov = stacked_fov.background * stacked_off.background.geom.region_mask(reg, inside=False)
    
    off_num = 0
    for i in range(0,24):
        if np.nansum(off.data[i]) == 0:
            off_num = i

    fov_num = 0
    for i in range(0,24):
        if np.nansum(fov.data[i]) == 0:
            fov_num = i

    if off_num < fov_num:
        for l in range(0,fov_num+1):
            off.data[l] = np.zeros(off.data[l].shape)

    if fov_num < off_num:
        for l in range(0,off_num+1):
            fov.data[l] = np.zeros(fov.data[l].shape)
            
    bkg_off = off.data.sum()
    bkg_fov = fov.data.sum()

    #Significance of the model:
    estimator_001 = ExcessMapEstimator(
    correlation_radius="0.1 deg",
    energy_edges=[0.7, 100] * u.TeV,
    selection_optional=[],)
    lima_maps_off = estimator_001.run(stacked_off)
    lima_maps_fov = estimator_001.run(stacked_fov)

    hap_exclusion_regions = get_excluded_regions(lima_maps_off["sqrt_ts"].geom.center_coord[0].value, lima_maps_off["sqrt_ts"].geom.center_coord[1].value, 5)
    excl_regions = []
    for source in hap_exclusion_regions:
        center = SkyCoord(source.ra, source.dec, unit='deg', frame='icrs')
        region = CircleSkyRegion(center=center, radius=source.radius*u.deg)
        excl_regions.append(region)
               

    bins=np.linspace(-6, 8, 131)
    excl_data = lima_maps_off["sqrt_ts"].geom.region_mask(regions=excl_regions, inside=False)
    significance_map_off = lima_maps_off["sqrt_ts"] * excl_data
    significance_off = significance_map_off.data[np.isfinite(significance_map_off.data)]
    significance_all = significance_map_off.data[np.isfinite(significance_map_off.data)]
    significance_data = significance_off
    mu, std = norm.fit(significance_data)

    excl_data = lima_maps_fov["sqrt_ts"].geom.region_mask(regions=excl_regions, inside=False)
    significance_map_off2 = lima_maps_fov["sqrt_ts"] * excl_data
    significance_off2 = significance_map_off2.data[np.isfinite(significance_map_off2.data)]
    significance_all2 = significance_map_off2.data[np.isfinite(significance_map_off2.data)]
    significance_data2 = significance_off2
    mu2, std2 = norm.fit(significance_data2)
        

    bkg_dev = (bkg_fov - bkg_off)/bkg_fov
    bkg_dev_error = np.sqrt((np.sqrt(bkg_off)/bkg_fov)**2 + ((-bkg_off*np.sqrt(bkg_fov))/bkg_fov**2)**2)
    
    new_entries = pd.DataFrame([[run, off_run.obs_id, bkg_off, bkg_fov, deviation, bkg_dev, bkg_dev_error, mu, mu2]], 
                            columns=['on run', 'off run', 'bkg_off', 'bkg_fov', 'deviation', 'bkg_dev', 'bkg_dev error', 'mu_off', 'mu_fov'])
    
    entries_sculp = pd.concat([entries_sculp, new_entries], ignore_index=True)


entries_sculp.to_csv('bkg_dev_cut_values_hess1_thight.csv', sep='\t')







