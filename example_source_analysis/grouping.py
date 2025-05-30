import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import runmatching_utilities as util


#load and access the database information:
database_path_woody = '/home/wecapstor1/caph/shared/hess/fits/database_image'
DB_trigger = pd.read_csv('{}/trigger_1223.csv'.format(database_path_woody))
DB_pix = pd.read_csv('{}/pixel_1223.csv'.format(database_path_woody))
DB_track = pd.read_csv('{}/tracking_1223.csv'.format(database_path_woody))
DB_atmo = pd.read_csv('{}/atmosphere_1223.csv'.format(database_path_woody))
DB_weather = pd.read_csv('{}/weather_1223.csv'.format(database_path_woody))
DB_muon = pd.read_csv('{}/muon_1223.csv'.format(database_path_woody), low_memory=False)
DB_header = pd.read_csv('{}/header_1223.csv'.format(database_path_woody))
DB_ADC = pd.read_csv('{}/ADC_1223.csv'.format(database_path_woody))
monitor_run_data = pd.read_csv('{}/data_1223.csv'.format(database_path_woody)).sort_values("Run", ascending=True)
DB_radio = pd.read_csv('{}/radiometer_1223.csv'.format(database_path_woody))

# Observation numbers after which hardware changes were made (example set for publication)
muon_phases2 = [20000, 40000, 60000, 120000, monitor_run_data['Run'][len(monitor_run_data)-1]]


def fetch_parameters(parameter_name, dataframe, column, run, tel=None):
    """
    Description

    Parameters:
    ----------
    - parameter_name: str
        Name of the parameter queried from the database
    - dataframe: pd.Dataframe
        Database which to query
    - column: str
        Column name of the parameter
    - run: int
        Observtation run number
    - tel: str
        If not None: Number of the telescope which should be queried

    Returns:
    -------
    Dictionary
"""
    if tel=='system':
        params_dict = np.abs(dataframe[(dataframe['Run']==run)][column].iloc[0])
    else:
        params_dict = {}
        for tel in range(1,5):
            params_dict[parameter_name + '_CT' + str(tel)] = [dataframe[(dataframe['Run']==run) & (dataframe['Telescope']==tel)][column].iloc[0]]
            params_dict = pd.DataFrame(params_dict)
    return params_dict




# mimic the 3 major HESS hardware phases and query the database for all observations in these periods
hess1_unfiltered = monitor_run_data[monitor_run_data['Run'] < 40000]['Run'].to_list()
hess2_unfiltered = monitor_run_data[(monitor_run_data['Run'] > 40000) & (monitor_run_data['Run'] < 60000)]['Run'].to_list()
hess1u_unfiltered = monitor_run_data[monitor_run_data['Run'] > 120000]['Run'].to_list()

# Quality select the observation runs
quality_selection = util.quality_selection(runlist=hess1_unfiltered, HessEra='HESS1', Config='stereo', FitsEra='HESS1', AnalysisType='spectral')
hess1 = quality_selection.quality_cuts(requireCT5=False, logBadRuns=False)
quality_selection = util.quality_selection(runlist=hess2_unfiltered, HessEra='HESS2', Config='stereo', FitsEra='HESS2', AnalysisType='spectral')
hess2 = quality_selection.quality_cuts(requireCT5=False, logBadRuns=False)
quality_selection = util.quality_selection(runlist=hess1u_unfiltered, HessEra='HESS2', Config='stereo', FitsEra='HESS1u', AnalysisType='spectral')
hess1u = quality_selection.quality_cuts(requireCT5=False, logBadRuns=False)


# Define which Parameters for which telescopes we query
params1 = pd.DataFrame(columns=['run',
                                'zenith', 'duration',
                                'NSB_CT1', 'NSB_CT2', 'NSB_CT3', 'NSB_CT4', 
                                'muon_eff_CT1', 'muon_eff_CT2', 'muon_eff_CT3', 'muon_eff_CT4', 
                                'trigger_CT1', 'trigger_CT2', 'trigger_CT3', 'trigger_CT4',
                                'transp_CT1', 'transp_CT2', 'transp_CT3','transp_CT4', 
                                'radio_CT1', 'radio_CT2', 'radio_CT3','radio_CT4',
                               ])


telescope_ids = [1, 2, 3, 4]

for run in hess1:
    # Check if the basic conditions are met and all data is filled
    if not DB_trigger[DB_trigger['Run'] == run].empty \
    and not monitor_run_data[monitor_run_data['Run'] == run].empty \
    and not DB_radio[DB_radio['Run'] == run].empty \
    and not DB_pix[DB_pix['Run'] == run].empty \
    and not DB_muon[DB_muon['Run'] == run].empty \
    and not DB_atmo[DB_atmo['Run'] == run].empty:
        all_conditions_met = True
    
        if not DB_atmo[DB_atmo['Run'] == run]['TransparencyCoefficient_CT1'].empty \
        and not DB_atmo[DB_atmo['Run'] == run]['TransparencyCoefficient_CT2'].empty \
        and not DB_atmo[DB_atmo['Run'] == run]['TransparencyCoefficient_CT3'].empty \
        and not DB_atmo[DB_atmo['Run'] == run]['TransparencyCoefficient_CT4'].empty:
    
            for telescope in telescope_ids:
                if DB_trigger[(DB_trigger['Run'] == run) & (DB_trigger['Telescope'] == telescope)].empty \
                and DB_radio[(DB_radio['Run'] == run) & (DB_radio['Telescope'] == telescope)].empty \
                and DB_muon[(DB_muon['Run'] == run) & (DB_muon['Telescope'] == telescope)].empty \
                and DB_pix[(DB_pix['Run'] == run) & (DB_pix['Telescope'] == telescope)].empty:
                
                    if DB_trigger[(DB_trigger['Run'] == run) & (DB_trigger['Telescope'] == telescope)]['True_Rate_mean'].empty \
                    and DB_radio[(DB_radio['Run'] == run) & (DB_radio['Telescope'] == telescope)]['Temperature_mean'].empty \
                    and DB_muon[(DB_muon['Run'] == run) & (DB_muon['Telescope'] == telescope)]['Efficiency_mean'].empty \
                    and DB_pix[(DB_pix['Run'] == run) & (DB_pix['Telescope'] == telescope)]['Ped_NSB_mean'].empty:

                        #Get all information from the database and fill into one dataframe
                        runnumber = pd.DataFrame([run], columns=['run'])
                        zenith_on = DB_trigger[DB_trigger['Run'] == run]['Mean_Zenith'].iloc[0] 
                        zenith = pd.DataFrame([zenith_on], columns=['zenith'])
                        duration_on = monitor_run_data[monitor_run_data['Run'] == run]['Duration'].iloc[0]
                        duration = pd.DataFrame([duration_on], columns=['duration'])
                    
                        NSB = fetch_parameters('NSB', DB_pix, 'Ped_NSB_mean', run)
                        muon = fetch_parameters('muon_eff', DB_muon, 'Efficiency_mean', run)
                        rate = fetch_parameters('trigger', DB_trigger, 'True_Rate_mean', run)
                        broken = fetch_parameters('broken', DB_pix, 'Num_Broken', run)
                        radio = fetch_parameters('radio', DB_radio, 'Temperature_mean', run)
                    
                        transparency = {}
                        for tel in range(1,5):
                            transparency['transp_CT' + str(tel)] =  [fetch_parameters('transp', DB_atmo, 'TransparencyCoefficient_CT' + str(tel), 
                                                                                      run, tel='system')]
                        transparency = pd.DataFrame(transparency)
                    
                        new_df = pd.merge(runnumber, zenith, left_index=True, right_index=True)
                        new_df = pd.merge(new_df, duration, left_index=True, right_index=True)
                        new_df = pd.merge(new_df, NSB, left_index=True, right_index=True)
                        new_df = pd.merge(new_df, muon, left_index=True, right_index=True)
                        new_df = pd.merge(new_df, rate, left_index=True, right_index=True)
                        new_df = pd.merge(new_df, transparency, left_index=True, right_index=True)
                        new_df = pd.merge(new_df, broken, left_index=True, right_index=True)
                        new_df = pd.merge(new_df, radio, left_index=True, right_index=True)
                    
                        params1 = pd.concat([params1, new_df], ignore_index=True)

params1.to_csv('grouped_vals_hess1.csv', sep='\t')


























