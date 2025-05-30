import pandas as pd
import numpy as np
import yaml
from astropy.coordinates import SkyCoord
from astropy import units as u
import os

#load and access the database information:
database_path_woody = '/home/wecapstor1/caph/shared/hess/fits/database_image'
DB_trigger = pd.read_csv('{}/trigger_0924.csv'.format(database_path_woody), delimiter='\t')
DB_pix = pd.read_csv('{}/pixel_0924.csv'.format(database_path_woody), delimiter='\t')
DB_track = pd.read_csv('{}/tracking_0924.csv'.format(database_path_woody), delimiter='\t')
DB_atmo = pd.read_csv('{}/atmosphere_0924.csv'.format(database_path_woody), delimiter='\t')
DB_weather = pd.read_csv('{}/weather_0924.csv'.format(database_path_woody), delimiter='\t')
DB_muon = pd.read_csv('{}/muon_0924.csv'.format(database_path_woody), delimiter='\t', low_memory=False)
DB_header = pd.read_csv('{}/header_0924.csv'.format(database_path_woody), delimiter='\t')
DB_ADC = pd.read_csv('{}/ADC_0924.csv'.format(database_path_woody), delimiter='\t')
monitor_run_data = pd.read_csv('{}/data_0924.csv'.format(database_path_woody), delimiter='\t').sort_values("Run", ascending=True)
DB_radio = pd.read_csv('{}/radiometer_0924.csv'.format(database_path_woody), delimiter='\t')


""" 
This is information specific to the HESS Muon Phases, as well as the Background Model Template developed by Mohrmann et. al. (2019)
"""
alt_bounds = [30.0, 37.5, 42.5, 47.5, 55., 65., 75., 90.]
zenith_bounds = [90-x for x in alt_bounds]
# Observation IDs after which hardware was changed (example set)
muon_phases = [20000, 40000, 60000, 128000, monitor_run_data['Run'][len(monitor_run_data)-1]]




def query_runs(TargetRA, TargetDec, Radius, TelPattern='fits'):
    """
    Get all observations of Targets in the specified radius around the target position: 
    
        --------------------------------------------
        Parameters:

            - TargetRA: float, Right Ascension of the target position in the 'icrs' frame in degree

            - TargetDec: float, Declination of the target postion in the 'icrs'frame in degree

            - Radius: The radius out to which you want to query observations in degree

            - TelPattern: float, either use Telescope Pattern used for the observation or choose 'fits' which corresponds to tel pattern 30 and 62
    """

    run_list = []

    # Specify the required telescope pattern for the OFF runs
    allowed_tel_pattern = []
    if TelPattern == 'fits':
        allowed_tel_pattern.extend([30, 62])
    else:
        allowed_tel_pattern.append([TelPattern])
    
    for i in range(0,len(monitor_run_data)):
        ra_tel = monitor_run_data['Target_RA'].iloc[i] + monitor_run_data['Offset_x'].iloc[i]
        dec_tel = monitor_run_data['Target_Dec'].iloc[i] + monitor_run_data['Offset_y'].iloc[i]
        name = monitor_run_data['Target_Name'].iloc[i]
        tel_pattern = monitor_run_data['Telescope_Pattern'].iloc[i]

        if tel_pattern in allowed_tel_pattern: 
            if (TargetRA -Radius <= ra_tel <= TargetRA +Radius) \
            and (TargetDec -Radius <= dec_tel <= TargetDec +Radius):
                run_list.append(monitor_run_data['Run'].iloc[i])

    return run_list


def query_off_runs(FitsEra, TelPattern='fits', dist_to_plane=10):
    """
    Get all observations in the selected era which can be used as OFF runs: 
    
        - Filter out galactic observations and observations with detected extended emission
        
        - Filter out calibration runs
        --------------------------------------------
        Parameters:

            - FitsEra: string, choose between 'HESS1', 'HESS2' and 'HESS1u'

            - TelPattern: float, either use Telescope Pattern used for the observation or choose 'fits' which corresponds to tel pattern 30 and 62
    """
    
    targets_with_detection = ['Fornax A west lobe', 'Fornax A east lobe', '1ES 0229+200', 'PSR B1055-52']

    # Specify the required telescope pattern for the OFF runs
    allowed_tel_pattern = []
    if TelPattern == 'fits':
        allowed_tel_pattern.extend([30, 62])
    else:
        allowed_tel_pattern.append([TelPattern])
    
    
    run_list = []
    # The start and end obs ids of each phase have been matched to the status of the FITS production (current status 2024)
    last_obs_HESS1 = 60000
    last_obs_HESS2 = 12000

    if FitsEra == 'HESS1':
        start_index = 0
        end_index = last_obs_HESS1
    elif FitsEra == 'HESS2':
        start_index = last_obs_HESS1
        end_index = last_obs_HESS2
    elif FitsEra == 'HESS1u':
        start_index = last_obs_HESS2
        end_index = monitor_run_data.iloc[-1]["Run"]

    monitor_run_data_era = monitor_run_data[monitor_run_data['Run'].between(start_index, end_index)]
    
    for i in range(0, len(monitor_run_data_era)):
        if not monitor_run_data.empty:          
            ra   = monitor_run_data_era['Target_RA'].iloc[i]
            dec  = monitor_run_data_era['Target_Dec'].iloc[i]
            name = monitor_run_data_era['Target_Name'].iloc[i]
            tel_pattern = monitor_run_data_era['Telescope_Pattern'].iloc[i]

            #Filter for Observation Runs of 'empty' sky regions taken with a specific telecope pattern:
            if monitor_run_data_era['RunType'].iloc[i] == 'ObservationRun' \
            and name not in targets_with_detection \
            and tel_pattern in allowed_tel_pattern:
                
                #filter for sensible observations
                if dec > -90 and dec < 90:
                    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs').galactic
                    
                    #filter for extragalactic runs
                    if coord.b.value > dist_to_plane or coord.b.value < -dist_to_plane:
                        run_list.append(monitor_run_data_era['Run'].iloc[i])
        
    return run_list




class quality_selection():
    """
        Apply quality selection cuts on your runlist
        --------------------------------------------
        Parameters:
            - HessEra: string, choose between 'HESS1'and 'HESS2'

            - Config: string, choose between 'mono', 'stereo' and 'hybrid'

            - FitsEra: string, choose between 'HESS1', 'HESS2' and 'HESS1u'

            - AnalysisTpye: string, choose between 'detection' and 'spectral'
    
            - requireCT5: bool
    
            - logBadRuns: bool
    """

    def __init__(self, runlist, HessEra, Config, FitsEra, AnalysisType):
        self.runs = runlist
        self.HessEra = HessEra
        self.Config = Config
        self.FitsEra = FitsEra
        self.AnalysisType = AnalysisType


    def checkHessEra(self):
        if (self.HessEra != "HESS1") and (self.HessEra != "HESS2"):
            raise Exception("ERROR! This HessEra '"+self.HessEra+"' doesn't exist. Please choose between 'HESS1' or 'HESS2'")


    def checkConfig(self):
        fHybrid, fStereo, fMono = False, False, False
        if self.Config == "hybrid":
            fHybrid = True
        elif self.Config == "stereo":
            fStereo = True
        elif self.Config == "mono":
            fMono = True
        else:
            print("ERROR! Unknown config '"+self.Config+"'. Please choose between 'hybrid','stereo' or 'mono'")
    
        return fHybrid, fStereo, fMono


    def filter_era(self):
        """Return only observations taken during the specified hardware period """
        
        last_obs_HESS1 = 85399
        last_obs_HESS2 = 123799
        if (self.FitsEra != "HESS1") and (self.FitsEra != "HESS2") and (self.FitsEra != "HESS1u"):
            raise Exception("ERROR! This FitsEra '"+self.FitsEra+"' doesn't exist. Please choose between 'HESS1', 'HESS2' or 'HESS1u'")
        if self.FitsEra == 'HESS1':
            filtered_runs = [x for x in self.runs if x <= last_obs_HESS1]
        if self.FitsEra == 'HESS2':
            filtered_runs = [x for x in self.runs if x > last_obs_HESS1 and x <=last_obs_HESS2]
        if self.FitsEra == 'HESS1u':
            filtered_runs = [x for x in self.runs if x > last_obs_HESS2]
        return filtered_runs


            

    def remove_3tel_runs(self):
        """ Remove observations taken with less than 4 Telescopes"""

        passing_runs = []
        for tested_run in self.runs:
            tested_run_pattern = monitor_run_data[monitor_run_data['Run']==tested_run]['Telescope_Pattern'].iloc[0]
            if tested_run_pattern == 30 or tested_run_pattern==62:
                passing_runs.append(tested_run)
            else:
                print('remove run', tested_run)
                
        return passing_runs


    def quality_cuts(self, requireCT5=False, logBadRuns=False, off_run_sel=False):
        """ 
        Apply quality cuts for each observation by checking different observation criteria.

        The criteria are taken from haptools.runselector and up to date with the current websummary quality flags
        
        """
    
        # First some safety checks (spelling errors etc.)
        self.checkHessEra()
        fHybrid, fStereo, fMono = self.checkConfig()
    
        # Deciding between different analysis types
        fSpectral, fDetection = False, False
        if self.AnalysisType == "spectral":
            fSpectral = True
            HESS_Quality_Criteria_file = "/home/woody/caph/mppi103h/runselection/quality_run_selection/"+self.HessEra+"_SPECTRAL_CRITERIA.yaml"
        elif self.AnalysisType == "detection":
            fDetection = True
            HESS_Quality_Criteria_file ="/home/woody/caph/mppi103h/runselection/quality_run_selection/"+self.HessEra+"_DETECTION_CRITERIA.yaml"
        else:
            print("WRONG! Choose between 'detection' or 'spectral'.")
            raise
    
        # Loading in the different Quality selection criteria
        HESS_Quality_Criteria =  open(HESS_Quality_Criteria_file,'r')
        HESS_Quality_Criteria = yaml.safe_load(HESS_Quality_Criteria)
        runs_failed = []
        runlist_selected = []

        #define a BadQualityRuns file
        if(logBadRuns):
            logFile = open("logBadQualityRuns.out", "w")
            logFile.write("Here you can find the 'bad quality' runs --> always suggested to check the websummary \n")
            logFile.close()

        #filter the run list to only contain observations in the specific hardware phase
        filtered_runs = self.filter_era()
        
        for run in filtered_runs:
            n_failed_criteria = 0
            failed_criteria_logs = []
    

            """ Check if the observation has a acceptable duration """
            tested_run = monitor_run_data[monitor_run_data['Run']==run]
            low_bound = HESS_Quality_Criteria["Duration"][0]
            high_bound = HESS_Quality_Criteria["Duration"][1]
            if tested_run.empty:
                runs_failed.append(run)
                n_failed_criteria += 1
                if(logBadRuns):
                    failed_criteria_logs.append("   - Missing data {Duration} ")
                    
            else:
                duration = tested_run['Duration'].iloc[0]
                if duration != 0 and (duration < low_bound or duration > high_bound):
                        runs_failed.append(run)
                        n_failed_criteria += 1
                        if(logBadRuns):
                            failed_criteria_logs.append(
                                "   - Duration criteria [{},{}] failed: run duration {} ".format(low_bound, \
                                                                                                 high_bound, \
                                                                                                 tested_run['Duration'].iloc[0])
                            )

    
    
            """ Check the participation fraction of the observation """
            tested_run = DB_header[DB_header['Run']==run]
            tel_pattern = monitor_run_data[monitor_run_data['Run']==run]['Telescope_Pattern'].iloc[0]
            low_bound = HESS_Quality_Criteria["Participation_frac_"+self.Config][0]
            high_bound = HESS_Quality_Criteria["Participation_frac_"+self.Config][1]
            failed_part_frac = []
            
            if tested_run.empty:
                runs_failed.append(run)
                n_failed_criteria += 1
                failed_criteria_logs.append("   - Missing data {Participation_frac} ")
    
            else:
                if(fMono):
                    tested_run = tested_run[tested_run["Telescope"] == 5]
                if((fStereo==True) & (requireCT5!=True)):
                    tested_run = tested_run[tested_run["Telescope"] != 5]
                for tel_id in np.unique(tested_run["Telescope"]):
                    tested_run_telid = tested_run[tested_run["Telescope"]==tel_id]
                    tel_participation = tested_run_telid['Participation_frac'].iloc[0]
                    if (self.Config=='mono' or self.Config=='hyrbid') \
                       and (tel_participation < low_bound \
                       or tel_participation > high_bound):
                        runs_failed.append(run)
                        n_failed_criteria += 1
                        if(logBadRuns):
                            failed_part_frac.append([low_bound, high_bound, tel_id, tel_participation])

                    if self.Config=='stereo' and tel_pattern == 30:
                        if tel_participation < low_bound or tel_participation > high_bound:
                            runs_failed.append(run)
                            n_failed_criteria += 1
                            if(logBadRuns):
                                failed_part_frac.append([low_bound, high_bound, tel_id, tel_participation])

                    if self.Config=='stereo' and tel_pattern == 62:
                        if tel_participation < HESS_Quality_Criteria["Participation_frac_hybrid"][0] \
                        or tel_participation > HESS_Quality_Criteria["Participation_frac_hybrid"][1]:
                            runs_failed.append(run)
                            n_failed_criteria += 1
                            if(logBadRuns):
                                failed_part_frac.append([low_bound, high_bound, tel_id, tel_participation])
                
                    if(logBadRuns):
                        if failed_part_frac != []:
                            failed_criteria_logs.append(
                                        """   - Participation fraction criteria [{},{}] failed for CT{} : \
                                        run Participation fraction of {} """.format(*failed_part_frac[0])
                                    )
    
    
    
            """ Check the tracking criteria of the observation"""
            tested_run = DB_track[DB_track['Run']==run]
            if(fMono):
                tested_run = tested_run[tested_run["Telescope"] == 5]
            if((fStereo==True) & (requireCT5!=True)):
                tested_run = tested_run[tested_run["Telescope"] != 5]
    
            for tel_id in np.unique(tested_run["Telescope"]):
                if tested_run.empty:
                    runs_failed.append(run)
                    n_failed_criteria += 1
                    failed_criteria_logs.append("   - Missing data {Tracking} for CT{} ".format(tel_id))
                    
                else:
                    
                    alt = tested_run[tested_run["Telescope"]==tel_id]["Alt_Dev_rms"].iloc[0]
                    az  = tested_run[tested_run["Telescope"]==tel_id]["Az_Dev_rms"].iloc[0]
                    ra  = tested_run[tested_run["Telescope"]==tel_id]["RA_Dev_mean"].iloc[0]
                    dec = tested_run[tested_run["Telescope"]==tel_id]["Dec_Dev_mean"].iloc[0]

                    
                    if (alt < HESS_Quality_Criteria["Alt_Dev_rms"][0]  or alt > HESS_Quality_Criteria["Az_Dev_rms"][1])   \
                    or (az  < HESS_Quality_Criteria["Az_Dev_rms"][0]   or az  > HESS_Quality_Criteria["Alt_Dev_rms"][1])  \
                    or (ra  < HESS_Quality_Criteria["RA_Dev_mean"][0]  or ra  > HESS_Quality_Criteria["RA_Dev_mean"][1])  \
                    or (dec < HESS_Quality_Criteria["Dec_Dev_mean"][0] or dec > HESS_Quality_Criteria["Dec_Dev_mean"][1]):
                        runs_failed.append(run)
                        n_failed_criteria += 1
                        if(logBadRuns):
                            failed_criteria_logs.append("   - Tracking criteria failed for CT{}".format(tel_id))
    

                
            """ Check the number of broken or turned of pixel in the Camera"""
            tested_run = DB_pix[DB_pix['Run']==run]
            
            if tested_run.empty:
                runs_failed.append(run)
                n_failed_criteria += 1
                failed_criteria_logs.append("   - Missing data {Broken pixel}")
                
            else:
                if(fMono):
                    tested_run = tested_run[tested_run["Telescope"] == 5]
                if((fStereo==True) & (requireCT5!=True)):
                    tested_run = tested_run[tested_run["Telescope"] != 5]
                for tel_id in np.unique(tested_run["Telescope"]):
                    tested_run_telid = tested_run[tested_run["Telescope"]==tel_id]
                    if tested_run_telid['Num_Hardware'].iloc[0] < HESS_Quality_Criteria["Num_Hardware_"+self.Config][0] \
                    or tested_run_telid['Num_Hardware'].iloc[0] > HESS_Quality_Criteria["Num_Hardware_"+self.Config][1] \
                    or tested_run_telid['Num_HV_Turned_Off'].iloc[0] < HESS_Quality_Criteria["Num_HV_Turned_Off_"+self.Config][0] \
                    or tested_run_telid['Num_HV_Turned_Off'].iloc[0] > HESS_Quality_Criteria["Num_HV_Turned_Off_"+self.Config][1]:
                        runs_failed.append(run)
                        n_failed_criteria += 1
                        if(logBadRuns):
                            failed_criteria_logs.append("   - Broken pixel criteria failed for CT{} ".format(tel_id))
    

                
            """ Check if the trigger rates are acceptable"""
            tested_run = DB_trigger[DB_trigger['Run']==run]
            if(fMono):
                tested_run = tested_run[tested_run["Telescope"] == 5]
                track_tel = DB_track.loc[DB_track['Run']==run].loc[DB_track["Telescope"]== 5]
            if((fStereo==True) & (requireCT5!=True)):
                tested_run = tested_run[tested_run["Telescope"] != 5]
                track_tel = DB_track.loc[DB_track['Run']==run].loc[DB_track["Telescope"]!= 5]
                
            if tested_run.empty:
                runs_failed.append(run)
                n_failed_criteria += 1
                failed_criteria_logs.append("   - Missing data {Trigger rate}")
                
            else:
                for tel_id in np.unique(track_tel['Telescope']):
                    if tested_run[tested_run["Telescope"]==tel_id].empty == False:
                        tested_run_telid = tested_run[tested_run["Telescope"]==tel_id]
                        if self.HessEra== 'HESS2' and (fSpectral):
                            if tested_run_telid['True_Rate_Delta_1'].iloc[0] < HESS_Quality_Criteria["True_Rate_Delta_1"][0] \
                            or tested_run_telid['True_Rate_Delta_1'].iloc[0] > HESS_Quality_Criteria["True_Rate_Delta_1"][1] \
                            or tested_run_telid['True_Rate_Delta_2'].iloc[0] < HESS_Quality_Criteria["True_Rate_Delta_2"][0] \
                            or tested_run_telid['True_Rate_Delta_2'].iloc[0] > HESS_Quality_Criteria["True_Rate_Delta_2"][1]:
                                runs_failed.append(run)
                                n_failed_criteria += 1
                                if(logBadRuns):
                                    failed_criteria_logs.append("   - Trigger rate criteria failed for CT{}".format(tel_id))
                        if self.HessEra== 'HESS1' and (fSpectral):
                            if tested_run_telid['True_Rate_Delta_1'].iloc[0] < HESS_Quality_Criteria["True_Rate_Delta_1"][0] \
                            or tested_run_telid['True_Rate_Delta_1'].iloc[0] > HESS_Quality_Criteria["True_Rate_Delta_1"][1] \
                            or tested_run_telid['True_Rate_Delta_2'].iloc[0] < HESS_Quality_Criteria["True_Rate_Delta_2"][0] \
                            or tested_run_telid['True_Rate_Delta_2'].iloc[0] > HESS_Quality_Criteria["True_Rate_Delta_2"][1]:
                                    runs_failed.append(run)
                                    n_failed_criteria += 1
                                    if(logBadRuns):
                                        failed_criteria_logs.append("   - Trigger rate criteria failed for CT{}".format(tel_id))
                                        
                    if tested_run[tested_run["Telescope"]==tel_id].empty == True: 
                        runs_failed.append(run)
                        n_failed_criteria += 1
                        failed_criteria_logs.append("   - Missing data {Trigger rate}")


            
            """ Check the state of the atmosphere during the observation"""
            if(fSpectral):
                tested_run = DB_atmo[DB_atmo['Run']==run]
                if tested_run.empty:
                    runs_failed.append(run)
                    n_failed_criteria += 1
                    failed_criteria_logs.append("   - Missing data {Transparency Coefficient}")
                    
                else:
                    if tested_run['TransparencyCoefficient_mean'].iloc[0] != 0 and \
                       (tested_run['TransparencyCoefficient_mean'].iloc[0] < HESS_Quality_Criteria['TransparencyCoefficient_mean'][0] \
                        or tested_run['TransparencyCoefficient_mean'].iloc[0] > HESS_Quality_Criteria['TransparencyCoefficient_mean'][1]):
                        runs_failed.append(run)
                        n_failed_criteria += 1
                        if(logBadRuns):
                            failed_criteria_logs.append("   - Atmospheric rate criteria failed")


            
            """Evaluate an overall observation list depending on your quality criteria selection"""       
            if logBadRuns and n_failed_criteria > 0:
                os.system("echo Run {} failed {} quality criteria: >> logBadQualityRuns.out".format(run,n_failed_criteria))
                os.system("echo ---------------------------------- >> logBadQualityRuns.out")
    
                if len(failed_criteria_logs) == 0:
                    os.system("echo MISSING DATA >> logBadQualityRuns.out")
                else:
                    for i in range(len(failed_criteria_logs)):
                        os.system("echo "+failed_criteria_logs[i]+">> logBadQualityRuns.out")
                os.system("echo   >> logBadQualityRuns.out")
    
    
        runlist_selected_runs = [x for x in filtered_runs if x not in runs_failed]

        if off_run_sel == True:
            return runlist_selected_runs
        else:
            runlist_selected = []
            for obs in runlist_selected_runs:
                tel_pattern = monitor_run_data[monitor_run_data['Run']==obs]['Telescope_Pattern'].iloc[0]
                runlist_selected.append([obs, tel_pattern])
        
            return runlist_selected



class run_matching():
    """
        Find a matching OFF run to your list of selected target observations.
        
        The matching parameters have been choosen by calculating a pearson correlation coefficient between various 
        
        parameters from the database and the background rate (application of bkg model template). If a correlation
        
        above 0.15 for all three hardware phases was found, the parameter is used for the matching. 
        
        The boundaries for the matching parameters have been aquired using the cut_values.ipynb notebook. 
        
        --------------------------------------------
        Parameters:
            - OnRuns: list, quality selected target observations
            
            - OffRuns: list, quality selected OFF observations
            
            - FitsEra: string, choose between 'HESS1', 'HESS2' and 'HESS1u'
            
            - duration: float, maximum validity range for the weighted difference in observation length between ON and OFF run
            
            - nsb: float, maximum validity range for the weighted difference in Night Sky Background between ON and OFF run
            
            - muon_eff: float, maximum validity range for the weighted difference in Muon Efficiency between ON and OFF run per telescope
            
            - transparency: float, maximum validity range for the weighted difference in the Transparency Coefficient between ON and OFF run
            
            - trig_rate: float, maximum validity range for the weighted difference in mean Trigger rate between ON and OFF run per telescope
            
            - radio: float, maximum validity range for the weighted difference in mean radiometer temperature between ON and OFF run per telescope
            
    """

    def __init__(self, OnRuns, OffRuns, FitsEra, duration, nsb, muon_eff, transparency, trig_rate, radio):
        self.OnRuns = OnRuns
        self.OffRuns = OffRuns
        self.FitsEra = FitsEra
        self.duration = duration
        self.nsb = nsb
        self.muon_eff = muon_eff
        self.transparency = transparency
        self.trig_rate = trig_rate
        self.radio = radio


    def check_if_DB_is_filled(self, run, data, column, telescope=None,):
        if telescope is None:
            return not data[data['Run'] ==  run][column].empty
        else:
            if not data[(data['Run'] == run)].empty:
                if not data[(data['Run'] == run) & (data['Telescope']==telescope)].empty:
                    return not data[(data['Run'] ==  run) & (data['Telescope'] == telescope)][column].empty
            else: 
                return False
                
            


    
    def find_bin(self, distribution, value):
        """ Helper function to estimate the correct zenith bin """
        
        for i in range(0,len(distribution)-1):
            distribution.sort()
            if value>distribution[i] and value<distribution[i+1]:
                return [distribution[i], distribution[i+1]]




    def test_run_NSB(self, on_run, off_run, quality):
        """ 
        Check if the NSB of ON and OFF run is comparable and return either 0 (not comparable) or 1 (comparable) 

        If the median is filled (post HESS1u), it will be used for this calculation, otherwise we will resort to using the mean Pedestal
        
        """
        
        number = 0
        CT_values = {}

        if quality == 1:
            for telescope in range(1, 5):
                if self.check_if_DB_is_filled( on_run, DB_pix, 'Ped_NSB_mean', telescope=telescope) \
                and self.check_if_DB_is_filled( off_run, DB_pix, 'Ped_NSB_mean', telescope=telescope):
                    CT_values[f'CT{telescope}'] = {
                        'on_med': DB_pix[(DB_pix['Run'] == on_run) & (DB_pix['Telescope'] == telescope)]['Ped_NSB_med'].iloc[0],
                        'off_med': DB_pix[(DB_pix['Run'] == off_run) & (DB_pix['Telescope'] == telescope)]['Ped_NSB_med'].iloc[0],
                        'on_mean': DB_pix[(DB_pix['Run'] == on_run) & (DB_pix['Telescope'] == telescope)]['Ped_NSB_mean'].iloc[0],
                        'off_mean': DB_pix[(DB_pix['Run'] == off_run) & (DB_pix['Telescope'] == telescope)]['Ped_NSB_mean'].iloc[0]
                    }
                else:
                    return 0
        
            # Check if median Pedestal is filled and use it for comparison
            med_pedestal = all(v == 0 for v in [d['on_med'] for d in list(CT_values.values())])
            if not med_pedestal:
                for telescope in range(1, 5):
                    CT_on = CT_values[f'CT{telescope}']['on_med']
                    CT_off = CT_values[f'CT{telescope}']['off_med']
                    if np.abs(CT_off - CT_on) / CT_on > self.nsb:
                        break
                    number = 1
            #otherwise use mean Pedestal
            else:
                for telescope in range(1, 5):
                    CT_on = CT_values[f'CT{telescope}']['on_mean']
                    CT_off = CT_values[f'CT{telescope}']['off_mean']
                    #exit the loop if the condition is not fulfilled
                    if np.abs(CT_off - CT_on) / CT_on > self.nsb:
                        break
    
                    #all conditions are fulfilled
                    number = 1

        if quality == 2:
            for telescope in range(1, 5):
                if self.check_if_DB_is_filled( on_run, DB_pix, 'Ped_NSB_mean', telescope=telescope) \
                and self.check_if_DB_is_filled( off_run, DB_pix, 'Ped_NSB_mean', telescope=telescope):
                    number = 1
    
        return number



    def test_run_transparency(self, on_run, off_run, quality):
        """ Check if the Transparency Coefficient of ON and OFF run is comparable and return either 0 (not comparable) or 1 (comparable) """
        
        number =0
        CT_values = {}
    
        # Fetch all necessary CT values
        for telescope in range(1, 5):
            if self.check_if_DB_is_filled( on_run, DB_atmo, 'TransparencyCoefficient_CT{}'.format(telescope)) \
            and self.check_if_DB_is_filled( off_run, DB_atmo, 'TransparencyCoefficient_CT{}'.format(telescope)):
                CT_values[f'CT{telescope}'] = {
                    'on': DB_atmo[(DB_atmo['Run'] == on_run)]['TransparencyCoefficient_CT' + str(telescope)].iloc[0],
                    'off': DB_atmo[(DB_atmo['Run'] == off_run)]['TransparencyCoefficient_CT' + str(telescope)].iloc[0]
                }
            else:
                return 0
        
        # Check if differences in transparency coefficients meet the specified criteria
        for telescope in range(1, 5):
            CT_on = CT_values[f'CT{telescope}']['on']
            CT_off = CT_values[f'CT{telescope}']['off']
            if np.abs(CT_off - CT_on) / CT_on > self.transparency:
                return 0
            number = 1
    
        return number


    def test_run_efficiency(self, on_run, off_run, quality):
        """ Check if the Muon Efficiency per telescope of ON and OFF run is comparable and return either 0 (not comparable) or 1 (comparable) """
        
        number = 0
        CT_values = {}

        if quality == 1:
            for telescope in range(1, 5):
                if self.check_if_DB_is_filled(on_run, DB_muon, 'Efficiency_mean', telescope=telescope) \
                and self.check_if_DB_is_filled(off_run, DB_muon, 'Efficiency_mean', telescope=telescope):
                    CT_values[f'CT{telescope}'] = {
                        'on': DB_muon[(DB_muon['Run'] == on_run) & (DB_muon['Telescope'] == telescope)]['Efficiency_mean'].iloc[0],
                        'off': DB_muon[(DB_muon['Run'] == off_run) & (DB_muon['Telescope'] == telescope)]['Efficiency_mean'].iloc[0],
                    }
                else:
                    return 0
    
            for telescope in range(1, 5):
                CT_on = CT_values[f'CT{telescope}']['on']
                CT_off = CT_values[f'CT{telescope}']['off']
                if np.abs(CT_off - CT_on) / CT_on > self.muon_eff:
                    return 0
                number = 1
            if quality == 2:
                for telescope in range(1, 5):
                    if self.check_if_DB_is_filled(on_run, DB_muon, 'Efficiency_mean', telescope=telescope) \
                    and self.check_if_DB_is_filled(off_run, DB_muon, 'Efficiency_mean', telescope=telescope):
                        number = 1
    
        return number


    def test_run_radiometer(self, on_run, off_run, quality):
        """ Check if the radiometer temperature of ON and OFF run is comparable and return either 0 (not comparable) or 1 (comparable) """
        
        number = 0
        CT_values = {}

        if on_run > 178589 and on_run < 183006:
            if quality == 1:
                for telescope in [1,2,4]:
                    if self.check_if_DB_is_filled(on_run, DB_radio, 'Temperature_mean', telescope=telescope) \
                    and self.check_if_DB_is_filled(off_run, DB_radio, 'Temperature_mean', telescope=telescope):
                        CT_values[f'CT{telescope}'] = {
                            'on': DB_radio[(DB_radio['Run'] == on_run) & (DB_radio['Telescope'] == telescope)]['Temperature_mean'].iloc[0],
                            'off': DB_radio[(DB_radio['Run'] == off_run) & (DB_radio['Telescope'] == telescope)]['Temperature_mean'].iloc[0],
                        }
                    else:
                        return 0
                for telescope in [1,2,4]:
                    CT_on = CT_values[f'CT{telescope}']['on']
                    CT_off = CT_values[f'CT{telescope}']['off']
                    if np.abs(np.abs(CT_off - CT_on) / CT_on) > self.radio:
                        return 0
                    number = 1
            if quality == 2:
                for telescope in range(1, 5):
                    if self.check_if_DB_is_filled(on_run, DB_radio, 'Temperature_mean', telescope=telescope) \
                    and self.check_if_DB_is_filled(off_run, DB_radio, 'Temperature_mean', telescope=telescope):
                        number = 1
            
        else:
            if quality == 1:
                for telescope in range(1, 5):
                    if self.check_if_DB_is_filled(on_run, DB_radio, 'Temperature_mean', telescope=telescope) \
                    and self.check_if_DB_is_filled(off_run, DB_radio, 'Temperature_mean', telescope=telescope):
                        CT_values[f'CT{telescope}'] = {
                            'on': DB_radio[(DB_radio['Run'] == on_run) & (DB_radio['Telescope'] == telescope)]['Temperature_mean'].iloc[0],
                            'off': DB_radio[(DB_radio['Run'] == off_run) & (DB_radio['Telescope'] == telescope)]['Temperature_mean'].iloc[0],
                        }
                    else:
                        return 0
        
                for telescope in range(1, 5):
                    CT_on = CT_values[f'CT{telescope}']['on']
                    CT_off = CT_values[f'CT{telescope}']['off']
                    if np.abs(np.abs(CT_off - CT_on) / CT_on) > self.radio:
                        return 0
                    number = 1
            if quality == 2:
                for telescope in range(1, 5):
                    if self.check_if_DB_is_filled(on_run, DB_radio, 'Temperature_mean', telescope=telescope) \
                    and self.check_if_DB_is_filled(off_run, DB_radio, 'Temperature_mean', telescope=telescope):
                        number = 1
    
        return number




    def test_run_trigger(self, on_run, off_run, quality):
        """ Check if the Trigger rate of ON and OFF run is comparable and return either 0 (not comparable) or 1 (comparable) """
        
        number = 0
        CT_values = {}
    
        for telescope in range(1, 5):
            if self.check_if_DB_is_filled(on_run, DB_trigger, 'True_Rate_mean', telescope=telescope) \
            and self.check_if_DB_is_filled(off_run, DB_trigger, 'True_Rate_mean', telescope=telescope):
                CT_values[f'CT{telescope}'] = {
                    'on': DB_trigger[(DB_trigger['Run'] == on_run) & (DB_trigger['Telescope'] == telescope)]['True_Rate_mean'].iloc[0],
                    'off': DB_trigger[(DB_trigger['Run'] == off_run) & (DB_trigger['Telescope'] == telescope)]['True_Rate_mean'].iloc[0],
                }
            else:
                return 0

        for telescope in range(1, 5):
            CT_on = CT_values[f'CT{telescope}']['on']
            CT_off = CT_values[f'CT{telescope}']['off']
            if np.abs(CT_off - CT_on) / CT_on > self.trig_rate:
                return 0
            number = 1
    
        return number


    def test_run_duration(self, on_run, off_run, quality):
        """ Check if the deadtime corrected observation time of ON and OFF run is comparable and return either 0 (not comparable) or 1 (comparable) """
        
        number = 0
        analysis_run = monitor_run_data[monitor_run_data['Run']==on_run] 
        analysis_off_run = monitor_run_data[(monitor_run_data['Run']==off_run)]
        
        run_livetime = analysis_run['Duration'].iloc[0] 
        if not analysis_off_run.empty:
            tested_run_livetime = analysis_off_run['Duration'].iloc[0]
            if np.abs(tested_run_livetime - run_livetime) / run_livetime < self.duration:            
                number = 1
        return number



    def test_run_zenith(self, on_run, off_run, quality):
        """
        The background model template is binned by the Zenith angle of the observations used for the template creation
        
        (see Table 3 in https://www.aanda.org/articles/aa/pdf/2019/12/aa36452-19.pdf)

        Check if the zenith angle of the OFF run is in the same bkg model bin as the ON run
        
        """
        
        number = 0
        analysis_run = DB_trigger[(DB_trigger['Run']==on_run)]
        analysis_off_run = DB_trigger[(DB_trigger['Run']==off_run)]

        run_zenith = analysis_run['Mean_Zenith'].iloc[0] 
        tested_run_zenith = analysis_off_run['Mean_Zenith'].iloc[0]

        bin_values = self.find_bin(zenith_bounds, run_zenith)
        
        if bin_values == None:
            if tested_run_zenith > 60 and tested_run_zenith < 70:
                number = 1
        else:
            #here we don't do matching by deviation but in the same background bin
            if tested_run_zenith > bin_values[0] and tested_run_zenith < bin_values[1]:
                number = 1
                
        return number


    def test_run_azimuth(self, on_run, off_run, quality):
        """
        The background model template is binned by the Azimuth angle of the observations used for the template creation
        
        (see Table 3 in https://www.aanda.org/articles/aa/pdf/2019/12/aa36452-19.pdf)

        Check if the Azimuth angle of the OFF run is in the same bkg model bin as the ON run
        
        """
        
        number = 0
        analysis_azimuth = DB_track[(DB_track['Run']==on_run)]['Az_mean'].iloc[0]
        tested_azimuth = DB_track[DB_track['Run']==off_run]['Az_mean'].iloc[0]
        
        if (-90 < analysis_azimuth <= 90):
            if (-90 < tested_azimuth <= 90) or (270 < tested_azimuth <= 360):
                number = 1
        elif (90 < analysis_azimuth < 270):
            if (90 < tested_azimuth < 270):
                number = 1
        elif (270 < analysis_azimuth <= 360):
            if (-90 < tested_azimuth < 90) or (270 < tested_azimuth <= 360):
                number = 1
                       
        return number


    def get_weights(self):
        """ 
        The influence of the respective matching parameter on the background rate is not equal. 
        
        Weights have been calculated to estimate this influence using the bkg rate computed by applying the bkg template from Mohrmann et al 2019

        These weights are stored per hardware phase, and averaged over the telescopes (only CT1-4), in this function.

        Sequence: duration, zenith, muon efficiency, NSB, transparency coefficient, trigger rate, radiometer temp

        """
        
        if self.FitsEra == 'HESS1':
            #first row are the old vals for the pearson correlation coefficient, new values are computed using the distance covariance
            #return [0.59, 0.06, 0.35, 0.35, 0.35, 0.35, 0.17, 0.17, 0.17, 0.17, 0.33, 0.33, 0.33, 0.33, 0.49, 0.49, 0.49, 0.49, 0.05, 0.05, 0.05, 0.05]
            return [0.50, 0.44, 0.49, 0.49, 0.49, 0.49, 0.40, 0.40, 0.40, 0.40, 0.36, 0.36, 0.36, 0.36, 0.45, 0.45, 0.45, 0.45, 0.38, 0.38, 0.38, 0.38]
        elif self.FitsEra == 'HESS2':
            #return [0.47, 0.29, 0.18, 0.18, 0.18, 0.18, 0.16, 0.16, 0.16, 0.16, 0.46, 0.46, 0.46, 0.46, 0.40, 0.40, 0.40, 0.40, 0.31, 0.31, 0.31, 0.31]
            return [0.49, 0.43, 0.49, 0.49, 0.49, 0.49, 0.40, 0.40, 0.40, 0.40, 0.36, 0.36, 0.36, 0.36, 0.45, 0.45, 0.45, 0.45, 0.38, 0.38, 0.38, 0.38]
        elif self.FitsEra == 'HESS1u':
            #return [0.36, 0.22, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.72, 0.72, 0.72, 0.72, 0.64, 0.64, 0.64, 0.64, 0.51, 0.51, 0.51, 0.51]
            return [0.52, 0.37, 0.21, 0.21, 0.21, 0.21, 0.23, 0.23, 0.23, 0.23, 0.60, 0.60, 0.60, 0.60, 0.59, 0.59, 0.59, 0.59, 0.43, 0.43, 0.43, 0.43]
        else:
            raise ValueError("Invalid era")

    

    def calculate_total_diff(self, diffs):
        """ Calculate the difference in the matching parameter between ON and OFF run including weights """
        
        weights = self.get_weights()
        total_diff = sum(weight * diff for weight, diff in zip(weights, diffs))
        return total_diff

    

    def compute_difference(self, run, tested_run, data, column, telescope=None, alternative_column=None):
        """ Compute the difference in On and OFF run for a respective matching parameter and telescope """
        
        if telescope is None:
            diff = np.abs(data[data['Run']==run][column].iloc[0] \
                          - data[data['Run']==tested_run][column].iloc[0]) \
                          / np.abs(data[data['Run']==run][column].iloc[0])
        else:
            if all(v == 0 for v in data[data['Run']==run][column].to_numpy()) == False:
                diff = np.abs(data[(data['Run']==run) & (data['Telescope']==telescope)][column].iloc[0] \
                              - data[(data['Run']==tested_run) & (data['Telescope']==telescope)][column].iloc[0]) \
                              / np.abs(data[(data['Run']==run) & (data['Telescope']==telescope)][column].iloc[0])
            else:
                diff = np.abs(data[(data['Run']==run) & (data['Telescope']==telescope)][alternative_column].iloc[0] \
                              - data[(data['Run']==tested_run) & (data['Telescope']==telescope)][alternative_column].iloc[0]) \
                              / np.abs(data[(data['Run']==run) & (data['Telescope']==telescope)][alternative_column].iloc[0])
        return diff

    

    def get_frac_run_deviation(self, run, tested_run):
        """ Compute the difference between ON and OFF run and estimate the fractional run deviation off these runs """

        all_diffs = []
        duration_diff = self.compute_difference(run, tested_run, monitor_run_data, 'Duration')
        zenith_diff = self.compute_difference(run, tested_run, DB_trigger, 'Mean_Zenith')                                

        NSB_diffs = {}
        efficiency_diffs = {}
        transparency_diffs = {}
        trigger_diffs = {}
        radio_diffs = {}
        
        for telescope_index in range(1, 5):
            NSB_diffs[f'NSB_diff_CT{telescope_index}'] \
            = self.compute_difference(run, tested_run, DB_pix, 'Ped_NSB_med', \
                                      telescope=telescope_index, alternative_column='Ped_NSB_mean')
            efficiency_diffs[f'efficiency_diff_CT{telescope_index}'] \
            = self.compute_difference(run, tested_run, DB_muon, 'Efficiency_mean', telescope=telescope_index)
            transparency_diffs[f'transparency_diff_CT{telescope_index}'] \
            = self.compute_difference(run, tested_run, DB_atmo, 'TransparencyCoefficient_CT{}'.format(telescope_index))
            trigger_diffs[f'trigger_diff_CT{telescope_index}'] \
            = self.compute_difference(run, tested_run, DB_trigger, 'True_Rate_mean', telescope=telescope_index)
            if run > 178589 and run < 183006 and telescope_index != 3:
                radio_diffs[f'radio_diff_CT{telescope_index}'] \
                = self.compute_difference(run, tested_run, DB_radio, 'Temperature_mean', telescope=telescope_index)
            elif run < 178589 or run > 183006: 
                radio_diffs[f'radio_diff_CT{telescope_index}'] \
                = self.compute_difference(run, tested_run, DB_radio, 'Temperature_mean', telescope=telescope_index)

        all_diffs.append(duration_diff)
        all_diffs.append(zenith_diff)
        all_diffs.extend(efficiency_diffs.values())
        all_diffs.extend(NSB_diffs.values())
        all_diffs.extend(transparency_diffs.values())
        all_diffs.extend(trigger_diffs.values())
        all_diffs.extend(radio_diffs.values())

        total_diff = self.calculate_total_diff(all_diffs)
        return total_diff
        



    def matching_operation(self):
        """ 
        For every On run in your run list, test all OFF runs and find the one with the smallest fractional run deviation.

        For every On run, each run in the OFF run list is tested and the one with the minimal deviation is identified.

        If a OFF run was identified as best matching for an ON run, it is removed from the testing pool for subsequent ON runs.
        
        Returns a list containing all run pairs found in this operation
        """
        
        run_pairs = []
        
        # Define the parameters that should be tested for the matching
        test_functions = [
            self.test_run_duration,
            self.test_run_zenith,
            self.test_run_azimuth,
            self.test_run_transparency,
            self.test_run_efficiency,
            self.test_run_NSB,
            self.test_run_trigger,
            self.test_run_radiometer
        ]
        
        for run in self.OnRuns:
            fulfilled_criteria =[]
            source_name = monitor_run_data[monitor_run_data['Run']==run]['Target_Name'].iloc[0]
            print('currently working on observation run:', run)
            
            for tested_run in self.OffRuns:
                # print('currently working on observation', tested_run, 'as off run')
                tested_name = monitor_run_data[monitor_run_data['Run']==tested_run]['Target_Name'].iloc[0]

                # check if the OFF run is not on the source and in the same muon phase as the ON run
                if tested_run not in self.OnRuns and tested_name != source_name:
                    for i in range(0, len(muon_phases2)):
                        if (muon_phases2[i] <= run <= muon_phases2[i+1]) and (muon_phases2[i] <= tested_run <= muon_phases2[i+1]):
                            # Run all test functions sequentially, the quality indicates which functions should be used for the matching
                            for test_function in test_functions:
                                number = test_function(run, tested_run, 1)
                                if number != 1:
                                    break
                            else:
                                # if our OFF run passed all the criteria, calculate the fractional run deviation
                                total_diff = self.get_frac_run_deviation(run, tested_run)
                                fulfilled_criteria.append([tested_run, total_diff])
                    
            if fulfilled_criteria != []:
                minimum_deviation = min(fulfilled_criteria, key=lambda p:p[1])[0]
                # print('Run', minimum_deviation, 'will be used as Off run for run', run)
                idx = np.where(np.array(fulfilled_criteria)[:,0] == minimum_deviation)[0][0]
                run_pairs.append([run, minimum_deviation, fulfilled_criteria[idx][1]])
                self.OffRuns.remove(minimum_deviation)    
                
            else:
                print('no run fulfilling all criteria can be found for run:', run, ', test only necassary ones')
                for tested_run in self.OffRuns:
                    tested_name = monitor_run_data[monitor_run_data['Run']==tested_run]['Target_Name'].iloc[0]
    
                    # check if the OFF run is not on the source and in the same muon phase as the ON run
                    if tested_run not in self.OnRuns and tested_name != source_name:
                        for i in range(0, len(muon_phases2)):
                            if (muon_phases2[i] <= run <= muon_phases2[i+1]) and (muon_phases2[i] <= tested_run <= muon_phases2[i+1]):
                                # Run all test functions sequentially, quality =2 indicates that only some functions are tested
                                for test_function in test_functions:
                                    number = test_function(run, tested_run, 2)
                                    if number != 1:
                                        break
                                else:
                                    # if our OFF run passed all the criteria, calculate the fractional run deviation
                                    total_diff = self.get_frac_run_deviation(run, tested_run)
                                    print(total_diff)
                                    fulfilled_criteria.append([tested_run, total_diff])
                        
                if fulfilled_criteria != []:
                    minimum_deviation = min(fulfilled_criteria, key=lambda p:p[1])[0]
    
                    idx = np.where(np.array(fulfilled_criteria)[:,0] == minimum_deviation)[0][0]
                    run_pairs.append([run, minimum_deviation, fulfilled_criteria[idx][1]])
                    self.OffRuns.remove(minimum_deviation)    
                    
                else:
                    print('no match found for run:', run, '- remove the run')
        return run_pairs    



    def matching_operation_systematics(self):
        """ 

        For every On run, each run in the OFF run list is tested and a fractional run deviation is calculated.

        If a OFF run was identified as best matching for an ON run, it is kept the testing pool for subsequent ON runs.
        
        Returns a list containing all run pairs, not just the best fitting ones, found in this operation
        """
        
        run_pairs = []
        
        # Define the parameters that should be tested for the matching
        test_functions = [
            self.test_run_duration,
            self.test_run_zenith,
            self.test_run_azimuth,
            self.test_run_transparency,
            self.test_run_efficiency,
            self.test_run_NSB,
            self.test_run_trigger,
            self.test_run_radiometer
        ]
        
        for run in self.OnRuns:
            fulfilled_criteria =[]
            source_name = monitor_run_data[monitor_run_data['Run']==run]['Target_Name'].iloc[0]
            print('currently working on observation run:', run)
            
            for tested_run in self.OffRuns:
                tested_name = monitor_run_data[monitor_run_data['Run']==tested_run]['Target_Name'].iloc[0]

                # check if the OFF run is not on the source and in the same muon phase as the ON run
                if tested_run not in self.OnRuns and tested_name != source_name:
                    for i in range(0, len(muon_phases2)):
                        if (muon_phases2[i] <= run <= muon_phases2[i+1]) and (muon_phases2[i] <= tested_run <= muon_phases2[i+1]):
                            # print('currently working on observation', tested_run, 'as off run')
                            # Run all test functions sequentially
                            for test_function in test_functions:
                                number = test_function(run, tested_run, 1)
                                if number != 1:
                                    break
                            else:
                                # if our OFF run passed all the criteria, calculate the fractional run deviation
                                total_diff = self.get_frac_run_deviation(run, tested_run)
                                fulfilled_criteria.append([run, tested_run, total_diff])
                    
            if fulfilled_criteria != []:
                run_pairs.extend(fulfilled_criteria)
        return run_pairs
 












































        