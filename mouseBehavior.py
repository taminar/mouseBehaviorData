# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:33:33 2019

@author: svc_ccg
"""

from psycopg2 import connect, extras
import numpy as np
import pandas as pd
import glob,os
from matplotlib import pyplot as plt
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe
import pickle
import scipy
import labTracksQuery as ltq


class mouseBehaviorData():
    
    def __init__(self, mouse_id=None, daysBeforeHandoff=28, saveDirectory=None):
        
        self.mouse_id = mouse_id
        self.daysBeforeHandoff = daysBeforeHandoff
        self.saveDirectory=saveDirectory
        self.queryLims()
        
        self.behavior_sessions = None
        
    def saveToPickle(self, saveDir=None, fileBase=None):
        if saveDir is None:
            saveDir = self.saveDirectory
        if fileBase is None:
            fileBase = str(self.mouse_id)
        filename = fileBase + '_behaviorHistory.pkl'
        savedict = {}
        for field in self.__dict__.keys():
            if field in ['cursor', 'con']:
                continue
            savedict[field] = self.__dict__[field]            
        
        with open(os.path.join(saveDir, filename), 'wb') as fp:
            pickle.dump(savedict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
    def loadFromPickle(self, filepath):
        with open(filepath, 'rb') as fp:
            loaddict = pickle.load(fp)
        
        for field in loaddict:
            self.__dict__[field] = loaddict[field]
    
    def save_dataframe_separately(self, saveDir=None, fileBase=None):
        if saveDir is None:
            saveDir = self.saveDirectory
        if fileBase is None:
            fileBase = str(self.mouse_id) + '_dataframe'
        
        filename = fileBase + '_behaviorHistory.pkl'
        self.beh_df.to_pickle(os.path.join(saveDir, filename))
        
    
    def queryLims(self):
        
        self.con = connect(
            dbname='lims2',
            user='limsreader',
            host='limsdb2',
            password='limsro',
            port=5432,
        )
        
        self.con.set_session(
            readonly=True, 
            autocommit=True,
        )
        
        self.cursor = self.con.cursor(
            cursor_factory=extras.RealDictCursor,
        )
    
    def getLimsID(self):
        query_template = '''
        SELECT * 
        FROM donors d
        WHERE d.external_donor_name = '{}'
        '''
    
        self.cursor.execute(query_template.format(self.mouse_id))
        lims_id = self.cursor.fetchone()['id']
        return lims_id

    
    def getBehaviorSessionsFromLimsID(self):
        query_template = '''
        SELECT *
        FROM behavior_sessions bs
        WHERE bs.donor_id = {}
        '''
    
        self.cursor.execute(query_template.format(self.lims_id))
        self.behavior_sessions = pd.DataFrame(self.cursor.fetchall())
        
    
    def getBehaviorSessionsForMouse(self):
        self.lims_id = self.getLimsID()
        self.getBehaviorSessionsFromLimsID()
    
    def getPicklePath(self, storage_directory):
        if storage_directory[1] != '/':
            storage_directory = '/' + storage_directory
        pp = glob.glob(os.path.join(storage_directory, '*.pkl'))
        if len(pp)>0:
            return pp[0]
        else:
            return None
        
    def getTrialsDF(self, pklpath):
        p = pd.read_pickle(pklpath)
        if 'behavior' in p['items']:
            core_data = data_to_change_detection_core(p)
            trials = create_extended_dataframe(
                    trials=core_data['trials'],
                    metadata=core_data['metadata'],
                    licks=core_data['licks'],
                    time=core_data['time'])
        else:
            print('Found non-behavior pickle file: ' + pklpath)
            trials = pd.DataFrame.from_dict({'stage':[None]})
            
        return trials
    
    def getRunning(self, pklpath):
        p = pd.read_pickle(pklpath)
        if 'behavior' in p['items']:
            core_data = data_to_change_detection_core(p)
            rtime = core_data['running']['time']
            rspeed = core_data['running']['speed']
        else:
            rtime = np.zeros(5).astype(float)
            rspeed = np.zeros(5).astype(float)
        
        return [rtime, rspeed]
    
    def calculate_dprime_engaged(trials, reward_rate_thresh = 1):
        
        engagedTrials = (trials['reward_rate'] >= 1) & (trials['response_type'] != 'aborted')
        engagedDF = trials.loc[engagedTrials]
        hits = np.sum(engagedDF['response_type'] == 'HIT')
        misses = np.sum(engagedDF['response_type'] == 'MISS')
        fas = np.sum(engagedDF['response_type'] == 'FA')
        crs = np.sum(engagedDF['response_type'] == 'CR')
        def trial_number_limit(p, N):
            if N == 0:
                return np.nan
            if not pd.isnull(p):
                p = np.max((p, 1. / (2 * N)))
                p = np.min((p, 1 - 1. / (2 * N)))
            return p
    
        hitRate = trial_number_limit(hits/float(hits + misses), hits + misses)
        faRate = trial_number_limit(fas/float(fas+crs), fas+crs)
    
        z = [scipy.stats.norm.ppf(r) for r in (hitRate,faRate)]
    
        return z[0] - z[1]
    
#    def calculate_dprime_engaged(self, trials, reward_rate_thresh = 1):
#        
#        engagedTrials = trials['reward_rate'] >= reward_rate_thresh
#        engagedDF = trials.loc[engagedTrials]
#        
#        hits = np.sum(engagedDF['response_type'] == 'HIT')
#        misses = np.sum(engagedDF['response_type'] == 'MISS')
#        fas = np.sum(engagedDF['response_type'] == 'FA')
#        crs = np.sum(engagedDF['response_type'] == 'CR')
#        
#        engagedTrialHitRate = hits/float(hits+misses)
#        engagedTrialFARate = fas/float(fas+crs)
#        
#        z = [scipy.stats.norm.ppf(r) for r in (engagedTrialHitRate,engagedTrialFARate)]
#        
#        return z[0] - z[1]
    
    def calculate_response_rate_engaged(self, trials, responseType='HIT', reward_rate_thresh = 1):
        engagedTrials = trials['reward_rate'] >= reward_rate_thresh
        engagedDF = trials.loc[engagedTrials]
        
        responses = np.sum(engagedDF['response_type'] == responseType)
        
        hits = np.sum(engagedDF['response_type'] == 'HIT')
        misses = np.sum(engagedDF['response_type'] == 'MISS')
        fas = np.sum(engagedDF['response_type'] == 'FA')
        crs = np.sum(engagedDF['response_type'] == 'CR')
        
        if responseType=='HIT' or responseType=='MISS':
            denom = hits + misses
        elif responseType=='FA' or responseType=='CR':
            denom = fas + crs
        else:
            denom = len(engagedDF)
        
        return responses/float(denom)
    
    def calculate_total_earned_rewards(self, trials):
        total_rewards = trials['cumulative_reward_number'].max()
        free_rewards = trials[trials['response_type']!='EARLY_RESPONSE']['auto_rewarded'].sum()
        
        return total_rewards - free_rewards
        
    
    def get_rig_name(self, row):
        rig_name = None
        
        #check if we can get rig from trials df
        if 'trials' in row:
            if 'rig_id' in row['trials']:
                rig_name = row['trials']['rig_id'][0]
        
        #otherwise get it from LIMs equip id
        else:
            equipID = row['equipment_id']
            if not equipID is None and not np.isnan(equipID):
                rig_name = pd.read_sql('select * from equipment where id = {}'.format(row['equipment_id']), self.con)['name']
        
        if rig_name is None:
            rig_name = 'unknown'
        
        return rig_name
    
    def get_mouse_metadata(self):
        mid = str(self.mouse_id)
        #get labtracks info
        q = ltq.get_labtracks_animals_entry(mid)
        params_to_extract = ['Maternal_Index', 'Paternal_Index', 'wean_date', 'birth_date']
        for p in params_to_extract:
            self.__dict__[p] = q[p]
            
        #get baseline weight
        self.baseline_weight = float(pd.read_sql('select * from donors where external_donor_name = \'%s\'' % mid, self.con)['baseline_weight_g'])
    
    def buildBehaviorDataframe(self, startDate=None, endDate=None, all_sessions=False, overwrite_behdf=False):
        if self.behavior_sessions is None:
            self.getBehaviorSessionsForMouse()
        
        #add common rig name
        self.behavior_sessions['rig'] = self.behavior_sessions.apply(lambda row: 
                self.get_rig_name(row), axis=1)
    
        #add pkl file paths
        self.behavior_sessions['pklfile'] = self.behavior_sessions.apply(lambda row: 
                self.getPicklePath(row['storage_directory']), axis=1)
    
    
        #pick out dates to analyze: it takes a bit of time to pull this data from the network, so limiting your
        #dates is helpful when possible. Right now I'm pulling from 'daysBeforeHandoff' to end
        if not hasattr(self, 'beh_df') or overwrite_behdf:
            if all_sessions:
                startDate = '1900'
                endDate = '2100'
            else:
                handoff = self.behavior_sessions[self.behavior_sessions['rig'].str.contains('NP')].iloc[-1]['created_at']
                if startDate is None:
                    startDate = handoff - pd.DateOffset(days=self.daysBeforeHandoff)
                
                if endDate is None:
                    endDate = self.behavior_sessions[self.behavior_sessions['rig'].str.contains('NP')].iloc[0]['created_at']
            
            
            toAnalyze = self.behavior_sessions[(self.behavior_sessions['created_at']>=startDate)&(self.behavior_sessions['created_at']<endDate)]
            toAnalyze['trials'] = toAnalyze.apply(lambda row: self.getTrialsDF(row['pklfile']), axis=1) #this trials object has all the info you need about the session
        
        else:
            toAnalyze = self.beh_df
        
        toAnalyze['stage'] = toAnalyze.apply(lambda row: row['trials']['stage'][0], axis=1) #add the training stage to the dataframe
        toAnalyze = toAnalyze.loc[toAnalyze['stage'].notnull()] #filtem.beh_df.apply(lambda row: (row['session_datetime_local'].tz_localize(None) - m.birth_date).days, axis=1)r out the passive pickle files that get added during recordings
        toAnalyze['running'] = toAnalyze.apply(lambda row: self.getRunning(row['pklfile']), axis=1)
        
        #Add some useful columns to dataframe: These don't require the pickle and maybe should be moved to separate function
        toAnalyze['session_datetime'] = toAnalyze.apply(lambda row: row['trials']['startdatetime'][0], axis=1)
        toAnalyze['session_datetime_local'] = toAnalyze.apply(lambda row: pd.to_datetime(row['trials']['startdatetime'][0]), axis=1)
        toAnalyze['session_datetime_utc'] = toAnalyze.apply(lambda row: pd.to_datetime(row['trials']['startdatetime'][0], utc=True), axis=1)
        toAnalyze = toAnalyze.sort_values('session_datetime_utc', ascending=False) #sort dataframe by date
        toAnalyze['timeFromLastSession'] = toAnalyze['session_datetime_utc'].diff(periods=-1).astype('timedelta64[s]')/3600
        toAnalyze['session_day_of_week'] = toAnalyze.apply(lambda row: row['session_datetime_local'].dayofweek, axis=1)
        
        #fill in rig names missing from lims
        toAnalyze['rig'] = toAnalyze.apply(lambda row: self.get_rig_name(row), axis=1)
        
        self.beh_df = toAnalyze
        self.calculate_behavior_metrics()
        self.add_metadata_to_dataframe()
        self.add_weight_and_water_history()
        
    def calculate_behavior_metrics(self):
        #Add behavior metrics without reconstituting from original pickle files (ie calling buildBehaviorDataframe)
        self.beh_df['hit_rate_engaged'] = self.beh_df.apply(lambda row: self.calculate_response_rate_engaged(row['trials'], responseType='HIT'), axis=1) 
        self.beh_df['FA_rate_engaged'] = self.beh_df.apply(lambda row: self.calculate_response_rate_engaged(row['trials'], responseType='FA'), axis=1) 
        self.beh_df['abort_rate_engaged'] = self.beh_df.apply(lambda row: self.calculate_response_rate_engaged(row['trials'], responseType='EARLY_RESPONSE'), axis=1) 
        self.beh_df['engaged_dprime'] = self.beh_df.apply(lambda row: self.calculate_dprime_engaged(row['trials']), axis=1) 
        self.beh_df['earned_rewards'] = self.beh_df.apply(lambda row: self.calculate_total_earned_rewards(row['trials']), axis=1)
        self.beh_df['total_rewards'] = self.beh_df.apply(lambda row: row['trials']['cumulative_reward_number'].max(), axis=1)
        
    def add_metadata_to_dataframe(self):
        if not hasattr(self, 'Maternal_Index'):
            self.get_mouse_metadata()
            
        self.beh_df['Maternal_Index'] = int(self.Maternal_Index)
        self.beh_df['Paternal_Index'] = int(self.Paternal_Index)
        self.beh_df['age'] = self.beh_df.apply(lambda row: (row['session_datetime_local'].tz_localize(None) - self.birth_date).days, axis=1)
        self.beh_df['baseline_weight'] = self.baseline_weight
    
    
    def add_weight_and_water_history(self):
        import mysql.connector
        import re
        import datetime
        mysql_conn = mysql.connector.connect(host='aibspi2', database='mpe', user='read', password='read')
        mysql_conn.autocommit = True
        mysql_cursor = mysql_conn.cursor(dictionary=True)
        
        not_clients = 'client_address not like "%desktop%" ' \
                          'and client_address not like "%ariell%"' \
                          'and client_address not like "%ben%"' \
                          'and client_address not like "%test%"'
        
        start_index=0

        search_string = 'WS_ml'
        program = 'mouse_director'
        limit_str = 'limit {num_matches}'.format(num_matches=1000)
        
        sql = 'select * from log_server where rowID > {start_index} and {not_clients} and logname like "%{program}%" and message like "%{mouseID}%"\
        and message like "%{search_string}%" order by rowID asc {limit_str}'.format(start_index=start_index,\
                            not_clients=not_clients, program=program, mouseID=self.mouse_id, search_string=search_string, limit_str=limit_str)
        
        
        mysql_cursor.execute( sql )
        sessions = mysql_cursor.fetchall()
        
        
        datadict = {'Wt_g': [],
                    'WE_ml': [],
                    'WS_ml': [],
                    'weight_datetime': []}
        
        for key in datadict:
            self.beh_df[key] = ""
            
        if len(sessions)==0:
            print('Did not find any water entries for this mouse')
        
        else:
            
            def parse_key_value(string, key, dtype=None, limiter=','):
                start_ind = re.split(key+limiter, string)
                value = re.split(limiter, start_ind[1])
                if dtype is not None:
                    return dtype(value[0])
                else:
                    return value[0]
            
            
            
            for sess in sessions:
#                datadict['datetime'].append(sess['datetime'])
#                message = sess['message']
#                for key in ['Wt_g', 'WE_ml', 'WS_ml']:
#                    datadict[key].append(parse_key_value(message, key))   
                message = sess['message']    
                weight_datetime = sess['datetime']
                for ind, session_date in enumerate(self.beh_df['session_datetime_local']):
                    if datetime.datetime.date(session_date) == datetime.datetime.date(weight_datetime):
                        for key in ['Wt_g', 'WE_ml', 'WS_ml']:
                            val = parse_key_value(message, ' ' + key)
                            self.beh_df[key].iloc[ind] = float(val) if val != '' else np.nan
                        self.beh_df['weight_datetime'].iloc[ind] = weight_datetime
        
            
        
    def standardizeDatatypes(self):
        
        self.beh_df['ophys_session_id'] = pd.to_numeric(self.beh_df['ophys_session_id'])
        
        #replace blanks with nans
        for col in ['Wt_g', 'WE_ml', 'WS_ml']:
            self.beh_df[col].replace(r'', np.nan, inplace=True)
            self.beh_df[col] = self.beh_df[col].astype('float')
    
        
    def plotResponseTypeProportions(self):
        #plot proportion of trials that were aborts, hits and false alarms
        for ir, row in self.beh_df.iterrows():
            fig, ax = plt.subplots()
            fig.suptitle(row['session_datetime_local'])
            [ax.plot(np.convolve(np.ones(50), row['trials']['response_type']==r, 'same')/50) for r in ['EARLY_RESPONSE', 'HIT', 'FA', 'MISS', 'CR']]
            ax.set_xlabel('trial num')
            ax.set_ylabel('proportion trials')
            ax.legend(['EARLY_RESPONSE', 'HIT', 'FA', 'MISS', 'CR'])
            
                
    def plotSessionHistory(self):
    
        def getColorAlphaFill(row):
            a = 1.0
            f = 'full'
            if 'NP' not in row['rig']:
                c = 'k'
            elif 'TRAINING' in row['stage']:
                c = 'm'
            else:
                c = 'g'
            
            if 'low_volume' in row['stage']:
                a = 0.3
                f = 'none'
            return c,a,f
        
        fig, ax = plt.subplots()
        for ir, row in self.beh_df.iterrows():  
            num_rewards = row['trials']['cumulative_reward_number'].max()
            c,a,f = getColorAlphaFill(row)
            ax.plot(row['session_datetime_local'], num_rewards, c+'o', alpha=a, fillstyle=f, mew=3)
        fig.suptitle(self.mouse_id)
        ax.set_xlabel('Sessions')
        ax.set_ylabel('num rewards')
        ax.set_xticks([row['session_datetime_local'] for _,row in self.beh_df.iterrows()])
        ax.set_xticklabels([row['session_datetime_local'].date() for _,row in self.beh_df.iterrows()])
        plt.xticks(rotation=90)
        
        
    def plotPerformanceByTimeFromLastSession(self):
        
        fig, ax = plt.subplots()
        ax.plot(self.beh_df['timeFromLastSession'], self.beh_df['cumulative_rewards'], 'o')
#        ax.set_xlim([18,28])
        ax.set_xlabel('Hours since last session')
        ax.set_ylabel('Number of rewards earned')
           


        











