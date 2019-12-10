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
        
        with open(os.path.join(self.saveDirectory, filename), 'wb') as fp:
            pickle.dump(savedict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
    def loadFromPickle(self, filepath):
        with open(filepath, 'rb') as fp:
            loaddict = pickle.load(fp)
        
        for field in loaddict:
            self.__dict__[field] = loaddict[field]
    
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
    
    
    def buildBehaviorDataframe(self, startDate=None, endDate=None):
        if self.behavior_sessions is None:
            self.getBehaviorSessionsForMouse()
        
    
        #add common rig name
        self.behavior_sessions['rig'] = self.behavior_sessions.apply(lambda row: 
                pd.read_sql('select * from equipment where id = {}'.format(row['equipment_id']), self.con)['name'], axis=1)
    
        #add pkl file paths
        self.behavior_sessions['pklfile'] = self.behavior_sessions.apply(lambda row: 
                self.getPicklePath(row['storage_directory']), axis=1)
    
    
        #pick out dates to analyze: it takes a bit of time to pull this data from the network, so limiting your
        #dates is helpful when possible. Right now I'm pulling from 'daysBeforeHandoff' to end
        handoff = self.behavior_sessions[self.behavior_sessions['rig'].str.contains('NP')].iloc[-1]['created_at']
        if startDate is None:
            startDate = handoff - pd.DateOffset(days=self.daysBeforeHandoff)
        
        if endDate is None:
            endDate = self.behavior_sessions[self.behavior_sessions['rig'].str.contains('NP')].iloc[0]['created_at']
        
        toAnalyze = self.behavior_sessions[(self.behavior_sessions['created_at']>=startDate)&(self.behavior_sessions['created_at']<endDate)]
        toAnalyze['trials'] = toAnalyze.apply(lambda row: self.getTrialsDF(row['pklfile']), axis=1) #this trials object has all the info you need about the session
        toAnalyze['stage'] = toAnalyze.apply(lambda row: row['trials']['stage'][0], axis=1) #add the training stage to the dataframe
        toAnalyze = toAnalyze.loc[toAnalyze['stage'].notnull()] #filter out the passive pickle files that get added during recordings
        toAnalyze['session_datetime'] = toAnalyze.apply(lambda row: row['trials']['startdatetime'][0], axis=1)
        
        #Add some useful columns to dataframe
        toAnalyze['session_datetime_local'] = toAnalyze.apply(lambda row: pd.to_datetime(row['trials']['startdatetime'][0]), axis=1)
        toAnalyze['session_datetime_utc'] = toAnalyze.apply(lambda row: pd.to_datetime(row['trials']['startdatetime'][0], utc=True), axis=1)
        toAnalyze['cumulative_rewards'] = toAnalyze.apply(lambda row: row['trials']['cumulative_reward_number'].max(), axis=1)
        toAnalyze['timeFromLastSession'] = toAnalyze['session_datetime_utc'].diff(periods=-1).astype('timedelta64[s]')/3600
            
        self.beh_df = toAnalyze
    
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
        ax.set_xlim([18,28])
        ax.set_xlabel('Hours since last session')
        ax.set_ylabel('Number of rewards earned')
        
#        
#        saveDir = r"C:\Users\svc_ccg\Desktop\Data\NP mouse behavior dfs"
#        self.beh_df.to_pickle(os.path.join(saveDir, str(self.mouse_id)+'_behavior.pkl'))
#        
#    
#    
























