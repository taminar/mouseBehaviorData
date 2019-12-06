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

con = connect(
    dbname='lims2',
    user='limsreader',
    host='limsdb2',
    password='limsro',
    port=5432,
)

con.set_session(
    readonly=True, 
    autocommit=True,
)

cursor = con.cursor(
    cursor_factory=extras.RealDictCursor,
)

def getLimsID(mouse_id):
    query_template = '''
    SELECT * 
    FROM donors d
    WHERE d.external_donor_name = '{}'
    '''

    cursor.execute(query_template.format(mouse_id))
    lims_id = cursor.fetchone()['id']
    return lims_id

def getBehaviorSessionsFromLimsID(lims_id):
    query_template = '''
    SELECT *
    FROM behavior_sessions bs
    WHERE bs.donor_id = {}
    '''

    cursor.execute(query_template.format(lims_id))
    behavior_sessions = pd.DataFrame(cursor.fetchall())
    return behavior_sessions

def getBehaviorSessionsForMouse(mouse_id):
    lims_id = getLimsID(mouse_id)
    return getBehaviorSessionsFromLimsID(lims_id)

def getPicklePath(storage_directory):
    if storage_directory[1] != '/':
        storage_directory = '/' + storage_directory
    pp = glob.glob(os.path.join(storage_directory, '*.pkl'))
    if len(pp)>0:
        return pp[0]
    else:
        return None
    
def getTrialsDF(pklpath):
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


mouse_id = '477120'
beh_sessions = getBehaviorSessionsForMouse(mouse_id)

#add common rig name
beh_sessions['rig'] = beh_sessions.apply(lambda row: 
        pd.read_sql('select * from equipment where id = {}'.format(row['equipment_id']), con)['name'], axis=1)

#add pkl file paths
beh_sessions['pklfile'] = beh_sessions.apply(lambda row: 
        getPicklePath(row['storage_directory']), axis=1)


#pick out dates to analyze: it takes a bit of time to pull this data from the network, so limiting your
#dates is helpful when possible. Right now I'm pulling from 'daysBeforeHandoff' to end
daysBeforeHandoff = 28

handoff = beh_sessions[beh_sessions['rig'].str.contains('NP')].iloc[-1]['created_at']
endDate = beh_sessions[beh_sessions['rig'].str.contains('NP')].iloc[0]['created_at']
startDate = handoff - pd.DateOffset(days=daysBeforeHandoff)

toAnalyze = beh_sessions[(beh_sessions['created_at']>=startDate)&(beh_sessions['created_at']<endDate)]
toAnalyze['trials'] = toAnalyze.apply(lambda row: getTrialsDF(row['pklfile']), axis=1) #this trials object has all the info you need about the session
toAnalyze['stage'] = toAnalyze.apply(lambda row: row['trials']['stage'][0], axis=1) #add the training stage to the dataframe
toAnalyze = toAnalyze.loc[toAnalyze['stage'].notnull()] #filter out the passive pickle files that get added during recordings
toAnalyze['session_datetime'] = toAnalyze.apply(lambda row: row['trials']['startdatetime'][0], axis=1)

#plot proportion of trials that were aborts, hits and false alarms
for ir, row in toAnalyze.iterrows():
    fig, ax = plt.subplots()
    fig.suptitle(row['created_at'])
    [ax.plot(np.convolve(np.ones(50), row['trials']['response_type']==r, 'same')/50) for r in ['EARLY_RESPONSE', 'HIT', 'FA', 'MISS', 'CR']]
    ax.set_xlabel('trial num')
    ax.set_ylabel('proportion trials')
    ax.legend(['EARLY_RESPONSE', 'HIT', 'FA', 'MISS', 'CR'])
    

toAnalyze['session_datetime_local'] = toAnalyze.apply(lambda row: pd.to_datetime(row['trials']['startdatetime'][0]), axis=1)
toAnalyze['session_datetime_utc'] = toAnalyze.apply(lambda row: pd.to_datetime(row['trials']['startdatetime'][0], utc=True), axis=1)
toAnalyze['cumulative_rewards'] = toAnalyze.apply(lambda row: row['trials']['cumulative_reward_number'].max(), axis=1)
toAnalyze['timeFromLastSession'] = toAnalyze['session_datetime_utc'].diff(periods=-1).astype('timedelta64[s]')/3600
fig, ax = plt.subplots()
ax.plot(toAnalyze['timeFromLastSession'], toAnalyze['cumulative_rewards'], 'o')
ax.set_xlim([18,28])
ax.set_xlabel('Hours since last session')
ax.set_ylabel('Number of rewards earned')


saveDir = r"C:\Users\svc_ccg\Desktop\Data\NP mouse behavior dfs"
toAnalyze.to_pickle(os.path.join(saveDir, str(mouse_id)+'_behavior.pkl'))






























