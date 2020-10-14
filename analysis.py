import numpy as np
import glob, os
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def checkforMouseBehaviorObject(directory, mouseID):

	print('checking {} for mouse object pkl'.format(directory))
	pkllist = [f for f in os.listdir(directory) if '.pkl' in f]
	mouseID = str(mouseID)
	pklpath = None
	for pkl in pkllist:
		if pkl[:6]==mouseID:
			pklpath = os.path.join(directory, pkl)
			print('Found existing mouseBehaviorObject'
			' for mouse {}'.format(mouseID))
	if pklpath is None:
		print('Did not find existing' 
			'mouseBehaviorObject for mouse {}'.format(mouseID))
	return pklpath


def filter_sessions_by_stage(beh_df, filter_string='HAB'):

	if isinstance(filter_string, list):
		filtered_df = beh_df.copy(deep=True)
		for fs in filter_string:
			filtered_df = filter_sessions_by_stage(filtered_df, fs)
	else:
		filtered = beh_df['stage'].str.contains(filter_string)
		filtered_df = beh_df.loc[filtered]
	
	return filtered_df


def filter_out_pretest(beh_df):

	filtered = beh_df['stage'].str.contains('pretest')

	return beh_df.loc[~filtered]


def filter_sessions_by_rig(beh_df, rig='NP'):

	filtered = beh_df['rig'].str.contains(rig)

	return beh_df.loc[filtered]


def plot_weight_over_time(beh_df):

	fig, ax = plt.subplots()
	ax.plot(beh_df['session_datetime_local'], beh_df['Wt_g'], 'k-o')
	ax.set_title('Wt_g')
	ax.tick_params(axis='x', labelrotation=45)


def plot_water_allotment(beh_df):

	fig, ax = plt.subplots()
	ax.plot(beh_df['session_datetime_local'], beh_df['WE_ml'], 'b-o')
	ax.plot(beh_df['session_datetime_local'], beh_df['WS_ml'], 'r-o')
	ax.plot(beh_df['session_datetime_local'], beh_df['WE_ml']+beh_df['WS_ml'], 'k-o')
	ax.legend(['WE_ml', 'WS_ml', 'Total'])
	ax.tick_params(axis='x', labelrotation=45)


def plot_inferred_presession_weight(beh_df, water_loss_during_session=0.3):

	post_wt = beh_df['Wt_g'].astype(float)
	earned_wt = beh_df['WE_ml'].astype(float)
	inferred_wt = post_wt - earned_wt + water_loss_during_session
	
	fig, ax = plt.subplots()
	ax.plot(beh_df['session_datetime_local'], inferred_wt, 'g-o')
	ax.set_title('Wt_g - WE_ml + {}: inferred pre-session weight'.format(water_loss_during_session))
	ax.tick_params(axis='x', labelrotation=45)


def plotSessionHistory(beh_df):
    
        def getColorAlphaFill(row):
            a = 1.0
            f = 'full'
            if 'NP' not in row['rig']:
                c = 'k'
            elif 'HAB' in row['stage']:
                c = 'm'
            elif 'EPHYS' in row['stage']:
                c = 'g'
            
            if '3uL' in row['stage']:
                a = 0.3
            
            return c,a,f

        #mouseID = str(mouseID)
        fig, ax = plt.subplots()
        fig.set_size_inches([12, 6])
        artists_for_legend = []
        labels_for_legend = []
        colors_used = []
        for ir, row in beh_df.iterrows():  
            num_rewards = row['trials']['cumulative_reward_number'].max()
            c,a,f = getColorAlphaFill(row)
            ax.plot(row['session_datetime_local'], num_rewards, c+'o', alpha=a, fillstyle=f, mew=3)
        
        ax.set_xlabel('Sessions')
        ax.set_ylabel('num rewards')
        ax.set_xticks([row['session_datetime_local'] for _,row in beh_df.iterrows()][::2])
        ax.set_xticklabels([row['session_datetime_local'].date() for _,row in beh_df.iterrows()], rotation=90)
        #title = mouseID + 'Rewards per Session'
        plt.tight_layout()
        
        k_patch = mpatches.Patch(color='k', label='NSB')
        m_patch = mpatches.Patch(color='m', label='HAB')
        g_patch = mpatches.Patch(color='g', label='EPHYS')

        ax.legend(handles=[k_patch, m_patch, g_patch])


def findSaturationTime(trialdf):
    startTrial = np.where(trialdf['cumulative_volume']>0)[0]
    saturationTrial = np.where(trialdf['cumulative_volume'] > 0.90*trialdf['cumulative_volume'].max())[0]
    if len(saturationTrial)==0:
        saturationTrial = 0
        startTrial = 0
    else:
        saturationTrial = saturationTrial[0]
        startTrial = startTrial[0]
    return np.array(trialdf['endtime'])[saturationTrial] - np.array(trialdf['endtime'])[startTrial]

def findAbortFraction(trialdf):
	trial_types = trialdf['trial_type']
	fraction_aborted = (np.sum(trial_types=='aborted'))/float(len(trial_types))

	return fraction_aborted
