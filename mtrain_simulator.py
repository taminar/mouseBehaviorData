from visual_behavior.change_detection.trials.session_metrics import \
        peak_dprime, trial_count_by_trial_type
from visual_behavior.change_detection.trials import session_metrics as metrics
from six import iteritems
import logging
import pandas as pd

logger = logging.getLogger()

def meets_hit_threshold_average(mouse):
    """requires the last 3 behavior sessions to meet an average hit threshold
    of 120 hits per behavioral sessions
    Parameters
    ----------
    mouse: mtrain_api.mouse.Mouse
        mtrain_api database interface object
    
    Returns
    -------
    bool
        meets criteria
    Notes
    -----
    - if there are less than 3 behavior sessions, this will return False
    """
    hit_threshold = 120

    try:
        trials_df = mouse.trials.copy(deep=True) \
            .sort_values(by=["startdatetime"]) \
            .groupby("behavior_session_uuid")
        
        n_unique_sessions = len(mouse.trials['behavior_session_uuid'].unique())

        num_hits_df = trials_df \
            .apply(trial_count_by_trial_type)

        def trials_to_startdatetime(session_trials):
            return session_trials["startdatetime"].iloc[0]

        startdatetime_df = trials_df \
            .apply(trials_to_startdatetime)
        
        joined_df = pd.DataFrame({
            "num_hits": num_hits_df,
            "startdatetime": startdatetime_df, 
        })

        sorted_joined = joined_df.sort_values(by=["startdatetime"])
        print(sorted_joined['num_hits'])
        recent = list(sorted_joined["num_hits"].iloc[-3:])
        meets_hit_threshold = (len(recent) > 2) & (sum(recent) > (hit_threshold * 3))

        logging.info(
            "meets hit threshold average - mouse id: %s\nrecent hit dataframe: %s\nresult: %s",
            mouse.LabTracks_ID,
            recent,
            meets_hit_threshold,
        )

        return meets_hit_threshold
    except Exception:
        logging.error("meets_hit_threshold_average error", exc_info=True)


def trial_translator(trial_type, response_type, auto_rewarded=False):
    if trial_type == 'aborted':
        return 'aborted'
    elif auto_rewarded == True:
        return 'auto_rewarded'
    elif trial_type == 'autorewarded':
        return 'auto_rewarded'
    elif trial_type == 'go':
        if response_type in ['HIT', True, 1]:
            return 'hit'
        else:
            return 'miss'
    elif trial_type == 'catch':
        if response_type in ['FA', True, 1]:
            return 'false_alarm'
        else:
            return 'correct_reject'


def assign_trial_description(trial, palette='trial_types'):
    return trial_translator(
        trial['trial_type'],
        trial['response'],
        trial['auto_rewarded'],
    )


# def trial_count_by_trial_type(session_trials, trial_type='hit'):
#     session_trials['full_trial_type'] = session_trials.apply(
#         assign_trial_description,
#         axis=1,
#     )
#     trial_count = session_trials \
#         .groupby('full_trial_type')['trial_length'] \
#         .count()
#     try:
#         # trial_count is a pandas.DataFrame of total trial counts with trial_type as column names 
#         return trial_count[trial_type]  
#     except KeyError:
#         return 0.0


from functools import wraps
def requires_daily_metrics(**metrics_to_calculate):
    """This decorator is for functions that require the mouse object to have a
    session_summary dataframe with specific
    It takes as keyword arguments the functions that are computed on each day of
    the dataframe.
    For example, the function `two_out_of_three_aint_bad` assumes that the mouse
    object has session_summary dataframe which has a column titled
    'dprime_peak'. If it does not, it will create it using the function
    passed in on each training day dataframe.
        @requires_daily_metrics(dprime_peak=metrics.peak_dprime)
        def two_out_of_three_aint_bad(mouse):
            criteria = (mouse.session_summary[-3:]['dprime_peak']>2).sum() > 1
            return criteria==True
    """

    def requires_metrics_decorator(func_needs_metrics):

        @wraps(func_needs_metrics)  # <- this is important to maintain attributes of the wrapped function
        def wrapper(anymouse):

            # logic to append columns to the existing session_summary
            try:
                def calculator(group):
                    group = group.sort_values(by=['startdatetime', 'index'])
                    result = {
                        metric: func(group)
                        for metric, func
                        in iteritems(metrics_to_calculate)
                        if metric not in anymouse.session_summary.columns
                    }
                    return pd.Series(result, name='metrics')

                new_summary_data = anymouse.trials \
                    .groupby('behavior_session_uuid') \
                    .apply(calculator)
                
               
                anymouse.session_summary = anymouse.session_summary.join(
                    new_summary_data,
                    rsuffix='_joined'
                )

            # logic to create a new session_summary from scratch
            except AttributeError as e:
                logger.debug('session_summary not found. creating a new one.')

                def calculator(group):
                    group = group.sort_values(by=['startdatetime', 'index'])
                    result = {
                        metric: func(group)
                        for metric, func
                        in iteritems(metrics_to_calculate)
                    }
                    return pd.Series(result, name='metrics')

                anymouse.session_summary = anymouse.trials \
                    .groupby('behavior_session_uuid') \
                    .apply(calculator) \
                    .reset_index()

                lookup = anymouse.trials.groupby('behavior_session_uuid').apply(lambda df: df['startdatetime'].unique()[0])

                anymouse.session_summary['startdatetime'] = (
                    anymouse.session_summary['behavior_session_uuid']
                    .map(lookup)
                )
                
                anymouse.session_summary.sort_values('startdatetime',inplace=True)
                anymouse.session_summary.reset_index(drop=True,inplace=True)

            return func_needs_metrics(anymouse)
        return wrapper
    return requires_metrics_decorator

@requires_daily_metrics(
    dprime_peak=metrics.peak_dprime,
    num_engaged_trials=metrics.num_contingent_trials,
)
def meets_engagement_criteria_logged(mouse):
    """Version of meets engagement criteria that includes logging
    Parameters
    ----------
    mouse: mtrain_api.mouse.Mouse
        mtrain_api database interface object
    
    Returns
    -------
    bool
        meets engagement criteria
    Notes
    -----
    - if there are less than 3 behavior sessions, this will return False
    """
    mouse.session_summary['engagement_criteria'] = (
        (mouse.session_summary['dprime_peak'] > 1.0)
        & (mouse.session_summary['num_engaged_trials'] > 100)
    )

    x = mouse.session_summary['engagement_criteria'].iloc[-3:]
    
    logging.info(
        'meets engagement criteria. mouse id: %s\ndprime peaks:\n%s\nn engaged trials: %s,  engagement: %s, criteria met: %s',
        mouse.LabTracks_ID,
        mouse.session_summary['dprime_peak'],
        mouse.session_summary['num_engaged_trials'],
        x,
        x.sum() == 3,
    )

    return x.sum() == 3

HIT_THRESHOLD = 120
def meets_engagement_and_hit_threshold(mouse):
    """requires that the last three behavior sessions pass both 
    meets_engagement_criteria and meets_hit_threshold_average.
    Parameters
    ----------
    mouse: mtrain_api.mouse.Mouse
        mtrain_api database interface object
    
    Returns
    -------
    bool
        meets both criteria
    Notes
    -----
    - if there are less than 3 behavior sessions, this will return False
    """
    try:
        engaged = meets_engagement_criteria_logged(mouse)
        logger.info("engaged: %s, mouse_id: %s", engaged, mouse.LabTracks_ID)
    except Exception as e:
        logger.error("meets enagement criteria critically failed.", exc_info=True)
        engaged = False
    
    try:
        at_hit_threshold = meets_hit_threshold_average(mouse)
        logger.info("at_hit_threshold: %s, mouse_id: %s", at_hit_threshold, mouse.LabTracks_ID)
    except Exception as e:
        logger.error("meets hit threshold average criteria critically failed.", exc_info=True)
        at_hit_threshold = False

    handoff_ready = engaged and at_hit_threshold

    logger.info("handoff ready: %s, mouse id: %s, hit threshold: %s", handoff_ready, mouse.LabTracks_ID, HIT_THRESHOLD)

    return handoff_ready