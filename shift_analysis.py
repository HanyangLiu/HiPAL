# shift estimation algorithm
# implementation of the paper https://pubmed.ncbi.nlm.nih.gov/30625502/
# partitioning accesses into chunks with 3 steps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# import tqdm


def remove_duplicates(access_logs):
    # TODO: Look at commit history to see why this is empty.
    return access_logs


class ShiftEstimation:
    def __init__(self, access_logs, T_basic=4, T_small_2=7, T_large_2=30, T_small_3=2, T_large_3=20):
        self.access_logs = access_logs
        self.T_basic = T_basic
        self.T_small_2 = T_small_2
        self.T_large_2 = T_large_2
        self.T_small_3 = T_small_3
        self.T_large_3 = T_large_3

    def shift_estimation(self):
        ACCESS_LOGS = self.access_logs
        intervals = pd.to_datetime(ACCESS_LOGS.ACCESS_TIME) - pd.to_datetime(ACCESS_LOGS.ACCESS_TIME.shift(1))
        intervals_hr = intervals / np.timedelta64(1, 'h')
        access_intervals = ACCESS_LOGS[['ACCESS_TIME']].assign(interval_to_last=intervals_hr.values)
        access_intervals.loc[0, 'interval_to_last'] = 0

        df_1 = self.iteration_1(df_in=access_intervals, thresh=self.T_basic)
        df_2 = self.iteration_2(df_in=df_1, T_small=self.T_small_2, T_large=self.T_large_2)
        df_shift = self.iteration_3(df_in=df_2, T_small=self.T_small_3, T_large=self.T_large_3)

        return df_shift


    def iteration_1(self, df_in, thresh):
        iter_1 = df_in[df_in.interval_to_last > thresh]
        df_out = pd.DataFrame(columns=['start', 'end'])
        #pbar = tqdm.tqdm(total=len(iter_1), desc="Iteration 1")
        for order, df_ind in enumerate(iter_1.index):
            start = df_in.loc[df_ind, 'ACCESS_TIME']
            if order == len(iter_1.index) - 1:
                end = df_in.loc[len(df_in) - 1, 'ACCESS_TIME']
            else:
                end = df_in.loc[iter_1.index[order + 1] - 1, 'ACCESS_TIME']
            df_out = df_out.append({
                'start': start,
                'end': end
            }, ignore_index=True)
            #pbar.update(1)
        df_out = df_out.append({
            'start': df_in.loc[0, 'ACCESS_TIME'],
            'end': df_in.loc[iter_1.index[0] - 1, 'ACCESS_TIME']
        }, ignore_index=True)
        #pbar.close()

        df_out = df_out.sort_values(by=['start', 'end'], ignore_index=True)
        df_out = df_out.assign(duration=(pd.to_datetime(df_out.end)
                               - pd.to_datetime(df_out.start)) / np.timedelta64(1, 'h'))
        df_out = df_out.assign(interval_to_last=(pd.to_datetime(df_out.start)
                               - pd.to_datetime(df_out.end.shift(1))) / np.timedelta64(1, 'h'))

        return df_out

    def iteration_2(self, df_in, T_small, T_large):
        df_in = df_in.fillna(value=0)
        df_out = pd.DataFrame(columns=['start', 'end', 'duration'])
        sum_duration = 0
        start = df_in.loc[0, 'start']
        #pbar = tqdm.tqdm(total=len(df_in), desc="Iteration 2")
        for order, df_ind in enumerate(df_in.index):

            end = df_in.loc[df_ind, 'end']
            sum_duration = sum_duration + df_in.loc[df_ind, 'duration']

            # margin case: last row
            if order == len(df_in) - 1:
                df_out = df_out.append({
                    'start': start,
                    'end': end,
                    'duration': sum_duration
                }, ignore_index=True)
                break

            # if interval less than 7 hr and aggregated length is less than 30
            if df_in.loc[df_ind + 1, 'interval_to_last'] > T_small or sum_duration + df_in.loc[
                df_ind + 1, 'duration'] > T_large:
                df_out = df_out.append({
                    'start': start,
                    'end': end,
                    'duration': sum_duration
                }, ignore_index=True)
                start = df_in.loc[df_ind + 1, 'start']
                sum_duration = 0
            #pbar.update(1)
        #pbar.close()
        df_out = df_out.assign(
            interval_to_last=(pd.to_datetime(df_out.start)
                              - pd.to_datetime(df_out.end.shift(1))) / np.timedelta64(1, 'h'))
        return df_out

    def iteration_3(self, df_in, T_small, T_large):
        df_in = df_in.fillna(value=0)
        df_out = pd.DataFrame(columns=['start', 'end', 'duration'])
        sum_duration = 0
        start = df_in.loc[0, 'start']
        #pbar = tqdm.tqdm(total=len(df_in), desc="Iteration 3")
        for order, df_ind in enumerate(df_in.index):

            end = df_in.loc[df_ind, 'end']
            sum_duration = sum_duration + df_in.loc[df_ind, 'duration']

            # margin case: last row
            if order == len(df_in) - 1:
                df_out = df_out.append({
                    'start': start,
                    'end': end,
                    'duration': sum_duration
                }, ignore_index=True)
                break

            # if length less than 2 hr and aggregated length is less than 20
            if df_in.loc[df_ind + 1, 'duration'] > T_small or sum_duration + df_in.loc[
                df_ind + 1, 'duration'] > T_large:
                df_out = df_out.append({
                    'start': start,
                    'end': end,
                    'duration': sum_duration
                }, ignore_index=True)
                start = df_in.loc[df_ind + 1, 'start']
                sum_duration = 0
            #pbar.update(1)
        #pbar.close()
        df_out = df_out.assign(interval_to_last=(pd.to_datetime(df_out.start)
                                                 - pd.to_datetime(df_out.end.shift(1))) / np.timedelta64(1, 'h'))
        df_out.loc[0, 'interval_to_last'] = 0

        return df_out


def shifts_work_hours_analysis(df, clean=True):
    """
    Returns a tuple with:
    - number of shifts worked
    - total hours worked
    - average hours per shift
    - average time gap between shifts

    for a dataframe of access_logs_complete
    :param df: clean, True if already cleaned
    :return:
    """
    if not clean:
        df = df[df.WORKSTATION_ID != "HAIKU"] # Get rid of irrelevant data.
        df = df[~df.WORKSTATION_ID.isnull()]
        df.reset_index(drop=True, inplace=True)
    estimator = ShiftEstimation(df)
    estimations = estimator.shift_estimation()
    shifts_worked = len(estimations) # Get the number of dataframes returned.
    total_hrs_worked = sum(estimations.duration)

    avg_hrs_shift = total_hrs_worked / shifts_worked

    avg_time_gap = np.average(estimations.interval_to_last)

    return (shifts_worked, total_hrs_worked, avg_hrs_shift, avg_time_gap)

if __name__ == "__main__":
    # data_dir = 'data/2019_wHNO_export'
    # data_dir = 'data/data_sunny_2019'
    data_dir = 'data/data_sunny_2020'

    # select a user
    user = 'M58530'  # Sunny
    # user = 'M61877'  # Mark

    # load files
    # access_logs_all = pd.read_csv(os.path.join(data_dir, 'access_log_complete.csv'))
    # access_logs_all = pd.read_csv(os.path.join(data_dir, '2019_processed_accesslog_wreport2.csv'))
    access_logs_all = pd.read_excel(os.path.join(data_dir, 'Access Log Mnemonics filtered 1.1.2020 through 10.21.2020 for Sunny without PHI.xlsx')); access_logs_all = access_logs_all.rename(columns={'Masked_Patient_Id': 'MASKED_PAT_ID'})
    access_logs = access_logs_all[access_logs_all['USER_ID'] == user]
    #access_logs = remove_duplicates(access_logs)
    access_logs = access_logs[access_logs.WORKSTATION_ID != "HAIKU"]
    access_logs = access_logs[~access_logs.WORKSTATION_ID.isnull()]
    access_logs.reset_index(drop=True, inplace=True)

    # shift estimation
    shift_estimator = ShiftEstimation(access_logs, T_basic=4, T_small_2=7, T_large_2=30, T_small_3=2, T_large_3=20)
    df_shift = shift_estimator.shift_estimation()

    # plot
    bw = 0.3
    # distribution of shift duration
    df_shift.duration.plot.kde(bw_method=bw, title='distribution of shift duration - ' + user)
    plt.xlim([0, 40])
    plt.xlabel('shift duration')
    plt.show()

    # distrubution of shift intervals
    df_shift.interval_to_last.plot.kde(bw_method=bw, title='distribution of shift intervals - ' + user)
    plt.xlim([0, 40])
    plt.xlabel('shift intervals')
    plt.show()

    # distribution of start & end of shifts
    start_time = pd.to_datetime(df_shift.start) - pd.to_datetime(df_shift.start).dt.normalize()
    end_time = pd.to_datetime(df_shift.end) - pd.to_datetime(df_shift.end).dt.normalize()
    plt.figure(figsize=[10, 5])
    start_time.dt.total_seconds().div(3600).astype(float).plot.kde(bw_method=bw)
    end_time.dt.total_seconds().div(3600).astype(float).plot.kde(bw_method=bw)
    plt.xlim([0, 24])
    plt.xlabel('Time in A Day (24 hr)')
    plt.title('start/end of shifts distribution in a day - ' + user)
    plt.legend()
    plt.show()
