# Imports

import numpy as np
import pandas as pd
import sklearn
import glob

from sklearn.preprocessing import MinMaxScaler


# Functions
def get_solcast(path):
    file = pd.read_csv(path)
    file.index = pd.to_datetime(file['PeriodEnd'])
    file.drop(columns=['PeriodEnd', 'PeriodStart', 'Period'], inplace=True)
    file = file.resample('15T').bfill()

    return file


def get_pv(path):
    file = pd.read_csv(path)
    file.index = pd.to_datetime(file['datetime_utc'])
    file.drop(columns=['datetime_utc'], inplace=True)
    file = file.resample('15T').mean()

    return file


def get_solcast_pv(df1, df2):
    """
    df1: PV dataframe
    df2: Solcast dataframe
    """

    # Filter both dataframes for 2019 and 2020
    try:
        temp_df1 = df1['2019':'2021-04-01']
        temp_df2 = df2['2019':'2021-04-01']

        # Check if data is complete. If not, match the smaller indexes
        if temp_df2.shape[0] < temp_df1.shape[0]:
            temp_df1 = temp_df1.loc['2019':'{}'.format(temp_df2.index[-1].tz_convert(None))]

        # Only considering 2019 and 2020 since data is complete for that period
        temp_data = pd.DataFrame({'PV': temp_df1['pv'].values}, index=temp_df1.index)

        for col in temp_df2.columns:
            temp_data[col] = temp_df2[col].shift(-1).values

        return temp_data
    except:
        temp_df1 = df1['2019':'2020']
        temp_df2 = df2['2019':'2020']

        # Check if data is complete. If not, match the smaller indexes
        if temp_df2.shape[0] < temp_df1.shape[0]:
            temp_df1 = temp_df1['2019':'{}'.format(temp_df2.index[-1].tz_convert(None))]

        # Only considering 2019 and 2020 since data is complete for that period
        temp_data = pd.DataFrame({'PV': temp_df1['pv'].values}, index=temp_df1.index)

        for col in temp_df2.columns:
            temp_data[col] = temp_df2[col].shift(-1).values

        return temp_data


def load_data(path: str):
    """
    path: path to the folder containing the data
    """

    # Get the PV and Solcast files
    temp_pv = get_pv('{}/pv.csv'.format(path))
    temp_solcast = get_solcast('{}/solcast.csv'.format(path))

    # Join the files into a single dataframe
    data = get_solcast_pv(temp_pv, temp_solcast)

    print('date range: {} - {}'.format(data.index[0], data.index[-1]))

    return data
