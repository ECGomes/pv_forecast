import pandas as pd
import numpy as np
import datetime


def time_2d(df: pd.DataFrame, day: bool = True) -> tuple:
    """
    Adds 2D time information for single days
    df: dataframe to add the information
    day: if True, return day information. If False, return year information
    """

    # Check if day or year
    if day:
        # Map the index into seconds
        timestamp_s = pd.to_datetime(df.index.values).map(datetime.datetime.timestamp)

        # Since we're calculating the cos and sin values from seconds
        # It's 60 seconds into 60 min into 24 hours per day
        day_calc = 24 * 60 * 60

        # Calculate the values
        day_x = np.cos((2 * np.pi / day_calc) * timestamp_s)
        day_y = np.sin((2 * np.pi / day_calc) * timestamp_s)

        return day_x, day_y

    else:
        day_year = df.index.dayofyear
        year_constant = 365.2524

        year_x = np.cos((2 * np.pi / year_constant) * day_year)
        year_y = np.sin((2 * np.pi / year_constant) * day_year)

        return year_x, year_y


# Aux Function for filtering data

def filter_by_points(df, frequency='D', num_points=1440, return_dictionary=False):
    df_dropped = df.dropna()
    grouper = df_dropped.groupby(pd.Grouper(freq=frequency))

    output = 0
    if return_dictionary:
        new_dict = {}
        for i in grouper:
            if len(i[1]) != num_points:
                pass
            else:
                new_dict[i[0]] = pd.DataFrame(i[1])
        output = new_dict
    else:
        new_df = pd.DataFrame({})
        for i in grouper:
            if len(i[1]) != num_points:
                pass
            else:
                new_df = pd.concat([new_df, pd.DataFrame(i[1])], axis=0)

        output = new_df

    return output
