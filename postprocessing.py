import pandas as pd


def split_season(df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy as to not affect the original dataframe
    temp_df = df.copy(deep=True)

    date = temp_df.index.month * 100 + temp_df.index.day
    temp_df['season'] = pd.cut(date, bins=[0, 321, 620, 922, 1220, 1300],
                               labels=['winter', 'spring', 'summer', 'autumn', 'winter '])

    temp_df['season'] = temp_df['season'].str.replace(' ', '')
    temp_df['season'] = temp_df['season'].astype('category')

    return temp_df
