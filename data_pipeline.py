import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import preprocessing
import data_loader


class DataPipeline:

    def __init__(self, path: str):
        self.data = data_loader.load_data(path)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scaler = None

    def _split(self):

        self.train_data = self.data.loc['2019']
        self.val_data = self.data.loc['2020-01':'2020-03']
        self.test_data = self.data.loc['2020-04':]

    def _preprocess(self):
        # Preprocess the data

        # Add time information
        self.data['day_x'], self.data['day_y'] = preprocessing.time_2d(self.data, day=True)
        self.data['year_x'], self.data['year_y'] = preprocessing.time_2d(self.data, day=False)

        # Filter by points
        self.data = preprocessing.filter_by_points(self.data, frequency='D', num_points=1440/15)

    def _normalize(self):
        self.scaler = MinMaxScaler()
        self.train_data = pd.DataFrame(self.scaler.fit_transform(self.train_data),
                                       columns=self.train_data.columns, index=self.train_data.index)

        self.val_data = pd.DataFrame(self.scaler.transform(self.val_data),
                                     columns=self.val_data.columns, index=self.val_data.index)

        self.test_data = pd.DataFrame(self.scaler.transform(self.test_data),
                                      columns=self.test_data.columns, index=self.test_data.index)

    def _format(self):
        x_train = self.train_data.drop(columns=['PV'], axis=1)
        y_train = self.train_data['PV']

        x_val = self.val_data.drop(columns=['PV'], axis=1)
        y_val = self.val_data['PV']

        x_test = self.test_data.drop(columns=['PV'], axis=1)
        y_test = self.test_data['PV']

        self.train_data = (x_train, y_train)
        self.val_data = (x_val, y_val)
        self.test_data = (x_test, y_test)

    def _do(self):

        self._preprocess()
        self._split()
        self._normalize()
        self._format()
