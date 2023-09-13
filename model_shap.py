import shap
import xgboost as xgb
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tabnet import StackedTabNetRegressor

import os
import joblib


class ModelShap:

    def __init__(self, train_x, train_y,
                 val_x, val_y,
                 test_x, test_y,
                 scaler,
                 n_trials=100, seed=None, study=None):
        # Get the data
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y

        # Get the scaler
        self.scaler = scaler

        # Get the parameters
        self.n_trials = n_trials
        self.seed = seed
        self.model = None
        self.study = study

        self.explainer = None
        self.shap_values = None
        self.model_memory_usage = None

        self.season_shap = None
        self.season_data = None

        # Placeholder for the report
        self.report = None

    def predict(self, x):
        return self.model.predict(x)

    def get_report(self):

        # Create list placeholders
        train_errors = []
        val_errors = []
        test_errors = []

        # Inverse the normalization -> Create a temporary scaler to reverse only the first column
        temp_scaler = MinMaxScaler()
        temp_scaler.min_, temp_scaler.scale_ = self.scaler.min_[0], self.scaler.scale_[0]

        # Do the predictions
        train_yhat = self.predict(self.train_x)
        val_yhat = self.predict(self.val_x)
        test_yhat = self.predict(self.test_x)

        # Correct negative values to 0
        train_yhat[train_yhat < 0] = 0
        val_yhat[val_yhat < 0] = 0
        test_yhat[test_yhat < 0] = 0

        # Reverse the normalization of the predictions and the actual values
        train_yhat = temp_scaler.inverse_transform(train_yhat.reshape(-1, 1))
        val_yhat = temp_scaler.inverse_transform(val_yhat.reshape(-1, 1))
        test_yhat = temp_scaler.inverse_transform(test_yhat.reshape(-1, 1))

        train_y = temp_scaler.inverse_transform(self.train_y.values.reshape(-1, 1))
        val_y = temp_scaler.inverse_transform(self.val_y.values.reshape(-1, 1))
        test_y = temp_scaler.inverse_transform(self.test_y.values.reshape(-1, 1))

        # Calculate the errors
        train_errors.append(mean_squared_error(train_y, train_yhat, squared=False))
        val_errors.append(mean_squared_error(val_y, val_yhat, squared=False))
        test_errors.append(mean_squared_error(test_y, test_yhat, squared=False))

        # Create the report
        report = pd.DataFrame({'RMSE Train': train_errors,
                               'RMSE Val': val_errors,
                               'RMSE Test': test_errors,
                               'Number of Features': self.train_x.shape[1],
                               'Avg Memory Usage': np.mean(self.model_memory_usage - self.model_memory_usage.min()),
                               'Execution Time': self.model_memory_usage.index[-1] - self.model_memory_usage.index[0],
                               })

        self.report = report

    @staticmethod
    def _split_season(df: pd.DataFrame) -> pd.DataFrame:
        # Create a copy as to not affect the original dataframe
        temp_df = df.copy(deep=True)

        date = temp_df.index.month * 100 + temp_df.index.day
        temp_df['season'] = pd.cut(date, bins=[0, 321, 620, 922, 1220, 1300],
                                   labels=['winter', 'spring', 'summer', 'autumn', 'winter '])

        temp_df['season'] = temp_df['season'].str.replace(' ', '')
        temp_df['season'] = temp_df['season'].astype('category')

        return temp_df

    def plot_shap(self, figsize=(20, 24), plot_type=None):

        fig = plt.figure(figsize=figsize)
        shap.summary_plot(self.shap_values.values, self.train_x,
                          plot_size=None, show=False, plot_type=plot_type)

        plt.tight_layout()
        plt.show()

    def plot_season_shap(self, figsize=(20, 24), plot_type=None):
        # Plot the SHAP values for each season
        fig = plt.figure(figsize=figsize)
        current_axs = 0

        for season in self.season_shap['season'].unique():
            plt.subplot(4, 1, current_axs + 1)

            season_shap = self.season_shap[self.season_shap['season'] == season].drop('season', axis=1)

            season_data = self.season_shap[self.season_shap['season'] == season].drop('season', axis=1)

            temp_axs = shap.summary_plot(season_shap.values, season_data,
                                         plot_size=None, show=False, plot_type=plot_type)

            plt.title('{} SHAP values'.format(season))
            current_axs += 1

        plt.tight_layout()
        plt.show()

    def save(self, path):

        if not os.path.exists(path):
            os.makedirs(path)

        # Save the DataFrames
        self.train_x.to_csv(os.path.join(path, 'train_x.csv'))
        self.train_y.to_csv(os.path.join(path, 'train_y.csv'))

        self.val_x.to_csv(os.path.join(path, 'val_x.csv'))
        self.val_y.to_csv(os.path.join(path, 'val_y.csv'))

        self.test_x.to_csv(os.path.join(path, 'test_x.csv'))
        self.test_y.to_csv(os.path.join(path, 'test_y.csv'))

        # Save the scaler
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))

        # Save the SHAP DataFrames
        self.shap_values.to_csv(os.path.join(path, 'shap_values.csv'))
        self.season_shap.to_csv(os.path.join(path, 'season_shap.csv'))

        # Save the memory usage
        self.model_memory_usage.to_csv(os.path.join(path, 'memory_usage.csv'))

        # Save the report
        self.report.to_csv(os.path.join(path, 'report.csv'))


class XGBShap(ModelShap):

    def __init__(self, train_x, train_y,
                 val_x, val_y,
                 test_x, test_y,
                 scaler,
                 n_trials=100, seed=None, study=None):
        super().__init__(train_x, train_y,
                         val_x, val_y,
                         test_x, test_y,
                         scaler,
                         n_trials=100, seed=None, study=None)

        self.history = None

    def _train(self):
        self.model = xgb.XGBRegressor(n_estimators=self.study.best_params['n_estimators'],
                                      max_depth=self.study.best_params['max_depth'],
                                      reg_alpha=self.study.best_params['reg_alpha'],
                                      reg_lambda=self.study.best_params['reg_lambda'])

        self.model_memory_usage = memory_usage((self.model.fit, (self.train_x, self.train_y)),
                                               timestamps=True)
        self.model_memory_usage = pd.DataFrame(self.model_memory_usage, columns=['Memory Usage', 'Timestamp'])
        self.model_memory_usage.index = pd.to_datetime(self.model_memory_usage['Timestamp'], unit='s')
        self.model_memory_usage = self.model_memory_usage.drop(columns=['Timestamp'], axis=1)

    def do(self):

        if self.study is None:
            self._parameter_sweep_xgb()
        self._train()
        self._shap_xgb()

        # Season split
        self.season_shap = self._split_season(self.shap_values)
        self.season_data = self._split_season(self.train_x)

        # Create the report
        self.get_report()

    def _parameter_sweep_xgb(self):
        # First do a parameter sweep with Optuna
        def create_model(trial):
            # Do search for n_estimators, max_depth, reg_alpha and reg_lambda
            sug_estimators = trial.suggest_int('n_estimators', 50, 5000)
            sug_depth = trial.suggest_int('max_depth', 10, 5000)
            sug_alpha = trial.suggest_float('reg_alpha', 1e-5, 1e-3)
            sug_lambda = trial.suggest_float('reg_lambda', 1e-5, 1e-3)

            sug_model = xgb.XGBRegressor(n_estimators=sug_estimators,
                                         max_depth=sug_depth,
                                         reg_alpha=sug_alpha,
                                         reg_lambda=sug_lambda)

            return sug_model

        def create_training(model):
            model.fit(self.train_x, self.train_y)

        def create_evaluation(model):
            temp_yhat = model.predict(self.val_x)
            return mean_squared_error(self.val_y, temp_yhat, squared=False)

        def create_objective(trial):
            # Instantiate the model
            temp_model = create_model(trial)

            # Train the model
            create_training(temp_model)

            # Evaluate model
            metrics_val = create_evaluation(temp_model)

            return metrics_val

        sampler = None
        if self.seed is not None:
            sampler = optuna.samplers.TPESampler(seed=self.seed)

        self.study = optuna.create_study(direction='minimize', sampler=sampler)
        self.study.optimize(create_objective, n_trials=self.n_trials, show_progress_bar=True)

    def _shap_xgb(self):
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.train_x)
        self.shap_values = pd.DataFrame(self.shap_values, columns=self.train_x.columns,
                                        index=self.train_x.index)

    def save(self, path):
        super().save(path)

        # Save the model
        joblib.dump(self.model, os.path.join(path, 'model.pkl'))


class TNShap(ModelShap):

    def __init__(self, train_x, train_y,
                 val_x, val_y,
                 test_x, test_y,
                 scaler,
                 n_trials=100, seed=None, study=None):
        super().__init__(train_x, train_y,
                         val_x, val_y,
                         test_x, test_y,
                         scaler,
                         n_trials=100, seed=None, study=None)

    @staticmethod
    def compile_and_fit(model, train_x, train_y, val_x, val_y,
                        max_epochs=1000, es_patience=200, rlr_patience=100,
                        adam_lr=0.001):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=es_patience,
                                                          mode='min')

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         patience=rlr_patience)

        model.compile(loss=tf.keras.losses.mean_squared_error,
                      optimizer=tf.optimizers.Adam(learning_rate=adam_lr),
                      metrics=[tf.metrics.MeanAbsoluteError(name='mae')])

        history = model.fit(train_x, train_y,
                            batch_size=32,
                            epochs=max_epochs,
                            validation_data=(val_x, val_y),
                            callbacks=[early_stopping, reduce_lr])
        return history

    def _parameter_sweep_tn(self):

        # Define the creation of the model
        def optuna_create_model(trial):
            # Do the search for n_estimators, max_depth, reg_alpha and reg_lambda
            sug_fdim = trial.suggest_int('feature_dim', 32, 72)
            sug_odim = trial.suggest_int('output_dim', 4, 12)
            sug_dsteps = trial.suggest_int('num_decision_steps', 1, 4)

            sug_model = StackedTabNetRegressor(feature_columns=None,
                                               num_layers=1,
                                               num_regressors=1,
                                               feature_dim=sug_fdim,
                                               num_features=8,
                                               output_dim=sug_odim,
                                               num_decision_steps=sug_dsteps,
                                               num_groups=1)

            return sug_model

        # Create the training for the model
        def optuna_create_training(model):
            self.compile_and_fit(model, self.train_x, self.train_y,
                                 self.val_x, self.val_y,
                                 max_epochs=50, rlr_patience=10, es_patience=20)

        def optuna_create_evaluation(model):
            yhat = model.predict(self.val_x)
            return mean_squared_error(self.val_y, yhat, squared=False)

        def optuna_objective(trial):
            model = optuna_create_model(trial)
            optuna_create_training(model)
            return optuna_create_evaluation(model)

        # Create the study
        if self.study is None:
            self.study = optuna.create_study(direction='minimize',
                                             sampler=optuna.samplers.TPESampler(seed=self.seed))

        # Run the study
        self.study.optimize(optuna_objective, n_trials=self.n_trials, show_progress_bar=True)

    def _train(self):

        model = StackedTabNetRegressor(feature_columns=None,
                                       num_layers=1,
                                       num_regressors=1,
                                       feature_dim=self.study.best_params['feature_dim'],
                                       num_features=8,
                                       output_dim=self.study.best_params['output_dim'],
                                       num_decision_steps=self.study.best_params['num_decision_steps'],
                                       num_groups=1)

        history = self.compile_and_fit(model, self.train_x, self.train_y,
                                       self.val_x, self.val_y,)

        self.model = model
        self.history = history

    def _shap_tn(self):
        self.explainer = shap.KernelExplainer(self.model, shap.sample(self.train_x, 10))
        self.shap_values = self.explainer.shap_values(self.train_x, check_additivity=False)
        self.shap_values = pd.DataFrame(self.shap_values, columns=self.train_x.columns,
                                        index=self.train_x.index)

    # Execute the pipeline
    def do(self):

        # Get the best model and train it
        self._parameter_sweep_tn()
        self.model_memory_usage = memory_usage((self._train, (), {}), timestamps=True)

        # SHAP values
        self._shap_tn()

        # Season split
        self.season_shap = self._split_season(self.shap_values)
        self.season_data = self._split_season(self.train_x)

        # Create the report
        self.get_report()

    def split_dataframe(self, df, input_width):
        current_list = []
        for i in df.columns[1:]:
            current_sequence = self.split_sequence(df[i], input_width, 1)
            current_list.append(current_sequence)

        stacked_list = np.stack(current_list, axis=-1)

        return stacked_list

    def do_windowing(self, df, input_width=1, y_col='PV'):
        """
        Perform the windowing on the dataframe
        df: input dataframe to perform windowing
        input_width: temporal dimension size or number of timesteps
        y_col: name of column for the prediction
        """

        temp_x = self.split_dataframe(df, input_width)
        temp_y = df[y_col].values[input_width - 1:-1]

        return temp_x, temp_y

    @staticmethod
    def split_sequence(sequence, n_steps, n_intervals):
        X = list()
        for i in np.arange(0, len(sequence), n_intervals):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x = sequence[i:end_ix]
            X.append(seq_x)
        return np.array(X)

    def save(self, path):
        super().save(path)

        # Save the model
        self.model.save(os.path.join(path, 'model'), save_format='tf')
