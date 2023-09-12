import shap
import xgboost as xgb
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
from sklearn.metrics import mean_squared_error


class XGBShap:

    def __init__(self, train_x, train_y, val_x, val_y, n_trials=100, seed=None, study=None):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.n_trials = n_trials
        self.seed = seed
        self.model = None
        self.study = study
        self.explainer = None
        self.shap_values = None
        self.model_memory_usage = None

        self.season_shap = None
        self.season_data = None

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
            return mean_squared_error(self.val_y, temp_yhat)

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
