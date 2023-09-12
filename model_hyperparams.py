# Define a training loop for UPACs

import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def parameter_sweep_xgb(train_x, train_y, val_x, val_y, n_trials=100):
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
        model.fit(train_x, train_y)

    def create_evaluation(model):
        temp_yhat = model.predict(val_x)
        return mean_squared_error(val_y, temp_yhat)

    def create_objective(trial):
        # Instantiate the model
        temp_model = create_model(trial)

        # Train the model
        create_training(temp_model)

        # Evaluate model
        metrics_val = create_evaluation(temp_model)

        return metrics_val

    study = optuna.create_study(direction='minimize')
    study.optimize(create_objective, n_trials=n_trials, show_progress_bar=True)

    return study
