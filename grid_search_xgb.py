import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/train_updated_1.csv")
    df = df[['Type_of_Day', 'is_weekday', 'is_weekend', 'dayofweek_sin', 'hour_cos', 'dayofweek_cos',
             'hour_sin', 'month_sin', 'month_cos', 'energy']]
    X = df.drop("energy", axis=1).values
    y = df.energy.values
    regressor = xgb.XGBRegressor()
    param_grid = {
        "eta": [0.1, 0.3, 0.5, 0.7, 0.9],
        'max_depth': [3, 5, 7, 9, 11, 15, 20]
    }
    model = model_selection.GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        verbose=10,
        n_jobs=1,
        cv=3
    )
    model.fit(X, y)
    print(f"Best score: {model.best_score_}")
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
