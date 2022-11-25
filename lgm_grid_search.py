import numpy as np
import pandas as pd
import lightgbm as lgm
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/train_updated_1.csv")
    df = df[['Type_of_Day', 'is_weekday', 'is_weekend', 'dayofweek_sin', 'hour_cos', 'dayofweek_cos',
             'hour_sin', 'month_sin', 'month_cos', 'year', 'dayofyear', 'quarter', 'season', 'energy']]
    X = df.drop("energy", axis=1).values
    y = df.energy.values
    regressor = lgm.LGBMRegressor()
    param_grid = {
        "max_depth": [3],
        'learning_rate': [0.3],
        'min_gain_to_split': [0.1],
        'min_data_in_leaf': [11],

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
