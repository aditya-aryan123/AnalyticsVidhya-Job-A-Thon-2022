import pandas as pd

import catboost
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('../input/train_updated_1.csv')
test_data = pd.read_csv('../input/test_updated.csv')
test_data_copy = test_data.copy()

train_data_copy = train_data[
    ['Type_of_Day', 'is_weekday', 'is_weekend', 'dayofweek_sin', 'hour_cos', 'dayofweek_cos',
     'hour_sin', 'month_sin', 'month_cos', 'energy', 'year', 'quarter', 'dayofyear', 'season']]
test_data_copy.drop(['row_id', 'datetime'], axis=1, inplace=True)
test_data_copy = test_data_copy[
    ['Type_of_Day', 'is_weekday', 'is_weekend', 'dayofweek_sin', 'hour_cos', 'dayofweek_cos',
     'hour_sin', 'month_sin', 'month_cos', 'year', 'quarter', 'dayofyear', 'season']]

X = train_data_copy.drop('energy', axis=1)
y = train_data_copy['energy']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

params = {
    'n_estimators': 50000,
    'learning_rate': 0.3,
    'eval_metric': "MSLE",
    'subsample': 0.7,
    'max_depth': 3,
    'use_best_model': True,
    'early_stopping_rounds': 500
}

train_pool = catboost.Pool(X_train, y_train)
validate_pool = catboost.Pool(X_val, y_val)

model = catboost.CatBoostRegressor(**params).fit(train_pool, eval_set=validate_pool, verbose=10)
pred = model.predict(X_val)
root_mean_squared_error = mean_squared_error(pred, y_val, squared=False)
print(root_mean_squared_error)

test_data_copy['prediction'] = model.predict(test_data_copy)
test_data = test_data.merge(test_data_copy['prediction'], how='left', right_index=True, left_index=True)
submission = test_data[['row_id', 'prediction']]
submission.rename(columns={'prediction': 'energy'}, inplace=True)
submission.to_csv('submission.csv', index=False)
