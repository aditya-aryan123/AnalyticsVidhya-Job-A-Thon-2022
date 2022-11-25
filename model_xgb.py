import pandas as pd
import xgboost as xgb
from sklearn import model_selection
from sklearn import metrics

train = pd.read_csv('../input/train_updated_1.csv')
train_copy = train.copy()
test = pd.read_csv('../input/test_updated.csv')
test_copy = test.copy()

train_copy = train_copy.copy()
train_copy = train_copy[['Type_of_Day', 'is_weekday', 'is_weekend', 'dayofweek_sin', 'hour_cos', 'dayofweek_cos',
                         'hour_sin', 'month_sin', 'month_cos', 'energy', 'year', 'quarter', 'dayofyear', 'season']]

test_copy.drop(['row_id', 'datetime'], axis=1, inplace=True)
test_copy = test_copy[['Type_of_Day', 'is_weekday', 'is_weekend', 'dayofweek_sin', 'hour_cos', 'dayofweek_cos',
                       'hour_sin', 'month_sin', 'month_cos', 'year', 'quarter', 'dayofyear', 'season']]

X = train_copy.drop('energy', axis=1)
y = train_copy['energy']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=42)

model = xgb.XGBRegressor(colsample_bylevel=0.8, colsample_bynode=0.1, colsample_bytree=0.9, eta=0.3, max_depth=5,
                         gamma=0, reg_alpha=10, reg_lambda=0)
model.fit(X_train, y_train)
pred = model.predict(X_test)
r_squared = metrics.r2_score(pred, y_test)
root_mean_squared_error = metrics.mean_squared_error(pred, y_test, squared=False)
print(r_squared)
print(root_mean_squared_error)

test_copy['prediction'] = model.predict(test_copy)
test = test.merge(test_copy['prediction'], how='left', left_index=True, right_index=True)
submission = test[['row_id', 'prediction']]
submission.rename(columns={'prediction': 'energy'}, inplace=True)
submission.to_csv('submission.csv', index=False)
