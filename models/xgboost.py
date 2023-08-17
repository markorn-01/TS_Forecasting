import xgboost as xgb
from sklearn.metrics import mean_squared_error
from data_process import *
from window_generate import *
from model_generate import *

df = get_data(csv_path="data/gold/LBMA-GOLD.csv")
df, datetime = prepare_data(df, label='USD (AM)', date='Date') 
df = add_time(df, datetime)
train_df, val_df, test_df, num_features = split_data(df)
train_df, val_df, test_df = normalize_data(train_df, val_df, test_df)

# Prepare data for XGBoost
X_train, y_train = [], []
for inputs, labels in train_df:
    X_train.append(inputs.numpy())
    y_train.append(labels.numpy())
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)

# Create and train XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train.reshape(X_train.shape[0], -1), y_train.reshape(y_train.shape[0], -1))

# Evaluate the model
X_val, y_val = [], []
for inputs, labels in train_df:
    X_val.append(inputs.numpy())
    y_val.append(labels.numpy())
X_val = np.concatenate(X_val, axis=0)
y_val = np.concatenate(y_val, axis=0)
val_predictions = model.predict(X_val.reshape(X_val.shape[0], -1))
mse = mean_squared_error(y_val.reshape(-1), val_predictions.reshape(-1))
print(f"Validation MSE: {mse}")

