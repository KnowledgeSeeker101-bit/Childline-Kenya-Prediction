"""
Childline Kenya Call Volume Prediction
Simple but effective pipeline using calendar features only
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📞 CHILDLINE KENYA CALL VOLUME PREDICTION")
print("="*80)

# Load data
print("\n📁 Loading Data...")
train_data = pd.read_csv('data/train.csv')
subs = pd.read_csv('data/Sample_Submission.csv')
print(f"✓ Train: {train_data.shape}")

# Preprocess
print("\n📊 Preprocessing...")
train_data['calldate'] = pd.to_datetime(train_data['calldate'])
train = train_data.groupby([train_data['calldate'].dt.floor('H')]).size().reset_index()
train.columns = ['datetime', 'num_calls']
train = train.set_index('datetime').sort_index()

subs['datetime'] = pd.to_datetime(subs['time_index'], format='%Y%m%d%H')
test = subs.set_index('datetime').sort_index()
test['num_calls'] = np.nan

print(f"✓ Training data: {train.shape}")

# Create features
print("\n🔧 Creating Features...")

def create_ts_features(df):
    df = df.copy()
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    return df[['hour', 'dayofweek', 'dayofyear', 'dayofmonth', 'weekofyear']]

X_train = create_ts_features(train)
y_train = train['num_calls']
X_test = create_ts_features(test)

print(f"✓ Features: {list(X_train.columns)}")

# Train model
print("\n🤖 Training XGBoost...")
model = xgb.XGBRegressor(
    n_estimators=42,
    learning_rate=0.11,
    gamma=0,
    subsample=0.74,
    colsample_bytree=1.0,
    max_depth=7,
    random_state=42
)
model.fit(X_train, y_train, verbose=False)
print("✓ Model trained")

# Evaluate
print("\n📊 Training Performance...")
train_pred = model.predict(X_train)
train_pred = np.clip(train_pred, 0, None)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_mae = mean_absolute_error(y_train, train_pred)
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE: {train_mae:.4f}")

# Predict
print("\n🔮 Making Predictions...")
predictions = model.predict(X_test)
predictions = np.clip(predictions, 0, None)

# Save submission
submission = pd.DataFrame({
    'time_index': subs['time_index'],
    'number_of_calls': predictions
})
submission.to_csv('outputs/submission.csv', index=False)
print("✓ Saved: outputs/submission.csv")

# Analysis
print("\n📊 Analysis:")
print(f"  Predictions - Mean: {predictions.mean():.2f}, Std: {predictions.std():.2f}")
print(f"  Training - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
print(f"  Ratio: {predictions.mean() / y_train.mean():.3f}")

# Feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
importance.to_csv('outputs/feature_importance.csv', index=False)

print("\n" + "="*80)
print("✅ PIPELINE COMPLETE")
print("="*80)
