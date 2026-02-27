import pandas as pd
import numpy as np
import json
import os
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Load data
df = pd.read_csv('data/fraudTrain.csv')

# Cleaning Plan 1: Drop Row ID immediately
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Cleaning Plan 2: Date Conversion
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])

# Cleaning Plan 3: Execute Feature Engineering Steps
# 1. Calculate Distance (Haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['distance_km'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

# 2. Calculate Age
df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

# 3. Extract Time Features
df['hour_of_day'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek

# 4. Binary Encoding (F=0, M=1)
df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# 5. Categorical Encoding (One-hot encode category and state)
df = pd.get_dummies(df, columns=['category', 'state'])

# Cleaning Plan 4: Column Removal (Drop all listed under "Columns To Drop")
cols_to_drop = [
    'cc_num', 'trans_num', 'first', 'last', 'street', 'city', 'zip', 
    'job', 'dob', 'merchant', 'trans_date_trans_time', 'unix_time', 
    'lat', 'long', 'merch_lat', 'merch_long'
]
# Note: 'Unnamed: 0' was already dropped in Plan Step 1
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# Cleaning Plan 5: Handle Missing Values (Median imputation for numerical features)
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Cleaning Plan 6: Feature Scaling
scaler = StandardScaler()
scale_cols = ['amt', 'city_pop', 'distance_km', 'age']
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Handle class imbalance with class weighting
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=df['is_fraud']
)
class_weights_dict = {0: float(weights[0]), 1: float(weights[1])}

# Save class weights
with open('data/class_weights.json', 'w') as f:
    json.dump(class_weights_dict, f)

# Save cleaned data
df.to_csv('data/cleaned_fraud.csv', index=False)