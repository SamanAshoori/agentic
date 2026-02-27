import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# Ensure the data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Load data
df = pd.read_csv('data/fraudTrain.csv')

print("Original shape:", df.shape)
print("Original class balance:")
print(df['is_fraud'].value_counts(normalize=True))

# Convert columns to datetime for calculations
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])

# 1. Haversine distance between cardholder and merchant
def haversine_np(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

df['distance'] = haversine_np(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

# 2. Datetime feature extraction
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df['month'] = df['trans_date_trans_time'].dt.month

# 3. Age calculation
df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

# 4. Drop identifying, redundant and leakage columns
columns_to_drop = [
    'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last',
    'street', 'city', 'zip', 'lat', 'long', 'merch_lat', 'merch_long',
    'job', 'dob', 'trans_num', 'unix_time'
]
df.drop(columns=columns_to_drop, inplace=True)

# 5. Encode categorical columns (keep human readable, no one-hot)
le = LabelEncoder()
for col in ['category', 'gender', 'state']:
    df[col] = le.fit_transform(df[col].astype(str))

# 6. Drop any remaining nulls
df.dropna(inplace=True)

# 7. Compute class weights (no SMOTE, no scaling)
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=df['is_fraud']
)
class_weights_dict = {
    "0": float(weights[0]),
    "1": float(weights[1])
}

print("\nCleaned shape:", df.shape)
print("Cleaned class balance:")
print(df['is_fraud'].value_counts(normalize=True))
print("\nColumns:", df.columns.tolist())
print("Class weights:", class_weights_dict)

# Save outputs
df.to_csv('data/cleaned_fraud.csv', index=False)
with open('data/class_weights.json', 'w') as f:
    json.dump(class_weights_dict, f)

print("\n✅ Saved cleaned_fraud.csv and class_weights.json")