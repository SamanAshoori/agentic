import pandas as pd
import numpy as np
from sklearn.utils import class_weight
import json

# Load the dataset
df = pd.read_csv('fraudTrain.csv')

# 1. Drop redundant/identifying/leakage risk columns
df = df.drop(['cc_num', 'trans_num', 'job', 'dob'], axis=1)

# 2. Feature Engineering from datetime
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df['month'] = df['trans_date_trans_time'].dt.month

# Drop the original datetime column
df = df.drop('trans_date_trans_time', axis=1)

# 3. Handle class imbalance using class weighting
class_counts = df['is_fraud'].value_counts()
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(df['is_fraud']), y=df['is_fraud'])
class_weights_dict = dict(zip(df['is_fraud'].unique(), class_weights))

# Save class weights to a JSON file
with open('class_weights.json', 'w') as f:
    json.dump({int(k): float(v) for k, v in class_weights_dict.items()}, f)

# Print class balance before handling imbalance
print("Class balance before:", class_counts)

# 4. Drop more redundant columns
df = df.drop(['street', 'city', 'state', 'zip', 'lat', 'long', 'merch_lat', 'merch_long'], axis=1)


# 5. Feature Engineering - Combine category and merchant
df['category_merchant'] = df['category'] + '_' + df['merchant']
df = df.drop(['category', 'merchant'], axis=1)

# 6. Handle class imbalance - this is already implemented via class_weight calculations
# Do NOT apply SMOTE or other oversampling techniques

# Print class balance after handling imbalance (no change to data, just info)
class_counts_after = df['is_fraud'].value_counts()
print("Class balance after:", class_counts_after)


# Print dataset shape
print("Dataset shape:", df.shape)

# Save the cleaned dataset to a CSV file
df.to_csv('cleaned_fraud.csv', index=False)