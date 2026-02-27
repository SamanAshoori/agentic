import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def haversine_np(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def evaluate_model():
    # 1. Load model
    model = joblib.load('data/model.pkl')

    # 2. Load raw test data
    df = pd.read_csv('data/fraudTest.csv')
    
    # 3. Preprocessing
    # Separate target
    y_test = df['is_fraud']
    
    # Pre-calculations before dropping columns
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    
    # Extract time features
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    
    # Compute age
    df['age'] = (df['trans_date_trans_time'].dt.year - df['dob'].dt.year)
    
    # Compute haversine distance
    df['distance'] = haversine_np(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    
    # Drop specified columns
    cols_to_drop = [
        'Unnamed: 0', 'trans_num', 'cc_num', 'unix_time', 'first', 'last', 
        'street', 'city', 'zip', 'lat', 'long', 'merch_lat', 'merch_long', 
        'job', 'dob', 'merchant', 'trans_date_trans_time'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Label Encoding for categorical features
    le = LabelEncoder()
    for col in ['category', 'gender', 'state']:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # 4. Load selected features
    with open('data/selected_features.json', 'r') as f:
        feature_data = json.load(f)
    selected_features = feature_data["selected_features"] if isinstance(feature_data, dict) else feature_data
    
    # Filter features for X_test
    X_test = df[selected_features]
    
    # 6. Generate predictions
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_probs >= 0.5).astype(int)
    
    # 7. Evaluate at default threshold (0.5)
    report_default = classification_report(y_test, y_pred_default, output_dict=True)
    cm_default = confusion_matrix(y_test, y_pred_default).tolist()
    roc_auc = roc_auc_score(y_test, y_probs)
    
    # 8. Tune decision threshold (0.1 to 0.9 in steps of 0.05)
    best_f1 = -1
    best_threshold = 0.5
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    for threshold in thresholds:
        current_preds = (y_probs >= threshold).astype(int)
        current_f1 = f1_score(y_test, current_preds)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
            
    # 9. Re-evaluate at optimal threshold
    y_pred_opt = (y_probs >= best_threshold).astype(int)
    report_opt = classification_report(y_test, y_pred_opt, output_dict=True)
    cm_opt = confusion_matrix(y_test, y_pred_opt).tolist()
    
    # 10. Save to data/eval_report.json
    eval_report = {
        "roc_auc": float(roc_auc),
        "default_threshold_0.5": {
            "confusion_matrix": cm_default,
            "classification_report": report_default,
            "fraud_recall": report_default["1"]["recall"]
        },
        "optimal_threshold_tuning": {
            "optimal_threshold": float(best_threshold),
            "confusion_matrix": cm_opt,
            "classification_report": report_opt,
            "fraud_recall": report_opt["1"]["recall"],
            "best_f1_score": float(best_f1)
        }
    }
    
    with open('data/eval_report.json', 'w') as f:
        json.dump(eval_report, f, indent=4)

if __name__ == "__main__":
    evaluate_model()