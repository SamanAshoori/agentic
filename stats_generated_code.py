import pandas as pd
import numpy as np
import json
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_classif

def main():
    # 1. Load the cleaned dataset
    try:
        df = pd.read_csv('data/cleaned_fraud.csv')
    except FileNotFoundError:
        print("Error: data/cleaned_fraud.csv not found.")
        return

    # Encode ALL string/object columns using LabelEncoder
    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Define target and predictors
    target = 'is_fraud'
    if target not in df.columns:
        print(f"Error: Target column {target} not found in dataset.")
        return

    X = df.drop(columns=[target])
    y = df[target]
    feature_names = X.columns.tolist()

    # 2. Compute correlation matrix and flag feature pairs with correlation > 0.85
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.85)]
    
    # 3. Run chi-squared tests for features against is_fraud
    # Using scipy.stats.chi2_contingency
    p_values = {}
    for col in feature_names:
        contingency_table = pd.crosstab(df[col], y)
        # Handle cases with only one category to avoid errors in chi2
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            p_values[col] = 1.0
        else:
            _, p, _, _ = chi2_contingency(contingency_table)
            p_values[col] = p

    # 4. Compute VIF using sklearn LinearRegression approach
    # vif = 1 / (1 - R^2)
    vif_scores = {}
    for feature in feature_names:
        X_feature = X[[feature]]
        X_others = X.drop(columns=[feature])
        
        model = LinearRegression()
        model.fit(X_others, X_feature)
        r_squared = model.score(X_others, X_feature)
        
        if r_squared >= 1.0:
            vif = float('inf')
        else:
            vif = 1 / (1 - r_squared)
        vif_scores[feature] = vif

    # 5. Rank features using mutual_info_classif from sklearn
    mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    mi_ranking = dict(zip(feature_names, mi_scores))

    # 6. Final Feature Selection Logic
    # Criteria:
    # - Not in the highly correlated list (one from each pair)
    # - P-value <= 0.05
    # - VIF <= 5
    selected_features = []
    reasons_rejected = {}

    for feature in feature_names:
        if feature in to_drop_corr:
            reasons_rejected[feature] = "High Correlation (>0.85)"
            continue
        if p_values[feature] > 0.05:
            reasons_rejected[feature] = f"Statistically Insignificant (p={p_values[feature]:.4f})"
            continue
        if vif_scores[feature] > 5:
            reasons_rejected[feature] = f"High Multicollinearity (VIF={vif_scores[feature]:.2f})"
            continue
        
        selected_features.append(feature)

    # Sort selected features by Mutual Information score for the final ranking
    selected_features.sort(key=lambda x: mi_ranking[x], reverse=True)

    # 7. Save recommended feature list to data/selected_features.json
    output_data = {
        "selected_features": selected_features,
        "feature_metrics": {
            f: {
                "p_value": p_values[f],
                "vif": vif_scores[f],
                "mutual_info": mi_ranking[f]
            } for f in selected_features
        }
    }

    with open('data/selected_features.json', 'w') as f:
        json.dump(output_data, f, indent=4)

    # 8. Print Summary Report
    print("-" * 60)
    print("STATISTICAL VALIDATION & FEATURE SELECTION REPORT")
    print("-" * 60)
    print(f"Total features evaluated: {len(feature_names)}")
    print(f"Features selected: {len(selected_features)}")
    print(f"Features rejected: {len(reasons_rejected)}")
    print("-" * 60)
    print(f"{'Feature':<25} | {'MI Score':<10} | {'VIF':<8} | {'P-Value':<8}")
    print("-" * 60)
    for feat in selected_features:
        print(f"{feat:<25} | {mi_ranking[feat]:.4f}     | {vif_scores[feat]:.2f}     | {p_values[feat]:.4f}")
    
    if reasons_rejected:
        print("\nREJECTED FEATURES SUMMARY:")
        for feat, reason in reasons_rejected.items():
            print(f"- {feat}: {reason}")
    
    print("-" * 60)
    print("Results saved to data/selected_features.json")

if __name__ == "__main__":
    main()