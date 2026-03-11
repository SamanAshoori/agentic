import json 
import os
import datetime
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Any
from scipy import stats as scipy_stats
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# ETL helpers (user-driven, no agent)
# ---------------------------------------------------------------------------

def get_column_details(csv_path: str) -> list[dict[str, Any]]:
    """Return detailed stats for every column in the dataset."""
    df = pd.read_csv(csv_path, nrows=2000)
    full_len = len(pd.read_csv(csv_path, usecols=[0]))

    columns = []
    for col in df.columns:
        s = df[col]
        info: dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "null_count": int(s.isnull().sum()),
            "null_pct": round(float(s.isnull().mean()) * 100, 2),
            "n_unique": int(s.nunique()),
            "total_rows": full_len,
            "sample_values": [str(v) for v in s.dropna().head(5).tolist()],
        }
        if pd.api.types.is_numeric_dtype(s):
            info["stats"] = {
                "mean": round(float(s.mean()), 4) if not s.isnull().all() else None,
                "median": round(float(s.median()), 4) if not s.isnull().all() else None,
                "std": round(float(s.std()), 4) if not s.isnull().all() else None,
                "min": float(s.min()) if not s.isnull().all() else None,
                "max": float(s.max()) if not s.isnull().all() else None,
            }
        else:
            vc = s.value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in vc.items()}
        columns.append(info)
    return columns


def run_target_stats(csv_path: str, target: str) -> dict[str, Any]:
    """Run statistical tests for every column against the target variable."""
    df = pd.read_csv(csv_path, nrows=50000)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    y = df[target]
    results = []
    for col in df.columns:
        if col == target:
            continue
        s = df[col].dropna()
        y_clean = y[s.index]
        entry: dict[str, Any] = {"column": col}
        try:
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
                groups = [s[y_clean == v] for v in y.unique()]
                if len(groups) == 2:
                    stat, p = scipy_stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                    entry["test"] = "Mann-Whitney U"
                else:
                    stat, p = scipy_stats.kruskal(*groups)
                    entry["test"] = "Kruskal-Wallis"
                entry["statistic"] = round(float(stat), 4)
                entry["p_value"] = float(p)
            else:
                ct = pd.crosstab(s, y_clean)
                stat, p, dof, _ = scipy_stats.chi2_contingency(ct)
                entry["test"] = "Chi-squared"
                entry["statistic"] = round(float(stat), 4)
                entry["p_value"] = float(p)
            entry["significant"] = entry["p_value"] < 0.05
        except Exception as e:
            entry["test"] = "error"
            entry["p_value"] = None
            entry["significant"] = False
            entry["error"] = str(e)
        results.append(entry)
    return {"target": target, "tests": results}


def apply_etl_decisions(csv_path: str, decisions) -> dict[str, Any]:
    """Drop user-marked columns, encode strings, save cleaned CSV + class weights."""
    df = pd.read_csv(csv_path)

    columns_to_drop = [c.column for c in decisions.columns if c.decision == "drop"]
    columns_to_keep = [c for c in df.columns if c not in columns_to_drop]
    if decisions.target not in columns_to_keep:
        columns_to_keep.append(decisions.target)

    cleaned = df[columns_to_keep].copy()

    # Encode string/object columns with LabelEncoder
    for col in cleaned.columns:
        if cleaned[col].dtype == "object" or str(cleaned[col].dtype) == "string":
            cleaned[col] = LabelEncoder().fit_transform(cleaned[col].astype(str))

    # Compute class weights
    target_vals = cleaned[decisions.target]
    classes = np.unique(target_vals)
    weights = compute_class_weight("balanced", classes=classes, y=target_vals)
    class_weights = {int(c): round(float(w), 6) for c, w in zip(classes, weights)}

    DATA.mkdir(exist_ok=True)
    cleaned_name = f"cleaned_{Path(csv_path).name}"
    cleaned.to_csv(DATA / cleaned_name, index=False)
    with open(DATA / "class_weights.json", "w") as f:
        json.dump(class_weights, f)

    return {
        "columns_kept": list(columns_to_keep),
        "columns_dropped": columns_to_drop,
        "cleaned_shape": [len(cleaned), len(cleaned.columns)],
        "class_weights": class_weights,
        "cleaned_path": str(DATA / cleaned_name),
    }


def get_dataset_summary(csv_path: str) -> dict[str, Any]:
    df = pd.read_csv(csv_path, nrows=5000)
    full_len = len(pd.read_csv(csv_path, usecols=[0]))
    column_info = []
    for col in df.columns:
        column_info.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_pct": round(float(df[col].isnull().mean()) * 100, 2),
            "n_unique": int(df[col].nunique()),
            "sample_values": [str(v) for v in df[col].dropna().head(3).tolist()],
        })
    return {
        "filename": os.path.basename(csv_path),
        "rows": full_len,
        "columns": len(df.columns),
        "column_info": column_info,
    }


# ---------------------------------------------------------------------------
# Stats stage — correlation, VIF, mutual information, feature selection
# ---------------------------------------------------------------------------

def run_stats(target: str, cleaned_path: str) -> dict[str, Any]:
    """Run full statistical feature analysis on cleaned data."""
    df = pd.read_csv(cleaned_path)

    # Encode any remaining string columns so numeric ops work
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    y = df[target]
    X = df.drop(columns=[target])

    # Correlation — flag pairs > 0.85
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    high_corr_drop = set()
    for col in upper.columns:
        for row in upper.index:
            if upper.loc[row, col] > 0.85:
                high_corr_pairs.append({"a": row, "b": col, "corr": round(float(upper.loc[row, col]), 4)})
                high_corr_drop.add(col)

    # Chi-squared p-values for all features
    p_values = {}
    for col in X.columns:
        try:
            ct = pd.crosstab(X[col], y)
            _, p, _, _ = scipy_stats.chi2_contingency(ct)
            p_values[col] = float(p)
        except Exception:
            p_values[col] = 1.0

    # VIF via sklearn LinearRegression
    vif_data = {}
    for col in X.columns:
        try:
            X_others = X.drop(columns=[col])
            r2 = LinearRegression().fit(X_others, X[col]).score(X_others, X[col])
            vif_data[col] = round(1.0 / (1.0 - r2), 4) if r2 < 1.0 else float("inf")
        except Exception:
            vif_data[col] = float("inf")

    # Mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_dict = {col: round(float(s), 6) for col, s in zip(X.columns, mi_scores)}

    # Build per-feature metrics and auto-select
    feature_metrics = {}
    selected = []
    excluded = {}
    for col in X.columns:
        feature_metrics[col] = {
            "p_value": p_values[col],
            "vif": vif_data[col],
            "mutual_info": mi_dict[col],
        }
        if col in high_corr_drop:
            excluded[col] = "high correlation (>0.85)"
        elif p_values[col] > 0.05:
            excluded[col] = f"not significant (p={p_values[col]:.4f})"
        elif vif_data[col] > 5:
            excluded[col] = f"high VIF ({vif_data[col]:.2f})"
        else:
            selected.append(col)

    # Sort selected by mutual information descending
    selected.sort(key=lambda c: mi_dict.get(c, 0), reverse=True)

    # Save outputs
    output = {
        "selected_features": selected,
        "feature_metrics": feature_metrics,
    }
    with open(DATA / "selected_features.json", "w") as f:
        json.dump(output, f, indent=2)

    return {
        "selected_features": selected,
        "feature_metrics": feature_metrics,
        "excluded": excluded,
        "high_corr_pairs": high_corr_pairs,
        "total_features": len(X.columns),
    }


# ---------------------------------------------------------------------------
# Model stage — train with user-configurable hyperparameters
# ---------------------------------------------------------------------------

def run_model(
    target: str,
    selected_features: list[str],
    cleaned_path: str,
    n_estimators: int = 100,
    max_depth: int = 10,
    class_weight_mode: str = "balanced",
    test_split: float = 0.2,
) -> dict[str, Any]:
    """Train a Random Forest on cleaned data with the given config."""
    df = pd.read_csv(cleaned_path)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    with open(DATA / "class_weights.json") as f:
        raw_weights = json.load(f)
    class_weights = {int(k): v for k, v in raw_weights.items()}

    y = df[target]
    X = df[selected_features]

    # Chronological split (no shuffle)
    split_idx = int(len(df) * (1 - test_split))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    cw = class_weights if class_weight_mode == "balanced" else class_weight_mode

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=cw,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    roc_auc = float(roc_auc_score(y_val, y_prob))
    target_recall = float(recall_score(y_val, y_pred, pos_label=1))

    importances = sorted(
        zip(selected_features, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    # Save model + metrics
    joblib.dump(model, DATA / "model.pkl")
    metrics = {
        "roc_auc": roc_auc,
        "target_recall": target_recall,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "feature_importances": {f: round(float(v), 6) for f, v in importances},
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
    }
    with open(DATA / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_metrics": metrics,
        "model_saved": True,
    }


# ---------------------------------------------------------------------------
# Evaluate stage — threshold tuning on validation or test data
# ---------------------------------------------------------------------------

def run_evaluate(
    target: str,
    selected_features: list[str],
    cleaned_path: str,
    test_csv_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate the trained model. Uses validation split from training data if no test CSV."""
    model = joblib.load(DATA / "model.pkl")

    if test_csv_path and Path(test_csv_path).exists():
        df = pd.read_csv(test_csv_path)
    else:
        df = pd.read_csv(cleaned_path)
        split_idx = int(len(df) * 0.8)
        df = df.iloc[split_idx:]

    # Encode non-numeric columns
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    y_test = df[target]
    # Only use features the model was trained on
    available = [f for f in selected_features if f in df.columns]
    X_test = df[available].values

    y_probs = model.predict_proba(X_test)[:, 1]

    # Default threshold 0.5
    y_pred_default = (y_probs >= 0.5).astype(int)
    cm_default = confusion_matrix(y_test, y_pred_default).tolist()
    report_default = classification_report(y_test, y_pred_default, output_dict=True, zero_division=0)
    roc_auc = float(roc_auc_score(y_test, y_probs))

    # Threshold tuning
    best_f1 = -1.0
    best_threshold = 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        preds = (y_probs >= t).astype(int)
        f = float(f1_score(y_test, preds, pos_label=1))
        if f > best_f1:
            best_f1 = f
            best_threshold = float(t)

    y_pred_opt = (y_probs >= best_threshold).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_opt).tolist()
    report_opt = classification_report(y_test, y_pred_opt, output_dict=True, zero_division=0)

    eval_report = {
        "roc_auc": roc_auc,
        "default_threshold_0.5": {
            "confusion_matrix": cm_default,
            "classification_report": report_default,
            "target_recall": float(report_default.get("1", {}).get("recall", 0)),
        },
        "optimal_threshold_tuning": {
            "optimal_threshold": round(best_threshold, 2),
            "confusion_matrix": cm_opt,
            "classification_report": report_opt,
            "target_recall": float(report_opt.get("1", {}).get("recall", 0)),
            "best_f1_score": round(best_f1, 6),
        },
    }

    with open(DATA / "eval_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    return {"eval_report": eval_report}


# ---------------------------------------------------------------------------
# Scoring stage — probability output with segments
# ---------------------------------------------------------------------------

SEGMENT_BOUNDS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEGMENT_NAMES = ["Low", "Medium", "High", "Very High"]


def run_scoring(
    target: str,
    selected_features: list[str],
    cleaned_path: str,
) -> dict[str, Any]:
    """Score every row in cleaned data: probability + segment columns. Return histogram data."""
    model = joblib.load(DATA / "model.pkl")
    df = pd.read_csv(cleaned_path)

    # Encode non-numeric
    encoders: dict[str, LabelEncoder] = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    available = [f for f in selected_features if f in df.columns]
    X = df[available].values
    y = df[target]

    probs = model.predict_proba(X)[:, 1]

    # Assign segments
    seg_numbers = np.digitize(probs, SEGMENT_BOUNDS[1:], right=False) + 1  # 1-4
    seg_numbers = np.clip(seg_numbers, 1, 4)
    seg_names = [SEGMENT_NAMES[n - 1] for n in seg_numbers]

    # Build output CSV
    df_out = pd.read_csv(cleaned_path)  # original (pre-encode) for readability
    df_out["probability"] = np.round(probs, 6)
    df_out["segment_name"] = seg_names
    df_out["segment_number"] = seg_numbers
    df_out.to_csv(DATA / "scored_output.csv", index=False)

    # Build histogram data for the frontend (binned by probability)
    n_bins = 40
    bin_edges = np.linspace(0, 1, n_bins + 1)
    hist_data = []
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        count_no = int(((y[mask] == 0).sum()))
        count_yes = int(((y[mask] == 1).sum()))
        hist_data.append({
            "bin_start": round(lo, 4),
            "bin_end": round(hi, 4),
            "count_no": count_no,
            "count_yes": count_yes,
            "count_total": count_no + count_yes,
        })

    avg_prob = float(np.mean(probs))

    # Segment summary
    segment_summary = []
    for seg_num in range(1, 5):
        mask = seg_numbers == seg_num
        segment_summary.append({
            "segment_number": seg_num,
            "segment_name": SEGMENT_NAMES[seg_num - 1],
            "count": int(mask.sum()),
            "avg_probability": round(float(probs[mask].mean()), 4) if mask.any() else 0,
            "target_rate": round(float(y[mask].mean()), 4) if mask.any() else 0,
        })

    return {
        "total_rows": len(df),
        "avg_probability": round(avg_prob, 4),
        "histogram": hist_data,
        "segments": segment_summary,
        "output_path": str(DATA / "scored_output.csv"),
    }


# ---------------------------------------------------------------------------
# Descriptives — SPSS-style class comparison
# ---------------------------------------------------------------------------

def run_descriptives(csv_path: str, target: str) -> dict[str, Any]:
    """Per-column descriptive statistics split by target class."""
    df = pd.read_csv(csv_path, nrows=50000)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found")

    y = df[target]

    # Target distribution
    vc = y.value_counts().sort_index()
    target_dist = [
        {"value": str(k), "count": int(v), "pct": round(100 * v / len(y), 2)}
        for k, v in vc.items()
    ]

    classes = sorted(y.unique())
    numeric_comparisons = []
    crosstabs = []

    for col in df.columns:
        if col == target:
            continue
        s = df[col].dropna()
        y_clean = y[s.index]

        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 5:
            by_class = []
            for cls in classes:
                subset = s[y_clean == cls]
                if len(subset) == 0:
                    continue
                by_class.append({
                    "class": str(cls),
                    "n": int(len(subset)),
                    "mean": round(float(subset.mean()), 4),
                    "sd": round(float(subset.std()), 4),
                    "median": round(float(subset.median()), 4),
                    "min": round(float(subset.min()), 4),
                    "max": round(float(subset.max()), 4),
                })
            numeric_comparisons.append({"column": col, "by_class": by_class})
            if len(numeric_comparisons) >= 20:
                break

    for col in df.columns:
        if col == target or (pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 5):
            continue
        s = df[col].dropna().astype(str)
        y_clean = y[s.index]
        try:
            ct = pd.crosstab(s, y_clean)
            chi2, p, dof, _ = scipy_stats.chi2_contingency(ct)
            n = ct.values.sum()
            cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if min(ct.shape) > 1 else 0.0
            top_cats = ct.index[:8].tolist()
            rows = []
            for cat in top_cats:
                row = {"category": str(cat)}
                for cls in ct.columns:
                    row[str(cls)] = int(ct.loc[cat, cls])
                row["total"] = int(ct.loc[cat].sum())
                rows.append(row)
            crosstabs.append({
                "column": col,
                "chi2": round(float(chi2), 4),
                "p_value": float(p),
                "cramers_v": round(float(cramers_v), 4),
                "n_categories": int(s.nunique()),
                "classes": [str(c) for c in ct.columns],
                "rows": rows,
            })
        except Exception:
            pass
        if len(crosstabs) >= 8:
            break

    return {
        "target": target,
        "n": int(len(df)),
        "target_distribution": target_dist,
        "numeric_comparisons": numeric_comparisons,
        "crosstabs": crosstabs,
    }


# ---------------------------------------------------------------------------
# Logistic Regression — coefficients, Wald stats, model fit, Hosmer-Lemeshow
# ---------------------------------------------------------------------------

def _hosmer_lemeshow(y_true: np.ndarray, y_prob: np.ndarray, n_groups: int = 10) -> dict[str, Any]:
    n = len(y_true)
    order = np.argsort(y_prob)
    y_s = y_true[order]
    p_s = y_prob[order]
    group_indices = np.array_split(np.arange(n), n_groups)
    hl_stat = 0.0
    group_data = []
    for i, g in enumerate(group_indices):
        obs_pos = int(y_s[g].sum())
        obs_neg = len(g) - obs_pos
        exp_pos = float(p_s[g].sum())
        exp_neg = len(g) - exp_pos
        if exp_pos > 0:
            hl_stat += (obs_pos - exp_pos) ** 2 / exp_pos
        if exp_neg > 0:
            hl_stat += (obs_neg - exp_neg) ** 2 / exp_neg
        group_data.append({
            "group": i + 1,
            "n": len(g),
            "observed_pos": obs_pos,
            "expected_pos": round(exp_pos, 2),
        })
    df_hl = n_groups - 2
    p_value = float(scipy_stats.chi2.sf(hl_stat, df=df_hl))
    return {
        "chi2": round(float(hl_stat), 4),
        "df": df_hl,
        "p_value": round(p_value, 4),
        "interpretation": "Good fit (p > 0.05)" if p_value > 0.05 else "Poor fit (p ≤ 0.05)",
        "groups": group_data,
    }


def run_logistic_regression(
    target: str,
    selected_features: list[str],
    cleaned_path: str,
    test_split: float = 0.2,
    max_iter: int = 1000,
) -> dict[str, Any]:
    """Fit logistic regression and return SPSS-style output."""
    df = pd.read_csv(cleaned_path)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    y = df[target].values
    X = df[selected_features].values

    split_idx = int(len(df) * (1 - test_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SklearnLR(C=1e4, max_iter=max_iter, solver="lbfgs", random_state=42)
        model.fit(X_train, y_train)

    p_hat = model.predict_proba(X_train)[:, 1]

    # Subsample for Hessian if large dataset
    if len(X_train) > 100_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), 100_000, replace=False)
        X_sub = X_train[idx]
        p_sub = p_hat[idx]
    else:
        X_sub = X_train
        p_sub = p_hat

    X_aug = np.column_stack([np.ones(len(X_sub)), X_sub])
    W = p_sub * (1 - p_sub)
    info_matrix = (X_aug * W[:, np.newaxis]).T @ X_aug

    try:
        cov_matrix = np.linalg.inv(info_matrix)
        ses = np.sqrt(np.clip(np.diag(cov_matrix), 0, None))
    except np.linalg.LinAlgError:
        ses = np.full(X_aug.shape[1], np.nan)

    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    with np.errstate(invalid="ignore", divide="ignore"):
        wald = np.where(ses > 0, (coefs / ses) ** 2, np.nan)
    p_values = np.where(np.isfinite(wald), scipy_stats.chi2.sf(wald, df=1), np.nan)
    ci_lower = coefs - 1.96 * ses
    ci_upper = coefs + 1.96 * ses

    names = ["(Intercept)"] + selected_features
    coefficients = []
    for i, name in enumerate(names):
        coefficients.append({
            "variable": name,
            "B": round(float(coefs[i]), 4),
            "SE": round(float(ses[i]), 4) if np.isfinite(ses[i]) else None,
            "wald": round(float(wald[i]), 4) if np.isfinite(wald[i]) else None,
            "df": 1,
            "p_value": float(p_values[i]) if np.isfinite(p_values[i]) else None,
            "exp_B": round(float(np.exp(coefs[i])), 4),
            "ci_lower_95": round(float(np.exp(ci_lower[i])), 4) if np.isfinite(ci_lower[i]) else None,
            "ci_upper_95": round(float(np.exp(ci_upper[i])), 4) if np.isfinite(ci_upper[i]) else None,
        })

    # Log-likelihood and pseudo R²
    eps = 1e-10
    ll_model = float(np.sum(y_train * np.log(p_hat + eps) + (1 - y_train) * np.log(1 - p_hat + eps)))
    p_null = float(np.mean(y_train))
    ll_null = float(len(y_train) * (p_null * np.log(p_null + eps) + (1 - p_null) * np.log(1 - p_null + eps)))
    n_train = len(y_train)
    k = len(selected_features) + 1
    neg2ll = -2 * ll_model
    aic = neg2ll + 2 * k
    bic = neg2ll + k * np.log(n_train)
    cox_snell = 1 - np.exp((2 / n_train) * (ll_null - ll_model))
    nagelkerke_denom = 1 - np.exp((2 / n_train) * ll_null)
    nagelkerke = cox_snell / nagelkerke_denom if nagelkerke_denom > 0 else float("nan")

    model_fit = {
        "neg2_log_likelihood": round(neg2ll, 4),
        "aic": round(aic, 4),
        "bic": round(bic, 4),
        "cox_snell_r2": round(float(cox_snell), 4),
        "nagelkerke_r2": round(float(nagelkerke), 4),
        "n_train": n_train,
        "n_val": len(y_val),
    }

    hl_result = _hosmer_lemeshow(y_train, p_hat)

    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_pred_val = (y_prob_val >= 0.5).astype(int)
    cm = confusion_matrix(y_val, y_pred_val)
    roc_auc = float(roc_auc_score(y_val, y_prob_val))
    report = classification_report(y_val, y_pred_val, output_dict=True, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    classification_table = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "pct_correct_0": round(100 * tn / (tn + fp), 1) if (tn + fp) > 0 else 0.0,
        "pct_correct_1": round(100 * tp / (tp + fn), 1) if (tp + fn) > 0 else 0.0,
        "overall_pct_correct": round(100 * (tn + tp) / (tn + fp + fn + tp), 1),
    }

    joblib.dump(model, DATA / "model.pkl")

    metrics = {
        "model_type": "LogisticRegression",
        "roc_auc": roc_auc,
        "target_recall": float(recall_score(y_val, y_pred_val, pos_label=1)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "coefficients": coefficients,
        "model_fit": model_fit,
        "hosmer_lemeshow": hl_result,
        "classification_table": classification_table,
        "train_shape": [len(X_train), len(selected_features)],
        "val_shape": [len(X_val), len(selected_features)],
    }
    with open(DATA / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {"model_metrics": metrics, "model_saved": True}


# ---------------------------------------------------------------------------
# HTML Report generation
# ---------------------------------------------------------------------------

def generate_html_report(data: dict[str, Any]) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    session = data.get("session", {})
    confirmed = session.get("confirmed_outputs", {})
    metrics = data.get("model_metrics") or {}
    eval_data = (data.get("eval_report") or {}).get("eval_report", {})
    features_data = data.get("selected_features") or {}

    dataset_info = confirmed.get("etl", {})
    target = dataset_info.get("target", "—")
    cols_kept = dataset_info.get("columns_kept", [])
    cols_dropped = dataset_info.get("columns_dropped", [])
    selected_feats = features_data.get("selected_features", confirmed.get("stats", {}).get("selected_features", []))

    model_type = metrics.get("model_type", "RandomForest")
    roc_auc = metrics.get("roc_auc")
    model_fit = metrics.get("model_fit", {})
    coefficients = metrics.get("coefficients", [])
    importances = metrics.get("feature_importances", {})
    hl = metrics.get("hosmer_lemeshow", {})
    cl_table = metrics.get("classification_table", {})

    opt = eval_data.get("optimal_threshold_tuning", {})
    eval_threshold = opt.get("optimal_threshold", "—")
    eval_recall = opt.get("target_recall")
    eval_precision = (opt.get("classification_report") or {}).get("1", {}).get("precision")
    eval_f1 = opt.get("best_f1_score")

    def fmt_p(v):
        if v is None:
            return "—"
        if v < 0.001:
            return "< .001"
        return f"{v:.4f}"

    def fmt_f(v, d=4):
        return f"{v:.{d}f}" if v is not None else "—"

    def sig_star(v):
        if v is None:
            return ""
        if v < 0.001:
            return " ***"
        if v < 0.01:
            return " **"
        if v < 0.05:
            return " *"
        return ""

    # Coefficient rows
    coef_rows = ""
    for c in coefficients:
        p = c.get("p_value")
        star = sig_star(p)
        bold_open = "<strong>" if (p is not None and p < 0.05) else ""
        bold_close = "</strong>" if bold_open else ""
        coef_rows += f"""
        <tr>
          <td>{bold_open}{c['variable']}{bold_close}</td>
          <td class="num">{fmt_f(c.get('B'))}</td>
          <td class="num">{fmt_f(c.get('SE'))}</td>
          <td class="num">{fmt_f(c.get('wald'))}</td>
          <td class="num">1</td>
          <td class="num sig">{fmt_p(p)}{star}</td>
          <td class="num">{fmt_f(c.get('exp_B'))}</td>
          <td class="num">{fmt_f(c.get('ci_lower_95'))}</td>
          <td class="num">{fmt_f(c.get('ci_upper_95'))}</td>
        </tr>"""

    coef_section = ""
    if coefficients:
        coef_section = f"""
    <div class="section">
      <h2>Variables in the Equation</h2>
      <table>
        <thead>
          <tr>
            <th></th>
            <th class="num">B</th>
            <th class="num">S.E.</th>
            <th class="num">Wald</th>
            <th class="num">df</th>
            <th class="num">Sig.</th>
            <th class="num">Exp(B)</th>
            <th class="num">95% CI Lower</th>
            <th class="num">95% CI Upper</th>
          </tr>
        </thead>
        <tbody>{coef_rows}</tbody>
      </table>
      <p class="footnote">* p &lt; .05 &nbsp;&nbsp; ** p &lt; .01 &nbsp;&nbsp; *** p &lt; .001</p>
    </div>"""

    model_fit_section = ""
    if model_fit:
        if model_type == "LogisticRegression":
            model_fit_section = f"""
    <div class="section">
      <h2>Model Fit Statistics</h2>
      <table>
        <thead><tr><th>−2 Log Likelihood</th><th class="num">AIC</th><th class="num">BIC</th><th class="num">Cox &amp; Snell R²</th><th class="num">Nagelkerke R²</th></tr></thead>
        <tbody>
          <tr>
            <td class="num">{fmt_f(model_fit.get('neg2_log_likelihood'), 3)}</td>
            <td class="num">{fmt_f(model_fit.get('aic'), 3)}</td>
            <td class="num">{fmt_f(model_fit.get('bic'), 3)}</td>
            <td class="num">{fmt_f(model_fit.get('cox_snell_r2'))}</td>
            <td class="num">{fmt_f(model_fit.get('nagelkerke_r2'))}</td>
          </tr>
        </tbody>
      </table>
    </div>"""

    hl_section = ""
    if hl:
        hl_section = f"""
    <div class="section">
      <h2>Hosmer and Lemeshow Test</h2>
      <table>
        <thead><tr><th class="num">Chi-square</th><th class="num">df</th><th class="num">Sig.</th><th>Interpretation</th></tr></thead>
        <tbody>
          <tr>
            <td class="num">{fmt_f(hl.get('chi2'))}</td>
            <td class="num">{hl.get('df', '—')}</td>
            <td class="num">{fmt_p(hl.get('p_value'))}</td>
            <td>{hl.get('interpretation', '—')}</td>
          </tr>
        </tbody>
      </table>
    </div>"""

    classification_table_section = ""
    if cl_table:
        classification_table_section = f"""
    <div class="section">
      <h2>Classification Table</h2>
      <table>
        <thead>
          <tr><th rowspan="2">Observed</th><th colspan="2" class="num">Predicted</th><th class="num">% Correct</th></tr>
          <tr><th class="num">0</th><th class="num">1</th><th></th></tr>
        </thead>
        <tbody>
          <tr><td>0</td><td class="num">{cl_table.get('tn', '—'):,}</td><td class="num">{cl_table.get('fp', '—'):,}</td><td class="num">{fmt_f(cl_table.get('pct_correct_0'), 1)}%</td></tr>
          <tr><td>1</td><td class="num">{cl_table.get('fn', '—'):,}</td><td class="num">{cl_table.get('tp', '—'):,}</td><td class="num">{fmt_f(cl_table.get('pct_correct_1'), 1)}%</td></tr>
          <tr class="total-row"><td><strong>Overall %</strong></td><td colspan="2"></td><td class="num"><strong>{fmt_f(cl_table.get('overall_pct_correct'), 1)}%</strong></td></tr>
        </tbody>
      </table>
    </div>"""

    # Feature importances (RF)
    importances_section = ""
    if importances and model_type == "RandomForest":
        top = list(importances.items())[:15]
        max_imp = top[0][1] if top else 1
        rows_html = ""
        for feat, imp in top:
            pct = round(imp * 100, 2)
            bar_w = round(100 * imp / max_imp, 1)
            rows_html += f'<tr><td>{feat}</td><td class="num">{pct}%</td><td><div class="bar" style="width:{bar_w}%"></div></td></tr>'
        importances_section = f"""
    <div class="section">
      <h2>Feature Importances</h2>
      <table>
        <thead><tr><th>Feature</th><th class="num">Importance</th><th>Relative</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

    features_section = ""
    if selected_feats:
        feat_rows = "".join(f"<tr><td>{f}</td></tr>" for f in selected_feats)
        features_section = f"""
    <div class="section">
      <h2>Selected Features ({len(selected_feats)})</h2>
      <table>
        <tbody>{feat_rows}</tbody>
      </table>
    </div>"""

    eval_section = ""
    if opt:
        eval_section = f"""
    <div class="section">
      <h2>Evaluation — Optimal Threshold</h2>
      <table>
        <thead><tr><th>Threshold</th><th class="num">ROC-AUC</th><th class="num">Recall (1)</th><th class="num">Precision (1)</th><th class="num">F1 (1)</th></tr></thead>
        <tbody>
          <tr>
            <td>{eval_threshold}</td>
            <td class="num">{fmt_f(roc_auc)}</td>
            <td class="num">{fmt_f(eval_recall)}</td>
            <td class="num">{fmt_f(eval_precision)}</td>
            <td class="num">{fmt_f(eval_f1)}</td>
          </tr>
        </tbody>
      </table>
    </div>"""

    etl_section = ""
    if cols_kept or cols_dropped:
        kept_rows = "".join(f"<tr><td>{c}</td><td class='badge-keep'>Keep</td></tr>" for c in cols_kept if c != target)
        drop_rows = "".join(f"<tr><td>{c}</td><td class='badge-drop'>Drop</td></tr>" for c in cols_dropped)
        etl_section = f"""
    <div class="section">
      <h2>ETL — Column Decisions</h2>
      <p><strong>Target:</strong> {target}</p>
      <table>
        <thead><tr><th>Column</th><th>Decision</th></tr></thead>
        <tbody>{kept_rows}{drop_rows}</tbody>
      </table>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>ML Pipeline Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color: #1a1a2e; background: #f5f7fa; }}
    .page {{ max-width: 960px; margin: 0 auto; padding: 2rem 2rem 4rem; }}
    .report-header {{ background: #1e3a5f; color: white; padding: 1.5rem 2rem; border-radius: 8px; margin-bottom: 1.5rem; }}
    .report-header h1 {{ font-size: 1.4rem; font-weight: 700; margin-bottom: 0.25rem; }}
    .report-header .meta {{ font-size: 0.8rem; opacity: 0.75; }}
    .section {{ background: white; border: 1px solid #dde3ee; border-radius: 6px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }}
    .section h2 {{ font-size: 0.95rem; font-weight: 700; color: #1e3a5f; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.4rem; margin-bottom: 0.9rem; text-transform: uppercase; letter-spacing: 0.04em; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{ background: #eef2f9; color: #334155; font-weight: 600; padding: 0.45rem 0.6rem; text-align: left; border-bottom: 1px solid #c8d4e8; font-size: 11px; }}
    td {{ padding: 0.4rem 0.6rem; border-bottom: 1px solid #eef0f4; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f8faff; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; font-family: 'Consolas', monospace; }}
    .sig {{ font-weight: 600; }}
    .total-row td {{ background: #f0f4fb; }}
    .footnote {{ font-size: 11px; color: #64748b; margin-top: 0.5rem; }}
    .badge-keep {{ color: #15803d; font-weight: 600; }}
    .badge-drop {{ color: #b91c1c; font-weight: 600; }}
    .bar {{ height: 12px; background: #3b82f6; border-radius: 2px; }}
    p {{ margin-bottom: 0.5rem; font-size: 13px; }}
    @media print {{
      body {{ background: white; }}
      .page {{ padding: 0; }}
      .section {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
<div class="page">
  <div class="report-header">
    <h1>ML Pipeline Report</h1>
    <div class="meta">Target: {target} &nbsp;|&nbsp; Model: {model_type} &nbsp;|&nbsp; Generated: {ts}</div>
  </div>
  {etl_section}
  {features_section}
  {model_fit_section}
  {coef_section}
  {hl_section}
  {classification_table_section}
  {importances_section}
  {eval_section}
</div>
</body>
</html>"""
