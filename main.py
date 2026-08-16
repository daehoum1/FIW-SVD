import gc
import os
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    RandomForestRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import StackingClassifier, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--kappa_list", type=str, default="1.0",
                    help="comma-separated kappa exponents (e.g., '0.8,1.0,1.2')")
parser.add_argument("--run_smote", action="store_true", help="include SMOTE runs for classification")
parser.add_argument("--save_dir", type=str, default="./", help="directory to save CSV files")
args, _ = parser.parse_known_args()

KAPPA_LIST = [float(x.strip()) for x in args.kappa_list.split(",") if x.strip()]
RUN_SMOTE = args.run_smote
SAVE_DIR = args.save_dir
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_CLS = os.path.join(SAVE_DIR, "classification_results_10.csv")
CSV_REG = os.path.join(SAVE_DIR, "regression_results_10.csv")

print(f"[INFO] Kappa sweep: {KAPPA_LIST}")
print(f"[INFO] SMOTE: {RUN_SMOTE}")
print(f"[INFO] Save dir: {SAVE_DIR}")

RANDOM_STATE = 2025
np.random.seed(RANDOM_STATE)
EPS = 1e-8

df = pd.read_csv("./city_day.csv")
use_cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3',
            'Benzene','Toluene','Xylene','AQI','AQI_Bucket','City','Date']
use_cols = [c for c in use_cols if c in df.columns]
df0 = df.loc[:, use_cols].copy()
if 'Date' in df0.columns:
    df0['Date'] = pd.to_datetime(df0['Date'], errors='coerce')

pollutants = [c for c in ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3',
                          'Benzene','Toluene','Xylene'] if c in df0.columns]

df1 = df0.dropna(subset=['AQI','AQI_Bucket']).copy()
all_nan_mask = df1[pollutants].isna().all(axis=1)
df1 = df1.loc[~all_nan_mask].copy()

X_raw = df1[pollutants].astype(np.float32)
y_cls_raw = df1['AQI_Bucket'].copy()
y_reg_raw = df1['AQI'].copy()

def iqr_to_nan(df_in: pd.DataFrame, cols=None, whisker: float = 1.5):
    df_out = df_in.copy()
    cols = cols or df_out.columns.tolist()
    for c in cols:
        s = df_out[c]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if np.isclose(iqr, 0.0):
            continue
        ll, ul = q1 - whisker*iqr, q3 + whisker*iqr
        df_out.loc[(s < ll) | (s > ul), c] = np.nan
    return df_out

X_iqr = iqr_to_nan(X_raw, cols=pollutants, whisker=1.5).astype(np.float32)

imp_knn = KNNImputer(n_neighbors=45)
imp_mice = IterativeImputer(random_state=RANDOM_STATE, max_iter=10, tol=1e-3, initial_strategy='median')

D1_full = pd.DataFrame(imp_knn.fit_transform(X_iqr), columns=pollutants, index=X_iqr.index).astype(np.float32)
D1_full.columns = [f"{c}_knn" for c in pollutants]
D2_full = pd.DataFrame(imp_mice.fit_transform(X_iqr), columns=pollutants, index=X_iqr.index).astype(np.float32)
D2_full.columns = [f"{c}_mice" for c in pollutants]

def get_xgb_importance(X, y, task, random_state=2025):
    if task == "cls":
        model = XGBClassifier(n_estimators=300, max_depth=6, random_state=random_state,
                              objective='multi:softprob', eval_metric='mlogloss', tree_method='hist')
    else:
        model = XGBRegressor(n_estimators=300, max_depth=6, random_state=random_state, tree_method='hist')
    model.fit(X, y)
    imp = model.feature_importances_
    imp = imp / (imp.sum() + EPS)
    return pd.Series(imp, index=X.columns)

def svd_impute_weighted(X, weights, rank=4, n_iter=3, kappa_exp=1.0, random_state=2025):
    w = weights.reindex(X.columns).fillna(0.0).astype(float)
    w = w / (w.sum() + EPS)
    scale = (w + EPS) ** (kappa_exp / 2.0)
    Xw = X * scale
    X_fill = Xw.fillna(Xw.median())
    rng = np.random.RandomState(random_state)
    for _ in range(n_iter):
        mu = X_fill.mean(axis=0)
        A = X_fill - mu
        n_comp = max(1, min(rank, A.shape[1]-1))
        svd = TruncatedSVD(n_components=n_comp, random_state=rng)
        Z = svd.fit_transform(A)
        Ahat = svd.inverse_transform(Z)
        X_hat = pd.DataFrame(Ahat, index=X.index, columns=X.columns) + mu
        X_fill = Xw.where(~Xw.isna(), X_hat)
    X_rec = X_fill / scale
    return X_rec.astype(np.float32)

le = LabelEncoder()
y_cls = le.fit_transform(y_cls_raw)

def make_classifiers(num_classes: int):
    rf  = RandomForestClassifier(n_estimators=300, max_depth=18, n_jobs=-1, random_state=RANDOM_STATE)
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                        colsample_bytree=0.8, tree_method='hist', random_state=RANDOM_STATE,
                        objective='multi:softprob', num_class=num_classes, eval_metric='mlogloss')
    et  = ExtraTreesClassifier(n_estimators=500, max_depth=18, n_jobs=-1, random_state=RANDOM_STATE)
    stk = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('et', et)],
        final_estimator=LogisticRegression(max_iter=1000), n_jobs=-1
    )
    return {"Random Forest": rf, "XGBoost": xgb, "ExtraTree": et, "Stacking": stk}

def make_regressors():
    rf  = RandomForestRegressor(n_estimators=300, max_depth=18, n_jobs=-1, random_state=RANDOM_STATE)
    xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                       colsample_bytree=0.8, tree_method='hist', random_state=RANDOM_STATE)
    et  = ExtraTreesRegressor(n_estimators=500, max_depth=18, n_jobs=-1, random_state=RANDOM_STATE)
    stk = StackingRegressor(
        estimators=[('rf', rf), ('xgb', xgb), ('et', et)],
        final_estimator=LinearRegression(), n_jobs=-1
    )
    return {"Random Forest": rf, "XGBoost": xgb, "ExtraTree": et, "Stacking": stk}

@dataclass
class ClsMetrics:
    Precision_macro: float
    Precision_weighted: float
    Recall_macro: float
    Recall_weighted: float
    F1_macro: float
    F1_weighted: float
    Accuracy: float

def eval_cls(y_true, y_pred) -> ClsMetrics:
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return ClsMetrics(prec_m*100, prec_w*100, rec_m*100, rec_w*100, f1_m*100, f1_w*100, acc*100)

@dataclass
class RegMetrics:
    MSE: float
    RMSE: float
    MAE: float
    R2: float

def eval_reg(y_true, y_pred) -> RegMetrics:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return RegMetrics(mse, rmse, mae, r2)

def make_random_splits_cls(y_encoded, n_splits=10,
                           train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                           base_seed=RANDOM_STATE):
    """Return list of (train_idx, val_idx, test_idx) with stratification."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9
    splits = []
    for i in range(n_splits):
        sss_test = StratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=base_seed + i
        )
        idx_all = np.arange(len(y_encoded))
        (trainval_idx, test_idx) = next(sss_test.split(np.zeros_like(y_encoded), y_encoded))

        y_trainval = y_encoded[trainval_idx]
        val_in_trainval = val_ratio / (train_ratio + val_ratio)
        sss_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_in_trainval, random_state=base_seed + 100 + i
        )
        (train_idx_rel, val_idx_rel) = next(sss_val.split(np.zeros_like(y_trainval), y_trainval))
        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]

        splits.append((train_idx, val_idx, test_idx))
    return splits

def make_random_splits_reg(n_samples, n_splits=5,
                           train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                           base_seed=RANDOM_STATE):
    """Return list of (train_idx, val_idx, test_idx) without stratification."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9
    splits = []
    idx_all = np.arange(n_samples)
    for i in range(n_splits):
        ss_test = ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=base_seed + i)
        (trainval_idx_rel, test_idx_rel) = next(ss_test.split(idx_all))
        trainval_idx = idx_all[trainval_idx_rel]
        test_idx = idx_all[test_idx_rel]

        val_in_trainval = val_ratio / (train_ratio + val_ratio)
        ss_val = ShuffleSplit(n_splits=1, test_size=val_in_trainval, random_state=base_seed + 100 + i)
        (train_idx_rel, val_idx_rel) = next(ss_val.split(trainval_idx))
        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]

        splits.append((train_idx, val_idx, test_idx))
    return splits

splits_cls = make_random_splits_cls(y_cls, n_splits=5)
splits_reg = make_random_splits_reg(n_samples=len(X_iqr), n_splits=5)

def append_cls_row(accum, imputer_name, balance, model_name, metrics: ClsMetrics):
    key = (imputer_name, balance, model_name)
    if key not in accum:
        accum[key] = { "Precision_macro": [], "Precision_weighted": [],
                       "Recall_macro": [], "Recall_weighted": [],
                       "F1_macro": [], "F1_weighted": [], "Accuracy": [] }
    d = accum[key]
    d["Precision_macro"].append(metrics.Precision_macro)
    d["Precision_weighted"].append(metrics.Precision_weighted)
    d["Recall_macro"].append(metrics.Recall_macro)
    d["Recall_weighted"].append(metrics.Recall_weighted)
    d["F1_macro"].append(metrics.F1_macro)
    d["F1_weighted"].append(metrics.F1_weighted)
    d["Accuracy"].append(metrics.Accuracy)

def append_reg_row(accum, imputer_name, model_name, metrics: RegMetrics):
    key = (imputer_name, model_name)
    if key not in accum:
        accum[key] = { "MSE": [], "RMSE": [], "MAE": [], "R2": [] }
    d = accum[key]
    d["MSE"].append(metrics.MSE)
    d["RMSE"].append(metrics.RMSE)
    d["MAE"].append(metrics.MAE)
    d["R2"].append(metrics.R2)

def agg_cls_accum_to_df(accum, kappa, smote_used):
    rows = []
    for (imp, bal, mdl), d in accum.items():
        row = {"kappa": kappa, "SMOTE_Used": smote_used, "Imputer": imp, "Balance": bal, "Model": mdl}
        for k, arr in d.items():
            arr = np.array(arr, dtype=float)
            row[f"{k}_mean"] = float(np.mean(arr))
            row[f"{k}_std"]  = float(np.std(arr, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)

def agg_reg_accum_to_df(accum, kappa, smote_used):
    rows = []
    for (imp, mdl), d in accum.items():
        row = {"kappa": kappa, "SMOTE_Used": smote_used, "Imputer": imp, "Model": mdl}
        for k, arr in d.items():
            arr = np.array(arr, dtype=float)
            row[f"{k}_mean"] = float(np.mean(arr))
            row[f"{k}_std"]  = float(np.std(arr, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)

for kappa in KAPPA_LIST:
    print(f"\n[RUN] κ = {kappa}")

    cls_accum = {}
    reg_accum = {}

    for fold_id, ((tr_c, va_c, te_c), (tr_r, va_r, te_r)) in enumerate(zip(splits_cls, splits_reg), start=1):
        w_cls = get_xgb_importance(X_iqr.iloc[tr_c][pollutants], y_cls[tr_c], "cls", RANDOM_STATE)
        D3_cls = svd_impute_weighted(X_iqr[pollutants], weights=w_cls, rank=4, n_iter=3,
                                     kappa_exp=kappa, random_state=RANDOM_STATE)
        D3_cls.columns = [f"{c}_svd" for c in pollutants]

        w_reg = get_xgb_importance(X_iqr.iloc[tr_r][pollutants], y_reg_raw.iloc[tr_r], "reg", RANDOM_STATE)
        D3_reg = svd_impute_weighted(X_iqr[pollutants], weights=w_reg, rank=4, n_iter=3,
                                     kappa_exp=kappa, random_state=RANDOM_STATE)
        D3_reg.columns = [f"{c}_svd" for c in pollutants]

        D4_cls = pd.concat([D1_full, D2_full, D3_cls], axis=1)
        D4_cls = D4_cls.loc[:, ~D4_cls.T.duplicated()]
        D4_cls = D4_cls.loc[:, D4_cls.nunique(dropna=False) > 1].astype(np.float32)

        D4_reg = pd.concat([D1_full, D2_full, D3_reg], axis=1)
        D4_reg = D4_reg.loc[:, ~D4_reg.T.duplicated()]
        D4_reg = D4_reg.loc[:, D4_reg.nunique(dropna=False) > 1].astype(np.float32)

        for name_cls, Xfeat_cls in [
            ("KNN", D1_full),
            ("MICE", D2_full),
            ("SVD(FIW-cls)", D3_cls),
            ("MM(FIW-cls)", D4_cls),
        ]:
            X_tr_c = Xfeat_cls.iloc[tr_c]
            X_te_c = Xfeat_cls.iloc[te_c]
            y_tr_c = y_cls[tr_c]
            y_te_c = y_cls[te_c]

            clfs = make_classifiers(num_classes=len(le.classes_))

            for mname, model in clfs.items():
                model.fit(X_tr_c, y_tr_c)
                y_hat = model.predict(X_te_c)
                append_cls_row(cls_accum, name_cls, "Imbalanced", mname, eval_cls(y_te_c, y_hat))

            if RUN_SMOTE:
                try:
                    def safe_k_neighbors(y):
                        binc = np.bincount(y)
                        min_class = np.min(binc)
                        return int(max(1, min(5, min_class - 1)))
                    k_smote = safe_k_neighbors(y_tr_c)
                    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_smote)
                    X_sm, y_sm = sm.fit_resample(X_tr_c, y_tr_c)
                    clfs2 = make_classifiers(num_classes=len(le.classes_))
                    for mname, model in clfs2.items():
                        model.fit(X_sm, y_sm)
                        y_hat = model.predict(X_te_c)
                        append_cls_row(cls_accum, name_cls, "SMOTE", mname, eval_cls(y_te_c, y_hat))
                except Exception as e:
                    print(f"[WARN][fold {fold_id}][{name_cls}] SMOTE skipped: {e}")

        for name_reg, Xfeat_reg in [
            ("KNN", D1_full),
            ("MICE", D2_full),
            ("SVD(FIW-reg)", D3_reg),
            ("MM(FIW-reg)", D4_reg),
        ]:
            X_tr_r = Xfeat_reg.iloc[tr_r]
            X_te_r = Xfeat_reg.iloc[te_r]
            y_tr_r = y_reg_raw.iloc[tr_r].values
            y_te_r = y_reg_raw.iloc[te_r].values

            regs = make_regressors()
            for mname, model in regs.items():
                model.fit(X_tr_r, y_tr_r)
                y_hat = model.predict(X_te_r)
                append_reg_row(reg_accum, name_reg, mname, eval_reg(y_te_r, y_hat))

        del D3_cls, D3_reg, D4_cls, D4_reg
        gc.collect()

    cls_all = agg_cls_accum_to_df(cls_accum, kappa=kappa, smote_used=RUN_SMOTE)
    reg_all = agg_reg_accum_to_df(reg_accum, kappa=kappa, smote_used=RUN_SMOTE)

    if os.path.exists(CSV_CLS):
        cls_all.to_csv(CSV_CLS, mode="a", header=False, index=False)
    else:
        cls_all.to_csv(CSV_CLS, mode="w", header=True, index=False)

    if os.path.exists(CSV_REG):
        reg_all.to_csv(CSV_REG, mode="a", header=False, index=False)
    else:
        reg_all.to_csv(CSV_REG, mode="w", header=True, index=False)

    print(f"[SAVED] κ={kappa} (10-split mean/std) → {CSV_CLS}, {CSV_REG}")

print("\nCompleted: 10 random splits (70/10/20) mean/std per metric (KNN/MICE + task-specific SVD(FIW) + MM(FIW), with optional SMOTE).")
