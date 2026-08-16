import gc
import os
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import ShuffleSplit
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--kappa_list", type=str, default="1.0")
parser.add_argument("--save_dir", type=str, default="./")
parser.add_argument("--data_path", type=str, default="./synthetic.csv")
args, _ = parser.parse_known_args()

KAPPA_LIST = [float(x.strip()) for x in args.kappa_list.split(",") if x.strip()]
SAVE_DIR = args.save_dir
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_REG = os.path.join(SAVE_DIR, "regression_results.csv")

RANDOM_STATE = 2025
np.random.seed(RANDOM_STATE)
EPS = 1e-8

df = pd.read_csv(args.data_path)

use_cols = [
    "CO(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "O3(GT)",
    "SO2(GT)",
    "PM2.5",
    "PM10",
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "WindDirection",
    "CO_NOx_Ratio",
    "NOx_NO2_Ratio",
    "Temp_Humidity_Index",
    "CO_MA3",
    "NO2_MA3",
    "O3_MA3",
    "Date",
    "Time",
]

use_cols = [c for c in use_cols if c in df.columns]
df0 = df.loc[:, use_cols].copy()

if "Date" in df0.columns:
    df0["Date"] = pd.to_datetime(df0["Date"], errors="coerce")


def calc_aqi_from_breakpoints(conc, breakpoints):
    if pd.isna(conc):
        return np.nan

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= conc <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (conc - c_low) + i_low

    return np.nan


EPA_BREAKPOINTS = {
    "PM2.5": [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ],
    "PM10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 604, 301, 500),
    ],
    "CO(GT)": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 50.4, 301, 500),
    ],
    "NO2(GT)": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 2049, 301, 500),
    ],
    "SO2(GT)": [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 1004, 301, 500),
    ],
    "O3(GT)": [
        (0, 54, 0, 50),
        (55, 70, 51, 100),
        (71, 85, 101, 150),
        (86, 105, 151, 200),
        (106, 200, 201, 300),
    ],
}


aqi_components = []

for col, breakpoints in EPA_BREAKPOINTS.items():
    if col in df0.columns:
        sub_col = f"{col}_AQI"
        df0[sub_col] = df0[col].apply(
            lambda x: calc_aqi_from_breakpoints(x, breakpoints)
        )
        aqi_components.append(sub_col)

df0["AirQualityIndex"] = df0[aqi_components].max(axis=1)

print("[INFO] Recomputed AirQualityIndex using U.S. EPA AQI breakpoints")
print(df0["AirQualityIndex"].describe())


target_col = "AirQualityIndex"

feature_cols = [
    "CO(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "O3(GT)",
    "SO2(GT)",
    "PM2.5",
    "PM10",
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "WindDirection",
    "CO_NOx_Ratio",
    "NOx_NO2_Ratio",
    "Temp_Humidity_Index",
    "CO_MA3",
    "NO2_MA3",
    "O3_MA3",
]

feature_cols = [c for c in feature_cols if c in df0.columns]


# -----------------------------
# Random Missing Injection
# target 생성 후 feature에만 missing 생성
# -----------------------------
missing_cols = [
    "CO(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "O3(GT)",
    "SO2(GT)",
    "PM2.5",
    "PM10",
    "CO_NOx_Ratio",
    "NOx_NO2_Ratio",
    "CO_MA3",
    "NO2_MA3",
    "O3_MA3",
]

missing_cols = [c for c in missing_cols if c in df0.columns]

missing_rate = 0.50
rng = np.random.RandomState(RANDOM_STATE)

for col in missing_cols:
    non_missing_idx = df0.index[df0[col].notna()]
    n_missing = int(len(non_missing_idx) * missing_rate)

    sampled_idx = rng.choice(
        non_missing_idx,
        size=n_missing,
        replace=False
    )

    df0.loc[sampled_idx, col] = np.nan

print("\n[INFO] Missing rate after injection:")
print(df0[missing_cols].isna().mean() * 100)


df1 = df0.dropna(subset=[target_col]).copy()

all_nan_mask = df1[feature_cols].isna().all(axis=1)
df1 = df1.loc[~all_nan_mask].copy()

X_raw = df1[feature_cols].astype(np.float32)
y_reg_raw = df1[target_col].astype(np.float32)

valid_cols = X_raw.columns[X_raw.notna().any(axis=0)].tolist()
X_raw = X_raw[valid_cols]
feature_cols = valid_cols

print(f"\n[INFO] Data path: {args.data_path}")
print(f"[INFO] #samples: {len(X_raw)}")
print(f"[INFO] Features: {feature_cols}")
print(f"[INFO] Target: {target_col}")

def iqr_to_nan(df_in: pd.DataFrame, cols=None, whisker: float = 1.5):
    df_out = df_in.copy()
    cols = cols or df_out.columns.tolist()

    for c in cols:
        s = df_out[c]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or np.isclose(iqr, 0.0):
            continue

        ll, ul = q1 - whisker * iqr, q3 + whisker * iqr
        df_out.loc[(s < ll) | (s > ul), c] = np.nan

    return df_out

X_iqr = iqr_to_nan(X_raw, cols=feature_cols, whisker=1.5).astype(np.float32)

imp_knn = KNNImputer(n_neighbors=45)
imp_mice = IterativeImputer(
    random_state=RANDOM_STATE,
    max_iter=10,
    tol=1e-3,
    initial_strategy="median"
)

D1_full = pd.DataFrame(
    imp_knn.fit_transform(X_iqr),
    columns=feature_cols,
    index=X_iqr.index
).astype(np.float32)
D1_full.columns = [f"{c}_knn" for c in feature_cols]

D2_full = pd.DataFrame(
    imp_mice.fit_transform(X_iqr),
    columns=feature_cols,
    index=X_iqr.index
).astype(np.float32)
D2_full.columns = [f"{c}_mice" for c in feature_cols]

def get_xgb_importance(X, y, random_state=2025):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=random_state
    )
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

        n_comp = max(1, min(rank, A.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=rng)

        Z = svd.fit_transform(A)
        Ahat = svd.inverse_transform(Z)

        X_hat = pd.DataFrame(Ahat, index=X.index, columns=X.columns) + mu
        X_fill = Xw.where(~Xw.isna(), X_hat)

    X_rec = X_fill / scale
    return X_rec.astype(np.float32)

def make_regressors():
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=RANDOM_STATE
    )

    et = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=18,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    stk = StackingRegressor(
        estimators=[("rf", rf), ("xgb", xgb), ("et", et)],
        final_estimator=LinearRegression(),
        n_jobs=-1
    )

    return {
        "Random Forest": rf,
        "XGBoost": xgb,
        "ExtraTree": et,
        "Stacking": stk
    }

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

def make_random_splits_reg(
    n_samples,
    n_splits=5,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    base_seed=RANDOM_STATE
):
    splits = []
    idx_all = np.arange(n_samples)

    for i in range(n_splits):
        ss_test = ShuffleSplit(
            n_splits=1,
            test_size=test_ratio,
            random_state=base_seed + i
        )
        trainval_idx_rel, test_idx_rel = next(ss_test.split(idx_all))
        trainval_idx = idx_all[trainval_idx_rel]
        test_idx = idx_all[test_idx_rel]

        val_in_trainval = val_ratio / (train_ratio + val_ratio)
        ss_val = ShuffleSplit(
            n_splits=1,
            test_size=val_in_trainval,
            random_state=base_seed + 100 + i
        )
        train_idx_rel, val_idx_rel = next(ss_val.split(trainval_idx))

        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]

        splits.append((train_idx, val_idx, test_idx))

    return splits

splits_reg = make_random_splits_reg(n_samples=len(X_iqr), n_splits=5)

def append_reg_row(accum, imputer_name, model_name, metrics: RegMetrics):
    key = (imputer_name, model_name)

    if key not in accum:
        accum[key] = {"MSE": [], "RMSE": [], "MAE": [], "R2": []}

    accum[key]["MSE"].append(metrics.MSE)
    accum[key]["RMSE"].append(metrics.RMSE)
    accum[key]["MAE"].append(metrics.MAE)
    accum[key]["R2"].append(metrics.R2)

def agg_reg_accum_to_df(accum, kappa):
    rows = []

    for (imp, mdl), d in accum.items():
        row = {
            "kappa": kappa,
            "Imputer": imp,
            "Model": mdl
        }

        for k, arr in d.items():
            arr = np.array(arr, dtype=float)
            row[f"{k}_mean"] = float(np.mean(arr))
            row[f"{k}_std"] = float(np.std(arr, ddof=1))

        rows.append(row)

    return pd.DataFrame(rows)

for kappa in KAPPA_LIST:
    print(f"\n[RUN] kappa = {kappa}")

    reg_accum = {}

    for fold_id, (tr_r, va_r, te_r) in enumerate(splits_reg, start=1):
        print(f"[Fold {fold_id}]")

        w_reg = get_xgb_importance(
            X_iqr.iloc[tr_r][feature_cols],
            y_reg_raw.iloc[tr_r],
            random_state=RANDOM_STATE
        )

        D3_reg = svd_impute_weighted(
            X_iqr[feature_cols],
            weights=w_reg,
            rank=4,
            n_iter=3,
            kappa_exp=kappa,
            random_state=RANDOM_STATE
        )
        D3_reg.columns = [f"{c}_svd" for c in feature_cols]

        D4_reg = pd.concat([D1_full, D2_full, D3_reg], axis=1)
        D4_reg = D4_reg.loc[:, ~D4_reg.T.duplicated()]
        D4_reg = D4_reg.loc[:, D4_reg.nunique(dropna=False) > 1].astype(np.float32)

        for name_reg, Xfeat_reg in [
            ("KNN", D1_full),
            ("MICE", D2_full),
            ("SVD(FIW-reg)", D3_reg),
            ("MM(FIW-reg)", D4_reg),
        ]:
            X_tr = Xfeat_reg.iloc[tr_r]
            X_te = Xfeat_reg.iloc[te_r]
            y_tr = y_reg_raw.iloc[tr_r].values
            y_te = y_reg_raw.iloc[te_r].values

            regs = make_regressors()

            for mname, model in regs.items():
                model.fit(X_tr, y_tr)
                y_hat = model.predict(X_te)

                append_reg_row(
                    reg_accum,
                    name_reg,
                    mname,
                    eval_reg(y_te, y_hat)
                )

        del D3_reg, D4_reg
        gc.collect()

    reg_all = agg_reg_accum_to_df(reg_accum, kappa=kappa)

    if os.path.exists(CSV_REG):
        reg_all.to_csv(CSV_REG, mode="a", header=False, index=False)
    else:
        reg_all.to_csv(CSV_REG, mode="w", header=True, index=False)

    print(f"[SAVED] kappa={kappa} → {CSV_REG}")

print("\n Completed: Synthetic AQI regression.")