import os
from pathlib import Path

home = Path.home()
cache_root = home / ".cache"
if cache_root.exists() and not cache_root.is_dir():
    backup_path = cache_root.with_name(".cache_backup")
    cache_root.rename(backup_path)
cache_base = cache_root / "ai_models"
os.environ["HF_HOME"] = str(cache_base / "huggingface")
os.environ["TORCH_HOME"] = str(cache_base / "torch")
os.environ["TIMM_HOME"] = str(cache_base / "timm")
for path in [Path(os.environ["HF_HOME"]), Path(os.environ["TORCH_HOME"]), Path(os.environ["TIMM_HOME"])]:
    path.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor, Pool

def median_ape(y_true, y_pred):
    ape = np.abs(y_true - y_pred) / np.clip(y_true, 1e-6, None)
    return np.median(ape)

def read_parquet(path):
    return pd.read_parquet(path)

def make_features(df: pd.DataFrame):
    out = df.copy()
    current_year = 2024
    if "year" in out.columns:
        out["age"] = np.clip(current_year - out["year"].fillna(current_year), 0, 60)
    for col in ["steering_wheel", "was_in_accident", "custom_cleared", "pts", "owners"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    if {"latitude", "longitude"}.issubset(out.columns):
        out["lat_bin"] = (out["latitude"] * 20).round()/20
        out["lon_bin"] = (out["longitude"] * 20).round()/20
    if "equipment" in out.columns:
        eq = out["equipment"].astype(str).fillna("None")
        top = eq.value_counts().head(50).index
        for k in top:
            out[f"eq_{k}"] = (eq == k).astype(int)
    text_cols = []
    for c in ["description"]:
        if c in out.columns:
            text_cols.append(c)
    return out, text_cols

train = read_parquet("train_dataset.parquet")
test  = read_parquet("test_dataset.parquet")

Xtr, text_cols = make_features(train)
Xts, _         = make_features(test)

y = Xtr["price_TARGET"].values
Xtr = Xtr.drop(columns=["price_TARGET"])

Xts = Xts.reindex(columns=Xtr.columns, fill_value=np.nan)

Xtr = Xtr.replace({None: np.nan})
Xts = Xts.replace({None: np.nan})

cat_cols = [c for c in Xtr.columns if Xtr[c].dtype == "object" and c not in text_cols]

for c in text_cols:
    Xtr[c] = Xtr[c].astype(str).fillna("")
    Xts[c] = Xts[c].astype(str).fillna("")

for c in cat_cols:
    Xtr[c] = Xtr[c].astype(str).fillna("__nan__")
    Xts[c] = Xts[c].astype(str).fillna("__nan__")

num_cols = [c for c in Xtr.columns if c not in set(text_cols + cat_cols)]
Xtr[num_cols] = Xtr[num_cols].fillna(-1)
Xts[num_cols] = Xts[num_cols].fillna(-1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(Xtr))
pred = np.zeros(len(Xts))

for tr_idx, vl_idx in kf.split(Xtr):
    tr_pool = Pool(
        Xtr.iloc[tr_idx], label=np.log1p(y[tr_idx]),
        cat_features=cat_cols, text_features=text_cols
    )
    vl_pool = Pool(
        Xtr.iloc[vl_idx], label=np.log1p(y[vl_idx]),
        cat_features=cat_cols, text_features=text_cols
    )
    model = CatBoostRegressor(
        loss_function="MAE",
        depth=8, learning_rate=0.05, iterations=1200,
        l2_leaf_reg=3, random_seed=42,
        verbose=200
    )
    model.fit(tr_pool, eval_set=vl_pool, use_best_model=True)
    oof[vl_idx] = np.expm1(model.predict(vl_pool))
    pred += np.expm1(model.predict(Pool(Xts, cat_features=cat_cols, text_features=text_cols))) / kf.n_splits

sub = pd.DataFrame({"ID": Xts["ID"], "target": pred})
sub.to_csv("submission.csv", index=False)
