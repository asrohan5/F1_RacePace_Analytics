import os
import sys
import pandas as pd
import numpy as np
import pickle
import re

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                   RidgeCV, LassoCV)
from lightgbm import LGBMRegressor, LGBMClassifier

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    TRAIN_PATH, VAL_PATH,
    TRAIN_SC_PATH, VAL_SC_PATH,
    CLASSIFIER_PATH, REGRESSOR_PATH,
    CLASSIFIER_SC_PATH, REGRESSOR_SC_PATH,
    TARGET_REGRESSION, TARGET_CLASSIFICATION,
    INITIAL_MODEL_PARAMS
)
import warnings
warnings.filterwarnings('ignore')


#-----------------------------------------------------------------------------------------------------------------------------------------

FEATURE_COLS = [
    "coasting_pct_delta", "full_throttle_pct_delta", "gear_shifts_delta",
    "avg_brake_zone_length_delta", "avg_entry_speed_delta",
    "brake_zone_count_delta", "tyre_life_delta", "tyre_life_x_coasting_delta",
    "stint_phase_delta", "abu_dhabi_gear_delta", "rolling_delta_3",
    "VER_coasting_pct", "HAM_coasting_pct",
    "VER_full_throttle_pct", "HAM_full_throttle_pct",
    "VER_gear_shifts", "HAM_gear_shifts",
    "VER_avg_brake_zone_length", "HAM_avg_brake_zone_length",
    "VER_avg_entry_speed", "HAM_avg_entry_speed",
    "VER_TyreLife", "HAM_TyreLife",
    "VER_stint_phase", "HAM_stint_phase",
    "same_compound", "VER_compound_enc", "HAM_compound_enc",
    "LapNumber", "race_enc"
]

# ─────────────────────────────────────────
# LOAD SPLITS
# ─────────────────────────────────────────

def load_splits():
    try:
        train = pd.read_csv(TRAIN_PATH)
        val   = pd.read_csv(VAL_PATH)
        log.info(f"Train: {train.shape} | Val: {val.shape}")
        return train, val
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────

def eval_regressor(model, X, y, label):
    pred = model.predict(X)
    mae  = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    r2   = r2_score(y, pred)
    log.info(f"  [{label}] MAE={mae:.4f}s  RMSE={rmse:.4f}s  R2={r2:.4f}")
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def eval_classifier(model, X, y, label):
    pred     = model.predict(X)
    pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else pred
    acc  = accuracy_score(y, pred)
    f1   = f1_score(y, pred, zero_division=0)
    auc  = roc_auc_score(y, pred_proba)
    log.info(f"  [{label}] Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return {"label": label, "Accuracy": acc, "F1": f1, "AUC": auc}


# ─────────────────────────────────────────
# AUTO-UPDATE config.py WITH BEST PARAMS
# ─────────────────────────────────────────

def update_config_best_params(best_reg_params, best_clf_params):
    """
    Reads config.py, replaces BEST_REGRESSOR_PARAMS and
    BEST_CLASSIFIER_PARAMS with the discovered best params.
    This way config.py is always the single source of truth.
    """
    try:
        config_path = os.path.join(os.getcwd(), "src", "config.py")
        config_path = os.path.normpath(config_path)

        with open(config_path, "r") as f:
            content = f.read()

        # Replace BEST_REGRESSOR_PARAMS block
        new_reg = f"BEST_REGRESSOR_PARAMS = {repr(best_reg_params)}"
        content = re.sub(
            r"BEST_REGRESSOR_PARAMS\s*=\s*\{[^}]*\}",
            new_reg, content, flags=re.DOTALL
        )

        # Replace BEST_CLASSIFIER_PARAMS block
        new_clf = f"BEST_CLASSIFIER_PARAMS = {repr(best_clf_params)}"
        content = re.sub(
            r"BEST_CLASSIFIER_PARAMS\s*=\s*\{[^}]*\}",
            new_clf, content, flags=re.DOTALL
        )

        with open(config_path, "w") as f:
            f.write(content)

        log.info(f"config.py updated with best params.")
        log.info(f"  BEST_REGRESSOR_PARAMS  = {best_reg_params}")
        log.info(f"  BEST_CLASSIFIER_PARAMS = {best_clf_params}")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# REGRESSION TRAINING
# ─────────────────────────────────────────

def train_regressors(X_train, y_train, X_val, y_val):
    try:
        log.info("=" * 60)
        log.info("REGRESSION MODELS")
        log.info("=" * 60)

        results = []

        # ── Baseline ──
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        results.append(eval_regressor(dummy, X_val, y_val, "Baseline-Mean"))

        # ── Linear Regression ──
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        results.append(eval_regressor(lr, X_val, y_val, "LinearRegression"))
        cv_lr = cross_val_score(lr, X_train, y_train,
                                cv=5, scoring="neg_mean_absolute_error")
        log.info(f"  [LinearRegression] CV MAE: {-cv_lr.mean():.4f} ± {cv_lr.std():.4f}")

        # ── RidgeCV — addresses LinearRegression CV instability ──
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
        ridge.fit(X_train, y_train)
        log.info(f"  RidgeCV selected alpha: {ridge.alpha_}")
        results.append(eval_regressor(ridge, X_val, y_val, "RidgeCV"))
        cv_ridge = cross_val_score(ridge, X_train, y_train,
                                   cv=5, scoring="neg_mean_absolute_error")
        log.info(f"  [RidgeCV] CV MAE: {-cv_ridge.mean():.4f} ± {cv_ridge.std():.4f}")

        # ── LassoCV — feature selection through sparsity ──
        lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], cv=5,
                        max_iter=5000, random_state=42)
        lasso.fit(X_train, y_train)
        n_zero = (lasso.coef_ == 0).sum()
        log.info(f"  LassoCV selected alpha: {lasso.alpha_:.4f} | "
                 f"zeroed features: {n_zero}/{len(lasso.coef_)}")
        results.append(eval_regressor(lasso, X_val, y_val, "LassoCV"))

        # ── RandomForest ──
        rf_params = INITIAL_MODEL_PARAMS.get("regressor", {})
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        results.append(eval_regressor(rf, X_val, y_val, "RandomForest-Initial"))

        log.info("  Running RandomizedSearchCV on RandomForest...")
        rf_param_grid = {
            "n_estimators"    : [50, 100, 200],
            "max_depth"       : [3, 4, 5, 6, None],
            "min_samples_leaf": [2, 5, 8, 10],
            "max_features"    : ["sqrt", "log2", 0.5],
        }
        rf_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            rf_param_grid, n_iter=20, cv=5,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=1
        )
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
        log.info(f"  RF best params: {rf_search.best_params_}")
        results.append(eval_regressor(best_rf, X_val, y_val, "RandomForest-Tuned"))

        # ── XGBoost ──
        log.info("  Running RandomizedSearchCV on XGBoost...")
        xgb_param_grid = {
            "n_estimators"    : [50, 100, 200],
            "max_depth"       : [2, 3, 4, 5],
            "learning_rate"   : [0.01, 0.05, 0.1, 0.2],
            "subsample"       : [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha"       : [0, 0.1, 0.5],
            "reg_lambda"      : [1, 2, 5],
        }
        xgb_search = RandomizedSearchCV(
            XGBRegressor(random_state=42, verbosity=0),
            xgb_param_grid, n_iter=20, cv=5,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=1
        )
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        log.info(f"  XGB best params: {xgb_search.best_params_}")
        results.append(eval_regressor(best_xgb, X_val, y_val, "XGBoost-Tuned"))

        # ── LightGBM — better regularisation for small datasets ──
        log.info("  Running RandomizedSearchCV on LightGBM...")
        lgbm_param_grid = {
            "n_estimators"    : [50, 100, 200],
            "max_depth"       : [3, 4, 5, 6],
            "learning_rate"   : [0.01, 0.05, 0.1, 0.2],
            "num_leaves"      : [15, 31, 63],
            "min_child_samples": [5, 10, 15, 20],
            "subsample"       : [0.6, 0.8, 1.0],
            "reg_alpha"       : [0, 0.1, 0.5],
        }
        lgbm_search = RandomizedSearchCV(
            LGBMRegressor(random_state=42, verbose=-1),
            lgbm_param_grid, n_iter=20, cv=5,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=1
        )
        lgbm_search.fit(X_train, y_train)
        best_lgbm = lgbm_search.best_estimator_
        log.info(f"  LGBM best params: {lgbm_search.best_params_}")
        results.append(eval_regressor(best_lgbm, X_val, y_val, "LightGBM-Tuned"))

        # ── Pick best by MAE on val (skip dummy) ──
        best_result = min(results[1:], key=lambda x: x["MAE"])
        log.info(f"\n  Best regressor on val: {best_result['label']} "
                 f"| MAE={best_result['MAE']:.4f}s")

        model_map = {
            "LinearRegression"    : lr,
            "RidgeCV"             : ridge,
            "LassoCV"             : lasso,
            "RandomForest-Initial": rf,
            "RandomForest-Tuned"  : best_rf,
            "XGBoost-Tuned"       : best_xgb,
            "LightGBM-Tuned"      : best_lgbm,
        }
        best_reg_model = model_map[best_result["label"]]

        if best_result["label"] in ["RandomForest-Tuned", "RandomForest-Initial"]:
            best_reg_params = rf_search.best_params_
        elif best_result["label"] == "XGBoost-Tuned":
            best_reg_params = xgb_search.best_params_
        elif best_result["label"] == "LightGBM-Tuned":
            best_reg_params = lgbm_search.best_params_
        elif best_result["label"] == "RidgeCV":
            best_reg_params = {"alpha": ridge.alpha_}
        elif best_result["label"] == "LassoCV":
            best_reg_params = {"alpha": lasso.alpha_}
        else:
            best_reg_params = {}

        return best_reg_model, best_reg_params, results

    except Exception as e:
        raise CustomException(e, sys)

# ─────────────────────────────────────────
# CLASSIFICATION TRAINING
# ─────────────────────────────────────────

def train_classifiers(X_train, y_train, X_val, y_val):
    try:
        log.info("=" * 60)
        log.info("CLASSIFICATION MODELS")
        log.info("=" * 60)

        results = []

        # ── Baseline ──
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        results.append(eval_classifier(dummy, X_val, y_val, "Baseline-MostFrequent"))

        # ── Logistic Regression ──
        clf_params = INITIAL_MODEL_PARAMS.get("classifier", {})
        lr = LogisticRegression(**clf_params)
        lr.fit(X_train, y_train)
        results.append(eval_classifier(lr, X_val, y_val, "LogisticRegression"))

        cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring="f1")
        log.info(f"  [LogisticRegression] CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── RandomForest with class_weight balanced ──
        log.info("  Running RandomizedSearchCV on RandomForest (balanced)...")
        rf_param_grid = {
            "n_estimators"    : [50, 100, 200],
            "max_depth"       : [3, 4, 5, 6, None],
            "min_samples_leaf": [2, 5, 8, 10],
            "max_features"    : ["sqrt", "log2", 0.5],
        }
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(class_weight="balanced", random_state=42),
            rf_param_grid, n_iter=20, cv=5,
            scoring="f1", random_state=42, n_jobs=1
        )
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
        log.info(f"  RF best params: {rf_search.best_params_}")
        results.append(eval_classifier(best_rf, X_val, y_val, "RandomForest-Tuned"))

        # ── XGBoost with scale_pos_weight for imbalance ──
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        log.info(f"  XGBoost scale_pos_weight = {scale_pos:.2f}")
        log.info("  Running RandomizedSearchCV on XGBoost...")
        xgb_param_grid = {
            "n_estimators"    : [50, 100, 200],
            "max_depth"       : [2, 3, 4, 5],
            "learning_rate"   : [0.01, 0.05, 0.1, 0.2],
            "subsample"       : [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha"       : [0, 0.1, 0.5],
            "reg_lambda"      : [1, 2, 5],
        }
        xgb_search = RandomizedSearchCV(
            XGBClassifier(scale_pos_weight=scale_pos,
                          random_state=42, verbosity=0,
                          eval_metric="logloss"),
            xgb_param_grid, n_iter=20, cv=5,
            scoring="f1", random_state=42, n_jobs=1
        )
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        log.info(f"  XGB best params: {xgb_search.best_params_}")
        results.append(eval_classifier(best_xgb, X_val, y_val, "XGBoost-Tuned"))

        # ── Pick best by F1 on val (skip dummy) ──
        best_result = max(results[1:], key=lambda x: x["F1"])
        log.info(f"\n  Best classifier on val: {best_result['label']} "
                 f"| F1={best_result['F1']:.4f}  AUC={best_result['AUC']:.4f}")

        model_map = {
            "LogisticRegression"  : lr,
            "RandomForest-Tuned"  : best_rf,
            "XGBoost-Tuned"       : best_xgb,
        }
        best_clf_model  = model_map[best_result["label"]]
        best_clf_params = (rf_search.best_params_
                           if "RandomForest" in best_result["label"]
                           else xgb_search.best_params_
                           if "XGBoost" in best_result["label"]
                           else {})

        return best_clf_model, best_clf_params, results

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def run_model_trainer():
    try:
        log.info("=" * 60)
        log.info("Starting model training")
        log.info("=" * 60)

        # ── FULL DATASET ──
        log.info("\n>>> FULL DATASET (all compound combinations)")
        train, val = load_splits()
        X_train = train[FEATURE_COLS].values
        X_val   = val[FEATURE_COLS].values
        y_train_reg = train[TARGET_REGRESSION].values
        y_val_reg   = val[TARGET_REGRESSION].values
        y_train_clf = train[TARGET_CLASSIFICATION].values
        y_val_clf   = val[TARGET_CLASSIFICATION].values

        best_reg, best_reg_params, reg_results = train_regressors(
            X_train, y_train_reg, X_val, y_val_reg
        )
        best_clf, best_clf_params, clf_results = train_classifiers(
            X_train, y_train_clf, X_val, y_val_clf
        )
        update_config_best_params(best_reg_params, best_clf_params)

        with open(REGRESSOR_PATH, "wb") as f:
            pickle.dump(best_reg, f)
        log.info(f"Full regressor saved → {REGRESSOR_PATH}")

        with open(CLASSIFIER_PATH, "wb") as f:
            pickle.dump(best_clf, f)
        log.info(f"Full classifier saved → {CLASSIFIER_PATH}")

        # ── SAME-COMPOUND SUBSET ──
        log.info("\n>>> SAME-COMPOUND SUBSET (driving style isolated)")
        train_sc = pd.read_csv(TRAIN_SC_PATH)
        val_sc   = pd.read_csv(VAL_SC_PATH)
        log.info(f"SC Train: {train_sc.shape} | SC Val: {val_sc.shape}")

        X_train_sc = train_sc[FEATURE_COLS].values
        X_val_sc   = val_sc[FEATURE_COLS].values
        y_train_sc_reg = train_sc[TARGET_REGRESSION].values
        y_val_sc_reg   = val_sc[TARGET_REGRESSION].values
        y_train_sc_clf = train_sc[TARGET_CLASSIFICATION].values
        y_val_sc_clf   = val_sc[TARGET_CLASSIFICATION].values

        best_reg_sc, best_reg_sc_params, reg_sc_results = train_regressors(
            X_train_sc, y_train_sc_reg, X_val_sc, y_val_sc_reg
        )
        best_clf_sc, best_clf_sc_params, clf_sc_results = train_classifiers(
            X_train_sc, y_train_sc_clf, X_val_sc, y_val_sc_clf
        )

        # Save SC models separately
        with open(REGRESSOR_SC_PATH,  "wb") as f:
            pickle.dump(best_reg_sc, f)
        with open(CLASSIFIER_SC_PATH, "wb") as f:
            pickle.dump(best_clf_sc, f)
        log.info(f"SC Regressor  saved → {REGRESSOR_SC_PATH}")
        log.info(f"SC Classifier saved → {CLASSIFIER_SC_PATH}")

        log.info("Model training complete.")
        return (best_reg, best_clf, reg_results, clf_results,
                best_reg_sc, best_clf_sc, reg_sc_results, clf_sc_results)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    (best_reg, best_clf, reg_results, clf_results,
     best_reg_sc, best_clf_sc, reg_sc_results, clf_sc_results) = run_model_trainer()

    print("\n--- FULL DATASET — REGRESSION ---")
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 58)
    for r in reg_results:
        print(f"{r['label']:<30} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>8.4f}")

    print("\n--- FULL DATASET — CLASSIFICATION ---")
    print(f"{'Model':<30} {'Accuracy':>10} {'F1':>8} {'AUC':>8}")
    print("-" * 60)
    for r in clf_results:
        print(f"{r['label']:<30} {r['Accuracy']:>10.4f} {r['F1']:>8.4f} {r['AUC']:>8.4f}")

    print("\n--- SAME-COMPOUND SUBSET — REGRESSION ---")
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 58)
    for r in reg_sc_results:
        print(f"{r['label']:<30} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>8.4f}")

    print("\n--- SAME-COMPOUND SUBSET — CLASSIFICATION ---")
    print(f"{'Model':<30} {'Accuracy':>10} {'F1':>8} {'AUC':>8}")
    print("-" * 60)
    for r in clf_sc_results:
        print(f"{r['label']:<30} {r['Accuracy']:>10.4f} {r['F1']:>8.4f} {r['AUC']:>8.4f}")