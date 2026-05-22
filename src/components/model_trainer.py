import os
import sys
import logging
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

import logger
from exception import CustomException
from config import (
    TRAIN_PATH, VAL_PATH,
    CLASSIFIER_PATH, REGRESSOR_PATH,
    TARGET_REGRESSION, TARGET_CLASSIFICATION,
    INITIAL_MODEL_PARAMS
)

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "coasting_pct_delta", "full_throttle_pct_delta", "gear_shifts_delta",
    "avg_brake_zone_length_delta", "avg_entry_speed_delta",
    "brake_zone_count_delta", "tyre_life_delta",
    "VER_coasting_pct", "HAM_coasting_pct",
    "VER_full_throttle_pct", "HAM_full_throttle_pct",
    "VER_gear_shifts", "HAM_gear_shifts",
    "VER_avg_brake_zone_length", "HAM_avg_brake_zone_length",
    "VER_avg_entry_speed", "HAM_avg_entry_speed",
    "VER_TyreLife", "HAM_TyreLife",
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
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "config.py")
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

        # Cross-val on train for LinearRegression
        cv_scores = cross_val_score(lr, X_train, y_train,
                                    cv=5, scoring="neg_mean_absolute_error")
        log.info(f"  [LinearRegression] CV MAE: {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── RandomForest ──
        rf_params = INITIAL_MODEL_PARAMS.get("regressor", {})
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        results.append(eval_regressor(rf, X_val, y_val, "RandomForest-Initial"))

        # ── RandomizedSearchCV on RandomForest ──
        log.info("  Running RandomizedSearchCV on RandomForest...")
        rf_param_grid = {
            "n_estimators" : [50, 100, 200],
            "max_depth"    : [3, 4, 5, 6, None],
            "min_samples_leaf": [2, 5, 8, 10],
            "max_features" : ["sqrt", "log2", 0.5],
        }
        rf_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            rf_param_grid, n_iter=20, cv=5,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=-1
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
            random_state=42, n_jobs=-1
        )
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        log.info(f"  XGB best params: {xgb_search.best_params_}")
        results.append(eval_regressor(best_xgb, X_val, y_val, "XGBoost-Tuned"))

        # ── Pick best by MAE on val ──
        best_result = min(results[1:], key=lambda x: x["MAE"])  # skip dummy
        log.info(f"\n  Best regressor on val: {best_result['label']} "
                 f"| MAE={best_result['MAE']:.4f}s")

        # Map label to model object
        model_map = {
            "LinearRegression"    : lr,
            "RandomForest-Initial": rf,
            "RandomForest-Tuned"  : best_rf,
            "XGBoost-Tuned"       : best_xgb,
        }
        best_reg_model  = model_map[best_result["label"]]
        best_reg_params = (rf_search.best_params_
                           if "RandomForest" in best_result["label"]
                           else xgb_search.best_params_
                           if "XGBoost" in best_result["label"]
                           else {})

        # Save best regressor
        with open(REGRESSOR_PATH, "wb") as f:
            pickle.dump(best_reg_model, f)
        log.info(f"  Best regressor saved → {REGRESSOR_PATH}")

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
            scoring="f1", random_state=42, n_jobs=-1
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
            scoring="f1", random_state=42, n_jobs=-1
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

        # Save best classifier
        with open(CLASSIFIER_PATH, "wb") as f:
            pickle.dump(best_clf_model, f)
        log.info(f"  Best classifier saved → {CLASSIFIER_PATH}")

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

        train, val = load_splits()

        X_train = train[FEATURE_COLS].values
        X_val   = val[FEATURE_COLS].values
        y_train_reg = train[TARGET_REGRESSION].values
        y_val_reg   = val[TARGET_REGRESSION].values
        y_train_clf = train[TARGET_CLASSIFICATION].values
        y_val_clf   = val[TARGET_CLASSIFICATION].values

        # Train regressors
        best_reg, best_reg_params, reg_results = train_regressors(
            X_train, y_train_reg, X_val, y_val_reg
        )

        # Train classifiers
        best_clf, best_clf_params, clf_results = train_classifiers(
            X_train, y_train_clf, X_val, y_val_clf
        )

        # Auto-update config.py
        update_config_best_params(best_reg_params, best_clf_params)

        log.info("Model training complete.")
        return best_reg, best_clf, reg_results, clf_results

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    best_reg, best_clf, reg_results, clf_results = run_model_trainer()

    print("\n--- REGRESSION RESULTS SUMMARY ---")
    print(f"{'Model':<30} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 58)
    for r in reg_results:
        print(f"{r['label']:<30} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>8.4f}")

    print("\n--- CLASSIFICATION RESULTS SUMMARY ---")
    print(f"{'Model':<30} {'Accuracy':>10} {'F1':>8} {'AUC':>8}")
    print("-" * 60)
    for r in clf_results:
        print(f"{r['label']:<30} {r['Accuracy']:>10.4f} {r['F1']:>8.4f} {r['AUC']:>8.4f}")