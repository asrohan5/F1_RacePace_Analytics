import os
import sys
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (mean_absolute_error, r2_score,
                             roc_auc_score, f1_score, accuracy_score)
import xgboost as xgb
import lightgbm as lgb

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TRAIN_SC_PATH, VAL_SC_PATH, TEST_SC_PATH,
    REGRESSOR_PATH, CLASSIFIER_PATH,
    REGRESSOR_SC_PATH, CLASSIFIER_SC_PATH,
    TARGET_REGRESSION, TARGET_CLASSIFICATION,
    INITIAL_MODEL_PARAMS,
)


# ─────────────────────────────────────────
# FEATURE COLUMNS
# Must match data_transformation.py FEATURE_COLS exactly
# ID and target columns are excluded from model input
# ─────────────────────────────────────────

ID_COLS = ["Race", "RoundNumber", "LapNumber",
           TARGET_REGRESSION, TARGET_CLASSIFICATION]

# Derived at load time from whatever columns remain after dropping ID_COLS
def get_feature_cols(df):
    return [c for c in df.columns if c not in ID_COLS]


# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────

def load_splits(sc=False):
    try:
        if sc:
            train = pd.read_csv(TRAIN_SC_PATH)
            val   = pd.read_csv(VAL_SC_PATH)
            test  = pd.read_csv(TEST_SC_PATH)
            label = "SC"
        else:
            train = pd.read_csv(TRAIN_PATH)
            val   = pd.read_csv(VAL_PATH)
            test  = pd.read_csv(TEST_PATH)
            label = "Full"

        log.info(f"[{label}] Loaded — train:{train.shape} val:{val.shape} "
                 f"test:{test.shape}")
        return train, val, test

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# GROUP K-FOLD CV
# LORO-aware: each race is one group → never mixed across folds
# Uses Race column as group label
# ─────────────────────────────────────────

def make_cv_groups(train_df):
    """Map Race to integer group labels for GroupKFold."""
    races  = train_df["Race"].values
    unique = sorted(set(races))
    race_to_int = {r: i for i, r in enumerate(unique)}
    return np.array([race_to_int[r] for r in races])


def loro_cv_score(model, X_train, y_train, groups, scoring, n_splits=None):
    """
    Run GroupKFold CV where each fold holds out one race.
    n_splits defaults to number of unique groups (true leave-one-race-out).
    Returns array of per-fold scores.
    """
    n_groups = len(set(groups))
    k = n_splits if n_splits else n_groups
    gkf = GroupKFold(n_splits=k)
    scores = cross_val_score(model, X_train, y_train,
                             cv=gkf, groups=groups,
                             scoring=scoring, n_jobs=1)
    return scores


# ─────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────

def eval_regressor(model, X, y, label):
    pred = model.predict(X)
    mae  = mean_absolute_error(y, pred)
    r2   = r2_score(y, pred)
    log.info(f"  [{label}] MAE={mae:.4f}s  R²={r2:.4f}")
    return mae, r2


def eval_classifier(model, X, y, label):
    pred      = model.predict(X)
    pred_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else pred
    acc  = accuracy_score(y, pred)
    f1   = f1_score(y, pred, zero_division=0)
    auc  = roc_auc_score(y, pred_prob)
    log.info(f"  [{label}] Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return acc, f1, auc


# ─────────────────────────────────────────
# REGRESSION TRAINING
# ─────────────────────────────────────────

def train_regressors(train_df, val_df, sc=False):
    try:
        label = "SC" if sc else "Full"
        log.info("=" * 60)
        log.info(f"REGRESSION — {label} dataset")
        log.info("=" * 60)

        feature_cols = get_feature_cols(train_df)
        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_REGRESSION].values
        X_val   = val_df[feature_cols].values
        y_val   = val_df[TARGET_REGRESSION].values
        groups  = make_cv_groups(train_df)

        results = {}

        # ── Baseline ────────────────────────────────────────
        log.info("\nBaseline — DummyRegressor (mean strategy)")
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        dummy_cv = loro_cv_score(dummy, X_train, y_train, groups,
                                 scoring="neg_mean_absolute_error")
        log.info(f"  CV MAE: {-dummy_cv.mean():.4f}s ± {dummy_cv.std():.4f}s")
        eval_regressor(dummy, X_val, y_val, "Val")
        results["dummy"] = {"model": dummy, "cv_mae": -dummy_cv.mean()}

        # ── Linear models ───────────────────────────────────
        log.info("\nLinearRegression")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_cv = loro_cv_score(lr, X_train, y_train, groups,
                              scoring="neg_mean_absolute_error")
        log.info(f"  CV MAE: {-lr_cv.mean():.4f}s ± {lr_cv.std():.4f}s")
        eval_regressor(lr, X_val, y_val, "Val")
        results["linear"] = {"model": lr, "cv_mae": -lr_cv.mean()}

        log.info("\nRidgeCV (alphas=[0.01, 0.1, 1, 10, 100])")
        ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
        ridge.fit(X_train, y_train)
        ridge_cv = loro_cv_score(ridge, X_train, y_train, groups,
                                 scoring="neg_mean_absolute_error")
        log.info(f"  Best alpha: {ridge.alpha_:.4f}")
        log.info(f"  CV MAE: {-ridge_cv.mean():.4f}s ± {ridge_cv.std():.4f}s")
        eval_regressor(ridge, X_val, y_val, "Val")
        results["ridge"] = {"model": ridge, "cv_mae": -ridge_cv.mean()}

        log.info("\nLassoCV (feature selection insight)")
        lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5, max_iter=5000)
        lasso.fit(X_train, y_train)
        n_nonzero = np.sum(lasso.coef_ != 0)
        log.info(f"  Best alpha: {lasso.alpha_:.6f}")
        log.info(f"  Non-zero coefficients: {n_nonzero}/{len(feature_cols)}")
        lasso_cv = loro_cv_score(lasso, X_train, y_train, groups,
                                 scoring="neg_mean_absolute_error")
        log.info(f"  CV MAE: {-lasso_cv.mean():.4f}s ± {lasso_cv.std():.4f}s")
        eval_regressor(lasso, X_val, y_val, "Val")
        results["lasso"] = {"model": lasso, "cv_mae": -lasso_cv.mean()}

        # Log which features Lasso kept
        kept = [(feature_cols[i], round(lasso.coef_[i], 4))
                for i in range(len(feature_cols)) if lasso.coef_[i] != 0]
        kept_sorted = sorted(kept, key=lambda x: abs(x[1]), reverse=True)
        log.info(f"  Lasso selected features (sorted by |coef|):")
        for feat, coef in kept_sorted:
            log.info(f"    {feat:<40} {coef:+.4f}")

        # ── RandomForest ────────────────────────────────────
        log.info("\nRandomForest — RandomizedSearchCV (LORO groups)")
        rf_param_grid = {
            "n_estimators"    : [100, 200, 300],
            "max_depth"       : [3, 5, 7, None],
            "min_samples_leaf": [4, 8, 16],
            "max_features"    : ["sqrt", "log2", 0.5],
        }
        rf_base = RandomForestRegressor(random_state=42, n_jobs=1)
        n_groups = len(set(groups))
        gkf = GroupKFold(n_splits=n_groups)
        rf_search = RandomizedSearchCV(
            rf_base, rf_param_grid,
            n_iter=20, cv=gkf,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=1, verbose=0
        )
        rf_search.fit(X_train, y_train, groups=groups)
        best_rf = rf_search.best_estimator_
        rf_cv_mae = -rf_search.best_score_
        log.info(f"  Best params : {rf_search.best_params_}")
        log.info(f"  CV MAE      : {rf_cv_mae:.4f}s")
        eval_regressor(best_rf, X_val, y_val, "Val")
        results["rf"] = {"model": best_rf, "cv_mae": rf_cv_mae}

        # ── XGBoost ─────────────────────────────────────────
        log.info("\nXGBoost — RandomizedSearchCV (LORO groups)")
        xgb_param_grid = {
            "n_estimators"     : [100, 200, 300],
            "max_depth"        : [3, 4, 5, 6],
            "learning_rate"    : [0.01, 0.05, 0.1, 0.2],
            "subsample"        : [0.6, 0.8, 1.0],
            "colsample_bytree" : [0.6, 0.8, 1.0],
            "reg_alpha"        : [0, 0.1, 1],
            "reg_lambda"       : [1, 5, 10],
        }
        xgb_base = xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=1)
        xgb_search = RandomizedSearchCV(
            xgb_base, xgb_param_grid,
            n_iter=20, cv=gkf,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=1, verbose=0
        )
        xgb_search.fit(X_train, y_train, groups=groups)
        best_xgb = xgb_search.best_estimator_
        xgb_cv_mae = -xgb_search.best_score_
        log.info(f"  Best params : {xgb_search.best_params_}")
        log.info(f"  CV MAE      : {xgb_cv_mae:.4f}s")
        eval_regressor(best_xgb, X_val, y_val, "Val")
        results["xgb"] = {"model": best_xgb, "cv_mae": xgb_cv_mae}

        # ── LightGBM ────────────────────────────────────────
        log.info("\nLightGBM — RandomizedSearchCV (LORO groups)")
        lgb_param_grid = {
            "n_estimators"  : [100, 200, 300],
            "max_depth"     : [3, 4, 5, 6, -1],
            "learning_rate" : [0.01, 0.05, 0.1, 0.2],
            "num_leaves"    : [15, 31, 63],
            "subsample"     : [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha"     : [0, 0.1, 1],
            "reg_lambda"    : [1, 5, 10],
        }
        lgb_base = lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=1)
        lgb_search = RandomizedSearchCV(
            lgb_base, lgb_param_grid,
            n_iter=20, cv=gkf,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=1, verbose=0
        )
        lgb_search.fit(X_train, y_train, groups=groups)
        best_lgb = lgb_search.best_estimator_
        lgb_cv_mae = -lgb_search.best_score_
        log.info(f"  Best params : {lgb_search.best_params_}")
        log.info(f"  CV MAE      : {lgb_cv_mae:.4f}s")
        eval_regressor(best_lgb, X_val, y_val, "Val")
        results["lgb"] = {"model": best_lgb, "cv_mae": lgb_cv_mae}

        # ── Select best by CV MAE ────────────────────────────
        log.info("\n--- REGRESSION SUMMARY (ranked by CV MAE) ---")
        ranked = sorted(results.items(), key=lambda x: x[1]["cv_mae"])
        for name, info in ranked:
            log.info(f"  {name:<10} CV MAE={info['cv_mae']:.4f}s")

        best_name, best_info = ranked[0]
        best_model = best_info["model"]
        log.info(f"\n  Winner: {best_name} (CV MAE={best_info['cv_mae']:.4f}s)")

        return best_model, results

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# CLASSIFICATION TRAINING
# ─────────────────────────────────────────

def train_classifiers(train_df, val_df, sc=False):
    try:
        label = "SC" if sc else "Full"
        log.info("=" * 60)
        log.info(f"CLASSIFICATION — {label} dataset")
        log.info("=" * 60)

        feature_cols = get_feature_cols(train_df)
        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_CLASSIFICATION].values
        X_val   = val_df[feature_cols].values
        y_val   = val_df[TARGET_CLASSIFICATION].values
        groups  = make_cv_groups(train_df)
        n_groups = len(set(groups))
        gkf = GroupKFold(n_splits=n_groups)

        results = {}

        # ── Baseline ────────────────────────────────────────
        log.info("\nBaseline — DummyClassifier (most_frequent)")
        dummy = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy.fit(X_train, y_train)
        dummy_cv = loro_cv_score(dummy, X_train, y_train, groups,
                                 scoring="roc_auc")
        log.info(f"  CV AUC: {dummy_cv.mean():.4f} ± {dummy_cv.std():.4f}")
        eval_classifier(dummy, X_val, y_val, "Val")
        results["dummy"] = {"model": dummy, "cv_auc": dummy_cv.mean()}

        # ── Logistic Regression ─────────────────────────────
        log.info("\nLogisticRegression")
        lr = LogisticRegression(
            C=INITIAL_MODEL_PARAMS["classifier"]["C"],
            max_iter=INITIAL_MODEL_PARAMS["classifier"]["max_iter"],
            random_state=42
        )
        lr.fit(X_train, y_train)
        lr_cv = loro_cv_score(lr, X_train, y_train, groups, scoring="roc_auc")
        log.info(f"  CV AUC: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
        eval_classifier(lr, X_val, y_val, "Val")
        results["logistic"] = {"model": lr, "cv_auc": lr_cv.mean()}

        # ── RandomForest ────────────────────────────────────
        log.info("\nRandomForest classifier — RandomizedSearchCV (LORO groups)")
        rf_param_grid = {
            "n_estimators"    : [100, 200, 300],
            "max_depth"       : [3, 5, 7, None],
            "min_samples_leaf": [4, 8, 16],
            "max_features"    : ["sqrt", "log2", 0.5],
        }
        rf_base = RandomForestClassifier(random_state=42, class_weight="balanced",
                                         n_jobs=1)
        rf_search = RandomizedSearchCV(
            rf_base, rf_param_grid,
            n_iter=20, cv=gkf,
            scoring="roc_auc",
            random_state=42, n_jobs=1, verbose=0
        )
        rf_search.fit(X_train, y_train, groups=groups)
        best_rf = rf_search.best_estimator_
        rf_cv_auc = rf_search.best_score_
        log.info(f"  Best params : {rf_search.best_params_}")
        log.info(f"  CV AUC      : {rf_cv_auc:.4f}")
        eval_classifier(best_rf, X_val, y_val, "Val")
        results["rf"] = {"model": best_rf, "cv_auc": rf_cv_auc}

        # ── XGBoost ─────────────────────────────────────────
        log.info("\nXGBoost classifier — RandomizedSearchCV (LORO groups)")
        xgb_param_grid = {
            "n_estimators"    : [100, 200, 300],
            "max_depth"       : [3, 4, 5, 6],
            "learning_rate"   : [0.01, 0.05, 0.1, 0.2],
            "subsample"       : [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha"       : [0, 0.1, 1],
            "reg_lambda"      : [1, 5, 10],
        }
        scale_pos = int((y_train == 0).sum()) / int((y_train == 1).sum())
        xgb_base = xgb.XGBClassifier(random_state=42, verbosity=0,
                                      scale_pos_weight=scale_pos, n_jobs=1,
                                      eval_metric="auc")
        xgb_search = RandomizedSearchCV(
            xgb_base, xgb_param_grid,
            n_iter=20, cv=gkf,
            scoring="roc_auc",
            random_state=42, n_jobs=1, verbose=0
        )
        xgb_search.fit(X_train, y_train, groups=groups)
        best_xgb = xgb_search.best_estimator_
        xgb_cv_auc = xgb_search.best_score_
        log.info(f"  Best params : {xgb_search.best_params_}")
        log.info(f"  CV AUC      : {xgb_cv_auc:.4f}")
        eval_classifier(best_xgb, X_val, y_val, "Val")
        results["xgb"] = {"model": best_xgb, "cv_auc": xgb_cv_auc}

        # ── LightGBM ────────────────────────────────────────
        log.info("\nLightGBM classifier — RandomizedSearchCV (LORO groups)")
        lgb_param_grid = {
            "n_estimators"    : [100, 200, 300],
            "max_depth"       : [3, 4, 5, 6, -1],
            "learning_rate"   : [0.01, 0.05, 0.1, 0.2],
            "num_leaves"      : [15, 31, 63],
            "subsample"       : [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha"       : [0, 0.1, 1],
            "reg_lambda"      : [1, 5, 10],
        }
        lgb_base = lgb.LGBMClassifier(random_state=42, verbosity=-1,
                                       class_weight="balanced", n_jobs=1)
        lgb_search = RandomizedSearchCV(
            lgb_base, lgb_param_grid,
            n_iter=20, cv=gkf,
            scoring="roc_auc",
            random_state=42, n_jobs=1, verbose=0
        )
        lgb_search.fit(X_train, y_train, groups=groups)
        best_lgb = lgb_search.best_estimator_
        lgb_cv_auc = lgb_search.best_score_
        log.info(f"  Best params : {lgb_search.best_params_}")
        log.info(f"  CV AUC      : {lgb_cv_auc:.4f}")
        eval_classifier(best_lgb, X_val, y_val, "Val")
        results["lgb"] = {"model": best_lgb, "cv_auc": lgb_cv_auc}

        # ── Select best by CV AUC ────────────────────────────
        log.info("\n--- CLASSIFICATION SUMMARY (ranked by CV AUC) ---")
        ranked = sorted(results.items(), key=lambda x: x[1]["cv_auc"], reverse=True)
        for name, info in ranked:
            log.info(f"  {name:<10} CV AUC={info['cv_auc']:.4f}")

        best_name, best_info = ranked[0]
        best_model = best_info["model"]
        log.info(f"\n  Winner: {best_name} (CV AUC={best_info['cv_auc']:.4f})")

        return best_model, results

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────

def save_model(model, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"  Saved → {path}")
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_training():
    try:
        log.info("=" * 60)
        log.info("Starting model training — Phase 2")
        log.info("Primary metric: CV MAE (regression), CV AUC (classification)")
        log.info("CV strategy   : Leave-One-Race-Out (GroupKFold by Race)")
        log.info("=" * 60)

        # ── Full dataset ─────────────────────────────────────
        log.info("\n>>> FULL DATASET")
        train_df, val_df, test_df = load_splits(sc=False)

        best_reg,  reg_results  = train_regressors(train_df, val_df, sc=False)
        best_clf,  clf_results  = train_classifiers(train_df, val_df, sc=False)

        save_model(best_reg, REGRESSOR_PATH)
        save_model(best_clf, CLASSIFIER_PATH)

        # ── SC dataset ──────────────────────────────────────
        log.info("\n>>> SAME-COMPOUND DATASET")
        sc_train_df, sc_val_df, sc_test_df = load_splits(sc=True)

        best_reg_sc, reg_sc_results = train_regressors(sc_train_df, sc_val_df, sc=True)
        best_clf_sc, clf_sc_results = train_classifiers(sc_train_df, sc_val_df, sc=True)

        save_model(best_reg_sc, REGRESSOR_SC_PATH)
        save_model(best_clf_sc, CLASSIFIER_SC_PATH)

        log.info("\n" + "=" * 60)
        log.info("Model training complete. 4 models saved.")
        log.info("=" * 60)

        return (best_reg, best_clf, reg_results, clf_results,
                best_reg_sc, best_clf_sc, reg_sc_results, clf_sc_results)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    (best_reg, best_clf, reg_results, clf_results,
     best_reg_sc, best_clf_sc, reg_sc_results, clf_sc_results) = run_training()

    # ── Final val comparison table ───────────────────────────
    log.info("\n" + "=" * 60)
    log.info("FINAL MODEL COMPARISON — VAL SET")
    log.info("=" * 60)

    train_df, val_df, _   = load_splits(sc=False)
    sc_train_df, sc_val_df, _ = load_splits(sc=True)

    feature_cols    = get_feature_cols(train_df)
    feature_cols_sc = get_feature_cols(sc_train_df)

    X_val    = val_df[feature_cols].values
    y_val_r  = val_df[TARGET_REGRESSION].values
    y_val_c  = val_df[TARGET_CLASSIFICATION].values

    X_sc_val    = sc_val_df[feature_cols_sc].values
    y_sc_val_r  = sc_val_df[TARGET_REGRESSION].values
    y_sc_val_c  = sc_val_df[TARGET_CLASSIFICATION].values

    log.info("\n[FULL — Regression] Val MAE / R²")
    for name, info in sorted(reg_results.items(),
                              key=lambda x: x[1]["cv_mae"]):
        m = info["model"]
        mae, r2 = mean_absolute_error(y_val_r, m.predict(X_val)), \
                  r2_score(y_val_r, m.predict(X_val))
        log.info(f"  {name:<10}  CV_MAE={info['cv_mae']:.4f}  "
              f"Val_MAE={mae:.4f}  Val_R²={r2:.4f}")

    log.info("\n[FULL — Classification] Val AUC / F1")
    for name, info in sorted(clf_results.items(),
                              key=lambda x: x[1]["cv_auc"], reverse=True):
        m = info["model"]
        pred_prob = m.predict_proba(X_val)[:, 1] \
                    if hasattr(m, "predict_proba") else m.predict(X_val)
        auc = roc_auc_score(y_val_c, pred_prob)
        f1  = f1_score(y_val_c, m.predict(X_val), zero_division=0)
        log.info(f"  {name:<10}  CV_AUC={info['cv_auc']:.4f}  "
              f"Val_AUC={auc:.4f}  Val_F1={f1:.4f}")

    log.info("\n[SC — Regression] Val MAE / R²")
    for name, info in sorted(reg_sc_results.items(),
                              key=lambda x: x[1]["cv_mae"]):
        m = info["model"]
        mae, r2 = mean_absolute_error(y_sc_val_r, m.predict(X_sc_val)), \
                  r2_score(y_sc_val_r, m.predict(X_sc_val))
        log.info(f"  {name:<10}  CV_MAE={info['cv_mae']:.4f}  "
              f"Val_MAE={mae:.4f}  Val_R²={r2:.4f}")

    log.info("\n[SC — Classification] Val AUC / F1")
    for name, info in sorted(clf_sc_results.items(),
                              key=lambda x: x[1]["cv_auc"], reverse=True):
        m = info["model"]
        pred_prob = m.predict_proba(X_sc_val)[:, 1] \
                    if hasattr(m, "predict_proba") else m.predict(X_sc_val)
        auc = roc_auc_score(y_sc_val_c, pred_prob)
        f1  = f1_score(y_sc_val_c, m.predict(X_sc_val), zero_division=0)
        log.info(f"  {name:<10}  CV_AUC={info['cv_auc']:.4f}  "
              f"Val_AUC={auc:.4f}  Val_F1={f1:.4f}")

    log.info("\nModels saved:")
    log.info(f"  regressor.pkl    → best full regression model")
    log.info(f"  classifier.pkl   → best full classification model")
    log.info(f"  regressor_sc.pkl → best SC regression model")
    log.info(f"  classifier_sc.pkl→ best SC classification model")