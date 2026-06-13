import os
import sys
import json
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import (LinearRegression, RidgeCV, LassoCV,
                                   ElasticNetCV, LogisticRegression)
from sklearn.svm import SVR, SVC
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               VotingRegressor, VotingClassifier)
from sklearn.calibration import CalibratedClassifierCV
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
    CV_SCORES_PATH,
    TARGET_REGRESSION, TARGET_CLASSIFICATION,
    INITIAL_MODEL_PARAMS,
)

ID_COLS = ["Race", "RoundNumber", "LapNumber",
           TARGET_REGRESSION, TARGET_CLASSIFICATION]


def get_feature_cols(df):
    return [c for c in df.columns if c not in ID_COLS]


def load_splits(sc=False):
    try:
        if sc:
            train = pd.read_csv(TRAIN_SC_PATH)
            val = pd.read_csv(VAL_SC_PATH)
            test = pd.read_csv(TEST_SC_PATH)
            label = "SC"
        else:
            train = pd.read_csv(TRAIN_PATH)
            val = pd.read_csv(VAL_PATH)
            test = pd.read_csv(TEST_PATH)
            label = "Full"

        log.info(f"[{label}] train:{train.shape} val:{val.shape} test:{test.shape}")
        return train, val, test

    except Exception as e:
        raise CustomException(e, sys)


def make_cv_groups(train_df):
    """Map each race to an integer group label for GroupKFold."""
    races = train_df["Race"].values
    unique = sorted(set(races))
    race_to_int = {r: i for i, r in enumerate(unique)}
    return np.array([race_to_int[r] for r in races])


def loro_cv_score(model, X_train, y_train, groups, scoring, n_splits=None):
    """
    GroupKFold cross-validation where each fold holds out one race entirely.
    This is leave-one-race-out CV — no lap from a held-out race appears in training.
    """
    n_groups = len(set(groups))
    k = n_splits if n_splits else n_groups
    gkf = GroupKFold(n_splits=k)
    scores = cross_val_score(model, X_train, y_train,
                             cv=gkf, groups=groups,
                             scoring=scoring, n_jobs=1)
    return scores


def eval_regressor(model, X, y, label):
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    log.info(f"  [{label}] MAE={mae:.4f}s  R2={r2:.4f}")
    return mae, r2


def eval_classifier(model, X, y, label):
    pred = model.predict(X)
    pred_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else pred
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, pred_prob)
    log.info(f"  [{label}] Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return acc, f1, auc


def train_regressors(train_df, val_df, sc=False):
    try:
        label = "SC" if sc else "Full"
        log.info(f"REGRESSION -- {label} dataset")

        feature_cols = get_feature_cols(train_df)
        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_REGRESSION].values
        X_val = val_df[feature_cols].values
        y_val = val_df[TARGET_REGRESSION].values
        groups = make_cv_groups(train_df)
        n_groups = len(set(groups))
        gkf = GroupKFold(n_splits=n_groups)
        results = {}
        cv_fold_scores = {}

        log.info("Baseline -- DummyRegressor (mean strategy)")
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        dummy_cv = loro_cv_score(dummy, X_train, y_train, groups,
                                 scoring="neg_mean_absolute_error")
        log.info(f"  CV MAE: {-dummy_cv.mean():.4f}s +/- {dummy_cv.std():.4f}s")
        _, dummy_val_r2 = eval_regressor(dummy, X_val, y_val, "Val")
        results["dummy"] = {"model": dummy, "cv_mae": -dummy_cv.mean(), "val_r2": dummy_val_r2}
        cv_fold_scores["dummy"] = (-dummy_cv).tolist()

        log.info("LinearRegression")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_cv = loro_cv_score(lr, X_train, y_train, groups,
                              scoring="neg_mean_absolute_error")
        log.info(f"  CV MAE: {-lr_cv.mean():.4f}s +/- {lr_cv.std():.4f}s")
        _, lr_val_r2 = eval_regressor(lr, X_val, y_val, "Val")
        results["linear"] = {"model": lr, "cv_mae": -lr_cv.mean(), "val_r2": lr_val_r2}
        cv_fold_scores["linear"] = (-lr_cv).tolist()

        log.info("RidgeCV")
        ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
        ridge.fit(X_train, y_train)
        ridge_cv = loro_cv_score(ridge, X_train, y_train, groups,
                                 scoring="neg_mean_absolute_error")
        log.info(f"  Best alpha: {ridge.alpha_:.4f}")
        log.info(f"  CV MAE: {-ridge_cv.mean():.4f}s +/- {ridge_cv.std():.4f}s")
        _, ridge_val_r2 = eval_regressor(ridge, X_val, y_val, "Val")
        results["ridge"] = {"model": ridge, "cv_mae": -ridge_cv.mean(), "val_r2": ridge_val_r2}
        cv_fold_scores["ridge"] = (-ridge_cv).tolist()

        log.info("LassoCV")
        lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5, max_iter=5000)
        lasso.fit(X_train, y_train)
        n_nonzero = np.sum(lasso.coef_ != 0)
        log.info(f"  Best alpha: {lasso.alpha_:.6f}")
        log.info(f"  Non-zero coefficients: {n_nonzero}/{len(feature_cols)}")
        lasso_cv = loro_cv_score(lasso, X_train, y_train, groups,
                                 scoring="neg_mean_absolute_error")
        log.info(f"  CV MAE: {-lasso_cv.mean():.4f}s +/- {lasso_cv.std():.4f}s")
        _, lasso_val_r2 = eval_regressor(lasso, X_val, y_val, "Val")
        results["lasso"] = {"model": lasso, "cv_mae": -lasso_cv.mean(), "val_r2": lasso_val_r2}
        cv_fold_scores["lasso"] = (-lasso_cv).tolist()

        kept = [(feature_cols[i], round(lasso.coef_[i], 4))
                for i in range(len(feature_cols)) if lasso.coef_[i] != 0]
        kept_sorted = sorted(kept, key=lambda x: abs(x[1]), reverse=True)
        log.info("  Lasso selected features (sorted by |coef|):")
        for feat, coef in kept_sorted:
            log.info(f"    {feat:<40} {coef:+.4f}")

        log.info("ElasticNetCV")
        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
            alphas=[0.001, 0.01, 0.1, 1],
            cv=5, max_iter=5000
        )
        enet.fit(X_train, y_train)
        enet_nonzero = np.sum(enet.coef_ != 0)
        log.info(f"  Best alpha={enet.alpha_:.6f}  l1_ratio={enet.l1_ratio_:.2f}")
        log.info(f"  Non-zero coefficients: {enet_nonzero}/{len(feature_cols)}")
        enet_cv = loro_cv_score(enet, X_train, y_train, groups,
                                scoring="neg_mean_absolute_error")
        log.info(f"  CV MAE: {-enet_cv.mean():.4f}s +/- {enet_cv.std():.4f}s")
        _, enet_val_r2 = eval_regressor(enet, X_val, y_val, "Val")
        results["elasticnet"] = {"model": enet, "cv_mae": -enet_cv.mean(), "val_r2": enet_val_r2}
        cv_fold_scores["elasticnet"] = (-enet_cv).tolist()

        log.info("SVR (RBF kernel)")
        svr_param_grid = {
            "C": [0.1, 0.5, 1, 5, 10, 50],
            "epsilon": [0.05, 0.1, 0.2, 0.3],
            "gamma": ["scale", "auto"],
        }
        svr_search = RandomizedSearchCV(
            SVR(kernel="rbf"),
            svr_param_grid,
            n_iter=30, cv=gkf,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=1, verbose=0
        )
        svr_search.fit(X_train, y_train, groups=groups)
        best_svr = svr_search.best_estimator_
        svr_cv_mae = -svr_search.best_score_
        log.info(f"  Best params: {svr_search.best_params_}")
        log.info(f"  CV MAE: {svr_cv_mae:.4f}s")
        _, svr_val_r2 = eval_regressor(best_svr, X_val, y_val, "Val")
        results["svr"] = {"model": best_svr, "cv_mae": svr_cv_mae, "val_r2": svr_val_r2}

        log.info("RandomForest Regressor")
        rf_param_grid = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [8, 16, 32],
            "max_features": ["sqrt", "log2", 0.4, 0.5],
            "min_impurity_decrease": [0.0, 0.001, 0.005],
        }
        rf_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=1),
            rf_param_grid,
            n_iter=50, cv=gkf,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=1, verbose=0
        )
        rf_search.fit(X_train, y_train, groups=groups)
        best_rf = rf_search.best_estimator_
        rf_cv_mae = -rf_search.best_score_
        log.info(f"  Best params: {rf_search.best_params_}")
        log.info(f"  CV MAE: {rf_cv_mae:.4f}s")
        _, rf_val_r2 = eval_regressor(best_rf, X_val, y_val, "Val")
        results["rf"] = {"model": best_rf, "cv_mae": rf_cv_mae, "val_r2": rf_val_r2}

        log.info("XGBoost Regressor")
        xgb_param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
            "colsample_bylevel": [0.5, 0.7, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.5, 1, 5, 10],
            "min_child_weight": [1, 3, 5, 10],
        }
        xgb_search = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=1),
            {k: v for k, v in xgb_param_grid.items() if k != "n_estimators"},
            n_iter=50, cv=gkf,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=1, verbose=0
        )
        xgb_search.fit(X_train, y_train, groups=groups)
        best_xgb_params = xgb_search.best_params_

        # The val set (Brazil and Qatar) was chosen as an adversarial hold-out,
        # not a calibration set. Using it for early stopping causes premature
        # termination because the model correctly cannot improve on those races
        # using style features alone. n_estimators is fixed at 300 which is
        # within the searched range and avoids overfitting on training data
        # while giving the model sufficient capacity.
        xgb_n_estimators = 300
        best_xgb = xgb.XGBRegressor(
            **best_xgb_params,
            n_estimators=xgb_n_estimators,
            random_state=42, verbosity=0, n_jobs=1
        )
        best_xgb.fit(X_train, y_train)
        xgb_cv_mae = -xgb_search.best_score_
        log.info(f"  Best params: {best_xgb_params}")
        log.info(f"  n_estimators: {xgb_n_estimators}")
        log.info(f"  CV MAE: {xgb_cv_mae:.4f}s")
        _, xgb_val_r2 = eval_regressor(best_xgb, X_val, y_val, "Val")
        results["xgb"] = {"model": best_xgb, "cv_mae": xgb_cv_mae, "val_r2": xgb_val_r2}

        log.info("LightGBM Regressor")
        lgb_param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [7, 15, 31],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.5, 1, 5, 10],
            "min_child_samples": [10, 20, 30, 50],
        }
        lgb_search = RandomizedSearchCV(
            lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=1),
            {k: v for k, v in lgb_param_grid.items() if k != "n_estimators"},
            n_iter=50, cv=gkf,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=1, verbose=0
        )
        lgb_search.fit(X_train, y_train, groups=groups)
        best_lgb_params = lgb_search.best_params_

        lgb_n_estimators = 300
        best_lgb = lgb.LGBMRegressor(
            **best_lgb_params,
            n_estimators=lgb_n_estimators,
            random_state=42, verbosity=-1, n_jobs=1
        )
        best_lgb.fit(X_train, y_train)
        lgb_cv_mae = -lgb_search.best_score_
        log.info(f"  Best params: {best_lgb_params}")
        log.info(f"  n_estimators: {lgb_n_estimators}")
        log.info(f"  CV MAE: {lgb_cv_mae:.4f}s")
        _, lgb_val_r2 = eval_regressor(best_lgb, X_val, y_val, "Val")
        results["lgb"] = {"model": best_lgb, "cv_mae": lgb_cv_mae, "val_r2": lgb_val_r2}

        # VotingRegressor calls fit() internally without an eval_set,
        # so constituent estimators must not require one. Using the same
        # fixed n_estimators as the individual models keeps the ensemble consistent.
        log.info("Ensemble VotingRegressor (XGB + LGB)")
        xgb_for_ensemble = xgb.XGBRegressor(
            **best_xgb_params,
            n_estimators=xgb_n_estimators,
            random_state=42, verbosity=0, n_jobs=1
        )
        lgb_for_ensemble = lgb.LGBMRegressor(
            **best_lgb_params,
            n_estimators=lgb_n_estimators,
            random_state=42, verbosity=-1, n_jobs=1
        )
        ensemble_reg = VotingRegressor([
            ("xgb", xgb_for_ensemble),
            ("lgb", lgb_for_ensemble),
        ])
        ensemble_reg.fit(X_train, y_train)
        ens_cv = loro_cv_score(ensemble_reg, X_train, y_train, groups,
                               scoring="neg_mean_absolute_error")
        ens_cv_mae = -ens_cv.mean()
        log.info(f"  CV MAE: {ens_cv_mae:.4f}s +/- {ens_cv.std():.4f}s")
        _, ens_val_r2 = eval_regressor(ensemble_reg, X_val, y_val, "Val")
        results["ensemble"] = {"model": ensemble_reg, "cv_mae": ens_cv_mae, "val_r2": ens_val_r2}
        cv_fold_scores["ensemble"] = (-ens_cv).tolist()

        log.info("REGRESSION SUMMARY (ranked by CV MAE)")
        ranked = sorted(results.items(), key=lambda x: x[1]["cv_mae"])
        for name, info in ranked:
            log.info(f"  {name:<12} CV MAE={info['cv_mae']:.4f}s  Val R2={info['val_r2']:.4f}")

        TIEBREAK_THRESHOLD = 0.01
        best_cv_mae = ranked[0][1]["cv_mae"]
        candidates = [(n, i) for n, i in ranked
                      if i["cv_mae"] <= best_cv_mae + TIEBREAK_THRESHOLD]

        if len(candidates) > 1:
            log.info(f"  {len(candidates)} models within {TIEBREAK_THRESHOLD}s of best CV MAE -- val R2 tiebreaker")
            candidates_sorted = sorted(candidates, key=lambda x: x[1]["val_r2"], reverse=True)
            best_name, best_info = candidates_sorted[0]
            candidate_summary = [(n, f"CV={i['cv_mae']:.4f} R2={i['val_r2']:.4f}")
                        for n, i in candidates]
            log.info(f"  Tiebreaker candidates: {candidate_summary}")
        else:
            best_name, best_info = ranked[0]

        log.info(f"  Winner: {best_name} (CV MAE={best_info['cv_mae']:.4f}s  Val R2={best_info['val_r2']:.4f})")

        return best_info["model"], results, cv_fold_scores

    except Exception as e:
        raise CustomException(e, sys)


def train_classifiers(train_df, val_df, sc=False):
    try:
        label = "SC" if sc else "Full"
        log.info(f"CLASSIFICATION -- {label} dataset")

        feature_cols = get_feature_cols(train_df)
        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_CLASSIFICATION].values
        X_val = val_df[feature_cols].values
        y_val = val_df[TARGET_CLASSIFICATION].values
        groups = make_cv_groups(train_df)
        n_groups = len(set(groups))
        gkf = GroupKFold(n_splits=n_groups)
        results = {}
        cv_fold_scores = {}

        log.info("Baseline -- DummyClassifier (most_frequent)")
        dummy = DummyClassifier(strategy="most_frequent", random_state=42)
        dummy.fit(X_train, y_train)
        dummy_cv = loro_cv_score(dummy, X_train, y_train, groups, scoring="roc_auc")
        log.info(f"  CV AUC: {dummy_cv.mean():.4f} +/- {dummy_cv.std():.4f}")
        eval_classifier(dummy, X_val, y_val, "Val")
        results["dummy"] = {"model": dummy, "cv_auc": dummy_cv.mean()}
        cv_fold_scores["dummy"] = dummy_cv.tolist()

        log.info("LogisticRegression (C searched via LORO CV)")
        best_lr_auc, best_lr, best_lr_C = 0, None, 1.0
        for C_val in [0.01, 0.1, 0.5, 1, 5, 10]:
            lr_cand = LogisticRegression(C=C_val, max_iter=2000,
                                         class_weight="balanced",
                                         random_state=42)
            lr_cv_scores = loro_cv_score(lr_cand, X_train, y_train,
                                         groups, scoring="roc_auc")
            if lr_cv_scores.mean() > best_lr_auc:
                best_lr_auc = lr_cv_scores.mean()
                best_lr = lr_cand
                best_lr_C = C_val
                best_lr_fold_scores = lr_cv_scores
        best_lr.fit(X_train, y_train)
        log.info(f"  Best C: {best_lr_C}  CV AUC: {best_lr_auc:.4f}")
        eval_classifier(best_lr, X_val, y_val, "Val")
        results["logistic"] = {"model": best_lr, "cv_auc": best_lr_auc}
        cv_fold_scores["logistic"] = best_lr_fold_scores.tolist()

        log.info("SVC (RBF kernel)")
        svc_param_grid = {
            "C": [0.1, 0.5, 1, 5, 10, 50],
            "gamma": ["scale", "auto"],
        }
        svc_search = RandomizedSearchCV(
            SVC(kernel="rbf", probability=True,
                class_weight="balanced", random_state=42),
            svc_param_grid,
            n_iter=12, cv=gkf,
            scoring="roc_auc",
            random_state=42, n_jobs=1, verbose=0
        )
        svc_search.fit(X_train, y_train, groups=groups)
        best_svc = svc_search.best_estimator_
        svc_cv_auc = svc_search.best_score_
        log.info(f"  Best params: {svc_search.best_params_}")
        log.info(f"  CV AUC: {svc_cv_auc:.4f}")
        eval_classifier(best_svc, X_val, y_val, "Val")
        results["svc"] = {"model": best_svc, "cv_auc": svc_cv_auc}

        log.info("RandomForest Classifier")
        rf_param_grid = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [8, 16, 32],
            "max_features": ["sqrt", "log2", 0.4, 0.5],
            "min_impurity_decrease": [0.0, 0.001, 0.005],
        }
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1),
            rf_param_grid,
            n_iter=50, cv=gkf,
            scoring="roc_auc",
            random_state=42, n_jobs=1, verbose=0
        )
        rf_search.fit(X_train, y_train, groups=groups)
        best_rf = rf_search.best_estimator_
        rf_cv_auc = rf_search.best_score_
        log.info(f"  Best params: {rf_search.best_params_}")
        log.info(f"  CV AUC: {rf_cv_auc:.4f}")
        eval_classifier(best_rf, X_val, y_val, "Val")
        results["rf"] = {"model": best_rf, "cv_auc": rf_cv_auc}

        log.info("XGBoost Classifier")
        scale_pos = int((y_train == 0).sum()) / int((y_train == 1).sum())
        xgb_param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
            "colsample_bylevel": [0.5, 0.7, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.5, 1, 5, 10],
            "min_child_weight": [1, 3, 5, 10],
        }
        xgb_search = RandomizedSearchCV(
            xgb.XGBClassifier(random_state=42, verbosity=0,
                               scale_pos_weight=scale_pos, n_jobs=1,
                               eval_metric="auc"),
            xgb_param_grid,
            n_iter=50, cv=gkf,
            scoring="roc_auc",
            random_state=42, n_jobs=1, verbose=0
        )
        xgb_search.fit(X_train, y_train, groups=groups)
        best_xgb = xgb_search.best_estimator_
        xgb_cv_auc = xgb_search.best_score_
        log.info(f"  Best params: {xgb_search.best_params_}")
        log.info(f"  CV AUC: {xgb_cv_auc:.4f}")
        eval_classifier(best_xgb, X_val, y_val, "Val")
        results["xgb"] = {"model": best_xgb, "cv_auc": xgb_cv_auc}

        log.info("LightGBM Classifier")
        lgb_param_grid = {
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "num_leaves": [7, 15, 31],
            "subsample": [0.6, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1],
            "reg_lambda": [0.5, 1, 5, 10],
            "min_child_samples": [10, 20, 30, 50],
            "n_estimators": [200, 300, 500],
        }
        lgb_search = RandomizedSearchCV(
            lgb.LGBMClassifier(random_state=42, verbosity=-1,
                               class_weight="balanced", n_jobs=1),
            lgb_param_grid,
            n_iter=50, cv=gkf,
            scoring="roc_auc",
            random_state=42, n_jobs=1, verbose=0
        )
        lgb_search.fit(X_train, y_train, groups=groups)
        best_lgb = lgb_search.best_estimator_
        lgb_cv_auc = lgb_search.best_score_
        log.info(f"  Best params: {lgb_search.best_params_}")
        log.info(f"  CV AUC: {lgb_cv_auc:.4f}")
        eval_classifier(best_lgb, X_val, y_val, "Val")
        results["lgb"] = {"model": best_lgb, "cv_auc": lgb_cv_auc}

        log.info("Ensemble VotingClassifier (XGB + LGB + RF, soft vote)")
        ensemble_clf = VotingClassifier([
            ("xgb", best_xgb),
            ("lgb", best_lgb),
            ("rf", best_rf),
        ], voting="soft")
        ensemble_clf.fit(X_train, y_train)
        ens_cv = loro_cv_score(ensemble_clf, X_train, y_train, groups,
                               scoring="roc_auc")
        ens_cv_auc = ens_cv.mean()
        log.info(f"  CV AUC: {ens_cv_auc:.4f} +/- {ens_cv.std():.4f}")
        eval_classifier(ensemble_clf, X_val, y_val, "Val")
        results["ensemble"] = {"model": ensemble_clf, "cv_auc": ens_cv_auc}
        cv_fold_scores["ensemble"] = ens_cv.tolist()

        log.info("CLASSIFICATION SUMMARY (ranked by CV AUC)")
        ranked = sorted(results.items(), key=lambda x: x[1]["cv_auc"], reverse=True)
        for name, info in ranked:
            log.info(f"  {name:<12} CV AUC={info['cv_auc']:.4f}")

        best_name, best_info = ranked[0]
        best_model = best_info["model"]
        log.info(f"  Winner: {best_name} (CV AUC={best_info['cv_auc']:.4f})")

        if best_name in ("xgb", "lgb", "rf"):
            calibrated = CalibratedClassifierCV(estimator=best_model,
                                                method="isotonic", cv=3)
            calibrated.fit(X_train, y_train)
            log.info("  Calibrated with isotonic regression.")
            eval_classifier(calibrated, X_val, y_val, "Val (calibrated)")
            return calibrated, results, cv_fold_scores
        else:
            log.info(f"  {best_name} produces well-calibrated probabilities by design.")
            return best_model, results, cv_fold_scores

    except Exception as e:
        raise CustomException(e, sys)


def save_model(model, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"Saved {path}")
    except Exception as e:
        raise CustomException(e, sys)


def save_cv_scores(all_cv_scores):
    """
    Persist per-fold CV scores for all models to a JSON artifact.
    Used by model_diagnostics to populate the overfitting profile CV bands
    with real fold-level variance rather than hardcoded values.
    """
    try:
        with open(CV_SCORES_PATH, "w") as f:
            json.dump(all_cv_scores, f, indent=2)
        log.info(f"CV fold scores saved to {CV_SCORES_PATH}")
    except Exception as e:
        raise CustomException(e, sys)


def run_training():
    try:
        log.info("Starting model training")
        log.info("Primary metric: CV MAE (regression), CV AUC (classification)")
        log.info("CV strategy: Leave-One-Race-Out via GroupKFold")

        train_df, val_df, test_df = load_splits(sc=False)
        best_reg, reg_results, reg_cv = train_regressors(train_df, val_df, sc=False)
        best_clf, clf_results, clf_cv = train_classifiers(train_df, val_df, sc=False)

        save_model(best_reg, REGRESSOR_PATH)
        save_model(best_clf, CLASSIFIER_PATH)

        sc_train_df, sc_val_df, sc_test_df = load_splits(sc=True)
        best_reg_sc, reg_sc_results, reg_sc_cv = train_regressors(sc_train_df, sc_val_df, sc=True)
        best_clf_sc, clf_sc_results, clf_sc_cv = train_classifiers(sc_train_df, sc_val_df, sc=True)

        save_model(best_reg_sc, REGRESSOR_SC_PATH)
        save_model(best_clf_sc, CLASSIFIER_SC_PATH)

        all_cv_scores = {
            "full_reg": reg_cv,
            "full_clf": clf_cv,
            "sc_reg": reg_sc_cv,
            "sc_clf": clf_sc_cv,
        }
        save_cv_scores(all_cv_scores)

        log.info("Model training complete. 4 models saved.")

        return (best_reg, best_clf, reg_results, clf_results,
                best_reg_sc, best_clf_sc, reg_sc_results, clf_sc_results)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    (best_reg, best_clf, reg_results, clf_results,
     best_reg_sc, best_clf_sc, reg_sc_results, clf_sc_results) = run_training()

    log.info("FINAL MODEL COMPARISON -- VAL SET")

    train_df, val_df, _ = load_splits(sc=False)
    sc_train_df, sc_val_df, _ = load_splits(sc=True)

    feature_cols = get_feature_cols(train_df)
    feature_cols_sc = get_feature_cols(sc_train_df)

    X_val = val_df[feature_cols].values
    y_val_r = val_df[TARGET_REGRESSION].values
    y_val_c = val_df[TARGET_CLASSIFICATION].values

    X_sc_val = sc_val_df[feature_cols_sc].values
    y_sc_val_r = sc_val_df[TARGET_REGRESSION].values
    y_sc_val_c = sc_val_df[TARGET_CLASSIFICATION].values

    log.info("Full Regression -- Val MAE / R2")
    for name, info in sorted(reg_results.items(), key=lambda x: x[1]["cv_mae"]):
        m = info["model"]
        mae = mean_absolute_error(y_val_r, m.predict(X_val))
        r2 = r2_score(y_val_r, m.predict(X_val))
        log.info(f"  {name:<10}  CV_MAE={info['cv_mae']:.4f}  Val_MAE={mae:.4f}  Val_R2={r2:.4f}")

    log.info("Full Classification -- Val AUC / F1")
    for name, info in sorted(clf_results.items(), key=lambda x: x[1]["cv_auc"], reverse=True):
        m = info["model"]
        pred_prob = m.predict_proba(X_val)[:, 1] if hasattr(m, "predict_proba") else m.predict(X_val)
        auc = roc_auc_score(y_val_c, pred_prob)
        f1 = f1_score(y_val_c, m.predict(X_val), zero_division=0)
        log.info(f"  {name:<10}  CV_AUC={info['cv_auc']:.4f}  Val_AUC={auc:.4f}  Val_F1={f1:.4f}")

    log.info("SC Regression -- Val MAE / R2")
    for name, info in sorted(reg_sc_results.items(), key=lambda x: x[1]["cv_mae"]):
        m = info["model"]
        mae = mean_absolute_error(y_sc_val_r, m.predict(X_sc_val))
        r2 = r2_score(y_sc_val_r, m.predict(X_sc_val))
        log.info(f"  {name:<10}  CV_MAE={info['cv_mae']:.4f}  Val_MAE={mae:.4f}  Val_R2={r2:.4f}")

    log.info("SC Classification -- Val AUC / F1")
    for name, info in sorted(clf_sc_results.items(), key=lambda x: x[1]["cv_auc"], reverse=True):
        m = info["model"]
        pred_prob = m.predict_proba(X_sc_val)[:, 1] if hasattr(m, "predict_proba") else m.predict(X_sc_val)
        auc = roc_auc_score(y_sc_val_c, pred_prob)
        f1 = f1_score(y_sc_val_c, m.predict(X_sc_val), zero_division=0)
        log.info(f"  {name:<10}  CV_AUC={info['cv_auc']:.4f}  Val_AUC={auc:.4f}  Val_F1={f1:.4f}")

    log.info("Models saved:")
    log.info(f"  regressor.pkl     best full regression model")
    log.info(f"  classifier.pkl    best full classification model")
    log.info(f"  regressor_sc.pkl  best SC regression model")
    log.info(f"  classifier_sc.pkl best SC classification model")