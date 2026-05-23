import os
import sys
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    TRAIN_PATH, VAL_PATH,
    REGRESSOR_PATH, CLASSIFIER_PATH,
    TARGET_REGRESSION, TARGET_CLASSIFICATION
)


FEATURE_COLS = [
    "coasting_pct_delta", "full_throttle_pct_delta", "gear_shifts_delta",
    "avg_brake_zone_length_delta", "avg_entry_speed_delta",
    "brake_zone_count_delta", "tyre_life_delta", "tyre_life_x_coasting_delta",
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
# LOAD DATA AND MODELS
# ─────────────────────────────────────────

def load_data_and_models():
    try:
        train = pd.read_csv(TRAIN_PATH)
        val   = pd.read_csv(VAL_PATH)

        with open(REGRESSOR_PATH,  "rb") as f:
            regressor = pickle.load(f)
        with open(CLASSIFIER_PATH, "rb") as f:
            classifier = pickle.load(f)

        log.info(f"Train: {train.shape} | Val: {val.shape}")
        log.info(f"Regressor  loaded : {type(regressor).__name__}")
        log.info(f"Classifier loaded : {type(classifier).__name__}")

        return train, val, regressor, classifier
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 1 — TRAIN vs VAL SCORE GAP
# The single most important overfitting check
# ─────────────────────────────────────────

def check_train_val_gap(model, X_train, y_train, X_val, y_val,
                        task, metric_fn, metric_name):
    """
    Compute train score and val score side by side.
    A large gap = overfitting. Both scores high = good fit.
    Both scores low = underfitting.
    """
    try:
        log.info("-" * 50)
        log.info(f"TRAIN vs VAL GAP — {task} ({metric_name})")
        log.info("-" * 50)

        train_pred = model.predict(X_train)
        val_pred   = model.predict(X_val)

        train_score = metric_fn(y_train, train_pred)
        val_score   = metric_fn(y_val,   val_pred)
        gap         = abs(train_score - val_score)

        log.info(f"  Train {metric_name} : {train_score:.4f}")
        log.info(f"  Val   {metric_name} : {val_score:.4f}")
        log.info(f"  Gap               : {gap:.4f}")

        # Interpret the gap
        if task == "regression":
            if gap > 0.15:
                log.info("  DIAGNOSIS: OVERFITTING — gap > 0.15s MAE. "
                         "Consider increasing min_samples_leaf or reducing max_depth.")
            elif train_score > 0.50 and val_score > 0.50:
                log.info("  DIAGNOSIS: UNDERFITTING — both train and val MAE are high. "
                         "Consider engineering more features.")
            else:
                log.info("  DIAGNOSIS: GOOD FIT — gap is acceptable.")
        else:
            if train_score > val_score + 0.10:
                log.info("  DIAGNOSIS: OVERFITTING — train F1 significantly above val F1.")
            elif train_score < 0.60 and val_score < 0.60:
                log.info("  DIAGNOSIS: UNDERFITTING — both train and val F1 are low.")
            else:
                log.info("  DIAGNOSIS: GOOD FIT — scores are consistent.")

        return train_score, val_score, gap

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 2 — CROSS VALIDATION ON FULL TRAIN
# More reliable than a single val split given small data
# ─────────────────────────────────────────

def cross_validate_models(regressor, classifier, X_train, y_train_reg, y_train_clf):
    try:
        log.info("-" * 50)
        log.info("CROSS VALIDATION (5-fold on train set)")
        log.info("-" * 50)

        # Regression — KFold (continuous target)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        reg_mae_scores  = -cross_val_score(regressor, X_train, y_train_reg,
                                            cv=kf, scoring="neg_mean_absolute_error")
        reg_rmse_scores = np.sqrt(-cross_val_score(regressor, X_train, y_train_reg,
                                                    cv=kf, scoring="neg_mean_squared_error"))
        reg_r2_scores   = cross_val_score(regressor, X_train, y_train_reg,
                                          cv=kf, scoring="r2")

        log.info(f"  Regressor CV MAE  : {reg_mae_scores.mean():.4f} ± {reg_mae_scores.std():.4f}")
        log.info(f"  Regressor CV RMSE : {reg_rmse_scores.mean():.4f} ± {reg_rmse_scores.std():.4f}")
        log.info(f"  Regressor CV R2   : {reg_r2_scores.mean():.4f} ± {reg_r2_scores.std():.4f}")
        log.info(f"  Regressor CV MAE per fold : {reg_mae_scores.round(4)}")

        # Classification — StratifiedKFold (preserves class ratio)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        clf_acc_scores = cross_val_score(classifier, X_train, y_train_clf,
                                          cv=skf, scoring="accuracy")
        clf_f1_scores  = cross_val_score(classifier, X_train, y_train_clf,
                                          cv=skf, scoring="f1")
        clf_auc_scores = cross_val_score(classifier, X_train, y_train_clf,
                                          cv=skf, scoring="roc_auc")

        log.info(f"  Classifier CV Acc : {clf_acc_scores.mean():.4f} ± {clf_acc_scores.std():.4f}")
        log.info(f"  Classifier CV F1  : {clf_f1_scores.mean():.4f} ± {clf_f1_scores.std():.4f}")
        log.info(f"  Classifier CV AUC : {clf_auc_scores.mean():.4f} ± {clf_auc_scores.std():.4f}")
        log.info(f"  Classifier CV F1 per fold : {clf_f1_scores.round(4)}")

        return {
            "reg_cv_mae"  : reg_mae_scores,
            "reg_cv_rmse" : reg_rmse_scores,
            "reg_cv_r2"   : reg_r2_scores,
            "clf_cv_acc"  : clf_acc_scores,
            "clf_cv_f1"   : clf_f1_scores,
            "clf_cv_auc"  : clf_auc_scores,
        }

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 3 — DETAILED VAL SET DIAGNOSTICS
# ─────────────────────────────────────────

def detailed_val_diagnostics(regressor, classifier,
                              X_val, y_val_reg, y_val_clf, val_df):
    try:
        log.info("-" * 50)
        log.info("DETAILED VAL SET DIAGNOSTICS")
        log.info("-" * 50)

        # ── Regression ──
        reg_pred   = regressor.predict(X_val)
        residuals  = y_val_reg - reg_pred

        log.info(f"\n  Regression residuals on val:")
        log.info(f"    mean  = {residuals.mean():.4f}s  (bias — should be near 0)")
        log.info(f"    std   = {residuals.std():.4f}s")
        log.info(f"    max   = {residuals.max():.4f}s")
        log.info(f"    min   = {residuals.min():.4f}s")

        # Large residual laps — what went wrong
        val_with_pred = val_df.copy()
        val_with_pred["reg_pred"]  = reg_pred
        val_with_pred["residual"]  = residuals

        large_errors = val_with_pred[abs(val_with_pred["residual"]) > 0.5].sort_values(
            "residual", key=abs, ascending=False
        )[["Race", "LapNumber", TARGET_REGRESSION, "reg_pred", "residual",
           "same_compound", "VER_compound_enc", "HAM_compound_enc"]]

        log.info(f"\n  Laps with residual > 0.5s ({len(large_errors)} laps):")
        if len(large_errors) > 0:
            log.info(f"\n{large_errors.to_string(index=False)}")
        else:
            log.info("  None — all predictions within 0.5s.")

        # ── Classification ──
        clf_pred  = classifier.predict(X_val)
        clf_proba = classifier.predict_proba(X_val)[:, 1]

        cm = confusion_matrix(y_val_clf, clf_pred)
        log.info(f"\n  Classification confusion matrix (val):")
        log.info(f"    TN={cm[0,0]}  FP={cm[0,1]}")
        log.info(f"    FN={cm[1,0]}  TP={cm[1,1]}")

        report = classification_report(y_val_clf, clf_pred,
                                        target_names=["HAM_faster", "VER_faster"])
        log.info(f"\n  Classification report (val):\n{report}")

        # Misclassified laps — what did the model get wrong
        val_with_pred["clf_pred"]  = clf_pred
        val_with_pred["clf_proba"] = clf_proba
        wrong = val_with_pred[val_with_pred[TARGET_CLASSIFICATION] != clf_pred][
            ["Race", "LapNumber", TARGET_CLASSIFICATION, "clf_pred", "clf_proba",
             "same_compound", "coasting_pct_delta", "tyre_life_delta"]
        ]
        log.info(f"\n  Misclassified laps ({len(wrong)}):")
        if len(wrong) > 0:
            log.info(f"\n{wrong.to_string(index=False)}")
        else:
            log.info("  None — perfect classification on val.")

        return val_with_pred

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 4 — LEARNING CURVE (manual)
# Train on increasing subsets, track train vs val MAE
# This is the definitive overfitting diagnostic
# ─────────────────────────────────────────

def compute_learning_curve(regressor, X_train, y_train_reg, X_val, y_val_reg):
    try:
        log.info("-" * 50)
        log.info("LEARNING CURVE — Train vs Val MAE as training size increases")
        log.info("-" * 50)

        # Use increasing fractions of train data
        fractions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        n_train   = len(X_train)

        curve_results = []

        for frac in fractions:
            n = max(10, int(n_train * frac))
            X_sub = X_train[:n]
            y_sub = y_train_reg[:n]

            # Refit a fresh model with same params on the subset
            import pickle as pkl
            model_copy = pkl.loads(pkl.dumps(regressor))
            model_copy.fit(X_sub, y_sub)

            train_mae = mean_absolute_error(y_sub,       model_copy.predict(X_sub))
            val_mae   = mean_absolute_error(y_val_reg,   model_copy.predict(X_val))

            curve_results.append({
                "n_samples" : n,
                "train_mae" : round(train_mae, 4),
                "val_mae"   : round(val_mae,   4),
                "gap"       : round(abs(val_mae - train_mae), 4)
            })

            log.info(f"  n={n:>3} | train_mae={train_mae:.4f}  "
                     f"val_mae={val_mae:.4f}  gap={abs(val_mae-train_mae):.4f}")

        # Interpret final gap
        final = curve_results[-1]
        log.info(f"\n  Final gap at full training size: {final['gap']:.4f}s")

        if final["gap"] > 0.15:
            log.info("  VERDICT: OVERFITTING detected.")
            log.info("  ACTION:  Increase min_samples_leaf, reduce max_depth, "
                     "or add regularization before test evaluation.")
        elif final["train_mae"] > 0.40:
            log.info("  VERDICT: UNDERFITTING detected.")
            log.info("  ACTION:  Engineer more features — "
                     "consider stint_phase, rolling lap delta, or compound interaction terms.")
        else:
            log.info("  VERDICT: FIT IS ACCEPTABLE — proceed to model_diagnostics.py")

        return curve_results

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 5 — OVERFITTING / UNDERFITTING SUMMARY
# ─────────────────────────────────────────

def print_fit_summary(train_reg_score, val_reg_score,
                      train_clf_score, val_clf_score, cv_results):
    try:
        log.info("=" * 60)
        log.info("FIT SUMMARY — FINAL VERDICT")
        log.info("=" * 60)

        reg_gap = abs(train_reg_score - val_reg_score)
        clf_gap = abs(train_clf_score - val_clf_score)

        log.info(f"\n  REGRESSION:")
        log.info(f"    Train MAE : {train_reg_score:.4f}s")
        log.info(f"    Val   MAE : {val_reg_score:.4f}s")
        log.info(f"    Gap       : {reg_gap:.4f}s")
        log.info(f"    CV MAE    : {cv_results['reg_cv_mae'].mean():.4f}s "
                 f"± {cv_results['reg_cv_mae'].std():.4f}s")

        log.info(f"\n  CLASSIFICATION:")
        log.info(f"    Train F1  : {train_clf_score:.4f}")
        log.info(f"    Val   F1  : {val_clf_score:.4f}")
        log.info(f"    Gap       : {clf_gap:.4f}")
        log.info(f"    CV F1     : {cv_results['clf_cv_f1'].mean():.4f} "
                 f"± {cv_results['clf_cv_f1'].std():.4f}")

        log.info("\n  RECOMMENDED NEXT STEP:")

        if reg_gap > 0.15:
            log.info("  → Regression is OVERFITTING.")
            log.info("    Try: increase min_samples_leaf (8-15), reduce max_depth (3-4),")
            log.info("    OR:  add L2 regularization via reg_alpha/reg_lambda in XGBoost.")
        elif train_reg_score > 0.40:
            log.info("  → Regression is UNDERFITTING.")
            log.info("    Try: add stint_phase feature, rolling lap delta, "
                     "compound interaction term.")
        else:
            log.info("  → Regression fit is ACCEPTABLE. Proceed to diagnostics.")

        if cv_results["clf_cv_f1"].mean() < 0.55:
            log.info("  → Classifier CV F1 is LOW — model is not generalizing well.")
            log.info("    Try: add interaction feature (coasting_pct * same_compound),")
            log.info("    OR:  use stint_phase to stratify predictions.")
        else:
            log.info("  → Classifier fit is ACCEPTABLE. Proceed to diagnostics.")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_validation():
    try:
        log.info("=" * 60)
        log.info("Starting model validation")
        log.info("=" * 60)

        train, val, regressor, classifier = load_data_and_models()

        X_train = train[FEATURE_COLS].values
        X_val   = val[FEATURE_COLS].values
        y_train_reg = train[TARGET_REGRESSION].values
        y_val_reg   = val[TARGET_REGRESSION].values
        y_train_clf = train[TARGET_CLASSIFICATION].values
        y_val_clf   = val[TARGET_CLASSIFICATION].values

        # Section 1 — Train vs Val gap
        train_reg_score, val_reg_score, reg_gap = check_train_val_gap(
            regressor, X_train, y_train_reg, X_val, y_val_reg,
            task="regression", metric_fn=mean_absolute_error, metric_name="MAE"
        )
        train_clf_score, val_clf_score, clf_gap = check_train_val_gap(
            classifier, X_train, y_train_clf, X_val, y_val_clf,
            task="classification", metric_fn=f1_score, metric_name="F1"
        )

        # Section 2 — Cross validation
        cv_results = cross_validate_models(
            regressor, classifier, X_train, y_train_reg, y_train_clf
        )

        # Section 3 — Detailed val diagnostics
        val_with_pred = detailed_val_diagnostics(
            regressor, classifier,
            X_val, y_val_reg, y_val_clf, val
        )

        # Section 4 — Learning curve
        curve_results = compute_learning_curve(
            regressor, X_train, y_train_reg, X_val, y_val_reg
        )

        # Section 5 — Final summary and recommendation
        print_fit_summary(
            train_reg_score, val_reg_score,
            train_clf_score, val_clf_score,
            cv_results
        )

        log.info("Model validation complete.")
        return cv_results, val_with_pred, curve_results

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    cv_results, val_with_pred, curve_results = run_validation()

    print("\n--- LEARNING CURVE TABLE ---")
    print(f"{'N Samples':>10} {'Train MAE':>12} {'Val MAE':>10} {'Gap':>8}")
    print("-" * 44)
    for row in curve_results:
        print(f"{row['n_samples']:>10} {row['train_mae']:>12.4f} "
              f"{row['val_mae']:>10.4f} {row['gap']:>8.4f}")

    print("\n--- CV SUMMARY ---")
    print(f"Regressor  CV MAE : "
          f"{cv_results['reg_cv_mae'].mean():.4f} ± {cv_results['reg_cv_mae'].std():.4f}")
    print(f"Classifier CV F1  : "
          f"{cv_results['clf_cv_f1'].mean():.4f} ± {cv_results['clf_cv_f1'].std():.4f}")
    print(f"Classifier CV AUC : "
          f"{cv_results['clf_cv_auc'].mean():.4f} ± {cv_results['clf_cv_auc'].std():.4f}")