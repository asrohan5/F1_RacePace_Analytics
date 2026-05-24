# test_pipeline.py
import sys
import pickle
import pandas as pd

from sklearn.metrics import mean_absolute_error, accuracy_score

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    TEST_PATH, TEST_SC_PATH,
    REGRESSOR_PATH, CLASSIFIER_PATH,
    REGRESSOR_SC_PATH, CLASSIFIER_SC_PATH,
    PREPROCESSOR_PATH,
    TARGET_REGRESSION, TARGET_CLASSIFICATION
)

import warnings
warnings.filterwarnings('ignore')


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


def load_artifacts():
    """Load saved models, scaler, and test splits."""
    try:
        test    = pd.read_csv(TEST_PATH)
        test_sc = pd.read_csv(TEST_SC_PATH)

        with open(REGRESSOR_PATH,     "rb") as f: reg    = pickle.load(f)
        with open(CLASSIFIER_PATH,    "rb") as f: clf    = pickle.load(f)
        with open(REGRESSOR_SC_PATH,  "rb") as f: reg_sc = pickle.load(f)
        with open(CLASSIFIER_SC_PATH, "rb") as f: clf_sc = pickle.load(f)
        with open(PREPROCESSOR_PATH,  "rb") as f: scaler = pickle.load(f)

        log.info(f"Test set loaded      : {test.shape}")
        log.info(f"SC Test set loaded   : {test_sc.shape}")
        log.info(f"Regressor loaded     : {type(reg).__name__}")
        log.info(f"Classifier loaded    : {type(clf).__name__}")
        log.info(f"SC Regressor loaded  : {type(reg_sc).__name__}")
        log.info(f"SC Classifier loaded : {type(clf_sc).__name__}")

        return test, test_sc, reg, clf, reg_sc, clf_sc, scaler

    except Exception as e:
        raise CustomException(e, sys)


def predict(model, X, label):
    """Run prediction and return results."""
    try:
        reg_pred  = None
        clf_pred  = None
        clf_proba = None

        # Check if regressor or classifier by trying predict_proba
        if hasattr(model, "predict_proba"):
            clf_pred  = model.predict(X)
            clf_proba = model.predict_proba(X)[:, 1]
            log.info(f"  [{label}] Classification predictions generated: {len(clf_pred)}")
            return clf_pred, clf_proba
        else:
            reg_pred = model.predict(X)
            log.info(f"  [{label}] Regression predictions generated: {len(reg_pred)}")
            return reg_pred, None

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_predictions(y_true, y_pred, task, label):
    """Compute and log metrics for a set of predictions."""
    try:
        from sklearn.metrics import (
            mean_absolute_error, r2_score,
            accuracy_score, f1_score, roc_auc_score
        )

        log.info(f"\n  [{label}] {task} metrics on test set:")

        if task == "regression":
            mae = mean_absolute_error(y_true, y_pred)
            r2  = r2_score(y_true, y_pred)
            log.info(f"    MAE = {mae:.4f}s")
            log.info(f"    R2  = {r2:.4f}")
            return {"MAE": mae, "R2": r2}

        else:
            pred, proba = y_pred
            acc = accuracy_score(y_true, pred)
            f1  = f1_score(y_true, pred, zero_division=0)
            auc = roc_auc_score(y_true, proba)
            log.info(f"    Accuracy = {acc:.4f}")
            log.info(f"    F1       = {f1:.4f}")
            log.info(f"    AUC      = {auc:.4f}")
            return {"Accuracy": acc, "F1": f1, "AUC": auc}

    except Exception as e:
        raise CustomException(e, sys)


def build_prediction_output(test_df, reg_pred, clf_pred, clf_proba, label):
    """
    Attach predictions back to the test dataframe.
    Returns a clean output dataframe with actuals and predictions side by side.
    """
    try:
        out = test_df[["Race", "LapNumber",
                        TARGET_REGRESSION,
                        TARGET_CLASSIFICATION]].copy()

        out["predicted_delta_sec"] = reg_pred.round(3)
        out["predicted_ver_faster"] = clf_pred
        out["ver_faster_probability"] = clf_proba.round(3)
        out["prediction_correct"] = (
            out["predicted_ver_faster"] == out[TARGET_CLASSIFICATION]
        ).astype(int)
        out["residual_sec"] = (
            out[TARGET_REGRESSION] - out["predicted_delta_sec"]
        ).round(3)

        log.info(f"\n  [{label}] Prediction output built: {out.shape}")
        log.info(f"  Correct predictions: "
                 f"{out['prediction_correct'].sum()} / {len(out)} "
                 f"({100 * out['prediction_correct'].mean():.1f}%)")

        return out

    except Exception as e:
        raise CustomException(e, sys)


def run_test_pipeline():
    try:
        log.info("=" * 60)
        log.info("TEST PIPELINE — START")
        log.info("=" * 60)

        # ── Load ──
        test, test_sc, reg, clf, reg_sc, clf_sc, scaler = load_artifacts()

        X_test    = test[FEATURE_COLS].values
        X_test_sc = test_sc[FEATURE_COLS].values

        y_test_reg    = test[TARGET_REGRESSION].values
        y_test_clf    = test[TARGET_CLASSIFICATION].values
        y_test_sc_reg = test_sc[TARGET_REGRESSION].values
        y_test_sc_clf = test_sc[TARGET_CLASSIFICATION].values

        # ── Full model predictions ──
        log.info("\n>>> FULL MODEL — Test Set")
        reg_pred, _         = predict(reg, X_test, "Full-Regressor")
        clf_results, _      = predict(clf, X_test, "Full-Classifier")
        clf_pred, clf_proba = clf_results, predict(clf, X_test, "Full-Classifier")[1]

        # Re-run cleanly
        reg_pred   = reg.predict(X_test)
        clf_pred   = clf.predict(X_test)
        clf_proba  = clf.predict_proba(X_test)[:, 1]

        full_reg_metrics = evaluate_predictions(
            y_test_reg, reg_pred, "regression", "Full"
        )
        full_clf_metrics = evaluate_predictions(
            y_test_clf, (clf_pred, clf_proba), "classification", "Full"
        )

        full_output = build_prediction_output(
            test, reg_pred, clf_pred, clf_proba, "Full"
        )

        # ── SC model predictions ──
        log.info("\n>>> SAME-COMPOUND MODEL — Test Set")
        reg_sc_pred  = reg_sc.predict(X_test_sc)
        clf_sc_pred  = clf_sc.predict(X_test_sc)
        clf_sc_proba = clf_sc.predict_proba(X_test_sc)[:, 1]

        sc_reg_metrics = evaluate_predictions(
            y_test_sc_reg, reg_sc_pred, "regression", "SC"
        )
        sc_clf_metrics = evaluate_predictions(
            y_test_sc_clf, (clf_sc_pred, clf_sc_proba), "classification", "SC"
        )

        sc_output = build_prediction_output(
            test_sc, reg_sc_pred, clf_sc_pred, clf_sc_proba, "SC"
        )

        # ── Final comparison ──
        log.info("\n" + "=" * 60)
        log.info("FINAL COMPARISON — Full vs Same-Compound")
        log.info("=" * 60)
        log.info(f"  {'Metric':<25} {'Full':>10} {'SC':>10}")
        log.info(f"  {'-'*47}")
        log.info(f"  {'Regression MAE':<25} "
                 f"{full_reg_metrics['MAE']:>10.4f} "
                 f"{sc_reg_metrics['MAE']:>10.4f}")
        log.info(f"  {'Regression R2':<25} "
                 f"{full_reg_metrics['R2']:>10.4f} "
                 f"{sc_reg_metrics['R2']:>10.4f}")
        log.info(f"  {'Classifier AUC':<25} "
                 f"{full_clf_metrics['AUC']:>10.4f} "
                 f"{sc_clf_metrics['AUC']:>10.4f}")
        log.info(f"  {'Classifier F1':<25} "
                 f"{full_clf_metrics['F1']:>10.4f} "
                 f"{sc_clf_metrics['F1']:>10.4f}")
                
        # ── Per-race breakdown — what manager actually wants to see ──
        log.info("\n--- PER-RACE TEST METRICS ---")
        log.info(f"  {'Race':<12} {'Full MAE':>10} {'Full Acc':>10} "
                f"{'SC MAE':>10} {'SC Acc':>10} {'N_full':>7} {'N_sc':>5}")
        log.info(f"  {'-'*62}")

        for race in test["Race"].unique():
            # Full model
            full_mask = test["Race"].values == race
            if full_mask.sum() > 0:
                f_mae = mean_absolute_error(
                    y_test_reg[full_mask], reg.predict(X_test[full_mask])
                )
                f_acc = accuracy_score(
                    y_test_clf[full_mask], clf.predict(X_test[full_mask])
                )
            else:
                f_mae = f_acc = float("nan")

            # SC model
            sc_mask = test_sc["Race"].values == race
            if sc_mask.sum() > 0:
                sc_mae = mean_absolute_error(
                    y_test_sc_reg[sc_mask], reg_sc.predict(X_test_sc[sc_mask])
                )
                sc_acc = accuracy_score(
                    y_test_sc_clf[sc_mask], clf_sc.predict(X_test_sc[sc_mask])
                )
            else:
                sc_mae = sc_acc = float("nan")

            log.info(f"  {race:<12} {f_mae:>10.4f} {f_acc:>10.4f} "
                    f"{sc_mae:>10.4f} {sc_acc:>10.4f} "
                    f"{full_mask.sum():>7} {sc_mask.sum():>5}")

        # ── High confidence prediction accuracy ──
        log.info("\n--- HIGH CONFIDENCE PREDICTIONS (probability > 0.80) ---")
        high_conf_mask = clf_proba > 0.80
        if high_conf_mask.sum() > 0:
            hc_acc = accuracy_score(
                y_test_clf[high_conf_mask], clf_pred[high_conf_mask]
            )
            log.info(f"  Full model: {high_conf_mask.sum()} predictions with prob > 0.80 | "
                    f"Accuracy = {hc_acc:.4f} "
                    f"({high_conf_mask.sum()}/{len(y_test_clf)} total test laps)")
        else:
            log.info("  No predictions with probability > 0.80 on test set")

        sc_high_conf = clf_sc_proba > 0.80
        if sc_high_conf.sum() > 0:
            sc_hc_acc = accuracy_score(
                y_test_sc_clf[sc_high_conf], clf_sc_pred[sc_high_conf]
            )
            log.info(f"  SC model  : {sc_high_conf.sum()} predictions with prob > 0.80 | "
                    f"Accuracy = {sc_hc_acc:.4f}")

        log.info("=" * 60)
        log.info("TEST PIPELINE — COMPLETE")
        log.info("=" * 60)

        return full_output, sc_output

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    full_output, sc_output = run_test_pipeline()

    print("\n--- FULL MODEL PREDICTIONS (Test Set) ---")
    print(full_output.to_string(index=False))

    print("\n--- SC MODEL PREDICTIONS (Test Set) ---")
    print(sc_output.to_string(index=False))

    print("\n--- PREDICTION ACCURACY ---")
    print(f"Full model correct: "
          f"{full_output['prediction_correct'].sum()}/{len(full_output)} "
          f"({100*full_output['prediction_correct'].mean():.1f}%)")
    print(f"SC model correct  : "
          f"{sc_output['prediction_correct'].sum()}/{len(sc_output)} "
          f"({100*sc_output['prediction_correct'].mean():.1f}%)")