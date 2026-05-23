import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.model_selection import KFold, StratifiedKFold
import shap

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    REGRESSOR_PATH, CLASSIFIER_PATH,
    ARTIFACTS_DIR,
    TARGET_REGRESSION, TARGET_CLASSIFICATION
)


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

# Human-readable feature labels for plots
FEATURE_LABELS = [
    "Coasting % Delta",        "Full Throttle % Delta",   "Gear Shifts Delta",
    "Brake Zone Length Delta",  "Entry Speed Delta",       "Brake Zone Count Delta",
    "Tyre Life Delta",
    "VER Coasting %",          "HAM Coasting %",
    "VER Full Throttle %",     "HAM Full Throttle %",
    "VER Gear Shifts",         "HAM Gear Shifts",
    "VER Brake Zone Length",   "HAM Brake Zone Length",
    "VER Entry Speed",         "HAM Entry Speed",
    "VER Tyre Life",           "HAM Tyre Life",
    "Same Compound",           "VER Compound",            "HAM Compound",
    "Lap Number",              "Race"
]


def save_fig(fig, filename):
    path = os.path.join(ARTIFACTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    log.info(f"Plot saved → {path}")


# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────

def load_all():
    try:
        train = pd.read_csv(TRAIN_PATH)
        val   = pd.read_csv(VAL_PATH)
        test  = pd.read_csv(TEST_PATH)

        with open(REGRESSOR_PATH,  "rb") as f:
            reg = pickle.load(f)
        with open(CLASSIFIER_PATH, "rb") as f:
            clf = pickle.load(f)

        log.info(f"Train={train.shape} | Val={val.shape} | Test={test.shape}")
        log.info(f"Regressor : {type(reg).__name__}")
        log.info(f"Classifier: {type(clf).__name__}")
        return train, val, test, reg, clf
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 1 — LEARNING CURVE
# ─────────────────────────────────────────

def plot_learning_curve(reg, X_train, y_train_reg, X_val, y_val_reg):
    try:
        log.info("Plotting learning curve...")
        fractions   = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        n_train     = len(X_train)
        train_maes  = []
        val_maes    = []
        sizes       = []

        for frac in fractions:
            n     = max(10, int(n_train * frac))
            X_sub = X_train[:n]
            y_sub = y_train_reg[:n]
            m     = pickle.loads(pickle.dumps(reg))
            m.fit(X_sub, y_sub)
            train_maes.append(mean_absolute_error(y_sub,      m.predict(X_sub)))
            val_maes.append(  mean_absolute_error(y_val_reg,  m.predict(X_val)))
            sizes.append(n)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sizes, train_maes, "o-", color="#1E3A8A", label="Train MAE")
        ax.plot(sizes, val_maes,   "o-", color="#15803D", label="Val MAE")
        ax.fill_between(sizes, train_maes, val_maes, alpha=0.1, color="gray",
                        label="Gap (overfit zone)")
        ax.axhline(0.358, color="red", linestyle="--", linewidth=0.8,
                   label="CV MAE (0.358s)")
        ax.set_xlabel("Training Samples")
        ax.set_ylabel("MAE (seconds)")
        ax.set_title("Learning Curve — Regression (RandomForest)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig(fig, "06_learning_curve.png")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 2 — RESIDUAL DIAGNOSTICS
# ─────────────────────────────────────────

def plot_residuals(reg, X_val, y_val_reg, val_df):
    try:
        log.info("Plotting residuals...")
        pred      = reg.predict(X_val)
        residuals = y_val_reg - pred

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Regression Residual Diagnostics (Val Set)", fontsize=13)

        # Actual vs Predicted
        ax = axes[0]
        ax.scatter(y_val_reg, pred, color="#1E3A8A", alpha=0.7, s=40)
        mn = min(y_val_reg.min(), pred.min()) - 0.2
        mx = max(y_val_reg.max(), pred.max()) + 0.2
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlabel("Actual Delta (s)")
        ax.set_ylabel("Predicted Delta (s)")
        ax.set_title("Actual vs Predicted")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Residuals vs Predicted
        ax = axes[1]
        ax.scatter(pred, residuals, color="#15803D", alpha=0.7, s=40)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted Delta (s)")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title("Residuals vs Predicted")
        ax.grid(True, alpha=0.3)

        # Residuals by Race
        ax = axes[2]
        colors_race = {"Bahrain": "#1E3A8A", "Spain": "#15803D", "AbuDhabi": "#B45309"}
        for race in val_df["Race"].unique():
            mask = val_df["Race"] == race
            ax.scatter(
                val_df.loc[mask, "LapNumber"],
                residuals[mask.values],
                label=race,
                color=colors_race.get(race, "gray"),
                alpha=0.8, s=40
            )
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Lap Number (scaled)")
        ax.set_ylabel("Residual (s)")
        ax.set_title("Residuals by Race")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_fig(fig, "07_residual_diagnostics.png")

        # Log summary
        log.info(f"  Residual mean  = {residuals.mean():.4f}s  (bias)")
        log.info(f"  Residual std   = {residuals.std():.4f}s")
        log.info(f"  Max error      = {abs(residuals).max():.4f}s")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 3 — CONFUSION MATRIX + ROC CURVE
# ─────────────────────────────────────────

def plot_classification_diagnostics(clf, X_val, y_val_clf):
    try:
        log.info("Plotting classification diagnostics...")
        pred      = clf.predict(X_val)
        proba     = clf.predict_proba(X_val)[:, 1]
        cm        = confusion_matrix(y_val_clf, pred)
        fpr, tpr, _ = roc_curve(y_val_clf, proba)
        roc_auc   = auc(fpr, tpr)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Classification Diagnostics (Val Set)", fontsize=13)

        # Confusion matrix
        disp = ConfusionMatrixDisplay(cm, display_labels=["HAM Faster", "VER Faster"])
        disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
        axes[0].set_title("Confusion Matrix")

        # ROC curve
        axes[1].plot(fpr, tpr, color="#1E3A8A", linewidth=2,
                     label=f"ROC (AUC = {roc_auc:.3f})")
        axes[1].plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        save_fig(fig, "08_classification_diagnostics.png")
        log.info(f"  Val AUC = {roc_auc:.4f}")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 4 — SHAP FEATURE IMPORTANCE (Regression)
# The portfolio showpiece — what drives lap time delta
# ─────────────────────────────────────────

def plot_shap_regression(reg, X_train, X_val):
    try:
        log.info("Computing SHAP values for regressor...")

        explainer   = shap.TreeExplainer(reg)
        shap_values = explainer.shap_values(X_val)

        # ── 4a: SHAP summary bar (mean absolute impact) ──
        fig, ax = plt.subplots(figsize=(9, 7))
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx    = np.argsort(mean_abs_shap)
        colors        = ["#1E3A8A" if i >= len(sorted_idx) - 5
                         else "#93C5FD" for i in range(len(sorted_idx))]

        ax.barh(
            [FEATURE_LABELS[i] for i in sorted_idx],
            mean_abs_shap[sorted_idx],
            color=colors
        )
        ax.set_xlabel("Mean |SHAP Value| (seconds impact on lap delta)")
        ax.set_title("Feature Importance — What Drives Lap Time Delta\n"
                     "(RandomForest Regressor, SHAP)")
        ax.grid(True, alpha=0.3, axis="x")
        save_fig(fig, "09a_shap_importance_reg.png")

        # ── 4b: SHAP beeswarm (direction of impact) ──
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_val,
            feature_names=FEATURE_LABELS,
            show=False, plot_type="dot"
        )
        plt.title("SHAP Beeswarm — Direction of Feature Impact on Lap Delta")
        plt.tight_layout()
        save_fig(plt.gcf(), "09b_shap_beeswarm_reg.png")

        # Log top 5 features
        top5_idx = np.argsort(mean_abs_shap)[::-1][:5]
        log.info("  Top 5 features by SHAP impact (regression):")
        for rank, i in enumerate(top5_idx, 1):
            log.info(f"    {rank}. {FEATURE_LABELS[i]:<35} "
                     f"mean |SHAP| = {mean_abs_shap[i]:.4f}s")

        return shap_values

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 5 — SHAP FEATURE IMPORTANCE (Classifier)
# ─────────────────────────────────────────

def plot_shap_classifier(clf, X_train, X_val):
    try:
        log.info("Computing SHAP values for classifier...")

        # LogisticRegression — use LinearExplainer
        explainer   = shap.LinearExplainer(clf, X_train,
                                            feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_val)

        fig, ax = plt.subplots(figsize=(9, 7))
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx    = np.argsort(mean_abs_shap)
        colors        = ["#15803D" if i >= len(sorted_idx) - 5
                         else "#86EFAC" for i in range(len(sorted_idx))]

        ax.barh(
            [FEATURE_LABELS[i] for i in sorted_idx],
            mean_abs_shap[sorted_idx],
            color=colors
        )
        ax.set_xlabel("Mean |SHAP Value| (impact on VER_faster probability)")
        ax.set_title("Feature Importance — What Predicts VER Faster\n"
                     "(LogisticRegression, SHAP)")
        ax.grid(True, alpha=0.3, axis="x")
        save_fig(fig, "10_shap_importance_clf.png")

        log.info("  Top 5 features by SHAP impact (classifier):")
        top5_idx = np.argsort(mean_abs_shap)[::-1][:5]
        for rank, i in enumerate(top5_idx, 1):
            log.info(f"    {rank}. {FEATURE_LABELS[i]:<35} "
                     f"mean |SHAP| = {mean_abs_shap[i]:.4f}")

        return shap_values

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 6 — DRIVER STYLE FINGERPRINT
# The fan-facing insight plot
# Average driving style metrics per driver per race
# ─────────────────────────────────────────

def plot_driver_fingerprint(train_df, val_df, test_df):
    try:
        log.info("Plotting driver style fingerprints...")

        # Combine all splits — use unscaled features.csv for this
        # We read from features.csv which has raw (unscaled) values
        from src.config import FEATURES_PATH
        features_df = pd.read_csv(FEATURES_PATH)

        style_metrics = {
            "Coasting %"        : ("VER_coasting_pct",         "HAM_coasting_pct"),
            "Full Throttle %"   : ("VER_full_throttle_pct",    "HAM_full_throttle_pct"),
            "Gear Shifts/Lap"   : ("VER_gear_shifts",          "HAM_gear_shifts"),
            "Brake Zone (m)"    : ("VER_avg_brake_zone_length","HAM_avg_brake_zone_length"),
            "Entry Speed (km/h)": ("VER_avg_entry_speed",      "HAM_avg_entry_speed"),
        }

        races = ["Bahrain", "Spain", "AbuDhabi"]
        fig, axes = plt.subplots(len(style_metrics), len(races),
                                 figsize=(15, 14), sharey="row")
        fig.suptitle("Driver Style Fingerprint — VER vs HAM per Race\n"
                     "(Average per lap, all clean green-flag laps)",
                     fontsize=13, y=1.01)

        colors = {"VER": "#1E3A8A", "HAM": "#15803D"}

        for row_idx, (metric_name, (ver_col, ham_col)) in enumerate(style_metrics.items()):
            for col_idx, race in enumerate(races):
                ax    = axes[row_idx][col_idx]
                rdata = features_df[features_df["Race"] == race]

                ver_vals = rdata[ver_col]
                ham_vals = rdata[ham_col]

                ax.hist(ver_vals, bins=15, alpha=0.6,
                        color=colors["VER"], label="VER")
                ax.hist(ham_vals, bins=15, alpha=0.6,
                        color=colors["HAM"], label="HAM")

                ax.axvline(ver_vals.mean(), color=colors["VER"],
                           linestyle="--", linewidth=1.5,
                           label=f"VER mean={ver_vals.mean():.2f}")
                ax.axvline(ham_vals.mean(), color=colors["HAM"],
                           linestyle="--", linewidth=1.5,
                           label=f"HAM mean={ham_vals.mean():.2f}")

                if row_idx == 0:
                    ax.set_title(race, fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(metric_name, fontsize=9)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=7)

                ax.grid(True, alpha=0.2)

        plt.tight_layout()
        save_fig(fig, "11_driver_fingerprint.png")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# TEST SET EVALUATION — the untouched holdout
# ─────────────────────────────────────────

def evaluate_test_set(reg, clf, X_test, y_test_reg, y_test_clf, test_df):
    try:
        log.info("=" * 60)
        log.info("TEST SET EVALUATION — Final holdout")
        log.info("=" * 60)

        # Regression
        reg_pred  = reg.predict(X_test)
        test_mae  = mean_absolute_error(y_test_reg, reg_pred)
        test_rmse = mean_squared_error(y_test_reg,  reg_pred) ** 0.5
        test_r2   = r2_score(y_test_reg, reg_pred)

        log.info(f"\n  REGRESSION on test set:")
        log.info(f"    MAE  = {test_mae:.4f}s")
        log.info(f"    RMSE = {test_rmse:.4f}s")
        log.info(f"    R2   = {test_r2:.4f}")

        # Classification
        clf_pred  = clf.predict(X_test)
        clf_proba = clf.predict_proba(X_test)[:, 1]
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        test_acc  = accuracy_score(y_test_clf, clf_pred)
        test_f1   = f1_score(y_test_clf, clf_pred, zero_division=0)
        test_auc  = roc_auc_score(y_test_clf, clf_proba)

        log.info(f"\n  CLASSIFICATION on test set:")
        log.info(f"    Accuracy = {test_acc:.4f}")
        log.info(f"    F1       = {test_f1:.4f}")
        log.info(f"    AUC      = {test_auc:.4f}")

        # Compare val vs test
        log.info(f"\n  Val vs Test comparison:")
        log.info(f"    Regression  MAE  — Val: 0.2515s | Test: {test_mae:.4f}s")
        log.info(f"    Classifier  AUC  — Val: 1.0000  | Test: {test_auc:.4f}")

        # Final verdict
        log.info(f"\n  FINAL VERDICT:")
        if abs(test_mae - 0.2515) < 0.10:
            log.info("    Regression generalizes well to test set. ✓")
        else:
            log.info("    Regression test MAE diverges from val — "
                     "consider more data or additional features.")

        if test_auc >= 0.80:
            log.info("    Classifier generalizes well to test set. ✓")
        else:
            log.info("    Classifier AUC dropped on test — "
                     "model is sensitive to lap composition. "
                     "Consider stratifying by same_compound.")

        # Test set prediction plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Test Set Evaluation", fontsize=13)

        # Actual vs predicted (regression)
        ax = axes[0]
        colors_race = {"Bahrain": "#1E3A8A", "Spain": "#15803D", "AbuDhabi": "#B45309"}
        for race in test_df["Race"].unique():
            mask = test_df["Race"].values == race
            ax.scatter(y_test_reg[mask], reg_pred[mask],
                       label=race, color=colors_race.get(race, "gray"),
                       alpha=0.8, s=50)
        mn = min(y_test_reg.min(), reg_pred.min()) - 0.2
        mx = max(y_test_reg.max(), reg_pred.max()) + 0.2
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Perfect")
        ax.set_xlabel("Actual Delta (s)")
        ax.set_ylabel("Predicted Delta (s)")
        ax.set_title(f"Regression — Test MAE={test_mae:.3f}s  R2={test_r2:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Predicted probability distribution (classifier)
        ax = axes[1]
        ax.hist(clf_proba[y_test_clf == 0], bins=10, alpha=0.6,
                color="#15803D", label="HAM Faster (actual)")
        ax.hist(clf_proba[y_test_clf == 1], bins=10, alpha=0.6,
                color="#1E3A8A", label="VER Faster (actual)")
        ax.axvline(0.5, color="red", linestyle="--", label="Decision boundary")
        ax.set_xlabel("Predicted Probability (VER Faster)")
        ax.set_ylabel("Count")
        ax.set_title(f"Classifier — Test AUC={test_auc:.3f}  F1={test_f1:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_fig(fig, "12_test_set_evaluation.png")

        return {
            "test_mae" : test_mae,
            "test_rmse": test_rmse,
            "test_r2"  : test_r2,
            "test_acc" : test_acc,
            "test_f1"  : test_f1,
            "test_auc" : test_auc,
        }

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_diagnostics():
    try:
        log.info("=" * 60)
        log.info("Starting model diagnostics")
        log.info("=" * 60)

        train, val, test, reg, clf = load_all()

        X_train = train[FEATURE_COLS].values
        X_val   = val[FEATURE_COLS].values
        X_test  = test[FEATURE_COLS].values

        y_train_reg = train[TARGET_REGRESSION].values
        y_val_reg   = val[TARGET_REGRESSION].values
        y_test_reg  = test[TARGET_REGRESSION].values

        y_train_clf = train[TARGET_CLASSIFICATION].values
        y_val_clf   = val[TARGET_CLASSIFICATION].values
        y_test_clf  = test[TARGET_CLASSIFICATION].values

        # Plot 1 — Learning curve
        plot_learning_curve(reg, X_train, y_train_reg, X_val, y_val_reg)

        # Plot 2 — Residuals
        plot_residuals(reg, X_val, y_val_reg, val)

        # Plot 3 — Classification diagnostics
        plot_classification_diagnostics(clf, X_val, y_val_clf)

        # Plot 4 — SHAP regression
        plot_shap_regression(reg, X_train, X_val)

        # Plot 5 — SHAP classifier
        plot_shap_classifier(clf, X_train, X_val)

        # Plot 6 — Driver fingerprint
        plot_driver_fingerprint(train, val, test)

        # Test set evaluation
        test_results = evaluate_test_set(
            reg, clf,
            X_test, y_test_reg, y_test_clf, test
        )

        log.info("=" * 60)
        log.info("Model diagnostics complete. All plots saved to artifacts/")
        log.info("=" * 60)

        return test_results

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    test_results = run_diagnostics()

    print("\n--- FINAL TEST SET RESULTS ---")
    print(f"Regression  MAE  : {test_results['test_mae']:.4f}s")
    print(f"Regression  RMSE : {test_results['test_rmse']:.4f}s")
    print(f"Regression  R2   : {test_results['test_r2']:.4f}")
    print(f"Classifier  Acc  : {test_results['test_acc']:.4f}")
    print(f"Classifier  F1   : {test_results['test_f1']:.4f}")
    print(f"Classifier  AUC  : {test_results['test_auc']:.4f}")