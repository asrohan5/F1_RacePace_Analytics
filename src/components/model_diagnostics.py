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
    TRAIN_SC_PATH, VAL_SC_PATH, TEST_SC_PATH,
    REGRESSOR_PATH, CLASSIFIER_PATH,
    REGRESSOR_SC_PATH, CLASSIFIER_SC_PATH,
    FEATURES_PATH, FEATURES_SAME_COMPOUND_PATH,
    ARTIFACTS_DIR,
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

FEATURE_LABELS = [
    "Coasting % Delta",         "Full Throttle % Delta",    "Gear Shifts Delta",
    "Brake Zone Length Delta",  "Entry Speed Delta",         "Brake Zone Count Delta",
    "Tyre Life Delta",          "TyreLife x Coasting Delta",
    "Stint Phase Delta",        "AbuDhabi Gear Delta",       "Rolling Delta 3",
    "VER Coasting %",           "HAM Coasting %",
    "VER Full Throttle %",      "HAM Full Throttle %",
    "VER Gear Shifts",          "HAM Gear Shifts",
    "VER Brake Zone Length",    "HAM Brake Zone Length",
    "VER Entry Speed",          "HAM Entry Speed",
    "VER Tyre Life",            "HAM Tyre Life",
    "VER Stint Phase",          "HAM Stint Phase",
    "Same Compound",            "VER Compound",              "HAM Compound",
    "Lap Number",               "Race"
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
        train    = pd.read_csv(TRAIN_PATH)
        val      = pd.read_csv(VAL_PATH)
        test     = pd.read_csv(TEST_PATH)
        train_sc = pd.read_csv(TRAIN_SC_PATH)
        val_sc   = pd.read_csv(VAL_SC_PATH)
        test_sc  = pd.read_csv(TEST_SC_PATH)

        with open(REGRESSOR_PATH,     "rb") as f: reg    = pickle.load(f)
        with open(CLASSIFIER_PATH,    "rb") as f: clf    = pickle.load(f)
        with open(REGRESSOR_SC_PATH,  "rb") as f: reg_sc = pickle.load(f)
        with open(CLASSIFIER_SC_PATH, "rb") as f: clf_sc = pickle.load(f)

        log.info(f"Full  — Train={train.shape} Val={val.shape} Test={test.shape}")
        log.info(f"SC    — Train={train_sc.shape} Val={val_sc.shape} Test={test_sc.shape}")
        log.info(f"Regressor     : {type(reg).__name__}")
        log.info(f"Classifier    : {type(clf).__name__}")
        log.info(f"SC Regressor  : {type(reg_sc).__name__}")
        log.info(f"SC Classifier : {type(clf_sc).__name__}")

        return train, val, test, train_sc, val_sc, test_sc, reg, clf, reg_sc, clf_sc
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

        # Residuals vs Predicted — coloured by same_compound
        # This directly tests whether large residuals cluster in compound-mismatch laps
        ax = axes[1]
        same_mask = val_df["same_compound"].values > 0.5

        ax.scatter(pred[same_mask],  residuals[same_mask],
                   color="#15803D", alpha=0.8, s=50, label="Same compound")
        ax.scatter(pred[~same_mask], residuals[~same_mask],
                   color="#DC2626", alpha=0.8, s=50, label="Diff compound",
                   marker="^")
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted Delta (s)")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title("Residuals vs Predicted\n(coloured by compound match)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Residuals by Race
        ax = axes[2]
        colors_race = {"Bahrain": "#1E3A8A", "Spain": "#15803D", "AbuDhabi": "#B45309"}
        for race in val_df["Race"].unique():
            mask = val_df["Race"].values == race
            ax.scatter(
                val_df.loc[mask, "LapNumber"],
                residuals[mask],
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

        log.info(f"  Residual mean  = {residuals.mean():.4f}s  (bias)")
        log.info(f"  Residual std   = {residuals.std():.4f}s")
        log.info(f"  Max error      = {abs(residuals).max():.4f}s")

        # Key finding: are large residuals in compound-mismatch laps?
        if same_mask.sum() > 0 and (~same_mask).sum() > 0:
            same_mae = mean_absolute_error(y_val_reg[same_mask],  pred[same_mask])
            diff_mae  = mean_absolute_error(y_val_reg[~same_mask], pred[~same_mask])
            log.info(f"\n  MAE by compound match:")
            log.info(f"    Same compound laps : {same_mae:.4f}s")
            log.info(f"    Diff compound laps : {diff_mae:.4f}s")
            log.info(f"    Ratio (diff/same)  : {diff_mae/same_mae:.2f}x worse on diff compound")
        else:
            log.info(f"\n  MAE by compound match: skipped — one group has 0 samples in val set")
        
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

def plot_shap_regression_sc(reg_sc, X_train_sc, X_val_sc):
    try:
        log.info("Computing SHAP values for SC regressor (LinearRegression)...")

        explainer   = shap.LinearExplainer(reg_sc, X_train_sc,
                                            feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_val_sc)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("SHAP Feature Importance — Same-Compound Laps\n"
                     "(Driving Style Isolated, LinearRegression)",
                     fontsize=13)

        # Bar chart
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx    = np.argsort(mean_abs_shap)
        colors        = ["#B45309" if i >= len(sorted_idx) - 5
                         else "#FCD34D" for i in range(len(sorted_idx))]

        axes[0].barh(
            [FEATURE_LABELS[i] for i in sorted_idx],
            mean_abs_shap[sorted_idx],
            color=colors
        )
        axes[0].set_xlabel("Mean |SHAP Value| (seconds impact on lap delta)")
        axes[0].set_title("Feature Importance — SC Model")
        axes[0].grid(True, alpha=0.3, axis="x")

        # Side-by-side comparison with full model SHAP
        # Compute full model SHAP for comparison
        full_explainer   = shap.TreeExplainer(
            pickle.loads(open(REGRESSOR_PATH, "rb").read())
        )
        full_shap_vals   = full_explainer.shap_values(X_val_sc)
        full_mean_abs    = np.abs(full_shap_vals).mean(axis=0)

        x      = np.arange(len(FEATURE_LABELS))
        width  = 0.35
        axes[1].barh(x - width/2, full_mean_abs,  width,
                     label="Full Dataset (XGBoost)",    color="#1E3A8A", alpha=0.7)
        axes[1].barh(x + width/2, mean_abs_shap,  width,
                     label="Same Compound (LinearReg)", color="#B45309", alpha=0.7)
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(FEATURE_LABELS, fontsize=8)
        axes[1].set_xlabel("Mean |SHAP Value|")
        axes[1].set_title("Full vs SC — Feature Importance Shift")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        save_fig(fig, "13_shap_comparison_full_vs_sc.png")

        log.info("  Top 5 SC features by SHAP impact:")
        top5 = np.argsort(mean_abs_shap)[::-1][:5]
        for rank, i in enumerate(top5, 1):
            log.info(f"    {rank}. {FEATURE_LABELS[i]:<35} "
                     f"mean |SHAP| = {mean_abs_shap[i]:.4f}s")

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

def evaluate_test_set_sc(reg_sc, clf_sc, X_test_sc,
                          y_test_sc_reg, y_test_sc_clf, test_sc_df):
    try:
        log.info("=" * 60)
        log.info("SC TEST SET EVALUATION — Same-compound holdout")
        log.info("=" * 60)

        from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                     r2_score, accuracy_score,
                                     f1_score, roc_auc_score)

        reg_pred  = reg_sc.predict(X_test_sc)
        test_mae  = mean_absolute_error(y_test_sc_reg, reg_pred)
        test_rmse = mean_squared_error(y_test_sc_reg,  reg_pred) ** 0.5
        test_r2   = r2_score(y_test_sc_reg, reg_pred)

        clf_pred  = clf_sc.predict(X_test_sc)
        clf_proba = clf_sc.predict_proba(X_test_sc)[:, 1]
        test_acc  = accuracy_score(y_test_sc_clf, clf_pred)
        test_f1   = f1_score(y_test_sc_clf, clf_pred, zero_division=0)
        test_auc  = roc_auc_score(y_test_sc_clf, clf_proba)

        log.info(f"\n  SC REGRESSION on test:")
        log.info(f"    MAE  = {test_mae:.4f}s")
        log.info(f"    RMSE = {test_rmse:.4f}s")
        log.info(f"    R2   = {test_r2:.4f}")
        log.info(f"\n  SC CLASSIFICATION on test:")
        log.info(f"    Accuracy = {test_acc:.4f}")
        log.info(f"    F1       = {test_f1:.4f}")
        log.info(f"    AUC      = {test_auc:.4f}")

        # Comparison table
        log.info(f"\n  Full vs SC — Test Set Comparison:")
        log.info(f"    {'Metric':<20} {'Full Dataset':>15} {'Same Compound':>15}")
        log.info(f"    {'-'*50}")
        log.info(f"    {'Reg MAE':<20} {'0.6679s':>15} {test_mae:>14.4f}s")
        log.info(f"    {'Reg R2':<20} {'0.4283':>15} {test_r2:>15.4f}")
        log.info(f"    {'Clf AUC':<20} {'0.8571':>15} {test_auc:>15.4f}")
        log.info(f"    {'Clf F1':<20} {'0.8000':>15} {test_f1:>15.4f}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("SC Test Set Evaluation — Same-Compound Laps Only", fontsize=13)

        colors_race = {"Bahrain": "#1E3A8A", "Spain": "#15803D", "AbuDhabi": "#B45309"}
        for race in test_sc_df["Race"].unique():
            mask = test_sc_df["Race"].values == race
            axes[0].scatter(y_test_sc_reg[mask], reg_pred[mask],
                            label=race, color=colors_race.get(race, "gray"),
                            alpha=0.8, s=60)
        mn = min(y_test_sc_reg.min(), reg_pred.min()) - 0.2
        mx = max(y_test_sc_reg.max(), reg_pred.max()) + 0.2
        axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Perfect")
        axes[0].set_xlabel("Actual Delta (s)")
        axes[0].set_ylabel("Predicted Delta (s)")
        axes[0].set_title(f"SC Regression — MAE={test_mae:.3f}s  R2={test_r2:.3f}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(clf_proba[y_test_sc_clf == 0], bins=8, alpha=0.6,
                     color="#15803D", label="HAM Faster (actual)")
        axes[1].hist(clf_proba[y_test_sc_clf == 1], bins=8, alpha=0.6,
                     color="#1E3A8A", label="VER Faster (actual)")
        axes[1].axvline(0.5, color="red", linestyle="--", label="Decision boundary")
        axes[1].set_xlabel("Predicted Probability (VER Faster)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"SC Classifier — AUC={test_auc:.3f}  F1={test_f1:.3f}")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        save_fig(fig, "14_sc_test_set_evaluation.png")

        return {
            "sc_test_mae" : test_mae,
            "sc_test_rmse": test_rmse,
            "sc_test_r2"  : test_r2,
            "sc_test_acc" : test_acc,
            "sc_test_f1"  : test_f1,
            "sc_test_auc" : test_auc,
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

        (train, val, test,
         train_sc, val_sc, test_sc,
         reg, clf, reg_sc, clf_sc) = load_all()

        X_train = train[FEATURE_COLS].values
        X_val   = val[FEATURE_COLS].values
        X_test  = test[FEATURE_COLS].values

        X_train_sc = train_sc[FEATURE_COLS].values
        X_val_sc   = val_sc[FEATURE_COLS].values
        X_test_sc  = test_sc[FEATURE_COLS].values

        y_train_reg    = train[TARGET_REGRESSION].values
        y_val_reg      = val[TARGET_REGRESSION].values
        y_test_reg     = test[TARGET_REGRESSION].values
        y_train_clf    = train[TARGET_CLASSIFICATION].values
        y_val_clf      = val[TARGET_CLASSIFICATION].values
        y_test_clf     = test[TARGET_CLASSIFICATION].values

        y_test_sc_reg  = test_sc[TARGET_REGRESSION].values
        y_test_sc_clf  = test_sc[TARGET_CLASSIFICATION].values

        # ── Full model diagnostics ──
        log.info("\n>>> FULL MODEL DIAGNOSTICS")
        plot_learning_curve(reg, X_train, y_train_reg, X_val, y_val_reg)
        plot_residuals(reg, X_val, y_val_reg, val)
        plot_classification_diagnostics(clf, X_val, y_val_clf)
        plot_shap_regression(reg, X_train, X_val)
        plot_shap_classifier(clf, X_train, X_val)
        plot_driver_fingerprint(train, val, test)
        full_results = evaluate_test_set(
            reg, clf, X_test, y_test_reg, y_test_clf, test
        )

        # ── SC model diagnostics ──
        log.info("\n>>> SAME-COMPOUND MODEL DIAGNOSTICS")
        plot_shap_regression_sc(reg_sc, X_train_sc, X_val_sc)
        sc_results = evaluate_test_set_sc(
            reg_sc, clf_sc, X_test_sc,
            y_test_sc_reg, y_test_sc_clf, test_sc
        )

        log.info("=" * 60)
        log.info("All diagnostics complete. Plots saved to artifacts/")
        log.info("=" * 60)

        return full_results, sc_results

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    full_results, sc_results = run_diagnostics()

    print("\n--- FULL DATASET TEST RESULTS ---")
    print(f"Regression  MAE  : {full_results['test_mae']:.4f}s")
    print(f"Regression  R2   : {full_results['test_r2']:.4f}")
    print(f"Classifier  AUC  : {full_results['test_auc']:.4f}")
    print(f"Classifier  F1   : {full_results['test_f1']:.4f}")

    print("\n--- SAME-COMPOUND TEST RESULTS ---")
    print(f"Regression  MAE  : {sc_results['sc_test_mae']:.4f}s")
    print(f"Regression  R2   : {sc_results['sc_test_r2']:.4f}")
    print(f"Classifier  AUC  : {sc_results['sc_test_auc']:.4f}")
    print(f"Classifier  F1   : {sc_results['sc_test_f1']:.4f}")

    print("\n--- IMPROVEMENT SUMMARY ---")
    reg_improvement = full_results['test_mae'] - sc_results['sc_test_mae']
    auc_improvement = sc_results['sc_test_auc'] - full_results['test_auc']
    print(f"Regression MAE improved by : {reg_improvement:+.4f}s "
          f"({'better' if reg_improvement > 0 else 'worse'})")
    print(f"Classifier AUC shifted by  : {auc_improvement:+.4f} "
          f"({'better' if auc_improvement > 0 else 'worse'})")