import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error, r2_score,
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TRAIN_SC_PATH, VAL_SC_PATH, TEST_SC_PATH,
    REGRESSOR_PATH, CLASSIFIER_PATH,
    REGRESSOR_SC_PATH, CLASSIFIER_SC_PATH,
    ARTIFACTS_DIR,
    TARGET_REGRESSION, TARGET_CLASSIFICATION,FEATURES_SAME_COMPOUND_PATH, FEATURES_PATH
)

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "validation_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ID_COLS = ["Race", "RoundNumber", "LapNumber",
           TARGET_REGRESSION, TARGET_CLASSIFICATION]

def get_feature_cols(df):
    return [c for c in df.columns if c not in ID_COLS]

# ─────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "#0f0f0f",
    "axes.facecolor"   : "#1a1a1a",
    "axes.edgecolor"   : "#444444",
    "axes.labelcolor"  : "#cccccc",
    "axes.titlecolor"  : "#ffffff",
    "xtick.color"      : "#888888",
    "ytick.color"      : "#888888",
    "text.color"       : "#cccccc",
    "grid.color"       : "#2a2a2a",
    "grid.linewidth"   : 0.6,
    "font.size"        : 10,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 10,
    "legend.framealpha": 0.3,
    "legend.edgecolor" : "#444444",
})

VER_COLOR  = "#3671C6"   # Red Bull blue
HAM_COLOR  = "#00D2BE"   # Mercedes teal
NEUTRAL    = "#e8e8e8"
ACCENT     = "#ff6b35"
GREY       = "#555555"

# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────

def load_all():
    try:
        train    = pd.read_csv(TRAIN_PATH)
        val      = pd.read_csv(VAL_PATH)
        test     = pd.read_csv(TEST_PATH)
        sc_train = pd.read_csv(TRAIN_SC_PATH)
        sc_val   = pd.read_csv(VAL_SC_PATH)
        sc_test  = pd.read_csv(TEST_SC_PATH)

        with open(REGRESSOR_PATH,    "rb") as f: reg     = pickle.load(f)
        with open(CLASSIFIER_PATH,   "rb") as f: clf     = pickle.load(f)
        with open(REGRESSOR_SC_PATH, "rb") as f: reg_sc  = pickle.load(f)
        with open(CLASSIFIER_SC_PATH,"rb") as f: clf_sc  = pickle.load(f)

        log.info("All splits and models loaded.")
        return (train, val, test, sc_train, sc_val, sc_test,
                reg, clf, reg_sc, clf_sc)
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────

def regression_metrics(model, X, y, label):
    pred = model.predict(X)
    mae  = mean_absolute_error(y, pred)
    r2   = r2_score(y, pred)
    bias = np.mean(pred - y)
    log.info(f"  [{label}] MAE={mae:.4f}s  R²={r2:.4f}  Bias={bias:+.4f}s")
    return pred, mae, r2, bias


def classification_metrics(model, X, y, label):
    pred      = model.predict(X)
    pred_prob = model.predict_proba(X)[:, 1]
    acc  = accuracy_score(y, pred)
    f1   = f1_score(y, pred, zero_division=0)
    auc  = roc_auc_score(y, pred_prob)
    cm   = confusion_matrix(y, pred)
    log.info(f"  [{label}] Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return pred, pred_prob, acc, f1, auc, cm


# ─────────────────────────────────────────
# PLOT 1 — REGRESSION: PREDICTED VS ACTUAL
# One panel per split (train / val / test)
# Both full and SC model
# ─────────────────────────────────────────

def plot_pred_vs_actual(splits_data, model_label, filename):
    """
    splits_data: list of (split_label, y_true, y_pred, mae, r2)
    """
    try:
        n = len(splits_data)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        fig.suptitle(f"Predicted vs Actual — {model_label}", fontsize=13,
                     color=NEUTRAL, y=1.01)

        for ax, (split_label, y_true, y_pred, mae, r2) in zip(axes, splits_data):
            lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.1
            ax.scatter(y_true, y_pred, alpha=0.55, s=18,
                       color=VER_COLOR, edgecolors="none")
            ax.plot([-lim, lim], [-lim, lim], "--", color=ACCENT,
                    linewidth=1.2, label="Perfect")
            ax.axhline(0, color=GREY, linewidth=0.6, linestyle=":")
            ax.axvline(0, color=GREY, linewidth=0.6, linestyle=":")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("Actual Δ lap time (s)")
            ax.set_ylabel("Predicted Δ lap time (s)")
            ax.set_title(f"{split_label}\nMAE={mae:.3f}s  R²={r2:.3f}")
            ax.legend(fontsize=8)
            ax.grid(True)

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved: {filename}")
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 2 — RESIDUALS BY RACE
# Shows which races the model struggles on
# ─────────────────────────────────────────

def plot_residuals_by_race(df_split, y_pred, split_label, model_label, filename):
    try:
        df = df_split.copy()
        df["residual"] = y_pred - df[TARGET_REGRESSION].values

        race_order = (df.groupby("Race")["residual"]
                      .apply(lambda x: x.abs().mean())
                      .sort_values(ascending=False).index.tolist())

        fig, ax = plt.subplots(figsize=(max(10, len(race_order) * 0.7), 5))
        sns.boxplot(data=df, x="Race", y="residual", order=race_order,
                    color=VER_COLOR, linecolor=NEUTRAL, linewidth=0.8,
                    fliersize=3, ax=ax)
        ax.axhline(0, color=ACCENT, linewidth=1.2, linestyle="--")
        ax.set_title(f"Residuals by Race — {model_label} ({split_label})")
        ax.set_xlabel("")
        ax.set_ylabel("Residual (pred − actual, s)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y")

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved: {filename}")
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 3 — ROC + PRECISION-RECALL
# ─────────────────────────────────────────

def plot_roc_pr(y_true, y_prob, split_label, model_label, filename):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val     = roc_auc_score(y_true, y_prob)
        prec, rec, _= precision_recall_curve(y_true, y_prob)
        baseline_pr = y_true.mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.suptitle(f"ROC & PR Curves — {model_label} ({split_label})",
                     color=NEUTRAL)

        # ROC
        ax1.plot(fpr, tpr, color=VER_COLOR, linewidth=2,
                 label=f"AUC = {auc_val:.3f}")
        ax1.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend()
        ax1.grid(True)

        # PR
        ax2.plot(rec, prec, color=HAM_COLOR, linewidth=2)
        ax2.axhline(baseline_pr, color=GREY, linestyle="--", linewidth=1,
                    label=f"Baseline ({baseline_pr:.2f})")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved: {filename}")
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 4 — CONFUSION MATRIX
# ─────────────────────────────────────────

def plot_confusion_matrix(cm, split_label, model_label, filename):
    try:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["HAM faster", "VER faster"],
                    yticklabels=["HAM faster", "VER faster"],
                    linewidths=0.5, ax=ax,
                    annot_kws={"size": 13, "color": NEUTRAL})
        ax.set_title(f"Confusion Matrix — {model_label} ({split_label})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved: {filename}")
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 5 — FEATURE IMPORTANCE
# Tree-based models: built-in gain importance
# ─────────────────────────────────────────

def plot_feature_importance(model, feature_cols, model_label, filename,
                             top_n=20):
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            log.warning(f"  {model_label} has no feature_importances_, skipping")
            return

        fi = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
        fi = fi.tail(top_n)

        fig, ax = plt.subplots(figsize=(7, top_n * 0.35 + 1))
        colors = [VER_COLOR if v >= fi.median() else GREY for v in fi.values]
        fi.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
        ax.set_title(f"Feature Importance (gain) — {model_label} | top {top_n}")
        ax.set_xlabel("Importance")
        ax.grid(True, axis="x")
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved: {filename}")
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 6 — TRAIN/VAL/TEST METRIC SUMMARY
# Side-by-side bars for MAE (regression) and AUC (classification)
# ─────────────────────────────────────────

def plot_metric_summary(metric_rows, ylabel, title, filename):
    """
    metric_rows: list of (split_label, full_val, sc_val)
    """
    try:
        labels = [r[0] for r in metric_rows]
        full   = [r[1] for r in metric_rows]
        sc     = [r[2] for r in metric_rows]
        x      = np.arange(len(labels))
        w      = 0.35

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - w/2, full, w, color=VER_COLOR, label="Full model",
               edgecolor="none")
        ax.bar(x + w/2, sc,   w, color=HAM_COLOR, label="SC model",
               edgecolor="none")

        for xi, v in zip(x - w/2, full):
            ax.text(xi, v + 0.003, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=8, color=NEUTRAL)
        for xi, v in zip(x + w/2, sc):
            ax.text(xi, v + 0.003, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=8, color=NEUTRAL)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y")
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved: {filename}")
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# PLOT 7 — ROLLING DELTA: ACTUAL VS PREDICTED
# Abu Dhabi test set only — lap-by-lap comparison
# ─────────────────────────────────────────


def plot_abu_dhabi_trace(test_raw, reg_pred, reg_sc_pred,
                          sc_test_raw, filename):
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
        fig.suptitle("Abu Dhabi 2021 — Predicted vs Actual Lap Time Delta",
                     color=NEUTRAL, fontsize=13)

        # Sort by LapNumber so trace reads left-to-right
        test_sorted = test_raw.sort_values("LapNumber").reset_index(drop=True)
        sc_sorted   = sc_test_raw.sort_values("LapNumber").reset_index(drop=True)

        # Use actual lap numbers as x-tick labels but integer positions as x
        laps_full   = test_sorted["LapNumber"].astype(int).values
        actual_full = test_sorted[TARGET_REGRESSION].values
        reg_pred_sorted = reg_pred[test_sorted.index]  # align predictions

        laps_sc   = sc_sorted["LapNumber"].astype(int).values
        actual_sc = sc_sorted[TARGET_REGRESSION].values
        reg_sc_pred_sorted = reg_sc_pred[sc_sorted.index]

        x_full = np.arange(len(laps_full))
        x_sc   = np.arange(len(laps_sc))

        ax1.plot(x_full, actual_full, color=NEUTRAL, linewidth=1.5,
                 label="Actual Δ", marker="o", markersize=3)
        ax1.plot(x_full, reg_pred_sorted, color=VER_COLOR, linewidth=1.5,
                 label="Predicted Δ (Full)", linestyle="--", marker="s", markersize=3)
        ax1.axhline(0, color=GREY, linewidth=0.8, linestyle=":")
        ax1.set_xticks(x_full[::3])
        ax1.set_xticklabels(x_full[::3] + 1)
        ax1.set_ylabel("Δ lap time (s)\n(VER − HAM)")
        ax1.set_title(f"Full model  |  MAE={mean_absolute_error(actual_full, reg_pred_sorted):.3f}s")
        ax1.legend(fontsize=8)
        ax1.grid(True)

        ax2.plot(x_sc, actual_sc, color=NEUTRAL, linewidth=1.5,
                 label="Actual Δ", marker="o", markersize=3)
        ax2.plot(x_sc, reg_sc_pred_sorted, color=HAM_COLOR, linewidth=1.5,
                 label="Predicted Δ (SC)", linestyle="--", marker="s", markersize=3)
        ax2.axhline(0, color=GREY, linewidth=0.8, linestyle=":")
        ax2.set_xticks(x_sc[::3])
        ax2.set_xticklabels(x_sc[::3] + 1)
        ax2.set_xlabel("Lap Number")
        ax2.set_ylabel("Δ lap time (s)\n(VER − HAM)")
        ax2.set_title(f"SC model  |  MAE={mean_absolute_error(actual_sc, reg_sc_pred_sorted):.3f}s"
                      f"  (n={len(sc_sorted)} SC laps)")
        ax2.legend(fontsize=8)
        ax2.grid(True)

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved: {filename}")
    except Exception as e:
        raise CustomException(e, sys)
    
# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
'''
def run_validation():
    try:
        log.info("=" * 60)
        log.info("Starting model validation — Phase 2")
        log.info("=" * 60)

        (train, val, test,
         sc_train, sc_val, sc_test,
         reg, clf, reg_sc, clf_sc) = load_all()
        
        # Load unscaled features for raw LapNumber (trace plot)
        features_raw    = pd.read_csv(FEATURES_PATH)
        features_sc_raw = pd.read_csv(FEATURES_SAME_COMPOUND_PATH)
        test_raw    = features_raw[features_raw["Race"] == "AbuDhabi"].copy()
        sc_test_raw = features_sc_raw[features_sc_raw["Race"] == "AbuDhabi"].copy()

        fc      = get_feature_cols(train)
        fc_sc   = get_feature_cols(sc_train)

        # ── Prepare arrays ────────────────────────────────────
        X_tr,  y_tr_r,  y_tr_c  = train[fc].values,    train[TARGET_REGRESSION].values,    train[TARGET_CLASSIFICATION].values
        X_val, y_val_r, y_val_c = val[fc].values,       val[TARGET_REGRESSION].values,       val[TARGET_CLASSIFICATION].values
        X_te,  y_te_r,  y_te_c  = test[fc].values,      test[TARGET_REGRESSION].values,      test[TARGET_CLASSIFICATION].values

        X_sc_tr,  y_sc_tr_r,  y_sc_tr_c  = sc_train[fc_sc].values, sc_train[TARGET_REGRESSION].values, sc_train[TARGET_CLASSIFICATION].values
        X_sc_val, y_sc_val_r, y_sc_val_c = sc_val[fc_sc].values,   sc_val[TARGET_REGRESSION].values,   sc_val[TARGET_CLASSIFICATION].values
        X_sc_te,  y_sc_te_r,  y_sc_te_c  = sc_test[fc_sc].values,  sc_test[TARGET_REGRESSION].values,  sc_test[TARGET_CLASSIFICATION].values

        # ─────────────────────────────────────────────────────
        # REGRESSION VALIDATION
        # ─────────────────────────────────────────────────────
        log.info("\n=== REGRESSION — FULL MODEL ===")
        pred_tr_r,  mae_tr,  r2_tr,  bias_tr  = regression_metrics(reg, X_tr,  y_tr_r,  "Train")
        pred_val_r, mae_val, r2_val, bias_val  = regression_metrics(reg, X_val, y_val_r, "Val")
        pred_te_r,  mae_te,  r2_te,  bias_te   = regression_metrics(reg, X_te,  y_te_r,  "TEST")

        log.info("\n=== REGRESSION — SC MODEL ===")
        pred_sc_tr_r,  mae_sc_tr,  r2_sc_tr,  _  = regression_metrics(reg_sc, X_sc_tr,  y_sc_tr_r,  "Train")
        pred_sc_val_r, mae_sc_val, r2_sc_val, _   = regression_metrics(reg_sc, X_sc_val, y_sc_val_r, "Val")
        pred_sc_te_r,  mae_sc_te,  r2_sc_te,  _   = regression_metrics(reg_sc, X_sc_te,  y_sc_te_r,  "TEST")

        # Overfitting check
        log.info("\n--- Overfitting check (train − test MAE gap) ---")
        log.info(f"  Full  : train={mae_tr:.4f}  val={mae_val:.4f}  "
                 f"test={mae_te:.4f}  gap={mae_te - mae_tr:.4f}s")
        log.info(f"  SC    : train={mae_sc_tr:.4f}  val={mae_sc_val:.4f}  "
                 f"test={mae_sc_te:.4f}  gap={mae_sc_te - mae_sc_tr:.4f}s")

        # Abu Dhabi note (SC test n=32)
        log.info(f"\n  NOTE: SC test set = {len(sc_test)} laps (Abu Dhabi SC laps only).")
        log.info(f"  SC test metrics have wider confidence intervals than full test ({len(test)} laps).")

        # ─────────────────────────────────────────────────────
        # CLASSIFICATION VALIDATION
        # ─────────────────────────────────────────────────────
        log.info("\n=== CLASSIFICATION — FULL MODEL ===")
        pred_tr_c,  prob_tr_c,  acc_tr,  f1_tr,  auc_tr,  cm_tr   = classification_metrics(clf, X_tr,  y_tr_c,  "Train")
        pred_val_c, prob_val_c, acc_val, f1_val, auc_val, cm_val   = classification_metrics(clf, X_val, y_val_c, "Val")
        pred_te_c,  prob_te_c,  acc_te,  f1_te,  auc_te,  cm_te    = classification_metrics(clf, X_te,  y_te_c,  "TEST")

        log.info("\n=== CLASSIFICATION — SC MODEL ===")
        pred_sc_tr_c,  prob_sc_tr_c,  acc_sc_tr,  f1_sc_tr,  auc_sc_tr,  cm_sc_tr  = classification_metrics(clf_sc, X_sc_tr,  y_sc_tr_c,  "Train")
        pred_sc_val_c, prob_sc_val_c, acc_sc_val, f1_sc_val, auc_sc_val, cm_sc_val  = classification_metrics(clf_sc, X_sc_val, y_sc_val_c, "Val")
        pred_sc_te_c,  prob_sc_te_c,  acc_sc_te,  f1_sc_te,  auc_sc_te,  cm_sc_te   = classification_metrics(clf_sc, X_sc_te,  y_sc_te_c,  "TEST")

        # ─────────────────────────────────────────────────────
        # FINAL RESULTS TABLE
        # ─────────────────────────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("FINAL HELD-OUT TEST RESULTS — ABU DHABI 2021")
        log.info("=" * 60)
        log.info(f"  Full Regression  : MAE={mae_te:.4f}s  R²={r2_te:.4f}  "
                 f"Bias={bias_te:+.4f}s")
        log.info(f"  SC   Regression  : MAE={mae_sc_te:.4f}s  R²={r2_sc_te:.4f}  "
                 f"(n={len(sc_test)})")
        log.info(f"  Full Classifier  : Acc={acc_te:.4f}  F1={f1_te:.4f}  "
                 f"AUC={auc_te:.4f}")
        log.info(f"  SC   Classifier  : Acc={acc_sc_te:.4f}  F1={f1_sc_te:.4f}  "
                 f"AUC={auc_sc_te:.4f}")

        # ─────────────────────────────────────────────────────
        # PLOTS
        # ─────────────────────────────────────────────────────
        log.info("\nGenerating validation plots...")

        # 1. Pred vs actual — full regression
        plot_pred_vs_actual([
            ("Train", y_tr_r,  pred_tr_r,  mae_tr,  r2_tr),
            ("Val",   y_val_r, pred_val_r, mae_val, r2_val),
            ("TEST",  y_te_r,  pred_te_r,  mae_te,  r2_te),
        ], "LGB Regressor (Full)", "reg_full_pred_vs_actual.png")

        # 2. Pred vs actual — SC regression
        plot_pred_vs_actual([
            ("Train", y_sc_tr_r,  pred_sc_tr_r,  mae_sc_tr,  r2_sc_tr),
            ("Val",   y_sc_val_r, pred_sc_val_r, mae_sc_val, r2_sc_val),
            ("TEST",  y_sc_te_r,  pred_sc_te_r,  mae_sc_te,  r2_sc_te),
        ], "XGB Regressor (SC)", "reg_sc_pred_vs_actual.png")

        # 3. Residuals by race — train (full model)
        plot_residuals_by_race(train, pred_tr_r,
                               "Train", "LGB Regressor (Full)",
                               "reg_full_residuals_train.png")

        # 4. ROC + PR — full classifier, val and test
        plot_roc_pr(y_val_c, prob_val_c, "Val",  "LGB Classifier (Full)",
                    "clf_full_roc_pr_val.png")
        plot_roc_pr(y_te_c,  prob_te_c,  "TEST", "LGB Classifier (Full)",
                    "clf_full_roc_pr_test.png")

        # 5. ROC + PR — SC classifier, val and test
        plot_roc_pr(y_sc_val_c, prob_sc_val_c, "Val",  "XGB Classifier (SC)",
                    "clf_sc_roc_pr_val.png")
        plot_roc_pr(y_sc_te_c,  prob_sc_te_c,  "TEST", "XGB Classifier (SC)",
                    "clf_sc_roc_pr_test.png")

        # 6. Confusion matrices — test set
        plot_confusion_matrix(cm_te,    "TEST", "LGB Classifier (Full)",
                              "clf_full_cm_test.png")
        plot_confusion_matrix(cm_sc_te, "TEST", "XGB Classifier (SC)",
                              "clf_sc_cm_test.png")

        # 7. Feature importance — all 4 models
        plot_feature_importance(reg,    fc,    "LGB Regressor (Full)",
                                "fi_reg_full.png")
        plot_feature_importance(clf,    fc,    "LGB Classifier (Full)",
                                "fi_clf_full.png")
        plot_feature_importance(reg_sc, fc_sc, "XGB Regressor (SC)",
                                "fi_reg_sc.png")
        plot_feature_importance(clf_sc, fc_sc, "XGB Classifier (SC)",
                                "fi_clf_sc.png")

        # 8. MAE summary bar chart
        plot_metric_summary([
            ("Train", mae_tr,    mae_sc_tr),
            ("Val",   mae_val,   mae_sc_val),
            ("TEST",  mae_te,    mae_sc_te),
        ], "MAE (s)", "Regression MAE — Train / Val / Test",
           "summary_mae.png")

        # 9. AUC summary bar chart
        plot_metric_summary([
            ("Train", auc_tr,    auc_sc_tr),
            ("Val",   auc_val,   auc_sc_val),
            ("TEST",  auc_te,    auc_sc_te),
        ], "ROC-AUC", "Classification AUC — Train / Val / Test",
           "summary_auc.png")

        # 10. Abu Dhabi lap-by-lap trace
        plot_abu_dhabi_trace(test_raw, pred_te_r, pred_sc_te_r,
                            sc_test_raw, "abu_dhabi_trace.png")

        log.info(f"\nAll plots saved to: {PLOTS_DIR}")
        log.info("=" * 60)
        log.info("Validation complete.")
        log.info("=" * 60)

        return {
            "reg_full" : {"mae_te": mae_te,    "r2_te": r2_te,    "bias_te": bias_te},
            "reg_sc"   : {"mae_te": mae_sc_te, "r2_te": r2_sc_te},
            "clf_full" : {"acc_te": acc_te,    "f1_te": f1_te,    "auc_te": auc_te},
            "clf_sc"   : {"acc_te": acc_sc_te, "f1_te": f1_sc_te, "auc_te": auc_sc_te},
        }

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    results = run_validation()

    log.info("\n" + "=" * 60)
    log.info("HELD-OUT TEST SUMMARY — ABU DHABI 2021")
    log.info("=" * 60)
    log.info(f"\nRegression (Full)  MAE : {results['reg_full']['mae_te']:.4f}s")
    log.info(f"Regression (Full)  R²  : {results['reg_full']['r2_te']:.4f}")
    log.info(f"Regression (Full)  Bias: {results['reg_full']['bias_te']:+.4f}s")
    log.info(f"\nRegression (SC)    MAE : {results['reg_sc']['mae_te']:.4f}s")
    log.info(f"Regression (SC)    R²  : {results['reg_sc']['r2_te']:.4f}")
    log.info(f"\nClassifier (Full)  Acc : {results['clf_full']['acc_te']:.4f}")
    log.info(f"Classifier (Full)  F1  : {results['clf_full']['f1_te']:.4f}")
    log.info(f"Classifier (Full)  AUC : {results['clf_full']['auc_te']:.4f}")
    log.info(f"\nClassifier (SC)    Acc : {results['clf_sc']['acc_te']:.4f}")
    log.info(f"Classifier (SC)    F1  : {results['clf_sc']['f1_te']:.4f}")
    log.info(f"Classifier (SC)    AUC : {results['clf_sc']['auc_te']:.4f}")
'''




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def run_validation():
    try:
        log.info("=" * 60)
        log.info("Starting model validation — Phase 2")
        log.info("=" * 60)

        (train, val, test,
         sc_train, sc_val, sc_test,
         reg, clf, reg_sc, clf_sc) = load_all()
        
        # Load unscaled features for raw LapNumber (trace plot)
        features_raw    = pd.read_csv(FEATURES_PATH)
        features_sc_raw = pd.read_csv(FEATURES_SAME_COMPOUND_PATH)
        test_raw    = features_raw[features_raw["Race"] == "AbuDhabi"].copy()
        sc_test_raw = features_sc_raw[features_sc_raw["Race"] == "AbuDhabi"].copy()

        fc      = get_feature_cols(train)
        fc_sc   = get_feature_cols(sc_train)

        # ── Prepare arrays ────────────────────────────────────
        X_tr,  y_tr_r,  y_tr_c  = train[fc].values,    train[TARGET_REGRESSION].values,    train[TARGET_CLASSIFICATION].values
        X_val, y_val_r, y_val_c = val[fc].values,       val[TARGET_REGRESSION].values,       val[TARGET_CLASSIFICATION].values
        X_te,  y_te_r,  y_te_c  = test[fc].values,      test[TARGET_REGRESSION].values,      test[TARGET_CLASSIFICATION].values

        X_sc_tr,  y_sc_tr_r,  y_sc_tr_c  = sc_train[fc_sc].values, sc_train[TARGET_REGRESSION].values, sc_train[TARGET_CLASSIFICATION].values
        X_sc_val, y_sc_val_r, y_sc_val_c = sc_val[fc_sc].values,   sc_val[TARGET_REGRESSION].values,   sc_val[TARGET_CLASSIFICATION].values
        X_sc_te,  y_sc_te_r,  y_sc_te_c  = sc_test[fc_sc].values,  sc_test[TARGET_REGRESSION].values,  sc_test[TARGET_CLASSIFICATION].values

        # ─────────────────────────────────────────────────────
        # REGRESSION VALIDATION
        # ─────────────────────────────────────────────────────
        log.info("\n=== REGRESSION — FULL MODEL ===")
        pred_tr_r,  mae_tr,  r2_tr,  bias_tr  = regression_metrics(reg, X_tr,  y_tr_r,  "Train")
        pred_val_r, mae_val, r2_val, bias_val  = regression_metrics(reg, X_val, y_val_r, "Val")
        pred_te_r,  mae_te,  r2_te,  bias_te   = regression_metrics(reg, X_te,  y_te_r,  "TEST")

        log.info("\n=== REGRESSION — SC MODEL ===")
        pred_sc_tr_r,  mae_sc_tr,  r2_sc_tr,  _  = regression_metrics(reg_sc, X_sc_tr,  y_sc_tr_r,  "Train")
        pred_sc_val_r, mae_sc_val, r2_sc_val, _   = regression_metrics(reg_sc, X_sc_val, y_sc_val_r, "Val")
        pred_sc_te_r,  mae_sc_te,  r2_sc_te,  _   = regression_metrics(reg_sc, X_sc_te,  y_sc_te_r,  "TEST")

        # Overfitting check
        log.info("\n--- Overfitting check (train − test MAE gap) ---")
        log.info(f"  Full  : train={mae_tr:.4f}  val={mae_val:.4f}  "
                 f"test={mae_te:.4f}  gap={mae_te - mae_tr:.4f}s")
        log.info(f"  SC    : train={mae_sc_tr:.4f}  val={mae_sc_val:.4f}  "
                 f"test={mae_sc_te:.4f}  gap={mae_sc_te - mae_sc_tr:.4f}s")

        # Abu Dhabi note (SC test n=32)
        log.info(f"\n  NOTE: SC test set = {len(sc_test)} laps (Abu Dhabi SC laps only).")
        log.info(f"  SC test metrics have wider confidence intervals than full test ({len(test)} laps).")

        # ─────────────────────────────────────────────────────
        # CLASSIFICATION VALIDATION
        # ─────────────────────────────────────────────────────
        log.info("\n=== CLASSIFICATION — FULL MODEL ===")
        pred_tr_c,  prob_tr_c,  acc_tr,  f1_tr,  auc_tr,  cm_tr   = classification_metrics(clf, X_tr,  y_tr_c,  "Train")
        pred_val_c, prob_val_c, acc_val, f1_val, auc_val, cm_val   = classification_metrics(clf, X_val, y_val_c, "Val")
        pred_te_c,  prob_te_c,  acc_te,  f1_te,  auc_te,  cm_te    = classification_metrics(clf, X_te,  y_te_c,  "TEST")

        log.info("\n=== CLASSIFICATION — SC MODEL ===")
        pred_sc_tr_c,  prob_sc_tr_c,  acc_sc_tr,  f1_sc_tr,  auc_sc_tr,  cm_sc_tr  = classification_metrics(clf_sc, X_sc_tr,  y_sc_tr_c,  "Train")
        pred_sc_val_c, prob_sc_val_c, acc_sc_val, f1_sc_val, auc_sc_val, cm_sc_val  = classification_metrics(clf_sc, X_sc_val, y_sc_val_c, "Val")
        pred_sc_te_c,  prob_sc_te_c,  acc_sc_te,  f1_sc_te,  auc_sc_te,  cm_sc_te   = classification_metrics(clf_sc, X_sc_te,  y_sc_te_c,  "TEST")

        # ─────────────────────────────────────────────────────
        # FINAL RESULTS TABLE
        # ─────────────────────────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("FINAL HELD-OUT TEST RESULTS — ABU DHABI 2021")
        log.info("=" * 60)
        log.info(f"  Full Regression  : MAE={mae_te:.4f}s  R²={r2_te:.4f}  "
                 f"Bias={bias_te:+.4f}s")
        log.info(f"  SC   Regression  : MAE={mae_sc_te:.4f}s  R²={r2_sc_te:.4f}  "
                 f"(n={len(sc_test)})")
        log.info(f"  Full Classifier  : Acc={acc_te:.4f}  F1={f1_te:.4f}  "
                 f"AUC={auc_te:.4f}")
        log.info(f"  SC   Classifier  : Acc={acc_sc_te:.4f}  F1={f1_sc_te:.4f}  "
                 f"AUC={auc_sc_te:.4f}")

        # ─────────────────────────────────────────────────────
        # PLOTS
        # ─────────────────────────────────────────────────────
        log.info("\nGenerating validation plots...")

        # 10. Abu Dhabi lap-by-lap trace
        plot_abu_dhabi_trace(test_raw, pred_te_r, pred_sc_te_r,
                            sc_test_raw, "abu_dhabi_trace.png")

        log.info(f"\nAll plots saved to: {PLOTS_DIR}")
        log.info("=" * 60)
        log.info("Validation complete.")
        log.info("=" * 60)

        return {
            "reg_full" : {"mae_te": mae_te,    "r2_te": r2_te,    "bias_te": bias_te},
            "reg_sc"   : {"mae_te": mae_sc_te, "r2_te": r2_sc_te},
            "clf_full" : {"acc_te": acc_te,    "f1_te": f1_te,    "auc_te": auc_te},
            "clf_sc"   : {"acc_te": acc_sc_te, "f1_te": f1_sc_te, "auc_te": auc_sc_te},
        }

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    results = run_validation()

    log.info("\n" + "=" * 60)
    log.info("HELD-OUT TEST SUMMARY — ABU DHABI 2021")
    log.info("=" * 60)
    log.info(f"\nRegression (Full)  MAE : {results['reg_full']['mae_te']:.4f}s")
    log.info(f"Regression (Full)  R²  : {results['reg_full']['r2_te']:.4f}")
    log.info(f"Regression (Full)  Bias: {results['reg_full']['bias_te']:+.4f}s")
    log.info(f"\nRegression (SC)    MAE : {results['reg_sc']['mae_te']:.4f}s")
    log.info(f"Regression (SC)    R²  : {results['reg_sc']['r2_te']:.4f}")
    log.info(f"\nClassifier (Full)  Acc : {results['clf_full']['acc_te']:.4f}")
    log.info(f"Classifier (Full)  F1  : {results['clf_full']['f1_te']:.4f}")
    log.info(f"Classifier (Full)  AUC : {results['clf_full']['auc_te']:.4f}")
    log.info(f"\nClassifier (SC)    Acc : {results['clf_sc']['acc_te']:.4f}")
    log.info(f"Classifier (SC)    F1  : {results['clf_sc']['f1_te']:.4f}")
    log.info(f"Classifier (SC)    AUC : {results['clf_sc']['auc_te']:.4f}")
