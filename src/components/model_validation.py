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
    TARGET_REGRESSION, TARGET_CLASSIFICATION,
)

PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "validation_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ID_COLS = ["Race", "RoundNumber", "LapNumber",
           TARGET_REGRESSION, TARGET_CLASSIFICATION]

AVG_LAP_TIME_SEC = 90.0

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#cccccc",
    "axes.titlecolor": "#ffffff",
    "xtick.color": "#888888",
    "ytick.color": "#888888",
    "text.color": "#cccccc",
    "grid.color": "#2a2a2a",
    "grid.linewidth": 0.6,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.framealpha": 0.3,
    "legend.edgecolor": "#444444",
})

VER_COLOR = "#3671C6"
HAM_COLOR = "#00D2BE"
NEUTRAL = "#e8e8e8"
ACCENT = "#ff6b35"
GREY = "#555555"


def get_feature_cols(df):
    return [c for c in df.columns if c not in ID_COLS]


def model_type_label(model):
    actual = model
    if hasattr(model, "calibrated_classifiers_"):
        actual = model.calibrated_classifiers_[0].estimator
    mapping = {
        "LGBMRegressor": "LGB",
        "LGBMClassifier": "LGB",
        "XGBRegressor": "XGB",
        "XGBClassifier": "XGB",
        "RandomForestRegressor": "RF",
        "RandomForestClassifier": "RF",
        "LogisticRegression": "Logistic",
        "SVR": "SVR",
        "SVC": "SVC",
        "VotingRegressor": "Ensemble",
        "VotingClassifier": "Ensemble",
    }
    return mapping.get(type(actual).__name__, type(actual).__name__)


def load_all():
    try:
        train = pd.read_csv(TRAIN_PATH)
        val = pd.read_csv(VAL_PATH)
        test = pd.read_csv(TEST_PATH)
        sc_train = pd.read_csv(TRAIN_SC_PATH)
        sc_val = pd.read_csv(VAL_SC_PATH)
        sc_test = pd.read_csv(TEST_SC_PATH)

        with open(REGRESSOR_PATH, "rb") as f: reg = pickle.load(f)
        with open(CLASSIFIER_PATH, "rb") as f: clf = pickle.load(f)
        with open(REGRESSOR_SC_PATH, "rb") as f: reg_sc = pickle.load(f)
        with open(CLASSIFIER_SC_PATH, "rb") as f: clf_sc = pickle.load(f)

        log.info("All splits and models loaded.")
        log.info(f"  Full reg: {model_type_label(reg)}")
        log.info(f"  Full clf: {model_type_label(clf)}")
        log.info(f"  SC reg: {model_type_label(reg_sc)}")
        log.info(f"  SC clf: {model_type_label(clf_sc)}")

        return (train, val, test, sc_train, sc_val, sc_test,
                reg, clf, reg_sc, clf_sc)
    except Exception as e:
        raise CustomException(e, sys)


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000, ci=0.95):
    """
    Bootstrap confidence interval on ROC-AUC.
    With n=44 test laps the point estimate has meaningful uncertainty.
    The CI communicates the plausible range around the reported AUC.
    """
    aucs = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    return lower, upper


def bootstrap_mae_ci(y_true, y_pred, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval on MAE."""
    maes = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        maes.append(mean_absolute_error(y_true[idx], y_pred[idx]))
    lower = np.percentile(maes, (1 - ci) / 2 * 100)
    upper = np.percentile(maes, (1 + ci) / 2 * 100)
    return lower, upper


def regression_metrics(model, X, y, label):
    pred = model.predict(X)
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    bias = np.mean(pred - y)
    dir_acc = np.mean(np.sign(pred) == np.sign(y))
    mae_pct = (mae / AVG_LAP_TIME_SEC) * 100
    ci_lo, ci_hi = bootstrap_mae_ci(y, pred)

    log.info(f"  [{label}] MAE={mae:.4f}s ({mae_pct:.2f}% of lap)  "
             f"R2={r2:.4f}  Bias={bias:+.4f}s  "
             f"DirAcc={dir_acc:.2%}  95%CI=[{ci_lo:.4f},{ci_hi:.4f}]")
    return pred, mae, r2, bias, dir_acc, mae_pct


def classification_metrics(model, X, y, label, threshold=0.5):
    pred_prob = model.predict_proba(X)[:, 1]
    pred = (pred_prob >= threshold).astype(int)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, pred_prob)
    cm = confusion_matrix(y, pred)

    tp = cm[1, 1] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)

    log.info(f"  [{label}] Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
             f"Prec={prec:.4f}  Rec={rec:.4f}  thresh={threshold:.2f}")
    return pred, pred_prob, acc, f1, auc, cm


def find_optimal_threshold(model, X_val, y_val):
    """Find the threshold on the val set that maximises F1."""
    probs = model.predict_proba(X_val)[:, 1]
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.2, 0.8, 0.02):
        preds = (probs >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    log.info(f"  Optimal threshold: {best_thresh:.2f} (val F1={best_f1:.4f})")
    return best_thresh


def save_fig(filename):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {filename}")


def plot_pred_vs_actual(splits_data, model_label, filename):
    try:
        n = len(splits_data)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        fig.suptitle(f"Predicted vs Actual -- {model_label}", fontsize=13,
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
            ax.set_xlabel("Actual delta lap time (s)")
            ax.set_ylabel("Predicted delta lap time (s)")
            ax.set_title(f"{split_label}\nMAE={mae:.3f}s  R2={r2:.3f}")
            ax.legend(fontsize=8)
            ax.grid(True)

        plt.tight_layout()
        save_fig(filename)
    except Exception as e:
        raise CustomException(e, sys)


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
        ax.set_title(f"Residuals by Race -- {model_label} ({split_label})")
        ax.set_xlabel("")
        ax.set_ylabel("Residual (pred minus actual, s)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y")
        plt.tight_layout()
        save_fig(filename)
    except Exception as e:
        raise CustomException(e, sys)


def plot_roc_pr(y_true, y_prob, split_label, model_label, filename):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        baseline_pr = y_true.mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.suptitle(f"ROC & PR Curves -- {model_label} ({split_label})",
                     color=NEUTRAL)

        ax1.plot(fpr, tpr, color=VER_COLOR, linewidth=2,
                 label=f"AUC = {auc_val:.3f}")
        ax1.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(rec, prec, color=HAM_COLOR, linewidth=2)
        ax2.axhline(baseline_pr, color=GREY, linestyle="--", linewidth=1,
                    label=f"Baseline ({baseline_pr:.2f})")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        save_fig(filename)
    except Exception as e:
        raise CustomException(e, sys)


def plot_confusion_matrix(cm, split_label, model_label, filename):
    try:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["HAM faster", "VER faster"],
                    yticklabels=["HAM faster", "VER faster"],
                    linewidths=0.5, ax=ax,
                    annot_kws={"size": 13, "color": NEUTRAL})
        ax.set_title(f"Confusion Matrix -- {model_label} ({split_label})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        save_fig(filename)
    except Exception as e:
        raise CustomException(e, sys)


def plot_feature_importance(model, feature_cols, model_label, filename, top_n=20):
    try:
        actual_model = model
        if hasattr(model, "calibrated_classifiers_"):
            actual_model = model.calibrated_classifiers_[0].estimator

        if type(actual_model).__name__ in ("VotingRegressor", "VotingClassifier"):
            imps = []
            for est in actual_model.estimators_:
                if hasattr(est, "feature_importances_"):
                    imps.append(est.feature_importances_)
            if not imps:
                log.warning(f"  {model_label}: no feature_importances_ in any constituent estimator, skipping")
                return
            importances = np.mean(imps, axis=0)
        elif hasattr(actual_model, "feature_importances_"):
            importances = actual_model.feature_importances_
        else:
            log.warning(f"  {model_label} has no feature_importances_, skipping")
            return

        fi = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
        fi = fi.tail(top_n)

        fig, ax = plt.subplots(figsize=(7, top_n * 0.35 + 1))
        colors = [VER_COLOR if v >= fi.median() else GREY for v in fi.values]
        fi.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
        ax.set_title(f"Feature Importance (gain) -- {model_label} | top {top_n}")
        ax.set_xlabel("Importance")
        ax.grid(True, axis="x")
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        save_fig(filename)
    except Exception as e:
        raise CustomException(e, sys)


def plot_metric_summary(metric_rows, ylabel, title, filename):
    try:
        labels = [r[0] for r in metric_rows]
        full = [r[1] for r in metric_rows]
        sc = [r[2] for r in metric_rows]
        x = np.arange(len(labels))
        w = 0.35

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - w/2, full, w, color=VER_COLOR, label="Full model",
               edgecolor="none")
        ax.bar(x + w/2, sc, w, color=HAM_COLOR, label="SC model",
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
        save_fig(filename)
    except Exception as e:
        raise CustomException(e, sys)


def plot_abu_dhabi_trace(test_df, reg_pred, reg_sc_pred, sc_test_df, filename):
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
        fig.suptitle("Abu Dhabi 2021 -- Predicted vs Actual Lap Time Delta",
                     color=NEUTRAL, fontsize=13)

        full_sorted = test_df.sort_values("LapNumber").reset_index(drop=True)
        sc_sorted = sc_test_df.sort_values("LapNumber").reset_index(drop=True)

        actual_full = full_sorted[TARGET_REGRESSION].values
        actual_sc = sc_sorted[TARGET_REGRESSION].values

        reg_pred_s = reg_pred[full_sorted.index] if len(reg_pred) == len(test_df) else reg_pred
        reg_sc_pred_s = reg_sc_pred[sc_sorted.index] if len(reg_sc_pred) == len(sc_test_df) else reg_sc_pred

        x_full = np.arange(1, len(actual_full) + 1)
        x_sc = np.arange(1, len(actual_sc) + 1)

        ax1.plot(x_full, actual_full, color=NEUTRAL, linewidth=1.5,
                 label="Actual delta", marker="o", markersize=3)
        ax1.plot(x_full, reg_pred_s, color=VER_COLOR, linewidth=1.5,
                 label="Predicted delta (Full)", linestyle="--", marker="s", markersize=3)
        ax1.axhline(0, color=GREY, linewidth=0.8, linestyle=":")
        ax1.set_xticks(x_full[::3])
        ax1.set_xticklabels(x_full[::3])
        ax1.set_ylabel("delta lap time (s)\n(VER minus HAM)")
        ax1.set_title(f"Full model  |  MAE={mean_absolute_error(actual_full, reg_pred_s):.3f}s")
        ax1.legend(fontsize=8)
        ax1.grid(True)

        ax2.plot(x_sc, actual_sc, color=NEUTRAL, linewidth=1.5,
                 label="Actual delta", marker="o", markersize=3)
        ax2.plot(x_sc, reg_sc_pred_s, color=HAM_COLOR, linewidth=1.5,
                 label="Predicted delta (SC)", linestyle="--", marker="s", markersize=3)
        ax2.axhline(0, color=GREY, linewidth=0.8, linestyle=":")
        ax2.set_xticks(x_sc[::3])
        ax2.set_xticklabels(x_sc[::3])
        ax2.set_xlabel("Lap (race order)")
        ax2.set_ylabel("delta lap time (s)\n(VER minus HAM)")
        ax2.set_title(f"SC model  |  MAE={mean_absolute_error(actual_sc, reg_sc_pred_s):.3f}s"
                      f"  (n={len(sc_sorted)} SC laps)")
        ax2.legend(fontsize=8)
        ax2.grid(True)

        plt.tight_layout()
        save_fig(filename)
    except Exception as e:
        raise CustomException(e, sys)


def run_validation():
    try:
        log.info("Starting model validation")

        (train, val, test,
         sc_train, sc_val, sc_test,
         reg, clf, reg_sc, clf_sc) = load_all()

        fc = get_feature_cols(train)
        fc_sc = get_feature_cols(sc_train)

        X_tr = train[fc].values
        y_tr_r = train[TARGET_REGRESSION].values
        y_tr_c = train[TARGET_CLASSIFICATION].values

        X_val = val[fc].values
        y_val_r = val[TARGET_REGRESSION].values
        y_val_c = val[TARGET_CLASSIFICATION].values

        X_te = test[fc].values
        y_te_r = test[TARGET_REGRESSION].values
        y_te_c = test[TARGET_CLASSIFICATION].values

        X_sc_tr = sc_train[fc_sc].values
        y_sc_tr_r = sc_train[TARGET_REGRESSION].values
        y_sc_tr_c = sc_train[TARGET_CLASSIFICATION].values

        X_sc_val = sc_val[fc_sc].values
        y_sc_val_r = sc_val[TARGET_REGRESSION].values
        y_sc_val_c = sc_val[TARGET_CLASSIFICATION].values

        X_sc_te = sc_test[fc_sc].values
        y_sc_te_r = sc_test[TARGET_REGRESSION].values
        y_sc_te_c = sc_test[TARGET_CLASSIFICATION].values

        reg_lbl = model_type_label(reg)
        clf_lbl = model_type_label(clf)
        reg_sc_lbl = model_type_label(reg_sc)
        clf_sc_lbl = model_type_label(clf_sc)

        log.info("REGRESSION -- FULL MODEL")
        pred_tr_r, mae_tr, r2_tr, bias_tr, dir_tr, _ = regression_metrics(reg, X_tr, y_tr_r, "Train")
        pred_val_r, mae_val, r2_val, bias_val, dir_val, _ = regression_metrics(reg, X_val, y_val_r, "Val")
        pred_te_r, mae_te, r2_te, bias_te, dir_te, mae_te_pct = regression_metrics(reg, X_te, y_te_r, "TEST")

        log.info("REGRESSION -- SC MODEL")
        pred_sc_tr_r, mae_sc_tr, r2_sc_tr, _, _, _ = regression_metrics(reg_sc, X_sc_tr, y_sc_tr_r, "Train")
        pred_sc_val_r, mae_sc_val, r2_sc_val, _, _, _ = regression_metrics(reg_sc, X_sc_val, y_sc_val_r, "Val")
        pred_sc_te_r, mae_sc_te, r2_sc_te, _, dir_sc_te, mae_sc_te_pct = regression_metrics(reg_sc, X_sc_te, y_sc_te_r, "TEST")

        log.info("Overfitting check (train to test MAE gap)")
        log.info(f"  Full: train={mae_tr:.4f}  val={mae_val:.4f}  test={mae_te:.4f}  gap={mae_te - mae_tr:.4f}s")
        log.info(f"  SC: train={mae_sc_tr:.4f}  val={mae_sc_val:.4f}  test={mae_sc_te:.4f}  gap={mae_sc_te - mae_sc_tr:.4f}s")
        log.info(f"  SC test set = {len(sc_test)} laps (Abu Dhabi SC laps only).")
        log.info(f"  SC test metrics have wider confidence intervals than full test ({len(test)} laps).")

        # SC regression result interpretation.
        # Abu Dhabi 2021 SC laps include the VSC and SC restart sequence where
        # both drivers were on strategically atypical tyres. Whether the SC model
        # is predictive on this test set depends on the actual metrics rather than
        # a fixed assumption -- the threshold below reflects the boundary between
        # genuine signal and noise given n=32 laps and the known distribution shift.
        log.info("SC regression result")
        log.info(f"  SC test n={len(sc_test)} laps  SC train n={len(sc_train)} laps across {sc_train['Race'].nunique()} races")
        log.info(f"  DirAcc={dir_sc_te:.2%}  R2={r2_sc_te:.4f}")
        if r2_sc_te < 0.05 or dir_sc_te < 0.55:
            log.info("  SC regression is not predictive at Abu Dhabi 2021 on this run.")
            log.info("  DirAcc near chance and R2 near zero indicate the SC laps are out-of-distribution.")
        else:
            log.info("  SC regression shows modest signal on this run.")
            log.info("  Abu Dhabi 2021 SC laps include VSC/SC restart sequence -- interpret with caution given n=32.")

        log.info("Finding optimal classification threshold on val set")
        thresh_full = find_optimal_threshold(clf, X_val, y_val_c)
        thresh_sc = find_optimal_threshold(clf_sc, X_sc_val, y_sc_val_c)

        log.info("CLASSIFICATION -- FULL MODEL")
        pred_tr_c, prob_tr_c, acc_tr, f1_tr, auc_tr, cm_tr = classification_metrics(
            clf, X_tr, y_tr_c, "Train", threshold=thresh_full)
        pred_val_c, prob_val_c, acc_val, f1_val, auc_val, cm_val = classification_metrics(
            clf, X_val, y_val_c, "Val", threshold=thresh_full)
        pred_te_c, prob_te_c, acc_te, f1_te, auc_te, cm_te = classification_metrics(
            clf, X_te, y_te_c, "TEST", threshold=thresh_full)

        auc_ci_lo, auc_ci_hi = bootstrap_auc_ci(y_te_c, prob_te_c)
        log.info(f"  [TEST] AUC 95% CI (bootstrap n=1000): [{auc_ci_lo:.4f}, {auc_ci_hi:.4f}]")

        log.info("CLASSIFICATION -- SC MODEL")
        pred_sc_tr_c, prob_sc_tr_c, acc_sc_tr, f1_sc_tr, auc_sc_tr, cm_sc_tr = classification_metrics(
            clf_sc, X_sc_tr, y_sc_tr_c, "Train", threshold=thresh_sc)
        pred_sc_val_c, prob_sc_val_c, acc_sc_val, f1_sc_val, auc_sc_val, cm_sc_val = classification_metrics(
            clf_sc, X_sc_val, y_sc_val_c, "Val", threshold=thresh_sc)
        pred_sc_te_c, prob_sc_te_c, acc_sc_te, f1_sc_te, auc_sc_te, cm_sc_te = classification_metrics(
            clf_sc, X_sc_te, y_sc_te_c, "TEST", threshold=thresh_sc)

        auc_sc_ci_lo, auc_sc_ci_hi = bootstrap_auc_ci(y_sc_te_c, prob_sc_te_c)
        log.info(f"  [TEST] SC AUC 95% CI (bootstrap n=1000): [{auc_sc_ci_lo:.4f}, {auc_sc_ci_hi:.4f}]")

        log.info("FINAL HELD-OUT TEST RESULTS -- ABU DHABI 2021")
        log.info(f"  Full Regression: MAE={mae_te:.4f}s ({mae_te_pct:.2f}% of lap)  R2={r2_te:.4f}  Bias={bias_te:+.4f}s  DirAcc={dir_te:.2%}")
        log.info(f"  SC Regression: MAE={mae_sc_te:.4f}s ({mae_sc_te_pct:.2f}% of lap)  R2={r2_sc_te:.4f}  DirAcc={dir_sc_te:.2%}  (n={len(sc_test)})")
        log.info(f"  Full Classifier: Acc={acc_te:.4f}  F1={f1_te:.4f}  AUC={auc_te:.4f}  95%CI=[{auc_ci_lo:.4f},{auc_ci_hi:.4f}]  thresh={thresh_full:.2f}")
        log.info(f"  SC Classifier: Acc={acc_sc_te:.4f}  F1={f1_sc_te:.4f}  AUC={auc_sc_te:.4f}  95%CI=[{auc_sc_ci_lo:.4f},{auc_sc_ci_hi:.4f}]  thresh={thresh_sc:.2f}")

        log.info("Generating validation plots...")

        plot_pred_vs_actual([
            ("Train", y_tr_r, pred_tr_r, mae_tr, r2_tr),
            ("Val", y_val_r, pred_val_r, mae_val, r2_val),
            ("TEST", y_te_r, pred_te_r, mae_te, r2_te),
        ], f"{reg_lbl} Regressor (Full)", "reg_full_pred_vs_actual.png")

        plot_pred_vs_actual([
            ("Train", y_sc_tr_r, pred_sc_tr_r, mae_sc_tr, r2_sc_tr),
            ("Val", y_sc_val_r, pred_sc_val_r, mae_sc_val, r2_sc_val),
            ("TEST", y_sc_te_r, pred_sc_te_r, mae_sc_te, r2_sc_te),
        ], f"{reg_sc_lbl} Regressor (SC)", "reg_sc_pred_vs_actual.png")

        plot_residuals_by_race(train, pred_tr_r, "Train",
                               f"{reg_lbl} Regressor (Full)",
                               "reg_full_residuals_train.png")

        plot_roc_pr(y_val_c, prob_val_c, "Val",
                    f"{clf_lbl} Classifier (Full)", "clf_full_roc_pr_val.png")
        plot_roc_pr(y_te_c, prob_te_c, "TEST",
                    f"{clf_lbl} Classifier (Full)", "clf_full_roc_pr_test.png")

        plot_roc_pr(y_sc_val_c, prob_sc_val_c, "Val",
                    f"{clf_sc_lbl} Classifier (SC)", "clf_sc_roc_pr_val.png")
        plot_roc_pr(y_sc_te_c, prob_sc_te_c, "TEST",
                    f"{clf_sc_lbl} Classifier (SC)", "clf_sc_roc_pr_test.png")

        plot_confusion_matrix(cm_te, "TEST",
                              f"{clf_lbl} Classifier (Full)", "clf_full_cm_test.png")
        plot_confusion_matrix(cm_sc_te, "TEST",
                              f"{clf_sc_lbl} Classifier (SC)", "clf_sc_cm_test.png")

        plot_feature_importance(reg, fc,
                                f"{reg_lbl} Regressor (Full)", "fi_reg_full.png")
        plot_feature_importance(clf, fc,
                                f"{clf_lbl} Classifier (Full)", "fi_clf_full.png")
        plot_feature_importance(reg_sc, fc_sc,
                                f"{reg_sc_lbl} Regressor (SC)", "fi_reg_sc.png")
        plot_feature_importance(clf_sc, fc_sc,
                                f"{clf_sc_lbl} Classifier (SC)", "fi_clf_sc.png")

        plot_metric_summary([
            ("Train", mae_tr, mae_sc_tr),
            ("Val", mae_val, mae_sc_val),
            ("TEST", mae_te, mae_sc_te),
        ], "MAE (s)", "Regression MAE -- Train / Val / Test", "summary_mae.png")

        plot_metric_summary([
            ("Train", auc_tr, auc_sc_tr),
            ("Val", auc_val, auc_sc_val),
            ("TEST", auc_te, auc_sc_te),
        ], "ROC-AUC", "Classification AUC -- Train / Val / Test", "summary_auc.png")

        plot_abu_dhabi_trace(test, pred_te_r, pred_sc_te_r,
                             sc_test, "abu_dhabi_trace.png")

        log.info(f"All plots saved to {PLOTS_DIR}")
        log.info("Validation complete.")

        return {
            "reg_full": {"mae_te": mae_te, "r2_te": r2_te, "bias_te": bias_te,
                         "dir_te": dir_te, "mae_te_pct": mae_te_pct},
            "reg_sc": {"mae_te": mae_sc_te, "r2_te": r2_sc_te,
                       "dir_te": dir_sc_te, "mae_te_pct": mae_sc_te_pct},
            "clf_full": {"acc_te": acc_te, "f1_te": f1_te, "auc_te": auc_te,
                         "auc_ci": (auc_ci_lo, auc_ci_hi), "thresh": thresh_full},
            "clf_sc": {"acc_te": acc_sc_te, "f1_te": f1_sc_te, "auc_te": auc_sc_te,
                       "auc_ci": (auc_sc_ci_lo, auc_sc_ci_hi), "thresh": thresh_sc},
        }

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    results = run_validation()

    log.info("HELD-OUT TEST SUMMARY -- ABU DHABI 2021")

    r = results["reg_full"]
    log.info(f"Regression (Full)  MAE: {r['mae_te']:.4f}s ({r['mae_te_pct']:.2f}% of lap time)")
    log.info(f"Regression (Full)  R2: {r['r2_te']:.4f}")
    log.info(f"Regression (Full)  Bias: {r['bias_te']:+.4f}s")
    log.info(f"Regression (Full)  DirAcc: {r['dir_te']:.2%}")

    r = results["reg_sc"]
    log.info(f"Regression (SC)  MAE: {r['mae_te']:.4f}s ({r['mae_te_pct']:.2f}% of lap time)")
    log.info(f"Regression (SC)  R2: {r['r2_te']:.4f}")
    log.info(f"Regression (SC)  DirAcc: {r['dir_te']:.2%}")

    r = results["clf_full"]
    log.info(f"Classifier (Full)  Acc: {r['acc_te']:.4f} (thresh={r['thresh']:.2f})")
    log.info(f"Classifier (Full)  F1: {r['f1_te']:.4f}")
    log.info(f"Classifier (Full)  AUC: {r['auc_te']:.4f}  95%CI=[{r['auc_ci'][0]:.4f},{r['auc_ci'][1]:.4f}]")

    r = results["clf_sc"]
    log.info(f"Classifier (SC)  Acc: {r['acc_te']:.4f} (thresh={r['thresh']:.2f})")
    log.info(f"Classifier (SC)  F1: {r['f1_te']:.4f}")
    log.info(f"Classifier (SC)  AUC: {r['auc_te']:.4f}  95%CI=[{r['auc_ci'][0]:.4f},{r['auc_ci'][1]:.4f}]")