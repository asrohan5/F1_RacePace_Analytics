import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, roc_auc_score
)

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH,
    TRAIN_SC_PATH, VAL_SC_PATH, TEST_SC_PATH,
    FEATURES_PATH, FEATURES_SAME_COMPOUND_PATH,
    REGRESSOR_PATH, CLASSIFIER_PATH,
    REGRESSOR_SC_PATH, CLASSIFIER_SC_PATH,
    ARTIFACTS_DIR, CV_SCORES_PATH,
    TARGET_REGRESSION, TARGET_CLASSIFICATION,
    VER_UPGRADE_ROUNDS, HAM_UPGRADE_ROUNDS,
    RACES,
)

DIAG_DIR = os.path.join(ARTIFACTS_DIR, "diagnostic_plots")
os.makedirs(DIAG_DIR, exist_ok=True)

ID_COLS = ["Race", "RoundNumber", "LapNumber",
           TARGET_REGRESSION, TARGET_CLASSIFICATION]

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
GREEN = "#44bb77"


def get_feature_cols(df):
    return [c for c in df.columns if c not in ID_COLS]


def save_fig(filename):
    path = os.path.join(DIAG_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {filename}")


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


def is_voting_model(model):
    actual = unwrap_model(model)
    return type(actual).__name__ in ("VotingClassifier", "VotingRegressor")


def is_tree_model(model):
    actual = model
    if hasattr(model, "calibrated_classifiers_"):
        actual = model.calibrated_classifiers_[0].estimator
    tree_types = (
        "LGBMRegressor", "LGBMClassifier",
        "XGBRegressor", "XGBClassifier",
        "RandomForestRegressor", "RandomForestClassifier",
        "VotingRegressor", "VotingClassifier",
    )
    return type(actual).__name__ in tree_types


def unwrap_model(model):
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def load_all():
    try:
        train = pd.read_csv(TRAIN_PATH)
        val = pd.read_csv(VAL_PATH)
        test = pd.read_csv(TEST_PATH)
        sc_train = pd.read_csv(TRAIN_SC_PATH)
        sc_val = pd.read_csv(VAL_SC_PATH)
        sc_test = pd.read_csv(TEST_SC_PATH)

        feat_raw = pd.read_csv(FEATURES_PATH)
        feat_sc_raw = pd.read_csv(FEATURES_SAME_COMPOUND_PATH)

        with open(REGRESSOR_PATH, "rb") as f: reg = pickle.load(f)
        with open(CLASSIFIER_PATH, "rb") as f: clf = pickle.load(f)
        with open(REGRESSOR_SC_PATH, "rb") as f: reg_sc = pickle.load(f)
        with open(CLASSIFIER_SC_PATH, "rb") as f: clf_sc = pickle.load(f)

        log.info("All data and models loaded.")
        log.info(f"  Full reg: {model_type_label(reg)}")
        log.info(f"  Full clf: {model_type_label(clf)}")
        log.info(f"  SC reg: {model_type_label(reg_sc)}")
        log.info(f"  SC clf: {model_type_label(clf_sc)}")

        return (train, val, test, sc_train, sc_val, sc_test,
                feat_raw, feat_sc_raw,
                reg, clf, reg_sc, clf_sc)
    except Exception as e:
        raise CustomException(e, sys)


def load_cv_scores():
    """
    Load per-fold LORO CV scores written by model_trainer.
    Falls back to empty dict if the file does not exist.
    """
    try:
        with open(CV_SCORES_PATH, "r") as f:
            return json.load(f)
    except Exception:
        log.warning(f"  CV scores file not found at {CV_SCORES_PATH}. Using fallback values.")
        return {}


def plot_shap(model, X, feature_cols, model_label, prefix, max_display=20):
    try:
        log.info(f"  Computing SHAP for {model_label}...")

        actual_model = unwrap_model(model)

        if is_voting_model(model):
            estimators = actual_model.estimators_
            shap_list = []
            for est in estimators:
                try:
                    exp = shap.TreeExplainer(est)
                    sv = exp.shap_values(X)
                    if isinstance(sv, list):
                        sv = sv[1]
                    if sv.ndim == 3:
                        sv = sv[:, :, 1]
                    if sv.ndim == 1:
                        sv = sv.reshape(1, -1)
                    if sv.ndim != 2 or sv.shape[1] != X.shape[1]:
                        log.warning(f"    Sub-estimator SHAP shape unexpected: {sv.shape} -- skipping")
                        continue
                    shap_list.append(sv)
                except Exception as sub_e:
                    log.warning(f"    Sub-estimator SHAP failed: {sub_e}")
            if not shap_list:
                log.warning(f"  SHAP: all sub-estimators failed for {model_label}")
                return
            shap_values = np.mean(shap_list, axis=0)

        elif is_tree_model(model):
            explainer = shap.TreeExplainer(actual_model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)

        else:
            explainer = shap.LinearExplainer(actual_model, X,
                                             feature_perturbation="interventional")
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)

        fig, ax = plt.subplots(figsize=(8, 6))
        mean_abs = np.abs(shap_values).mean(axis=0)
        fi = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=True)
        fi = fi.tail(max_display)
        colors = [VER_COLOR if v >= fi.median() else GREY for v in fi.values]
        fi.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
        ax.set_title(f"SHAP Mean |value| -- {model_label} | top {max_display}")
        ax.set_xlabel("mean(|SHAP value|)")
        ax.grid(True, axis="x")
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        save_fig(f"{prefix}_shap_bar.png")

        fig = plt.figure(figsize=(9, 7))
        fig.patch.set_facecolor("#0f0f0f")
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_cols,
            max_display=max_display,
            show=False,
            plot_type="dot",
            color_bar=True,
        )
        ax = plt.gca()
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#888888", labelsize=8)
        ax.set_title(f"SHAP Beeswarm -- {model_label}", color=NEUTRAL)
        plt.tight_layout()
        save_fig(f"{prefix}_shap_beeswarm.png")

    except Exception as e:
        log.warning(f"  SHAP failed for {model_label}: {e}")


def plot_calibration(clf, clf_sc,
                     X_val, y_val, X_sc_val, y_sc_val,
                     X_te, y_te, X_sc_te, y_sc_te):
    try:
        clf_label = model_type_label(clf)
        clf_sc_label = model_type_label(clf_sc)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("Classifier Calibration Curves", color=NEUTRAL)

        for ax, (model, X_v, y_v, X_t, y_t, label) in zip(axes, [
            (clf, X_val, y_val, X_te, y_te, f"Full ({clf_label})"),
            (clf_sc, X_sc_val, y_sc_val, X_sc_te, y_sc_te, f"SC ({clf_sc_label})"),
        ]):
            for X, y, split_label, color in [
                (X_v, y_v, "Val", HAM_COLOR),
                (X_t, y_t, "TEST", VER_COLOR),
            ]:
                prob = model.predict_proba(X)[:, 1]
                frac_pos, mean_pred = calibration_curve(y, prob, n_bins=8,
                                                        strategy="quantile")
                ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.8,
                        color=color, label=split_label)

            ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1, label="Perfect")
            ax.set_title(label)
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives (VER faster)")
            ax.legend(fontsize=8)
            ax.grid(True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        save_fig("calibration_curves.png")
    except Exception as e:
        raise CustomException(e, sys)


def plot_overfitting_profile(metrics):
    """
    Plots train/val/test MAE and AUC with LORO CV fold variance bands.
    CV values and std are loaded from the JSON artifact written by model_trainer.
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("Overfitting Profile -- Train / Val / Test", color=NEUTRAL)

        ax = axes[0]
        splits = ["Train", "Val", "Test"]
        for key, color, label in [
            ("full_reg", VER_COLOR, f"Full ({metrics['full_reg']['model_label']})"),
            ("sc_reg", HAM_COLOR, f"SC ({metrics['sc_reg']['model_label']})"),
        ]:
            vals = [metrics[key]["train"], metrics[key]["val"], metrics[key]["test"]]
            ax.plot(splits, vals, marker="o", color=color, linewidth=2, label=label)
            cv_m = metrics[key]["cv_mean"]
            cv_s = metrics[key]["cv_std"]
            ax.axhspan(cv_m - cv_s, cv_m + cv_s, alpha=0.12, color=color,
                       label=f"{label} CV +/-1sd")

        ax.set_title("Regression MAE (s)")
        ax.set_ylabel("MAE (s)")
        ax.legend(fontsize=8)
        ax.grid(True)

        ax = axes[1]
        for key, color, label in [
            ("full_clf", VER_COLOR, f"Full ({metrics['full_clf']['model_label']})"),
            ("sc_clf", HAM_COLOR, f"SC ({metrics['sc_clf']['model_label']})"),
        ]:
            vals = [metrics[key]["train"], metrics[key]["val"], metrics[key]["test"]]
            ax.plot(splits, vals, marker="o", color=color, linewidth=2, label=label)
            cv_m = metrics[key]["cv_mean"]
            cv_s = metrics[key]["cv_std"]
            ax.axhspan(cv_m - cv_s, cv_m + cv_s, alpha=0.12, color=color,
                       label=f"{label} CV +/-1sd")

        ax.set_title("Classification AUC")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True)

        plt.tight_layout()
        save_fig("overfitting_profile.png")
    except Exception as e:
        raise CustomException(e, sys)


def plot_bias_by_era(feat_raw, reg, fc_train):
    try:
        from src.config import VAL_RACES, TEST_RACE, EXCLUDE_ROUNDS
        train_raw = feat_raw[
            ~feat_raw["Race"].isin(VAL_RACES + [TEST_RACE]) &
            ~feat_raw["RoundNumber"].isin(EXCLUDE_ROUNDS)
        ].copy()

        def era(rnd):
            if rnd < 7:   return "Pre-upgrade\n(R1-6)"
            if rnd < 10:  return "VER ahead\n(R7-9)"
            if rnd < 13:  return "Near parity\n(R10-12)"
            if rnd < 18:  return "HAM upgrade\n(R13-17)"
            return "HAM power token\n(R18-21)"

        train_raw["era"] = train_raw["RoundNumber"].apply(era)

        train_scaled = pd.read_csv(TRAIN_PATH)
        fc = get_feature_cols(train_scaled)
        preds = reg.predict(train_scaled[fc].values)
        actuals = train_scaled[TARGET_REGRESSION].values
        residuals = preds - actuals

        if len(train_raw) != len(train_scaled):
            log.warning("  train_raw and train_scaled row count mismatch -- skipping era bias plot")
            return

        train_raw = train_raw.reset_index(drop=True)
        train_raw["residual"] = residuals
        train_raw["actual"] = actuals
        train_raw["pred"] = preds

        era_order = [
            "Pre-upgrade\n(R1-6)",
            "VER ahead\n(R7-9)",
            "Near parity\n(R10-12)",
            "HAM upgrade\n(R13-17)",
            "HAM power token\n(R18-21)",
        ]
        era_order = [e for e in era_order if e in train_raw["era"].unique()]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        reg_label = model_type_label(reg)
        fig.suptitle(f"Regression Bias by Upgrade Era -- {reg_label} (Train)",
                     color=NEUTRAL)

        ax = axes[0]
        era_bias = train_raw.groupby("era")["residual"].mean().reindex(era_order)
        colors = [VER_COLOR if v >= 0 else HAM_COLOR for v in era_bias.values]
        era_bias.plot(kind="bar", ax=ax, color=colors, edgecolor="none")
        ax.axhline(0, color=ACCENT, linewidth=1.2, linestyle="--")
        ax.set_title("Mean Residual (pred minus actual) by Era")
        ax.set_ylabel("Mean residual (s)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis="y")

        ax = axes[1]
        era_groups = [train_raw[train_raw["era"] == e]["residual"].values
                      for e in era_order]
        bp = ax.boxplot(era_groups, patch_artist=True,
                        medianprops=dict(color=ACCENT, linewidth=1.5),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor=GREY))
        for patch in bp["boxes"]:
            patch.set_facecolor(VER_COLOR)
            patch.set_alpha(0.6)
        ax.axhline(0, color=ACCENT, linewidth=1.2, linestyle="--")
        ax.set_xticklabels(era_order, fontsize=7)
        ax.set_title("Residual Distribution by Era")
        ax.set_ylabel("Residual (pred minus actual, s)")
        ax.grid(True, axis="y")

        plt.tight_layout()
        save_fig("bias_by_era.png")
    except Exception as e:
        raise CustomException(e, sys)


def plot_per_race_accuracy(feat_raw, clf, train, val, test):
    try:
        all_splits = pd.concat([train, val, test], ignore_index=True)
        fc = get_feature_cols(train)

        all_splits["pred_prob"] = clf.predict_proba(all_splits[fc].values)[:, 1]
        all_splits["pred"] = clf.predict(all_splits[fc].values)
        all_splits["correct"] = (
            all_splits["pred"] == all_splits[TARGET_CLASSIFICATION]
        ).astype(int)

        race_map = feat_raw[["Race", "RoundNumber"]].drop_duplicates()
        all_splits = all_splits.merge(race_map, on=["Race", "RoundNumber"], how="left")

        race_acc = (all_splits.groupby("Race")["correct"]
                    .mean()
                    .sort_values(ascending=True))

        val_races = val["Race"].unique().tolist()
        test_races = test["Race"].unique().tolist()

        def split_tag(race):
            if race in test_races: return "TEST"
            if race in val_races:  return "Val"
            return "Train"

        colors = []
        for race in race_acc.index:
            tag = split_tag(race)
            if tag == "TEST":   colors.append(ACCENT)
            elif tag == "Val":  colors.append(HAM_COLOR)
            else:               colors.append(VER_COLOR)

        clf_label = model_type_label(clf)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.barh(race_acc.index, race_acc.values, color=colors, edgecolor="none")
        ax.axvline(0.5, color=GREY, linewidth=1, linestyle="--")
        ax.axvline(race_acc.mean(), color=NEUTRAL, linewidth=1,
                   linestyle=":", label=f"Mean={race_acc.mean():.2f}")

        legend_patches = [
            mpatches.Patch(color=VER_COLOR, label="Train"),
            mpatches.Patch(color=HAM_COLOR, label="Val"),
            mpatches.Patch(color=ACCENT, label="TEST"),
        ]
        ax.legend(handles=legend_patches, fontsize=8)
        ax.set_title(f"Per-Race Classifier Accuracy -- Full {clf_label} Model")
        ax.set_xlabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.grid(True, axis="x")

        plt.tight_layout()
        save_fig("per_race_accuracy.png")
    except Exception as e:
        raise CustomException(e, sys)


def plot_feature_correlation(train, top_n=20):
    try:
        fc = get_feature_cols(train)
        variances = train[fc].var().sort_values(ascending=False)
        top_cols = variances.head(top_n).index.tolist()
        corr = train[top_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, ax=ax, cmap="coolwarm",
                    center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".1f", annot_kws={"size": 6},
                    linewidths=0.3, square=True,
                    cbar_kws={"shrink": 0.7})
        ax.set_title(f"Feature Correlation (top {top_n} by variance) -- Train")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

        plt.tight_layout()
        save_fig("feature_correlation.png")
    except Exception as e:
        raise CustomException(e, sys)


def plot_rolling_confidence(test, clf, feat_raw):
    try:
        fc = get_feature_cols(test)
        probs = clf.predict_proba(test[fc].values)[:, 1]
        actuals = test[TARGET_CLASSIFICATION].values

        test_raw = feat_raw[feat_raw["Race"] == "AbuDhabi"].copy()
        test_raw = test_raw.sort_values("LapNumber").reset_index(drop=True)

        if len(test_raw) != len(test):
            log.warning("  Abu Dhabi raw/scaled length mismatch -- using sequential index")
            laps = np.arange(1, len(test) + 1)
        else:
            laps = np.arange(1, len(test_raw) + 1)

        probs_sorted = probs
        actuals_sorted = actuals

        clf_label = model_type_label(clf)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(
            f"Abu Dhabi 2021 -- Classifier Confidence & Correctness ({clf_label})",
            color=NEUTRAL
        )

        ax1.axhline(0.5, color=GREY, linewidth=0.8, linestyle="--")
        ax1.fill_between(laps, 0.5, probs_sorted,
                         where=probs_sorted >= 0.5,
                         alpha=0.3, color=VER_COLOR, label="P(VER faster)")
        ax1.fill_between(laps, probs_sorted, 0.5,
                         where=probs_sorted < 0.5,
                         alpha=0.3, color=HAM_COLOR, label="P(HAM faster)")
        ax1.plot(laps, probs_sorted, color=NEUTRAL, linewidth=1.2)
        ax1.set_ylabel("P(VER faster)")
        ax1.set_ylim(0, 1)
        ax1.legend(fontsize=8)
        ax1.grid(True)

        correct = actuals_sorted == (probs_sorted >= 0.5).astype(int)
        ax2.scatter(laps[correct], np.ones(correct.sum()),
                    color=GREEN, s=40, zorder=3, label="Correct")
        ax2.scatter(laps[~correct], np.ones((~correct).sum()),
                    color=ACCENT, s=40, marker="x", zorder=3, label="Wrong")
        ax2.set_yticks([])
        ax2.set_xlabel("Lap (race order)")
        ax2.set_title(f"Accuracy={correct.mean():.2%}  "
                      f"AUC={roc_auc_score(actuals_sorted, probs_sorted):.3f}")
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="x")

        plt.tight_layout()
        save_fig("rolling_confidence_abu_dhabi.png")
    except Exception as e:
        raise CustomException(e, sys)


def plot_season_narrative(feat_raw, reg, train, val, test):
    try:
        all_splits = pd.concat([train, val, test], ignore_index=True)
        fc = get_feature_cols(train)
        all_preds = reg.predict(all_splits[fc].values)
        all_splits["pred"] = all_preds

        race_rnd = feat_raw[["Race", "RoundNumber"]].drop_duplicates()
        all_splits = all_splits.merge(race_rnd, on=["Race", "RoundNumber"], how="left")

        race_summary = (all_splits
                        .groupby(["Race", "RoundNumber"])
                        .agg(actual_median=(TARGET_REGRESSION, "median"),
                             pred_median=("pred", "median"),
                             n=("pred", "count"))
                        .reset_index()
                        .sort_values("RoundNumber"))

        x = np.arange(len(race_summary))
        races = race_summary["Race"].values
        rnd = race_summary["RoundNumber"].values

        reg_label = model_type_label(reg)
        fig, ax = plt.subplots(figsize=(16, 6))

        ax.fill_between(x, 0, race_summary["actual_median"].values,
                        where=race_summary["actual_median"].values < 0,
                        alpha=0.2, color=VER_COLOR)
        ax.fill_between(x, 0, race_summary["actual_median"].values,
                        where=race_summary["actual_median"].values >= 0,
                        alpha=0.2, color=HAM_COLOR)

        ax.plot(x, race_summary["actual_median"].values,
                color=NEUTRAL, linewidth=2, marker="o", markersize=5,
                label="Actual median delta")
        ax.plot(x, race_summary["pred_median"].values,
                color=ACCENT, linewidth=1.5, linestyle="--",
                marker="s", markersize=4,
                label=f"Predicted median delta ({reg_label})")

        ax.axhline(0, color=GREY, linewidth=0.8, linestyle=":")

        for r in VER_UPGRADE_ROUNDS:
            idx = np.where(rnd == r)[0]
            if len(idx):
                ax.axvline(idx[0], color=VER_COLOR, linewidth=1.2,
                           linestyle=":", alpha=0.7)
                ax.text(idx[0] + 0.1, ax.get_ylim()[1] * 0.85,
                        f"RB R{r}", color=VER_COLOR, fontsize=7, rotation=90)

        for r in HAM_UPGRADE_ROUNDS:
            idx = np.where(rnd == r)[0]
            if len(idx):
                ax.axvline(idx[0], color=HAM_COLOR, linewidth=1.2,
                           linestyle=":", alpha=0.7)
                ax.text(idx[0] + 0.1, ax.get_ylim()[1] * 0.6,
                        f"Merc R{r}", color=HAM_COLOR, fontsize=7, rotation=90)

        val_races = val["Race"].unique().tolist()
        test_races = test["Race"].unique().tolist()
        for i, race in enumerate(races):
            if race in test_races:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.12, color=ACCENT)
            elif race in val_races:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color=HAM_COLOR)

        ax.set_xticks(x)
        ax.set_xticklabels(races, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Median delta lap time (s)\n(VER minus HAM, negative = VER faster)")
        ax.set_title("2021 Season -- Actual vs Predicted Median Lap Time Delta")

        legend_patches = [
            mpatches.Patch(color=HAM_COLOR, alpha=0.3, label="Val races"),
            mpatches.Patch(color=ACCENT, alpha=0.3, label="Test race (Abu Dhabi)"),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + legend_patches,
                  labels + ["Val races", "Test race"],
                  fontsize=8, loc="upper right")
        ax.grid(True, axis="y")

        plt.tight_layout()
        save_fig("season_narrative.png")
    except Exception as e:
        raise CustomException(e, sys)


def plot_partial_dependence(model, X_train, feature_cols, model_label,
                             prefix, top_features=None):
    try:
        if top_features is None:
            actual_model = unwrap_model(model)

            if is_voting_model(model):
                imps = []
                for est in actual_model.estimators_:
                    if hasattr(est, "feature_importances_"):
                        imps.append(est.feature_importances_)
                    elif hasattr(est, "coef_"):
                        coef = est.coef_
                        if coef.ndim > 1:
                            coef = coef[0]
                        imps.append(np.abs(coef))
                if imps:
                    imp = np.mean(imps, axis=0)
                    top_idx = np.argsort(imp)[::-1][:3]
                else:
                    log.warning(f"  PDP: no importances found for {model_label} -- skipping")
                    return
            elif hasattr(actual_model, "feature_importances_"):
                imp = actual_model.feature_importances_
                top_idx = np.argsort(imp)[::-1][:3]
            elif hasattr(actual_model, "coef_"):
                coef = actual_model.coef_
                if coef.ndim > 1:
                    coef = coef[0]
                top_idx = np.argsort(np.abs(coef))[::-1][:3]
            else:
                log.warning(f"  PDP: cannot determine top features for {model_label} -- skipping")
                return
            top_features = [feature_cols[i] for i in top_idx]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Partial Dependence -- {model_label}", color=NEUTRAL)

        for ax, feat in zip(axes, top_features):
            feat_idx = feature_cols.index(feat)
            grid = np.linspace(X_train[:, feat_idx].min(),
                               X_train[:, feat_idx].max(), 60)

            X_tmp = X_train.copy()
            pdp_vals = []
            for val in grid:
                X_tmp[:, feat_idx] = val
                if hasattr(model, "predict_proba"):
                    pdp_vals.append(model.predict_proba(X_tmp)[:, 1].mean())
                else:
                    pdp_vals.append(model.predict(X_tmp).mean())

            ax.plot(grid, pdp_vals, color=VER_COLOR, linewidth=2)
            ax.axhline(np.mean(pdp_vals), color=GREY, linewidth=0.8,
                       linestyle="--")
            ax.set_title(feat, fontsize=9)
            ax.set_xlabel("Feature value (scaled)")
            ax.set_ylabel("Predicted output (mean)")
            ax.grid(True)

        plt.tight_layout()
        save_fig(f"{prefix}_pdp.png")
    except Exception as e:
        log.warning(f"  PDP failed for {model_label}: {e}")


def run_diagnostics():
    try:
        log.info("Starting model diagnostics")

        (train, val, test,
         sc_train, sc_val, sc_test,
         feat_raw, feat_sc_raw,
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

        # SHAP computed on test set to reflect what drove Abu Dhabi predictions.
        # Training SHAP shows what the model learned; test SHAP shows what it used
        # on the held-out race. Both are produced for comparison.
        log.info("[1/9] SHAP values")
        shap_sample_n = min(200, len(X_tr))
        train_idx = np.random.default_rng(42).integers(0, len(X_tr), size=shap_sample_n)

        plot_shap(reg, X_tr[train_idx], fc,
                  f"{reg_lbl} Regressor (Full) -- Train", "reg_full_train")
        plot_shap(reg, X_te, fc,
                  f"{reg_lbl} Regressor (Full) -- Test", "reg_full_test")
        plot_shap(clf, X_tr[train_idx], fc,
                  f"{clf_lbl} Classifier (Full) -- Train", "clf_full_train")
        plot_shap(clf, X_te, fc,
                  f"{clf_lbl} Classifier (Full) -- Test", "clf_full_test")
        plot_shap(reg_sc, X_sc_tr[:shap_sample_n], fc_sc,
                  f"{reg_sc_lbl} Regressor (SC) -- Train", "reg_sc_train")
        plot_shap(clf_sc, X_sc_tr[:shap_sample_n], fc_sc,
                  f"{clf_sc_lbl} Classifier (SC) -- Train", "clf_sc_train")

        log.info("[2/9] Calibration curves")
        plot_calibration(clf, clf_sc,
                         X_val, y_val_c, X_sc_val, y_sc_val_c,
                         X_te, y_te_c, X_sc_te, y_sc_te_c)

        log.info("[3/9] Overfitting profile")
        cv_scores = load_cv_scores()

        def cv_stats(task_key, model_name):
            """Extract mean and std from the JSON cv_scores for a given task and model."""
            task = cv_scores.get(task_key, {})
            fold_scores = task.get(model_name, [])
            if fold_scores:
                return float(np.mean(fold_scores)), float(np.std(fold_scores))
            return None, None

        full_reg_cv_mean, full_reg_cv_std = cv_stats("full_reg", "ensemble")
        sc_reg_cv_mean, sc_reg_cv_std = cv_stats("sc_reg", "xgb")
        full_clf_cv_mean, full_clf_cv_std = cv_stats("full_clf", "ensemble")
        sc_clf_cv_mean, sc_clf_cv_std = cv_stats("sc_clf", "ensemble")

        # Fall back to trainer log values if JSON did not have the expected keys
        full_reg_cv_mean = full_reg_cv_mean or 0.4731
        full_reg_cv_std = full_reg_cv_std or 0.1544
        sc_reg_cv_mean = sc_reg_cv_mean or 0.4170
        sc_reg_cv_std = sc_reg_cv_std or 0.1272
        full_clf_cv_mean = full_clf_cv_mean or 0.7431
        full_clf_cv_std = full_clf_cv_std or 0.1131
        sc_clf_cv_mean = sc_clf_cv_mean or 0.7463
        sc_clf_cv_std = sc_clf_cv_std or 0.1288

        metrics = {
            "full_reg": {
                "model_label": reg_lbl,
                "train": mean_absolute_error(y_tr_r, reg.predict(X_tr)),
                "val": mean_absolute_error(y_val_r, reg.predict(X_val)),
                "test": mean_absolute_error(y_te_r, reg.predict(X_te)),
                "cv_mean": full_reg_cv_mean,
                "cv_std": full_reg_cv_std,
            },
            "sc_reg": {
                "model_label": reg_sc_lbl,
                "train": mean_absolute_error(y_sc_tr_r, reg_sc.predict(X_sc_tr)),
                "val": mean_absolute_error(y_sc_val_r, reg_sc.predict(X_sc_val)),
                "test": mean_absolute_error(y_sc_te_r, reg_sc.predict(X_sc_te)),
                "cv_mean": sc_reg_cv_mean,
                "cv_std": sc_reg_cv_std,
            },
            "full_clf": {
                "model_label": clf_lbl,
                "train": roc_auc_score(y_tr_c, clf.predict_proba(X_tr)[:, 1]),
                "val": roc_auc_score(y_val_c, clf.predict_proba(X_val)[:, 1]),
                "test": roc_auc_score(y_te_c, clf.predict_proba(X_te)[:, 1]),
                "cv_mean": full_clf_cv_mean,
                "cv_std": full_clf_cv_std,
            },
            "sc_clf": {
                "model_label": clf_sc_lbl,
                "train": roc_auc_score(y_sc_tr_c, clf_sc.predict_proba(X_sc_tr)[:, 1]),
                "val": roc_auc_score(y_sc_val_c, clf_sc.predict_proba(X_sc_val)[:, 1]),
                "test": roc_auc_score(y_sc_te_c, clf_sc.predict_proba(X_sc_te)[:, 1]),
                "cv_mean": sc_clf_cv_mean,
                "cv_std": sc_clf_cv_std,
            },
        }
        plot_overfitting_profile(metrics)

        log.info("[4/9] Bias by upgrade era")
        plot_bias_by_era(feat_raw, reg, fc)

        log.info("[5/9] Per-race classifier accuracy")
        plot_per_race_accuracy(feat_raw, clf, train, val, test)

        log.info("[6/9] Feature correlation heatmap")
        plot_feature_correlation(train)

        log.info("[7/9] Rolling classifier confidence -- Abu Dhabi")
        plot_rolling_confidence(test, clf, feat_raw)

        log.info("[8/9] Season narrative plot")
        plot_season_narrative(feat_raw, reg, train, val, test)

        log.info("[9/9] Partial dependence plots")
        plot_partial_dependence(reg, X_tr, fc,
                                f"{reg_lbl} Regressor (Full)", "reg_full")
        plot_partial_dependence(clf, X_tr, fc,
                                f"{clf_lbl} Classifier (Full)", "clf_full")
        plot_partial_dependence(reg_sc, X_sc_tr, fc_sc,
                                f"{reg_sc_lbl} Regressor (SC)", "reg_sc")
        plot_partial_dependence(clf_sc, X_sc_tr, fc_sc,
                                f"{clf_sc_lbl} Classifier (SC)", "clf_sc")

        log.info(f"All diagnostic plots saved to {DIAG_DIR}")
        log.info("Diagnostics complete.")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_diagnostics()