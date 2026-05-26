import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.logger import logging as log
from src.exception import CustomException
from src.config import (
    LAPS_RAW_PATH, TELEMETRY_RAW_PATH, ARTIFACTS_DIR,
    RACE_METADATA_PATH,
    THROTTLE_OFF_THRESHOLD, FULL_THROTTLE_THRESHOLD,
    MIN_BRAKE_ZONE_LENGTH_M,
    DRIVERS, ALL_DRIVERS, TEAMMATE_PAIRS,
    EXCLUDE_ROUNDS, EXCLUDE_FROM_PAIRING, EXCLUDE_FROM_TEAMMATE,
    LOW_SAMPLE_ROUNDS,
    STINT_LENGTH_MAP,
    TEST_RACE, VAL_RACES,
)



COLORS = {"VER": "#1E3A8A", "HAM": "#15803D",
          "PER": "#7C3AED", "BOT": "#B45309"}


def save_fig(fig, filename):
    path = os.path.join(ARTIFACTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    log.info(f"Plot saved → {path}")


# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────

def load_data():
    try:
        laps = pd.read_csv(LAPS_RAW_PATH)
        meta = pd.read_csv(RACE_METADATA_PATH)
        log.info(f"Laps loaded      : {laps.shape}")
        log.info(f"Metadata loaded  : {meta.shape}")
        return laps, meta
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 1 — DATA QUALITY AT SCALE
# How many usable paired laps per race after exclusions?
# ─────────────────────────────────────────

def eda_data_quality(laps, meta):
    try:
        log.info("=" * 60)
        log.info("SECTION 1 — Data Quality at Scale")
        log.info("=" * 60)

        # Green flag laps only
        green = laps[laps["TrackStatus"] == 1].copy()
        dropped = len(laps) - len(green)
        log.info(f"Green flag filter: {len(laps)} → {len(green)} laps "
                 f"(dropped {dropped} non-green, "
                 f"{100*dropped/len(laps):.1f}%)")

        # Laps per driver per race
        counts = green.groupby(["Race", "Driver"]).size().unstack(fill_value=0)
        log.info(f"\nLaps per driver per race (green flag only):\n{counts.to_string()}")

        # Identify races where VER or HAM has fewer than 10 clean laps
        ver_sparse = counts[counts.get("VER", 0) < 10].index.tolist()
        ham_sparse = counts[counts.get("HAM", 0) < 10].index.tolist()
        log.info(f"\nRaces with VER < 10 clean laps: {ver_sparse}")
        log.info(f"Races with HAM < 10 clean laps: {ham_sparse}")

        # Paired lap count per race (inner join VER + HAM on LapNumber)
        ver_laps = green[green["Driver"] == "VER"][["Race", "LapNumber", "RoundNumber"]]
        ham_laps = green[green["Driver"] == "HAM"][["Race", "LapNumber"]]
        paired   = ver_laps.merge(ham_laps, on=["Race", "LapNumber"], how="inner")

        # Remove excluded pairing rounds
        paired = paired[~paired["RoundNumber"].isin(EXCLUDE_FROM_PAIRING)]

        paired_counts = paired.groupby("Race").size().reset_index(name="PairedLaps")
        log.info(f"\nPaired laps per race (after exclusions):")
        log.info(f"\n{paired_counts.to_string(index=False)}")
        log.info(f"\nTotal usable paired laps: {len(paired)}")
        log.info(f"(Phase 1 had 138 — Phase 2 has {len(paired)})")

        # Plot paired lap counts per race
        fig, ax = plt.subplots(figsize=(14, 5))
        races_sorted = paired_counts.sort_values("PairedLaps", ascending=False)
        colors = ["#DC2626" if r in [TEST_RACE] else
                  "#F59E0B" if r in VAL_RACES else
                  "#1E3A8A"
                  for r in races_sorted["Race"]]
        ax.bar(races_sorted["Race"], races_sorted["PairedLaps"], color=colors)
        ax.set_xlabel("Race")
        ax.set_ylabel("Paired Laps (VER + HAM, same lap number)")
        ax.set_title("Usable Paired Laps per Race\n"
                     "(Blue=Train, Orange=Val, Red=Test)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        # Add legend
        from matplotlib.patches import Patch
        legend = [Patch(color="#1E3A8A", label="Train"),
                  Patch(color="#F59E0B", label="Val"),
                  Patch(color="#DC2626", label="Test")]
        ax.legend(handles=legend)
        save_fig(fig, "p2_01_paired_laps_per_race.png")

        return green, paired

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 2 — LAP TIME DELTA AT SCALE
# Is the target variable well-behaved across all 21 races?
# ─────────────────────────────────────────

def eda_lap_delta(laps, meta):
    try:
        log.info("=" * 60)
        log.info("SECTION 2 — Lap Time Delta Across All Races")
        log.info("=" * 60)

        green = laps[laps["TrackStatus"] == 1].copy()

        ver = green[green["Driver"] == "VER"][["Race", "RoundNumber",
                                               "LapNumber", "LapTimeSec",
                                               "Compound", "TyreLife"]].copy()
        ham = green[green["Driver"] == "HAM"][["Race", "LapNumber",
                                               "LapTimeSec", "Compound",
                                               "TyreLife"]].copy()

        ver.columns = ["Race", "RoundNumber", "LapNumber", "VER_LapTime",
                       "VER_Compound", "VER_TyreLife"]
        ham.columns = ["Race", "LapNumber", "HAM_LapTime",
                       "HAM_Compound", "HAM_TyreLife"]

        delta = ver.merge(ham, on=["Race", "LapNumber"], how="inner")
        delta = delta.merge(meta[["Race", "upgrade_delta"]], on="Race", how="left")

        # Remove excluded races
        excluded_races = laps[laps["RoundNumber"].isin(
            EXCLUDE_FROM_PAIRING)]["Race"].unique()
        delta = delta[~delta["Race"].isin(excluded_races)]

        delta["LapTimeDelta"] = delta["VER_LapTime"] - delta["HAM_LapTime"]
        delta["VER_Faster"]   = (delta["LapTimeDelta"] < 0).astype(int)
        delta["SameCompound"] = (delta["VER_Compound"] == delta["HAM_Compound"]).astype(int)

        log.info(f"\nTotal paired laps: {len(delta)}")
        log.info(f"LapTimeDelta stats:")
        log.info(f"  mean = {delta['LapTimeDelta'].mean():.3f}s")
        log.info(f"  std  = {delta['LapTimeDelta'].std():.3f}s")
        log.info(f"  min  = {delta['LapTimeDelta'].min():.3f}s")
        log.info(f"  max  = {delta['LapTimeDelta'].max():.3f}s")

        # Class balance
        ver_faster = delta["VER_Faster"].sum()
        ham_faster = len(delta) - ver_faster
        log.info(f"\nClass balance:")
        log.info(f"  VER faster: {ver_faster} ({100*ver_faster/len(delta):.1f}%)")
        log.info(f"  HAM faster: {ham_faster} ({100*ham_faster/len(delta):.1f}%)")

        # Same compound rate
        sc_rate = delta["SameCompound"].mean() * 100
        log.info(f"\nSame compound rate: {sc_rate:.1f}%")
        log.info(f"  (Phase 1 was 72.5% — Phase 2 is {sc_rate:.1f}%)")

        # Per-race delta summary
        log.info(f"\nPer-race delta (VER - HAM):")
        race_summary = delta.groupby("Race").agg(
            mean_delta=("LapTimeDelta", "mean"),
            std_delta =("LapTimeDelta", "std"),
            n_laps    =("LapTimeDelta", "count"),
            ver_faster_pct=("VER_Faster", "mean")
        ).round(3)
        log.info(f"\n{race_summary.to_string()}")

        # Upgrade delta effect on lap delta
        log.info(f"\nUpgrade delta effect on lap time delta:")
        upg_effect = delta.groupby("upgrade_delta")["LapTimeDelta"].agg(
            ["mean", "std", "count"]
        ).round(3)
        log.info(f"\n{upg_effect.to_string()}")
        log.info("  upgrade_delta=-1: Mercedes upgrade advantage")
        log.info("  upgrade_delta=0 : Both at same upgrade level")
        log.info("  upgrade_delta=+1: Red Bull upgrade advantage")

        # Plot 1 — delta distribution overall
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Lap Time Delta Distribution — Full Season 2021", fontsize=13)

        axes[0].hist(delta["LapTimeDelta"], bins=50, color="#1E3A8A", alpha=0.7)
        axes[0].axvline(0, color="red", linestyle="--", label="Zero (equal)")
        axes[0].axvline(delta["LapTimeDelta"].mean(), color="orange",
                        linestyle="--", label=f"Mean={delta['LapTimeDelta'].mean():.3f}s")
        axes[0].set_xlabel("VER - HAM Lap Time (s)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Overall Distribution")
        axes[0].legend()

        # Plot 2 — per-race mean delta with upgrade context
        race_order = delta.groupby("Race")["LapTimeDelta"].mean().sort_values().index
        means = [delta[delta["Race"]==r]["LapTimeDelta"].mean() for r in race_order]
        bar_colors = ["#15803D" if m < 0 else "#1E3A8A" for m in means]
        axes[1].barh(list(race_order), means, color=bar_colors, alpha=0.8)
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Mean Lap Delta (s) — Negative = HAM Faster")
        axes[1].set_title("Mean Delta per Race\n(Green=HAM faster, Blue=VER faster)")
        axes[1].grid(True, alpha=0.3, axis="x")

        save_fig(fig, "p2_02_lap_delta_distribution.png")

        return delta

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 3 — DRIVING STYLE ACROSS ALL RACES
# Do Phase 1 patterns hold at full season scale?
# ─────────────────────────────────────────

def eda_style_consistency(laps):
    try:
        log.info("=" * 60)
        log.info("SECTION 3 — Driving Style Consistency Across All Races")
        log.info("=" * 60)

        # We compute lap-level coasting and full throttle from laps data
        # Note: telemetry is not loaded here — we use per-race aggregates
        # from the pre-computed laps file which only has timing data.
        # Style metrics require telemetry and will be computed in
        # data_transformation.py. Here we check what lap-level data tells us.

        green = laps[(laps["TrackStatus"] == 1) &
                     (laps["Driver"].isin(DRIVERS))].copy()

        # Lap time consistency per driver per race
        # A proxy for pace: lower std = more consistent laps
        consistency = green.groupby(["Race", "Driver"])["LapTimeSec"].agg(
            ["mean", "std", "count"]
        ).round(3)

        log.info(f"\nLap time consistency (mean, std, count) per driver per race:")
        log.info(f"\n{consistency.to_string()}")

        # TyreLife distribution per race — are stints comparable?
        tyre_life = green.groupby(["Race", "Driver"])["TyreLife"].agg(
            ["mean", "max"]
        ).round(1)
        log.info(f"\nTyre life per driver per race (mean, max):")
        log.info(f"\n{tyre_life.to_string()}")

        # Compound usage per race
        compound_counts = green.groupby(
            ["Race", "Driver", "Compound"]
        ).size().unstack(fill_value=0)
        log.info(f"\nCompound usage per driver per race:")
        log.info(f"\n{compound_counts.to_string()}")

        # Plot — lap time std per race (pace consistency)
        ver_std = consistency.xs("VER", level="Driver")["std"] if "VER" in \
            consistency.index.get_level_values("Driver") else pd.Series()
        ham_std = consistency.xs("HAM", level="Driver")["std"] if "HAM" in \
            consistency.index.get_level_values("Driver") else pd.Series()

        common_races = ver_std.index.intersection(ham_std.index)
        x = range(len(common_races))

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(list(x), [ver_std[r] for r in common_races],
                "o-", color=COLORS["VER"], label="VER std", alpha=0.8)
        ax.plot(list(x), [ham_std[r] for r in common_races],
                "o-", color=COLORS["HAM"], label="HAM std", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(list(common_races), rotation=45, ha="right")
        ax.set_ylabel("Lap Time Std (s)")
        ax.set_title("Lap Time Consistency per Race — VER vs HAM\n"
                     "Lower = More Consistent Pace")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig(fig, "p2_03_pace_consistency.png")

        return consistency

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 4 — UPGRADE EFFECT VALIDATION
# Does upgrade_delta predict who is faster?
# ─────────────────────────────────────────

def eda_upgrade_effect(delta_df, meta):
    try:
        log.info("=" * 60)
        log.info("SECTION 4 — Upgrade Effect on Lap Time Delta")
        log.info("=" * 60)

        # Per upgrade_delta level: mean delta, % VER faster
        upg = delta_df.groupby("upgrade_delta").agg(
            mean_delta    =("LapTimeDelta", "mean"),
            std_delta     =("LapTimeDelta", "std"),
            ver_faster_pct=("VER_Faster",   "mean"),
            n_laps        =("LapTimeDelta", "count")
        ).round(3)

        log.info(f"\nUpgrade delta → Lap time delta relationship:")
        log.info(f"\n{upg.to_string()}")

        log.info(f"\n  INTERPRETATION:")
        for udelta, row in upg.iterrows():
            if udelta < 0:
                car = "Mercedes upgrade advantage"
            elif udelta > 0:
                car = "Red Bull upgrade advantage"
            else:
                car = "Both at same upgrade level"
            log.info(f"  upgrade_delta={udelta:+d} ({car}): "
                     f"mean delta={row['mean_delta']:.3f}s, "
                     f"VER faster {100*row['ver_faster_pct']:.1f}% of laps "
                     f"(n={int(row['n_laps'])})")

        # Is the upgrade_delta a useful feature?
        # If mean_delta changes monotonically with upgrade_delta, it is useful
        deltas = upg["mean_delta"].values
        is_monotone = all(deltas[i] <= deltas[i+1]
                          for i in range(len(deltas)-1)) or \
                      all(deltas[i] >= deltas[i+1]
                          for i in range(len(deltas)-1))
        log.info(f"\n  Upgrade delta monotonically predicts lap delta: {is_monotone}")
        if is_monotone:
            log.info("  → upgrade_delta is a VALID feature — include in transformation")
        else:
            log.info("  → upgrade_delta is NOT monotone — check for confounds")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Car Upgrade Level Effect on Lap Time Delta", fontsize=13)

        upgrade_levels = upg.index.tolist()
        axes[0].bar([str(u) for u in upgrade_levels],
                    upg["mean_delta"],
                    color=["#15803D" if u < 0 else
                           "#6B7280" if u == 0 else
                           "#1E3A8A" for u in upgrade_levels],
                    alpha=0.8)
        axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Upgrade Delta (VER level - HAM level)")
        axes[0].set_ylabel("Mean Lap Delta (s)")
        axes[0].set_title("Mean Delta by Upgrade Level")
        axes[0].grid(True, alpha=0.3, axis="y")

        axes[1].bar([str(u) for u in upgrade_levels],
                    upg["ver_faster_pct"] * 100,
                    color="#1E3A8A", alpha=0.8)
        axes[1].axhline(50, color="red", linestyle="--",
                        linewidth=1, label="50% (equal)")
        axes[1].set_xlabel("Upgrade Delta (VER level - HAM level)")
        axes[1].set_ylabel("% Laps VER Faster")
        axes[1].set_title("VER Win Rate by Upgrade Level")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        save_fig(fig, "p2_04_upgrade_effect.png")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 5 — COMPOUND MISMATCH AT SCALE
# ─────────────────────────────────────────

def eda_compound_mismatch(delta_df):
    try:
        log.info("=" * 60)
        log.info("SECTION 5 — Compound Mismatch Analysis")
        log.info("=" * 60)

        sc_rate     = delta_df["SameCompound"].mean() * 100
        mismatch_n  = (delta_df["SameCompound"] == 0).sum()
        same_n      = (delta_df["SameCompound"] == 1).sum()

        log.info(f"Same compound laps : {same_n} ({sc_rate:.1f}%)")
        log.info(f"Diff compound laps : {mismatch_n} ({100-sc_rate:.1f}%)")

        # Per-race mismatch rate
        race_sc = delta_df.groupby("Race")["SameCompound"].mean() * 100
        log.info(f"\nSame compound rate per race:")
        log.info(f"\n{race_sc.round(1).to_string()}")

        # Mean delta on same vs different compound
        same_delta = delta_df[delta_df["SameCompound"]==1]["LapTimeDelta"]
        diff_delta = delta_df[delta_df["SameCompound"]==0]["LapTimeDelta"]
        log.info(f"\nMean delta — same compound : {same_delta.mean():.3f}s "
                 f"(std={same_delta.std():.3f}s)")
        log.info(f"Mean delta — diff compound : {diff_delta.mean():.3f}s "
                 f"(std={diff_delta.std():.3f}s)")
        log.info(f"Std ratio (diff/same)      : {diff_delta.std()/same_delta.std():.2f}x "
                 f"more variable on diff compound")

        log.info(f"\n  RECOMMENDATION:")
        if sc_rate >= 60:
            log.info(f"  Same-compound subset ({same_n} laps) is sufficient for SC model.")
        else:
            log.info(f"  Same-compound subset ({same_n} laps) may be too small — check.")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        race_sc_sorted = race_sc.sort_values()
        colors = ["#DC2626" if v < 50 else "#1E3A8A" for v in race_sc_sorted]
        ax.barh(race_sc_sorted.index, race_sc_sorted, color=colors, alpha=0.8)
        ax.axvline(50, color="black", linestyle="--", linewidth=1,
                   label="50% threshold")
        ax.set_xlabel("Same Compound %")
        ax.set_title("Same Compound Rate per Race\n"
                     "(Red = < 50% same compound, high mismatch)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")
        save_fig(fig, "p2_05_compound_mismatch.png")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 6 — TEAMMATE STYLE RESIDUALS AT SCALE
# Are Phase 1 coasting/braking patterns consistent across all races?
# Note: This section uses lap-level data only (no telemetry).
# Telemetry-based style metrics computed in data_transformation.
# Here we check tyre life and lap time as proxies.
# ─────────────────────────────────────────

def eda_teammate_proxy(laps):
    try:
        log.info("=" * 60)
        log.info("SECTION 6 — Teammate Proxy Analysis")
        log.info("=" * 60)
        log.info("Note: Full telemetry-based style metrics computed in transformation.")
        log.info("This section checks lap time proxy for teammate comparison.")

        green = laps[laps["TrackStatus"] == 1].copy()

        # For each race, compare VER vs PER and HAM vs BOT lap time mean
        # If VER consistently beats PER and HAM consistently beats BOT,
        # it tells us these are the faster drivers in their respective cars
        # (which we already know but want to confirm in the data)

        results = []
        for race in green["Race"].unique():
            race_data = green[green["Race"] == race]
            for driver, teammate in TEAMMATE_PAIRS.items():
                drv_laps = race_data[race_data["Driver"] == driver]["LapTimeSec"]
                tmm_laps = race_data[race_data["Driver"] == teammate]["LapTimeSec"]

                if len(drv_laps) < 5 or len(tmm_laps) < 5:
                    continue

                results.append({
                    "Race"          : race,
                    "Driver"        : driver,
                    "Teammate"      : teammate,
                    "Driver_mean"   : drv_laps.mean(),
                    "Teammate_mean" : tmm_laps.mean(),
                    "Delta_vs_tmm"  : drv_laps.mean() - tmm_laps.mean(),
                })

        proxy_df = pd.DataFrame(results)

        log.info(f"\nLap time delta vs teammate (Driver - Teammate):")
        log.info(f"Negative = Driver faster than teammate")
        summary = proxy_df.groupby("Driver")["Delta_vs_tmm"].agg(
            ["mean", "std", "min", "max", "count"]
        ).round(3)
        log.info(f"\n{summary.to_string()}")

        # Count how often each driver is faster than teammate
        proxy_df["Driver_faster"] = (proxy_df["Delta_vs_tmm"] < 0).astype(int)
        win_rate = proxy_df.groupby("Driver")["Driver_faster"].mean() * 100
        log.info(f"\nDriver faster than teammate (% of races):")
        log.info(f"\n{win_rate.round(1).to_string()}")

        # Plot
        fig, ax = plt.subplots(figsize=(14, 5))
        for driver, color in [("VER", COLORS["VER"]), ("HAM", COLORS["HAM"])]:
            subset = proxy_df[proxy_df["Driver"] == driver].sort_values("Race")
            ax.plot(subset["Race"], subset["Delta_vs_tmm"],
                    "o-", color=color, label=f"{driver} vs teammate",
                    alpha=0.8)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Race")
        ax.set_ylabel("Lap Time Delta vs Teammate (s)")
        ax.set_title("Driver Lap Time vs Teammate — Full Season\n"
                     "Negative = Driver faster than teammate")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig(fig, "p2_06_teammate_proxy.png")

        return proxy_df

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 7 — CONFIG RECOMMENDATIONS
# ─────────────────────────────────────────

def eda_config_recommendations(delta_df, laps):
    try:
        log.info("=" * 60)
        log.info("SECTION 7 — Config Recommendations for Transformation")
        log.info("=" * 60)

        # SC subset size
        sc_n = (delta_df["SameCompound"] == 1).sum()
        total_n = len(delta_df)
        log.info(f"\n  [DATASET SIZE]")
        log.info(f"    Total paired laps : {total_n}")
        log.info(f"    SC paired laps    : {sc_n} ({100*sc_n/total_n:.1f}%)")

        # Class balance
        ver_pct = delta_df["VER_Faster"].mean()
        log.info(f"\n  [CLASS BALANCE] VER_Faster = {ver_pct:.2f}")
        if ver_pct < 0.40 or ver_pct > 0.60:
            log.info("    Recommendation: use class_weight='balanced'")
        else:
            log.info("    Recommendation: class balance acceptable")

        # Non-green flag rate
        non_green_pct = (laps["TrackStatus"] != 1).mean() * 100
        log.info(f"\n  [TRACK STATUS] Non-green = {non_green_pct:.1f}%")
        log.info("    Recommendation: filter TrackStatus == 1")

        # Upgrade delta distribution
        upg_dist = delta_df["upgrade_delta"].value_counts().sort_index()
        log.info(f"\n  [UPGRADE DELTA DISTRIBUTION]")
        log.info(f"\n{upg_dist.to_string()}")
        log.info("    Recommendation: include upgrade_delta as feature — "
                 "sufficient variance across levels")

        # Target distribution
        log.info(f"\n  [TARGET DISTRIBUTION]")
        log.info(f"    mean = {delta_df['LapTimeDelta'].mean():.3f}s")
        log.info(f"    std  = {delta_df['LapTimeDelta'].std():.3f}s")
        log.info(f"    skew = {delta_df['LapTimeDelta'].skew():.3f}")
        if abs(delta_df["LapTimeDelta"].skew()) > 1.0:
            log.info("    WARNING: Target is skewed — check for outlier races")
        else:
            log.info("    Target distribution is acceptable")

        log.info("\n" + "=" * 60)
        log.info("Update config.py INITIAL_MODEL_PARAMS after reviewing EDA")
        log.info("Then run data_transformation.py")
        log.info("=" * 60)

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_eda():
    try:
        log.info("=" * 60)
        log.info("Starting EDA — Phase 2 (Full 2021 Season)")
        log.info("=" * 60)

        laps, meta = load_data()

        green, paired = eda_data_quality(laps, meta)
        delta_df      = eda_lap_delta(laps, meta)
        consistency   = eda_style_consistency(laps)
        eda_upgrade_effect(delta_df, meta)
        eda_compound_mismatch(delta_df)
        proxy_df      = eda_teammate_proxy(laps)
        eda_config_recommendations(delta_df, laps)

        log.info("EDA complete. All plots saved to artifacts/")
        return delta_df, proxy_df

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    delta_df, proxy_df = run_eda()