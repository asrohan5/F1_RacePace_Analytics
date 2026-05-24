import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.logger import logging as log
from src.exception import CustomException

from src.config import (
    LAPS_RAW_PATH, TELEMETRY_RAW_PATH, ARTIFACTS_DIR,
    THROTTLE_OFF_THRESHOLD, FULL_THROTTLE_THRESHOLD,
    MIN_BRAKE_ZONE_LENGTH_M, DRIVERS,
    CACHE_DIR, SEASON, RACES, SESSION_TYPE
)


# ─────────────────────────────────────────
# HELPER — save figure
# ─────────────────────────────────────────

def save_fig(fig, filename):
    path = os.path.join(ARTIFACTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    log.info(f"Plot saved → {path}")


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────

def load_data():
    try:
        laps = pd.read_csv(LAPS_RAW_PATH)
        tel  = pd.read_parquet(TELEMETRY_RAW_PATH)
        log.info(f"Laps loaded      : {laps.shape}")
        log.info(f"Telemetry loaded : {tel.shape}")
        return laps, tel
    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 1 — LAPS: distributions, compound, degradation, track status
# ─────────────────────────────────────────

def eda_laps(laps):
    try:
        log.info("=" * 60)
        log.info("SECTION 1 — Laps EDA")
        log.info("=" * 60)

        races   = laps["Race"].unique()
        drivers = laps["Driver"].unique()
        colors  = {"VER": "#1E3A8A", "HAM": "#15803D"}

        # ── 1a: Lap time distribution per driver per race ──
        fig, axes = plt.subplots(1, len(races), figsize=(15, 4), sharey=False)
        fig.suptitle("Lap Time Distribution per Driver per Race", fontsize=13)

        for i, race in enumerate(races):
            ax = axes[i]
            for driver in drivers:
                data = laps[(laps["Race"] == race) & (laps["Driver"] == driver)]["LapTimeSec"]
                ax.hist(data, bins=15, alpha=0.6, label=driver, color=colors[driver])
                log.info(f"  {race} | {driver} | mean={data.mean():.3f}s  std={data.std():.3f}s  "
                         f"min={data.min():.3f}s  max={data.max():.3f}s  count={len(data)}")
            ax.set_title(race)
            ax.set_xlabel("Lap Time (s)")
            ax.set_ylabel("Count")
            ax.legend()

        save_fig(fig, "01a_laptime_distribution.png")

        # ── 1b: Tyre compound usage per driver per race ──
        fig, axes = plt.subplots(1, len(races), figsize=(15, 4))
        fig.suptitle("Tyre Compound Usage per Driver per Race", fontsize=13)

        for i, race in enumerate(races):
            ax = axes[i]
            race_laps = laps[laps["Race"] == race]
            compound_counts = race_laps.groupby(["Driver", "Compound"]).size().unstack(fill_value=0)
            compound_counts.plot(kind="bar", ax=ax, legend=(i == 0))
            ax.set_title(race)
            ax.set_xlabel("")
            ax.set_ylabel("Lap Count")
            ax.tick_params(axis="x", rotation=0)
            log.info(f"  {race} compound counts:\n{compound_counts.to_string()}")

        save_fig(fig, "01b_compound_usage.png")

        # ── 1c: Tyre life vs lap time (degradation) per driver per race ──
        fig, axes = plt.subplots(1, len(races), figsize=(15, 4))
        fig.suptitle("Tyre Life vs Lap Time (Degradation)", fontsize=13)

        for i, race in enumerate(races):
            ax = axes[i]
            for driver in drivers:
                data = laps[(laps["Race"] == race) & (laps["Driver"] == driver)]
                ax.scatter(data["TyreLife"], data["LapTimeSec"],
                           alpha=0.6, label=driver, color=colors[driver], s=20)
            ax.set_title(race)
            ax.set_xlabel("Tyre Life (laps)")
            ax.set_ylabel("Lap Time (s)")
            ax.legend()

        save_fig(fig, "01c_tyre_degradation.png")

        # ── 1d: Track status value counts ──
        log.info("\n  Track Status value counts (1=green, 2=yellow, 4=SC, 5=red, 6=VSC):")
        status_counts = laps.groupby(["Race", "TrackStatus"]).size().reset_index(name="count")
        log.info(f"\n{status_counts.to_string(index=False)}")

        non_green = laps[laps["TrackStatus"] != 1]
        log.info(f"\n  Non-green flag laps: {len(non_green)} out of {len(laps)} total "
                 f"({100 * len(non_green) / len(laps):.1f}%)")

        fig, ax = plt.subplots(figsize=(8, 4))
        status_counts_pivot = status_counts.pivot(index="Race", columns="TrackStatus", values="count").fillna(0)
        status_counts_pivot.plot(kind="bar", ax=ax)
        ax.set_title("Track Status Distribution per Race")
        ax.set_xlabel("")
        ax.set_ylabel("Lap Count")
        ax.tick_params(axis="x", rotation=0)
        save_fig(fig, "01d_track_status.png")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 2 — TELEMETRY: Distance alignment + samples per lap
# ─────────────────────────────────────────

def eda_telemetry_alignment(tel):
    try:
        log.info("=" * 60)
        log.info("SECTION 2 — Telemetry Distance Alignment")
        log.info("=" * 60)

        # Samples per lap per driver per race
        samples_per_lap = (
            tel.groupby(["Race", "Driver", "LapNumber"])
            .size()
            .reset_index(name="SampleCount")
        )

        for race in tel["Race"].unique():
            for driver in DRIVERS:
                subset = samples_per_lap[
                    (samples_per_lap["Race"] == race) &
                    (samples_per_lap["Driver"] == driver)
                ]["SampleCount"]
                log.info(f"  {race} | {driver} | samples/lap — "
                         f"mean={subset.mean():.1f}  min={subset.min()}  max={subset.max()}")

        # Flag laps with very few samples (less than 50% of median)
        median_samples = samples_per_lap["SampleCount"].median()
        thin_laps = samples_per_lap[samples_per_lap["SampleCount"] < median_samples * 0.5]
        log.info(f"\n  Median samples/lap: {median_samples:.0f}")
        log.info(f"  Laps with < 50% of median samples: {len(thin_laps)}")
        if len(thin_laps) > 0:
            log.info(f"\n{thin_laps.to_string(index=False)}")

        # Distance range per lap
        dist_stats = (
            tel.groupby(["Race", "Driver", "LapNumber"])["Distance"]
            .agg(["min", "max"])
            .reset_index()
        )
        dist_stats["range"] = dist_stats["max"] - dist_stats["min"]

        for race in tel["Race"].unique():
            subset = dist_stats[dist_stats["Race"] == race]["range"]
            log.info(f"\n  {race} | Distance range/lap — "
                     f"mean={subset.mean():.1f}m  min={subset.min():.1f}m  max={subset.max():.1f}m")

        # Plot samples per lap
        fig, axes = plt.subplots(1, len(tel["Race"].unique()), figsize=(15, 4))
        fig.suptitle("Telemetry Samples per Lap per Driver", fontsize=13)
        colors = {"VER": "#1E3A8A", "HAM": "#15803D"}

        for i, race in enumerate(tel["Race"].unique()):
            ax = axes[i]
            for driver in DRIVERS:
                subset = samples_per_lap[
                    (samples_per_lap["Race"] == race) &
                    (samples_per_lap["Driver"] == driver)
                ]
                ax.plot(subset["LapNumber"], subset["SampleCount"],
                        label=driver, color=colors[driver], alpha=0.8)
            ax.set_title(race)
            ax.set_xlabel("Lap Number")
            ax.set_ylabel("Sample Count")
            ax.legend()

        save_fig(fig, "02_samples_per_lap.png")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 3 — TELEMETRY: Brake noise + zone length distribution
# ─────────────────────────────────────────

def eda_brake_zones(tel):
    try:
        log.info("=" * 60)
        log.info("SECTION 3 — Brake Zone Analysis")
        log.info("=" * 60)

        all_zone_lengths = []

        for (race, driver, lap_num), lap_tel in tel.groupby(["Race", "Driver", "LapNumber"]):
            lap_tel = lap_tel.sort_values("Distance").reset_index(drop=True)

            # Identify brake zone starts and lengths
            brake = lap_tel["Brake"].astype(bool)
            zone_id = (brake != brake.shift()).cumsum()
            brake_zones = lap_tel[brake].groupby(zone_id[brake])["Distance"].agg(
                lambda x: x.max() - x.min()
            )

            for length in brake_zones.values:
                all_zone_lengths.append({
                    "Race": race, "Driver": driver,
                    "LapNumber": lap_num, "ZoneLength": length
                })

        zone_df = pd.DataFrame(all_zone_lengths)

        log.info(f"\n  Total brake zone events detected: {len(zone_df)}")
        log.info(f"  Zone length stats (metres):")
        log.info(f"    min    = {zone_df['ZoneLength'].min():.1f}")
        log.info(f"    median = {zone_df['ZoneLength'].median():.1f}")
        log.info(f"    mean   = {zone_df['ZoneLength'].mean():.1f}")
        log.info(f"    max    = {zone_df['ZoneLength'].max():.1f}")

        # How many zones are below various thresholds (noise candidates)
        for threshold in [5, 10, 20, 30]:
            count = (zone_df["ZoneLength"] < threshold).sum()
            pct   = 100 * count / len(zone_df)
            log.info(f"    Zones < {threshold}m: {count} ({pct:.1f}%) — likely noise")

        log.info(f"\n  Current MIN_BRAKE_ZONE_LENGTH_M = {MIN_BRAKE_ZONE_LENGTH_M}m")

        # Plot brake zone length distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Brake Zone Length Distribution", fontsize=13)

        axes[0].hist(zone_df["ZoneLength"], bins=50, color="#1E3A8A", alpha=0.7)
        axes[0].axvline(MIN_BRAKE_ZONE_LENGTH_M, color="red", linestyle="--",
                        label=f"Current threshold ({MIN_BRAKE_ZONE_LENGTH_M}m)")
        axes[0].set_xlabel("Zone Length (m)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("All Zones")
        axes[0].legend()

        # Zoomed in on short zones (noise region)
        axes[1].hist(zone_df[zone_df["ZoneLength"] < 100]["ZoneLength"],
                     bins=40, color="#15803D", alpha=0.7)
        axes[1].axvline(MIN_BRAKE_ZONE_LENGTH_M, color="red", linestyle="--",
                        label=f"Current threshold ({MIN_BRAKE_ZONE_LENGTH_M}m)")
        axes[1].set_xlabel("Zone Length (m)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Zoomed: Zones < 100m")
        axes[1].legend()

        save_fig(fig, "03_brake_zone_lengths.png")

        return zone_df

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 4 — TELEMETRY: Signal distributions per driver per race
# ─────────────────────────────────────────

def eda_telemetry_signals(tel):
    try:
        log.info("=" * 60)
        log.info("SECTION 4 — Telemetry Signal Distributions")
        log.info("=" * 60)

        colors = {"VER": "#1E3A8A", "HAM": "#15803D"}
        races  = tel["Race"].unique()

        # ── 4a: Speed distribution per driver per race ──
        fig, axes = plt.subplots(1, len(races), figsize=(15, 4))
        fig.suptitle("Speed Distribution per Driver per Race", fontsize=13)

        for i, race in enumerate(races):
            ax = axes[i]
            for driver in DRIVERS:
                data = tel[(tel["Race"] == race) & (tel["Driver"] == driver)]["Speed"]
                ax.hist(data, bins=40, alpha=0.6, label=driver, color=colors[driver])
                log.info(f"  Speed | {race} | {driver} | "
                         f"mean={data.mean():.1f}  min={data.min():.1f}  max={data.max():.1f}")
            ax.set_title(race)
            ax.set_xlabel("Speed (km/h)")
            ax.legend()

        save_fig(fig, "04a_speed_distribution.png")

        # ── 4b: Throttle distribution ──
        fig, axes = plt.subplots(1, len(races), figsize=(15, 4))
        fig.suptitle("Throttle Distribution per Driver per Race", fontsize=13)

        for i, race in enumerate(races):
            ax = axes[i]
            for driver in DRIVERS:
                data = tel[(tel["Race"] == race) & (tel["Driver"] == driver)]["Throttle"]
                ax.hist(data, bins=30, alpha=0.6, label=driver, color=colors[driver])
                full_pct = (data >= FULL_THROTTLE_THRESHOLD).mean() * 100
                log.info(f"  Throttle | {race} | {driver} | "
                         f"mean={data.mean():.1f}  full_throttle%={full_pct:.1f}%")
            ax.set_title(race)
            ax.set_xlabel("Throttle (%)")
            ax.legend()

        save_fig(fig, "04b_throttle_distribution.png")

        # ── 4c: Gear shift count per lap per driver per race ──
        gear_shifts = (
            tel.groupby(["Race", "Driver", "LapNumber"])
            .apply(lambda x: (x["nGear"].diff().abs() > 0).sum(), include_groups=False)
            .reset_index(name="GearShifts")
        )

        fig, axes = plt.subplots(1, len(races), figsize=(15, 4))
        fig.suptitle("Gear Shifts per Lap per Driver", fontsize=13)

        for i, race in enumerate(races):
            ax = axes[i]
            for driver in DRIVERS:
                data = gear_shifts[
                    (gear_shifts["Race"] == race) & (gear_shifts["Driver"] == driver)
                ]["GearShifts"]
                ax.hist(data, bins=20, alpha=0.6, label=driver, color=colors[driver])
                log.info(f"  GearShifts | {race} | {driver} | "
                         f"mean={data.mean():.1f}  min={data.min()}  max={data.max()}")
            ax.set_title(race)
            ax.set_xlabel("Gear Shifts per Lap")
            ax.legend()

        save_fig(fig, "04c_gear_shifts.png")

        # ── 4d: Coasting % per lap per driver per race ──
        def coasting_pct(group):
            coasting = ((group["Throttle"] < THROTTLE_OFF_THRESHOLD) &
                        (~group["Brake"].astype(bool)))
            return coasting.mean() * 100

        coast_df = (
            tel.groupby(["Race", "Driver", "LapNumber"])
            .apply(coasting_pct, include_groups=False)
            .reset_index(name="CoastingPct")
        )

        fig, axes = plt.subplots(1, len(races), figsize=(15, 4))
        fig.suptitle("Coasting % per Lap per Driver", fontsize=13)

        for i, race in enumerate(races):
            ax = axes[i]
            for driver in DRIVERS:
                data = coast_df[
                    (coast_df["Race"] == race) & (coast_df["Driver"] == driver)
                ]["CoastingPct"]
                ax.hist(data, bins=20, alpha=0.6, label=driver, color=colors[driver])
                log.info(f"  Coasting% | {race} | {driver} | "
                         f"mean={data.mean():.2f}%  min={data.min():.2f}%  max={data.max():.2f}%")
            ax.set_title(race)
            ax.set_xlabel("Coasting % of Lap")
            ax.legend()

        save_fig(fig, "04d_coasting_pct.png")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 5 — TARGET: lap delta, class balance, lap symmetry, compound overlap
# ─────────────────────────────────────────

def eda_target(laps):
    try:
        log.info("=" * 60)
        log.info("SECTION 5 — Target Variable Analysis")
        log.info("=" * 60)

        races = laps["Race"].unique()

        # ── 5a: Lap count symmetry — find paired laps ──
        ver_laps = laps[laps["Driver"] == "VER"][["Race", "LapNumber"]].copy()
        ham_laps = laps[laps["Driver"] == "HAM"][["Race", "LapNumber"]].copy()

        paired = ver_laps.merge(ham_laps, on=["Race", "LapNumber"], how="inner")
        log.info(f"\n  Total VER laps : {len(ver_laps)}")
        log.info(f"  Total HAM laps : {len(ham_laps)}")
        log.info(f"  Paired laps    : {len(paired)}")
        log.info(f"  Unpaired laps  : {len(ver_laps) + len(ham_laps) - 2 * len(paired)}")

        for race in races:
            v = set(ver_laps[ver_laps["Race"] == race]["LapNumber"])
            h = set(ham_laps[ham_laps["Race"] == race]["LapNumber"])
            only_ver = v - h
            only_ham = h - v
            log.info(f"  {race} — VER only laps: {sorted(only_ver)} | HAM only laps: {sorted(only_ham)}")

        # ── 5b: Build lap delta on paired laps ──
        ver_data = laps[laps["Driver"] == "VER"][["Race", "LapNumber", "LapTimeSec",
                                                   "Compound", "TyreLife"]].copy()
        ham_data = laps[laps["Driver"] == "HAM"][["Race", "LapNumber", "LapTimeSec",
                                                   "Compound", "TyreLife"]].copy()

        ver_data.columns = ["Race", "LapNumber", "VER_LapTime", "VER_Compound", "VER_TyreLife"]
        ham_data.columns = ["Race", "LapNumber", "HAM_LapTime", "HAM_Compound", "HAM_TyreLife"]

        delta_df = ver_data.merge(ham_data, on=["Race", "LapNumber"])
        delta_df["LapTimeDelta"] = delta_df["VER_LapTime"] - delta_df["HAM_LapTime"]
        delta_df["VER_Faster"]   = (delta_df["LapTimeDelta"] < 0).astype(int)

        log.info(f"\n  Lap time delta (VER - HAM) stats:")
        log.info(f"    mean   = {delta_df['LapTimeDelta'].mean():.3f}s")
        log.info(f"    std    = {delta_df['LapTimeDelta'].std():.3f}s")
        log.info(f"    min    = {delta_df['LapTimeDelta'].min():.3f}s")
        log.info(f"    max    = {delta_df['LapTimeDelta'].max():.3f}s")

        for race in races:
            subset = delta_df[delta_df["Race"] == race]["LapTimeDelta"]
            log.info(f"  {race} delta — mean={subset.mean():.3f}s  std={subset.std():.3f}s")

        # ── 5c: Class balance ──
        ver_faster_count = delta_df["VER_Faster"].sum()
        ham_faster_count = len(delta_df) - ver_faster_count
        log.info(f"\n  Class balance (VER_Faster):")
        log.info(f"    VER faster (1): {ver_faster_count} laps ({100*ver_faster_count/len(delta_df):.1f}%)")
        log.info(f"    HAM faster (0): {ham_faster_count} laps ({100*ham_faster_count/len(delta_df):.1f}%)")

        # ── 5d: Compound overlap check ──
        same_compound = (delta_df["VER_Compound"] == delta_df["HAM_Compound"]).sum()
        diff_compound = len(delta_df) - same_compound
        log.info(f"\n  Compound overlap:")
        log.info(f"    Same compound laps : {same_compound} ({100*same_compound/len(delta_df):.1f}%)")
        log.info(f"    Diff compound laps : {diff_compound} ({100*diff_compound/len(delta_df):.1f}%)")

        diff_rows = delta_df[delta_df["VER_Compound"] != delta_df["HAM_Compound"]][
            ["Race", "LapNumber", "VER_Compound", "HAM_Compound", "LapTimeDelta"]
        ]
        if len(diff_rows) > 0:
            log.info(f"\n  Laps where compounds differ:\n{diff_rows.to_string(index=False)}")

        # ── 5e: Plots ──
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle("Target Variable Analysis", fontsize=13)

        # Delta distribution
        axes[0].hist(delta_df["LapTimeDelta"], bins=30, color="#1E3A8A", alpha=0.7)
        axes[0].axvline(0, color="red", linestyle="--", label="Zero (equal)")
        axes[0].set_xlabel("VER - HAM Lap Time (s)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Lap Time Delta Distribution")
        axes[0].legend()

        # Delta across race laps per race
        colors_race = {"Bahrain": "#1E3A8A", "Spain": "#15803D", "AbuDhabi": "#B45309"}
        for race, grp in delta_df.groupby("Race"):
            axes[1].plot(grp["LapNumber"], grp["LapTimeDelta"],
                         label=race, color=colors_race.get(race, "gray"), alpha=0.8)
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_xlabel("Lap Number")
        axes[1].set_ylabel("VER - HAM (s)")
        axes[1].set_title("Delta Across Race Laps")
        axes[1].legend()

        # Class balance bar
        axes[2].bar(["HAM Faster (0)", "VER Faster (1)"],
                    [ham_faster_count, ver_faster_count],
                    color=["#15803D", "#1E3A8A"])
        axes[2].set_ylabel("Lap Count")
        axes[2].set_title("Class Balance")

        save_fig(fig, "05_target_analysis.png")

        return delta_df

    except Exception as e:
        raise CustomException(e, sys)
    
# ─────────────────────────────────────────
# SECTION 7 — TEAMMATE COMPARISON
# Separates driver style from car design philosophy
# VER vs PER (same Red Bull), HAM vs BOT (same Mercedes)
# ─────────────────────────────────────────

def eda_teammate_comparison(session_cache_dir):
    """
    Pull Perez and Bottas telemetry for the same 3 races.
    Compute within-team style residuals:
        VER_style = VER_metric - PER_metric  (pure driver, same car)
        HAM_style = HAM_metric - BOT_metric  (pure driver, same car)
    Compare VER_style vs HAM_style.
    If VER coasts more than Perez in the same car -> that is VER's choice.
    If VER and Perez coast equally -> that is the Red Bull car requiring it.
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache(session_cache_dir)

        log.info("=" * 60)
        log.info("SECTION 7 — Teammate Comparison (Style vs Car)")
        log.info("=" * 60)

        # Drivers to compare: {driver: teammate}
        teammate_pairs = {"VER": "PER", "HAM": "BOT"}
        all_drivers    = ["VER", "PER", "HAM", "BOT"]

        style_records = []

        for round_number, race_label in RACES:
            log.info(f"\n  Loading {race_label} for teammate comparison...")
            session = fastf1.get_session(SEASON, round_number, SESSION_TYPE)
            session.load(telemetry=True, laps=True, weather=False, messages=False)

            driver_metrics = {}

            for driver in all_drivers:
                try:
                    laps = session.laps.pick_drivers(driver).pick_quicklaps().copy()
                    if len(laps) < 5:
                        log.info(f"    {driver} has fewer than 5 clean laps in "
                                 f"{race_label} — skipping")
                        continue

                    coasting_vals   = []
                    full_thr_vals   = []
                    gear_shift_vals = []
                    brake_len_vals  = []

                    for _, lap in laps.iterrows():
                        try:
                            tel = lap.get_telemetry()
                        except Exception:
                            continue

                        tel = tel.sort_values("Distance").reset_index(drop=True)
                        brake    = tel["Brake"].astype(bool)
                        throttle = tel["Throttle"]
                        gear     = tel["nGear"]

                        # Coasting %
                        coasting = ((throttle < THROTTLE_OFF_THRESHOLD) & (~brake))
                        coasting_vals.append(coasting.mean() * 100)

                        # Full throttle %
                        full_thr_vals.append(
                            (throttle >= FULL_THROTTLE_THRESHOLD).mean() * 100
                        )

                        # Gear shifts
                        gear_shift_vals.append((gear.diff().abs() > 0).sum())

                        # Avg brake zone length
                        zone_id = (brake != brake.shift()).cumsum()
                        brake_lengths = tel[brake].groupby(
                            zone_id[brake])["Distance"].agg(
                            lambda x: x.max() - x.min()
                        )
                        real_zones = brake_lengths[
                            brake_lengths >= MIN_BRAKE_ZONE_LENGTH_M
                        ]
                        brake_len_vals.append(
                            real_zones.mean() if len(real_zones) > 0 else 0.0
                        )

                    if len(coasting_vals) == 0:
                        continue

                    driver_metrics[driver] = {
                        "coasting_pct"      : np.mean(coasting_vals),
                        "full_throttle_pct" : np.mean(full_thr_vals),
                        "gear_shifts"       : np.mean(gear_shift_vals),
                        "brake_zone_length" : np.mean(brake_len_vals),
                    }
                    log.info(f"    {driver} | coasting={np.mean(coasting_vals):.2f}% | "
                             f"full_thr={np.mean(full_thr_vals):.2f}% | "
                             f"gear_shifts={np.mean(gear_shift_vals):.1f} | "
                             f"brake_len={np.mean(brake_len_vals):.1f}m")

                except Exception as ex:
                    log.warning(f"    Could not process {driver} in {race_label}: {ex}")
                    continue

            # Compute within-team residuals
            for driver, teammate in teammate_pairs.items():
                if driver in driver_metrics and teammate in driver_metrics:
                    for metric in ["coasting_pct", "full_throttle_pct",
                                   "gear_shifts", "brake_zone_length"]:
                        style_records.append({
                            "Race"     : race_label,
                            "Driver"   : driver,
                            "Teammate" : teammate,
                            "Metric"   : metric,
                            "Driver_val"  : driver_metrics[driver][metric],
                            "Teammate_val": driver_metrics[teammate][metric],
                            "Style_residual": (driver_metrics[driver][metric] -
                                               driver_metrics[teammate][metric]),
                        })

        if len(style_records) == 0:
            log.warning("  No teammate data computed — skipping Section 7 plots.")
            return

        style_df = pd.DataFrame(style_records)

        log.info("\n  Within-team style residuals (Driver - Teammate, same car):")
        log.info("  Positive = Driver does MORE of this than teammate")
        summary = style_df.groupby(["Driver", "Metric"])["Style_residual"].mean()
        log.info(f"\n{summary.round(3).to_string()}")

        # ── Plot: style residuals per metric per race ──
        metrics    = ["coasting_pct", "full_throttle_pct",
                      "gear_shifts", "brake_zone_length"]
        metric_labels = ["Coasting % (Driver - Teammate)",
                         "Full Throttle % (Driver - Teammate)",
                         "Gear Shifts (Driver - Teammate)",
                         "Brake Zone Length m (Driver - Teammate)"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Driver Style vs Teammate — Same Car Comparison\n"
            "VER vs PER (Red Bull) | HAM vs BOT (Mercedes)\n"
            "Positive = Driver does MORE of this than teammate → driver choice, not car",
            fontsize=12
        )
        axes = axes.flatten()
        colors = {"VER": "#1E3A8A", "HAM": "#15803D"}
        races  = style_df["Race"].unique()
        x      = np.arange(len(races))
        width  = 0.35

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i]
            for j, driver in enumerate(["VER", "HAM"]):
                vals = []
                for race in races:
                    row = style_df[
                        (style_df["Driver"] == driver) &
                        (style_df["Metric"] == metric) &
                        (style_df["Race"]   == race)
                    ]
                    vals.append(row["Style_residual"].values[0]
                                if len(row) > 0 else 0.0)
                offset = (j - 0.5) * width
                ax.bar(x + offset, vals, width,
                       label=f"{driver} - teammate",
                       color=colors[driver], alpha=0.8)

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(races)
            ax.set_ylabel(label)
            ax.set_title(label.split("(")[0].strip())
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")

        plt.tight_layout()
        save_fig(fig, "15_teammate_style_comparison.png")

        # ── Log interpretation ──
        log.info("\n  INTERPRETATION:")
        for driver in ["VER", "HAM"]:
            log.info(f"\n  {driver}:")
            for metric in metrics:
                avg_residual = style_df[
                    (style_df["Driver"] == driver) &
                    (style_df["Metric"] == metric)
                ]["Style_residual"].mean()
                direction = "MORE" if avg_residual > 0 else "LESS"
                log.info(f"    {metric:<25} {direction} than teammate "
                         f"by {abs(avg_residual):.3f} (avg across races)")

        log.info("\n  → Residuals that are consistently positive/negative across all 3 races")
        log.info("    are DRIVER choices. Residuals that vary by race are CAR/STRATEGY effects.")

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# SECTION 6 — SUMMARY: config recommendations
# ─────────────────────────────────────────

def print_config_recommendations(laps, tel, zone_df, delta_df):
    try:
        log.info("=" * 60)
        log.info("SECTION 6 — CONFIG RECOMMENDATIONS")
        log.info("=" * 60)

        # Recommend MIN_BRAKE_ZONE_LENGTH_M based on noise analysis
        p10 = zone_df["ZoneLength"].quantile(0.10)
        log.info(f"\n  [THRESHOLD] MIN_BRAKE_ZONE_LENGTH_M:")
        log.info(f"    10th percentile of all brake zone lengths = {p10:.1f}m")
        log.info(f"    Recommendation: set to {max(10, round(p10)):.0f}m "
                 f"(filters bottom 10% as noise)")

        # Class balance recommendation
        ver_pct = delta_df["VER_Faster"].mean()
        log.info(f"\n  [CLASS BALANCE] VER_Faster = {ver_pct:.2f}")
        if ver_pct < 0.40 or ver_pct > 0.60:
            log.info("    Recommendation: use class_weight='balanced' in classifier")
        else:
            log.info("    Recommendation: class balance is acceptable, no reweighting needed")

        # Non-green laps
        non_green_pct = (laps["TrackStatus"] != 1).mean() * 100
        log.info(f"\n  [TRACK STATUS] Non-green flag laps = {non_green_pct:.1f}%")
        log.info("    Recommendation: filter TrackStatus == 1 in data_transformation.py")

        # Compound overlap
        ver_data = laps[laps["Driver"] == "VER"][["Race", "LapNumber", "Compound"]]
        ham_data = laps[laps["Driver"] == "HAM"][["Race", "LapNumber", "Compound"]]
        ver_data.columns = ["Race", "LapNumber", "VER_Compound"]
        ham_data.columns = ["Race", "LapNumber", "HAM_Compound"]
        merged = ver_data.merge(ham_data, on=["Race", "LapNumber"])
        diff_pct = (merged["VER_Compound"] != merged["HAM_Compound"]).mean() * 100
        log.info(f"\n  [COMPOUND MISMATCH] {diff_pct:.1f}% of paired laps have different compounds")
        if diff_pct > 5:
            log.info("    Recommendation: add 'same_compound' flag as feature in transformation")
        else:
            log.info("    Recommendation: compound mismatch is minimal, safe to include compound as feature")

        # Sample count consistency
        samples = tel.groupby(["Race", "Driver", "LapNumber"]).size()
        median_s = samples.median()
        thin_pct = (samples < median_s * 0.5).mean() * 100
        log.info(f"\n  [TELEMETRY SAMPLES] Median samples/lap = {median_s:.0f}")
        log.info(f"    Thin laps (<50% median): {thin_pct:.1f}%")
        if thin_pct > 2:
            log.info("    Recommendation: filter thin laps in data_transformation.py")
        else:
            log.info("    Recommendation: sample count is consistent, no filtering needed")

        log.info("\n" + "=" * 60)
        log.info("Update config.py INITIAL_MODEL_PARAMS based on the above before running data_transformation.py")
        log.info("=" * 60)

    except Exception as e:
        raise CustomException(e, sys)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_eda():


    try:
        laps, tel = load_data()

        eda_laps(laps)
        eda_telemetry_alignment(tel)
        zone_df  = eda_brake_zones(tel)
        eda_telemetry_signals(tel)
        delta_df = eda_target(laps)
        print_config_recommendations(laps, tel, zone_df, delta_df)

        log.info("EDA complete. All plots saved to artifacts/")
        
        try:
            from src.config import CACHE_DIR, SESSION_TYPE
            eda_teammate_comparison(CACHE_DIR)
        except Exception as e:
            log.warning(f"Section 7 skipped: {e}")
    
        return laps, tel, zone_df, delta_df

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_eda()