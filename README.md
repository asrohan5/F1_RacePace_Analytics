# F1 Driving Style Analysis — VER vs HAM (2021)
### Decoding the Championship Battle Through Telemetry
# Phase 1
---

## Problem Statement

**Client Brief:**
> Max Verstappen and Lewis Hamilton finished the 2021 season tied on points going into the final lap of the final race. Both are widely considered the greatest drivers of their generation, yet they drive completely differently. We want to understand: what does each driver's raw driving style look like in data — how they use the throttle, when they brake, how they manage gears — and whether those style differences explain performance gaps across the three most decisive races of the season.

**Technical Derivation:**
> Using FastF1 telemetry for VER and HAM across Bahrain (R1), Spain (R4), and Abu Dhabi (R22) of the 2021 season, extract lap-level and micro-zone-level features from Speed, Throttle, Brake, nGear, and RPM channels sampled at ~5Hz. Segment each lap into braking zones, coasting zones, and throttle application zones identified from raw telemetry signals. Engineer driver-level style metrics including braking point distance, brake zone length, trail-brake indicators, coasting percentage, throttle aggression, and gear shift timing. Control for the car confound by comparing each driver against their teammate (VER vs Perez, HAM vs Bottas) in the same car. Build two supervised models — one on all laps (strategy-aware) and one on same-compound laps only (driving style isolated) — to predict lap time delta and who will be faster. Use SHAP to explain which style differences matter most. Validate on held-out test laps.

---

## Races Selected and Why

| Race | Round | Why It Matters |
|---|---|---|
| Bahrain GP | R1 | Season opener — VER and HAM on different strategies from lap 1. HAM ran Hard for 40 laps, VER ran Medium for 31. Baseline driving style with maximum compound mismatch. |
| Spanish GP | R4 | HAM won from P2 via a dramatic undercut. Late-race soft tyre stint gave HAM a 3.5s/lap advantage. Maximum strategy variance in the dataset. |
| Abu Dhabi GP | R22 | The championship decider. Both drivers in maximum attack mode, different tyre starts, the famous final lap. VER coasted 2.76% more per lap than HAM — widest gap of the season. |

---

## Data

| Source | FastF1 Python API (v3.8.3) |
|---|---|
| Season | 2021 |
| Races | Bahrain (R1), Spain (R4), Abu Dhabi (R22) |
| Drivers | VER, HAM (primary) + PER, BOT (teammate control) |
| Raw laps | 305 rows, 8 columns |
| Raw telemetry | 239,811 samples at ~5Hz, 9 columns |
| Paired laps | 138 matched lap pairs (VER + HAM, same lap number) |
| Same-compound subset | 99 paired laps (71.7%) |

**Key data constraint acknowledged:** Brake is boolean in 2021 FastF1 data — not continuous pressure. Braking style is proxied through zone start distance, zone length in metres, and entry speed at brake point. This is the same data a junior analyst at an F1 team would get from the public telemetry feed.

---

## Feature Engineering

30 features engineered across 4 categories:

**Delta features (VER − HAM per lap):**
- `coasting_pct_delta` — % of lap in coasting (neither throttle nor brake)
- `full_throttle_pct_delta` — % of lap at ≥98% throttle
- `gear_shifts_delta` — gear change count difference
- `avg_brake_zone_length_delta` — mean braking zone length difference (metres)
- `avg_entry_speed_delta` — mean speed at brake point difference (km/h)
- `tyre_life_delta` — raw age difference
- `tyre_life_x_coasting_delta` — interaction: does coasting advantage hold under tyre pressure? (clipped ±50)
- `stint_phase_delta` — tyre life as fraction of expected compound stint (captures non-linear degradation)
- `abu_dhabi_gear_delta` — Abu Dhabi specific gear shift interaction (EDA showed 4-shift gap in final race)
- `rolling_delta_3` — 3-lap rolling mean delta (temporal momentum)

**Individual driver features:** Absolute values for each metric per driver (VER and HAM separately)

**Contextual features:** `same_compound` flag, compound encoding, lap number (fuel proxy), race encoding

**Targets:**
- Regression: `lap_time_delta_sec` = VER LapTime − HAM LapTime (continuous, seconds)
- Classification: `ver_faster` = 1 if VER faster that lap (binary)

---

## Models

Two model pairs trained — one for each question:

| Question | Model | Dataset |
|---|---|---|
| How large is the lap time gap and what drives it? | LightGBM Regressor | Full 138 paired laps |
| Who is faster this lap? | Logistic Regression | Full 138 paired laps |
| Same question but with tyres controlled for | Linear Regression | 99 same-compound laps |
| Same question but with tyres controlled for | Logistic Regression | 99 same-compound laps |

**Why LightGBM over XGBoost for the full model:** Better built-in regularisation (`min_child_samples`) for 95 training rows. LightGBM val MAE=0.315s vs XGBoost val MAE=0.322s.

**Why LinearRegression wins the SC model:** When the tyre confound is removed, the relationship between driving style and lap delta becomes linear. Linear model wins on 68 SC training rows because it does not overfit the way tree models do on small data. This is itself a finding — the relationship is linear when measured correctly.

**Why LassoCV zeroed 25/30 features:** Only 5 features contain real signal in the full dataset model. This is evidence of feature redundancy — the driving style signals are correlated with each other and with tyre life. The SC model zeroes 24/30 features for the same reason.

---

## Validation Approach

**Train/Val/Test split:** Chronological per race (70/15/15). Laps are time-ordered — random splitting would leak future lap context into training.

**Cross-validation:** Custom per-race chronological CV (train on first 70% of each race, validate on remaining 30%). This respects temporal ordering within races and avoids leaking between-race patterns.

**Why CV scores are lower than val scores:** The chronological CV puts early-race laps in training and late-race laps in validation for each fold. Late-race laps have different tyre states and fuel loads than early-race laps — this is the correct and honest estimate of out-of-sample performance.

---

## Results

### Model Performance

| Metric | Full Model | Same-Compound Model |
|---|---|---|
| Regressor | LightGBM | Linear Regression |
| Test MAE | 0.733s | **0.384s** |
| Test R² | 0.276 | **0.750** |
| Classifier | Logistic Regression | Logistic Regression |
| Test AUC | 0.838 | 0.855 |
| Test F1 | 0.813 | 0.714 |
| Chrono-CV MAE | 0.587s ± 0.189s | — |
| Chrono-CV AUC | 0.898 ± 0.107 | — |

### Per-Race Test Performance

| Race | Full MAE | SC MAE | Full Acc | SC Acc |
|---|---|---|---|---|
| Bahrain | 0.332s | 0.413s | 43% | 50% |
| Spain | 1.119s | **0.415s** | 100% | 100% |
| Abu Dhabi | 0.694s | **0.318s** | 71% | 60% |

Spain full model MAE=1.119s is the honest penalty for compound mismatches in the test set. Spain SC MAE=0.415s confirms the SC model handles it correctly. **Both models correctly predicted the direction (who was faster) on all 8 Spain test laps** — the issue is magnitude, not direction.

---

## Key Findings

### Finding 1 — VER's coasting is a driver choice, not a car requirement

Teammate comparison (Plot 15) confirms:
- VER coasts **+0.5% more** than Perez across all 3 races in the same Red Bull
- HAM coasts **−3.2% less** than Bottas across all 3 races in the same Mercedes
- Both residuals are consistent across all 3 races → driver choices, not car characteristics

VER deliberately lifts off the throttle earlier entering corners to rotate the car on its front axle — a technique that works on the Red Bull's design philosophy. HAM deliberately keeps the throttle on longer (trail throttle) to maintain rear stability — a Mercedes-specific technique. These are conscious stylistic choices, not car demands.

### Finding 2 — The interaction between tyre age and coasting is the primary predictor

SHAP top feature across both regression and classification: `TyreLife × Coasting Delta`.

**What this means:** VER's coasting advantage is most pronounced on fresh tyres. As tyres degrade, the coasting benefit shrinks. When VER's tyres are significantly older than HAM's, the coasting gap disappears and HAM's higher throttle application becomes the dominant signal. The interaction feature captured a non-linear relationship that raw tyre age alone could not express.

### Finding 3 — HAM brakes earlier and longer, VER brakes later and shorter

From the driver fingerprint (Plot 11):
- HAM average brake zone: 96m — 7m longer than VER across all races
- VER average entry speed at brake point: higher across all three races
- In Abu Dhabi specifically: HAM brake zone 100.6m vs VER 102.2m — closest gap of the three races, reflecting the championship pressure both drivers were under

### Finding 4 — Full Throttle % Delta is the #3 most important feature

VER applies full throttle on 50.4% of each lap vs HAM's 49.2% — a 1.2% difference. In a 90-second lap this is approximately 1 second of additional full-throttle running per lap across 58 Abu Dhabi laps. The SHAP beeswarm shows this pushes the delta positive (VER faster) when VER's full-throttle percentage is higher than HAM's.

### Finding 5 — Removing the tyre confound improved regression R² by 172%

Full model R²=0.276 → SC model R²=0.750 on test. The same-compound model explains 75% of lap time delta variance from driving style features alone on held-out data. This is the core methodological contribution of the project: demonstrating that controlling for a confounding variable (compound mismatch) transforms a mediocre model into a meaningful one.

### Finding 6 — Stint phase matters more than raw tyre age

`Stint Phase Delta` (tyre life as fraction of expected compound stint length) outperforms raw `Tyre Life Delta` in SHAP importance on the SC model. A 15-lap Medium tyre is not the same as a 15-lap Hard tyre — stint phase normalises for this. This was identified only after building the interaction feature first.

---

## Limitations of Phase 1

**1. Brake is boolean in 2021 data.**
Brake pressure magnitude became available in FastF1 only from 2025. The braking style analysis uses zone length and entry speed as proxies. These are defensible but indirect measures.

**2. Lap-level averaging removes corner-level signal.**
Every feature is averaged across the full lap. VER's advantage at specific corners (Bahrain Turn 4, Spain Turn 5) is invisible to this model. Corner-level features require significantly more data (Phase 2).

**3. 138 paired laps is a small dataset.**
Chronological CV R²=−0.43 on some folds — the model has not seen enough variation to generalise to all race scenarios. Phase 2 with all 22 races of 2021 addresses this directly.

**4. Car philosophy confound is partially, not fully, controlled.**
The teammate analysis shows VER coasts more than Perez consistently. However, Perez and VER also have different driving styles — a perfect control group would require swapping cars, which is impossible. The teammate comparison is the best available proxy.

**5. Rolling delta leaks temporal information.**
`rolling_delta_3` uses the previous 3 laps' actual deltas. In a true real-time prediction scenario this feature would require storing race history. It is flagged as a race-context feature, not a pre-race prediction feature.

---

## Phase 2 Roadmap

| Addition | Impact | Effort |
|---|---|---|
| All 22 races of 2021 (~800–1000 paired laps) | Major — enables corner-level features and reliable CV | High |
| Major upgrade flags (Spain floor, Brazil ICE token) | Controls for car performance step-changes | Low |
| Corner-level braking and throttle features | Reveals track-specific style differences | High |
| LightGBM for SC model (currently LinearRegression) | May improve SC regression when data is larger | Low |
| Perez/Bottas as additional training subjects | Doubles sample size, validates teammate control | Medium |

---
