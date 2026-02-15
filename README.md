# F1 Race Pace Analytics

Been a fan of F1, applying my tech knowledge to extract insights and make my watch much interesting

### Objective
This project implements a production-grade data pipeline to answer "What drives lap times in F1 races, and how can teams use these insights for better strategy decisions?" using real 2024 race data from FastF1. 


Phase A focuses on a single race (Brazil GP). Phase B extends to multi-race generalization (4 GPs).


## Phase B: SMART Goal Definition (Phase A is provided after this)

### SMART Goal
**By February 28, 2026, develop and validate a cross-circuit lap time prediction model that generalizes across 4 distinct 2024 F1 Grand Prix races (Bahrain, Saudi Arabian, Australian, Belgian), achieving a test MAE ≤ 0.75 seconds on a held-out circuit while maintaining ≥85% R², using the same 5 core features from Phase A (LapTimeDelta, TireAge, RaceProgress%, ConsistencyScore, Compound_Encoded).** 

**Specific:** Multi-race generalization of Phase A Random Forest model.  
**Measurable:** MAE ≤ 0.75s, R² ≥ 0.85 on held-out test circuit.  
**Achievable:** Reuses existing pipeline + 3 additional races (already ingested).  
**Relevant:** Tests real-world model deployment across track conditions.  
**Time-bound:** Complete by Feb 28, 2026 (2 weeks).

***

## Primary Objectives

1. **Validate Phase A model generalization**  
   Train Phase A Random Forest on 3 races, test on 1 held-out race (Belgian GP).  
   **Success:** MAE ≤ 0.75s (vs Phase A single-race MAE 0.63s).

2. **Benchmark model families**  
   Compare Linear Regression (baseline), Random Forest, XGBoost.  
   **Success:** RF/XGBoost outperform Linear by ≥10% R².

3. **Quantify circuit effects**  
   Measure feature importance stability across tracks.  
   **Success:** Top 3 features (LapTimeDelta, TireAge, RaceProgress%) rank consistently.

***

## Secondary Objectives

1. **Circuit-specific insights**  
   Compare feature importances: Does TireAge matter more on high-degradation tracks?  
   
2. **Strategy benchmarking**  
   Compare model predictions vs actual lap times for top/mid-field drivers.

3. **Model diagnostics**  
   Residual analysis: Identify prediction failure modes by circuit/driver.

***

## KPIs & Success Metrics

| **Metric**       | **Phase A (Brazil)**     | **Phase B Target**         | **Rationale** 
|------------      |---------------------     |--------------------        |-------------- 
| Test MAE         | 0.63s                    | ≤ 0.75s                    | Slight degradation acceptable for generalization 
| Test R²          | 0.91                     | ≥ 0.85                     | 85% explained variance across circuits 
| Feature Stability| N/A                      | Top 3 features consistent  | Model learns generalizable patterns 
| Train/Test Gap   | N/A                      | ≤ 0.15 MAE                 | Minimal overfitting 
| Circuits Covered | 1                        | 4                          | Bahrain (street), Saudi (street), Australia (street), Belgium (traditional) 



## Why This Goal is Valuable for Recruiters

1. **Production ML mindset:** Cross-validation, generalization testing, multi-model comparison.
2. **Business relevance:** Circuit-agnostic strategy insights for teams.
3. **Technical depth:** End-to-end from raw data → deployable model.
4. **Quantifiable success:** Clear KPIs with realistic targets.
5. **Scalable design:** Pipeline ready for 24-race season.


## Phase A: Brazil 2024 Grand Prix Analysis
## Technical Architecture

The pipeline follows this flow:

Raw F1 Data (FastF1 API) leads to cached Parquet + JSON Metadata (4 races × 800-1100 laps each).

This leads to cleaned race pace, pit strategy, driver summary datasets.

This leads to 65+ engineered features across tire, driver, race context domains.

This leads to Random Forest (MAE: 0.63s, R²: 0.91) and multi-model comparison (Phase B).

This leads to visualizations and executive summary for non-technical stakeholders.

## Implementation Steps

### Phase A: Brazil GP (2024 Sao Paulo Grand Prix)

#### Data Ingestion (data_ingestion/fetch_data.py)

Fetched official FastF1 timing data: 1,135 valid laps × 20 drivers.  Error handling for missing drivers (e.g., Driver #23). 

#### Data Cleaning (data_processing/clean_laps.py)

Raw to cleaned pipeline: Filter accurate laps only (IsAccurate=True). Convert LapTime to LapTimeSeconds (timedelta to numeric). Extract pit-in/out laps for strategy analysis. Driver-level summaries (laps, avg/std/best). Output: 943 valid race-pace laps ready for modeling.

#### Feature Engineering (modeling/feature_engineering.py)

65+ engineered features across 4 domains:

Tire Management: TireAge, LapTimeDelta, Compound_Encoded. Quantifies degradation, pit timing signals.

Driver Performance: PaceVsAvg, ConsistencyScore, GapFromBest. Driver stability, coaching targets.

Race Context: RacePhase (Early/Mid/Late), RaceProgress%. Strategy shifts by stint phase.

Aggregates: AvgLap, StdLap, BestLap (driver-level). Benchmarks vs teammates.

#### Baseline Modeling

RandomForestRegressor(n_estimators=150, max_depth=10). Test MAE: 0.63 seconds. R²: 0.91. Top features: LapTimeDelta (pace change) > RaceProgress% > TireAge.



### Phase B: Multi-Race Generalization (4 GPs)

#### Batch Processing Extension

Races processed: Bahrain, Saudi Arabian, Australian, Belgian Grand Prix (2024). Total dataset: ~4,000+ race-pace laps across 4 circuits.

In Progress .....

