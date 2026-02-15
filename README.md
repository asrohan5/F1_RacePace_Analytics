# F1 Race Pace Analytics

Been a fan of F1, applying my tech knowledge to extract insights and make my watch much interesting

## Phase A: Brazil 2024 Grand Prix Analysis

### Objective
This project implements a production-grade data pipeline to answer "What drives lap times in F1 races, and how can teams use these insights for better strategy decisions?" using real 2024 race data from FastF1. 

Understand which factors most strongly influence race lap times and tire performance using data-driven modeling. Specifically, I aimed to:
- Quantify the impact of tire age, race progression, and driver consistency on lap time
- Build an interpretable predictive model to support strategic decision-making
- Validate the business value of proactive tire management and pit stop timing


Phase A focuses on a single race (Brazil GP). Phase B extends to multi-race generalization (4 GPs).

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

