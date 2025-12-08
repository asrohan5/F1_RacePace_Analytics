# F1 Race Data Business Analytics Project

Been a fan of F1, applying my tech knowledge to extract insights and make my watch much interesting

## Phase A: Brazil 2024 Grand Prix Analysis

### Objective
Understand which factors most strongly influence race lap times and tire performance using data-driven modeling. Specifically, I aimed to:
- Quantify the impact of tire age, race progression, and driver consistency on lap time
- Build an interpretable predictive model to support strategic decision-making
- Validate the business value of proactive tire management and pit stop timing



## Project Structure

- `/data_ingestion`: Scripts and notebooks for fetching, caching, and validating raw F1 data.
- `/data_processing`: Scripts for cleaning, merging, and transforming session data, telemetry, and external sources.
- `/eda`: Exploratory Data Analysis—initial insights, business storytelling, and visualization.
- `/modeling`: Feature engineering and development of relevant, business-driven statistical/ML models.
- `/reports`: Non-technical business summaries, dashboards, and communication materials.
- `main.py`: Entry point for workflow orchestration.
- `README.md`: This evolving documentation.


### Data & Approach
**Data Source:** 2024 São Paulo (Brazil) Grand Prix - Race session via FastF1  
**Sessions analyzed:** 1,135 laps across 19 drivers  
**Valid race pace laps:** 943 laps  


## Insights sought and brought out 

This section maps the original business questions to Phase A deliverables and outcomes.


**1. What metrics and features matter most for predicting performance loss due to tire wear or inconsistent pace?**
- **Answer:** The predictive model identified the top factors:
  - **LapTimeDelta (56.2%):** Lap-to-lap pace changes are the strongest predictor of performance shifts
  - **TireAge (16.3%):** Tire age directly correlates with slower lap times, quantifying wear impact
  - **ConsistencyScore (3.2%):** Driver consistency measurably affects lap-to-lap stability
- **Evidence:** Model R² = 0.913, MAE = 0.633s validates feature relevance
- **Business Use:** Teams can now quantify cost of tire wear and optimize pit stop windows

**2. How can race and telemetry data enable optimal strategy and pit stop decisions?**
- **Answer:** We demonstrated that:
  - Pit stop data (68 pit-related laps) can be extracted and analyzed from session data
  - TireAge is a quantifiable driver of pace; older tires = slower laps
  - RaceProgress% shows that late-race conditions differ significantly from early race (24.3% importance)
- **Evidence:** Feature importance and EDA lap progression plots
- **Business Use:** Data-driven pit stop timing based on tire age thresholds and race phase

**3. How can driver data be used to drive targeted coaching and improve lap times?**
- **Answer:** Driver consistency analysis revealed:
  - Most consistent drivers: RUS, NOR, VER, OCO (StdLap 2.2–2.7s)
  - Less consistent drivers: COL, HUL (higher StdLap, fewer laps)
  - Consistency is measurable and correlates with better tire management
- **Evidence:** Driver summary table with AvgLap, StdLap, BestLap metrics
- **Business Use:** Target coaching on consistency and tire management for underperforming drivers

**4. How can data visualizations make complex findings easy to understand for business audiences?**
- **Answer:** Generated three key visualizations:
  - **Pace Distribution Plot:** Shows lap time ranges by driver; identifies outliers and consistency
  - **Lap Progression Plot:** Trends over race distance; reveals tire wear and pace strategies
  - **Feature Importance Chart:** Clearly shows which factors matter (LapTimeDelta, RaceProgress%, TireAge)
- **Evidence:** Plots saved in `eda/plots/` and `modeling/plots/`
- **Business Use:** Stakeholders can instantly grasp which factors drive performance

**5. Which telemetry features most accurately predict pending pit stops or in-lap performance drops?**
- **Answer (Partial):** The model shows:
  - **LapTimeDelta** (lap-to-lap change) is the strongest early indicator of performance drops
  - Large positive deltas signal deterioration; can trigger pit stop consideration
  - TireAge provides predictive context for when drops are likely to occur
- **Evidence:** Feature importance (LapTimeDelta 56.2%) + residual analysis
- **Business Use:** Real-time monitoring of lap-to-lap changes as pit stop trigger signals

### Key Results

#### Model Performance
1. Test MAE - 0.633 seconds : Average prediction error is less than one second per lap
2. Test R² - 0.913 : Model explains ~91% of lap time variation 
3. Features Used - 5 engineered features : Minimal, interpretable feature set 

#### Feature Importance: What Drives Lap Time?
The model identified the following factors as most influential in determining lap times:

 **LapTimeDelta** - 56.2% - Lap-to-lap pace changes are the strongest early indicator of performance shifts. This allows teams to identify tire drop-off, driver issues, or strategic inflection points in real-time. 
 **RaceProgress%** - 24.3% - Race phase significantly affects pace. Early-race conditions differ from mid/late-race due to fuel load, tire wear progression, and strategy dynamics. 
 **TireAge** - 16.3% - Older tires demonstrably slow cars. This validates the business value of well-timed pit stops and tire management strategies. 
 **ConsistencyScore** - 3.2% - Driver consistency has a measurable but smaller impact; more reliable drivers show less lap-to-lap volatility. 
 **Compound_Encoded** - 0.0% - Tire compound did not differentiate performance in this race, suggesting converged strategies or similar compound usage across the grid. 

### EDA Insights

**Driver Consistency Analysis:**
- Most consistent drivers (lowest lap time variation): RUS, NOR, VER, OCO
- Less consistent drivers: COL, HUL (fewer total laps, possibly due to incidents)
- Insight: Driver consistency is a measurable trait and can inform team strategy and coaching focus

**Lap Progression Trends:**
- Top drivers show relatively stable pace throughout the race
- Mid-field drivers show greater pace variability, indicating tire management challenges
- Late-race performance diverges due to fuel and tire combinations

**Pit Stop Strategy:**
- 68 pit-related laps identified across 19 drivers
- Pit stop timing varied significantly; teams employed different strategies
- Model shows tire age directly influences pace, supporting data-driven pit stop optimization

### Business Implications

1. **Real-Time Strategy Support:** Teams can use lap-to-lap pace variation as an early warning system for pit stops or car adjustments.
2. **Tire Management Validation:** The model quantifies the cost of tire wear; strategic pit stops directly improve race outcomes.
3. **Driver Performance Coaching:** Consistency is measurable; drivers with high consistency scores manage tires and conditions better.
4. **Circuit-Specific Insights:** Tire compound did not matter in this race; future phases will investigate whether this is circuit-dependent.

### Files Generated
- `data/raw/2024_Brazil_R/`: Raw session data (laps, telemetry, metadata)
- `data/processed/2024_Brazil_*.parquet`: Cleaned and processed datasets
- `data/modeling/2024_Brazil_features.parquet`: Engineered features for modeling
- `eda/plots/pace_distribution.png`: Distribution of lap times by driver
- `eda/plots/lap_progression.png`: Lap time trends for top 5 drivers
- `modeling/plots/feature_importance.png`: Feature importance visualization


## Phase B: Ongoing and will be updated soon ...