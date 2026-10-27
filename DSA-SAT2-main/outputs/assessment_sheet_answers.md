# DS Sprint Skill Assessment Submission Sheet
## CH15 – Road Safety Insight: Understand Accidents. Discover Risk Patterns. Support Safer Roads

> **Assessment Disclosure**: All metrics, distributions, and models in this submission are derived strictly from the supplied synthetic benchmark datasets (`accidents.csv`, `roads.csv`, `weather.csv`, `traffic.csv`, `vehicles.csv`). Findings reflect recorded collision occurrences within the synthetic cohort rather than real-world municipal fatality rates or exposure-adjusted risks.

---

### ROUND 1 – DATA UNDERSTANDING, PREPARATION & EDA

#### 1. Problem Statement
The objective of CH15 (Road Safety Insight) is to analyze multi-source municipal road traffic accident records to discover spatial, temporal, environmental, and infrastructural risk patterns associated with accident frequency and injury severity. By integrating fragmented tabular datasets, performing exploratory data analysis, and engineering predictive features, the project builds a multi-class machine learning classification pipeline to predict accident severity (`Minor`, `Serious`, `Fatal`) and operationalizes actionable road safety intelligence via an automated Excel analytical workbook and executive reporting.

#### 2. Dataset Details: name, source, record count, attribute count
- **Data Source**: CH15 Road Safety Insight Synthetic Assessment Benchmark (`data/raw/`).
- **Raw Datasets (5 CSV files)**:
  1. `accidents.csv`: **6,040 rows**, **5 attributes** (`incident_id`, `date`, `time`, `location`, `severity`).
  2. `roads.csv`: **6,000 rows**, **4 attributes** (`incident_id`, `road_type`, `junction_type`, `speed_limit_kmph`).
  3. `weather.csv`: **6,000 rows**, **4 attributes** (`incident_id`, `rainfall_mm`, `visibility_km`, `weather_condition`).
  4. `traffic.csv`: **6,000 rows**, **3 attributes** (`incident_id`, `traffic_volume_vph`, `peak_flag`).
  5. `vehicles.csv`: **6,000 rows**, **3 attributes** (`incident_id`, `vehicle_category`, `vehicle_count`).
- **Cleaned & Merged Dataset (`data/processed/cleaned_accidents.csv`)**:
  - **Record Count**: **5,976 records** (100% verified 1-to-1 inner join across all 5 tables).
  - **Attribute Count**: **23 attributes** (15 integrated domain attributes + 8 engineered temporal/risk indicators: `incident_id`, `date`, `time`, `location`, `severity`, `road_type`, `junction_type`, `speed_limit_kmph`, `rainfall_mm`, `visibility_km`, `weather_condition`, `traffic_volume_vph`, `peak_flag`, `vehicle_category`, `vehicle_count`, `hour_of_day`, `day_of_week`, `day_of_week_num`, `month`, `month_name`, `is_weekend`, `time_period`, `is_severe`).

#### 3. Data Preprocessing: actual missing-value, duplicate and outlier treatment
- **Duplicate Removal**: Identified and removed **40 duplicate records** in `accidents.csv` based on `incident_id` (6,040 raw records reduced to 6,000 unique incident records).
- **Invalid Severity Exclusion**: Identified and excluded **24 non-standard severity records** (`Unknown`: 8, `Critical`: 6, `0`: 5, `5`: 5), preserving only standard classes (`Minor`, `Serious`, `Fatal`), resulting in **5,976 valid incident records**.
- **Missing Value Imputation**:
  - `location`: **149 missing values** in the valid cohort (151 in raw data) imputed with `'Unknown Location'` to preserve records without introducing spatial bias.
  - `weather_condition`: **300 missing observations** (5.0%) imputed with the categorical mode (`'Clear'`).
  - `rainfall_mm`: **366 missing values** (6.1%) imputed with median rainfall grouped by `weather_condition` (0.0 mm fallback).
  - `visibility_km`: **397 missing values** (6.6%) imputed with median visibility grouped by `weather_condition` (8.0 km fallback).
- **Outlier and Anomaly Treatment**:
  - `road_type`: Standardized **115 records** with irregular whitespace and mixed casing into 5 canonical categories (`Urban Arterial`, `Collector Road`, `Service Road`, `Highway Link`, `Residential Street`).
  - `speed_limit_kmph`: Identified **18 extreme speed limit outliers** outside the realistic municipal boundary of `[20, 80]` km/h (raw min: -20 km/h, raw max: 1,000 km/h) and imputed them using the median speed limit of the corresponding `road_type`.
  - `traffic_volume_vph`: Identified **24 extreme traffic volume outliers** outside the realistic boundary of `[50, 10000]` vehicles per hour (raw min: -100 vph, raw max: 130,000 vph) and imputed them using the grouped median of `(road_type, peak_flag)`.
- **Integration**: Performed 1-to-1 cardinality verification across all 5 tables using `incident_id`, producing a clean, integrated dataset of **5,976 records** with zero remaining missing values.

#### 4. EDA Findings: two actual findings with counts or percentages
- **Finding 1 (Hotspot Divergence — Volume vs. Lethality)**:
  Central Market Junction recorded the highest overall accident volume across the network with **445 incidents** (7.45% of total accidents; including 147 severe incidents, 33.03% severe share). However, East Lake Road exhibited the highest fatal crash count with **24 fatal crashes** (7.50% fatal crash share across 320 recorded incidents), followed by Outer Loop - Sector A with **19 fatal crashes** (6.93% fatal crash share across 274 recorded incidents). High-volume minor-collision junctions diverge from high-lethality arterial corridors.
- **Finding 2 (Environmental Risk Amplification Under Dense Fog)**:
  Under Clear conditions (**3,022 incidents**), the fatal crash share is **3.18%** (96 fatal crashes) and the combined severe crash share is **28.79%** (870 severe incidents). In contrast, under dense Fog conditions (**321 incidents**), the fatal crash proportion surges to **8.10%** (26 fatal crashes) and the combined severe crash share reaches **45.48%** (146 severe crashes: 120 serious, 26 fatal). Compromised visibility dramatically compounds collision severity.

---

### ROUND 2 – FEATURE ENGINEERING & MACHINE LEARNING

#### 1. Selected Features: actual input variables and target variable
- **Target Variable**:
  - `severity`: Multi-class categorical target with 3 classes (`Minor`: 4,100 records [68.61%], `Serious`: 1,645 records [27.53%], `Fatal`: 231 records [3.87%]).
- **Selected Input Features (16 features total)**:
  - **Numerical Features (8)**: `hour_of_day`, `month`, `is_weekend`, `speed_limit_kmph`, `rainfall_mm`, `visibility_km`, `traffic_volume_vph`, `vehicle_count`.
  - **Categorical Features (8)**: `day_of_week`, `time_period`, `location`, `road_type`, `junction_type`, `weather_condition`, `peak_flag`, `vehicle_category`.
- **Leakage Prevention**:
  - Direct identifiers (`incident_id`), raw timestamps (`date`, `time`), and target-derived indicators (`is_severe`) were strictly excluded.
  - Imputation and scaling transformers were fitted strictly on the training partition ($N=4,780$) and applied without refitting to the test partition ($N=1,196$).

#### 2. ML Algorithm Used
- **Algorithm**: `RandomForestClassifier` encapsulated in a Scikit-Learn `Pipeline` with `ColumnTransformer` (StandardScaler for numerical features, OneHotEncoder with `handle_unknown="ignore"` for categorical features, and median/mode SimpleImputers).
- **Hyperparameters**: `n_estimators=150`, `max_depth=10`, `min_samples_leaf=2`, `class_weight="balanced"`, `random_state=42`.

#### 3. Justification
- **Tabular Non-Linear Relationships**: Random Forest ensembles effectively capture non-linear interactions among heterogeneous environmental, temporal, and road features without requiring distributional assumptions.
- **Overfitting Control**: Restricting tree depth (`max_depth=10`) and setting `min_samples_leaf=2` prevents the ensemble from memorizing noisy synthetic tabular combinations.
- **Handling Severe Class Imbalance**: With fatal crashes constituting only 3.87% of the dataset, standard unweighted classifiers suffer complete majority-class collapse. The cost-sensitive `class_weight="balanced"` hyperparameter penalizes misclassification inversely proportional to class frequencies, forcing the trees to split on rare severe-incident signals.
- **Interpretability & Feature Importances**: Provides robust Gini importance metrics that directly inform engineering risk priorities.

#### 4. Model Performance: actual accuracy, F1-score, and other relevant metrics
All metrics evaluated on the untouched 20% held-out stratified test set ($N=1,196$ records):
- **Overall Accuracy**: **63.38%** (0.6338) (758 correct classifications out of 1,196).
- **Macro F1-Score**: **0.3670** (0.37)
- **Weighted F1-Score**: **0.6027** (0.60)
- **Macro Precision**: **0.3870** | **Macro Recall**: **0.3670**
- **Weighted Precision**: **0.5860** | **Weighted Recall**: **0.6338**
- **Class-Wise Metrics**:
  - **Minor Severity** (True Support: 821):
    - Precision: **0.7086** (70.86%)
    - Recall: **0.8295** (82.95%)
    - F1-Score: **0.7643**
  - **Serious Severity** (True Support: 329):
    - Precision: **0.3472** (34.72%)
    - Recall: **0.2280** (22.80%)
    - F1-Score: **0.2752**
  - **Fatal Severity** (True Support: 46):
    - Precision: **0.1053** (10.53%)
    - Recall: **0.0435** (4.35%)
    - F1-Score: **0.0615**
- **Confusion Matrix** (Mathematical verification: row sums equal true class supports; column sums equal predicted counts):
  ```
                   Predicted Minor   Predicted Serious   Predicted Fatal   Total True
  Actual Minor           681                132                 8             821
  Actual Serious         245                 75                 9             329
  Actual Fatal            35                  9                 2              46
  Total Predicted        961                216                19            1196
  ```
  - *Diagonal (Correct)*: $681 + 75 + 2 = 758$ / $1,196 = 63.38\%$.

---

### ROUND 3 – INTELLIGENCE LAYER, VISUALIZATION & PROTOTYPE

#### 1. Visualization: names of charts actually generated
14 publication-quality 300-DPI charts generated in `outputs/charts/`:
1. `01_severity_distribution.png` — Overall breakdown of Minor, Serious, and Fatal incidents.
2. `02_top_accident_hotspots.png` — Top 10 accident locations sorted by total volume and severe incident counts.
3. `03_hourly_accident_distribution.png` — 24-hour crash frequency histogram highlighting commuter peaks.
4. `04_day_of_week_distribution.png` — Weekly crash distribution across weekdays and weekends.
5. `05_weather_condition_severity.png` — Severity proportion across weather conditions.
6. `06_road_type_accident_severity.png` — Incident volume and fatal crash proportion by road type.
7. `07_junction_type_accident_severity.png` — Crash distribution across 6 junction configurations.
8. `08_speed_limit_by_severity.png` — Boxplot and mean nominal speed limit comparison across severity classes.
9. `09_vehicle_category_severity.png` — Crash counts and severe-incident proportions by vehicle type.
10. `10_traffic_volume_by_peak_flag.png` — Traffic flow distributions during peak versus off-peak hours.
11. `11_time_period_severity_breakdown.png` — Incident volumes and fatal proportions across four daily time blocks.
12. `12_correlation_heatmap.png` — Correlation matrix of continuous numerical variables.
13. `13_confusion_matrix.png` — 3x3 heatmap of model predictions versus ground truth on test data.
14. `14_feature_importance.png` — Top 15 Random Forest feature importances.

#### 2. Key Insights: two actual findings with numerical evidence
- **Key Insight 1 (Temporal Disconnect: Commuter Volume Peaks vs. Nighttime Fatal Proportion)**:
  Commuter rush hours concentrate the vast majority of collisions (**60.64%**, 3,624 incidents), led by 18:00 (608 incidents) and 08:00 (547 incidents). However, Night hours (21:00 to 04:59) represent only **9.59%** of total accidents (573 incidents) but exhibit the highest fatal crash proportion at **4.71%** (27 fatal crashes), nearly double the Afternoon fatal share of **2.70%** (37 fatal crashes across 1,372 incidents). Rush hours generate frequent low-speed minor fender-benders, whereas off-peak night driving exhibits higher kinetic-energy collisions.
- **Key Insight 2 (Vulnerable Road Users and High-Speed Road Infrastructure)**:
  Incidents involving Motorcycles (**1,602 crashes**) suffered a **5.06% fatal crash share** (81 fatal crashes) and a **30.65% serious injury share** (491 serious crashes), producing a **35.71% combined severe share**—substantially higher than passenger cars (**3.48% fatal**, 28.21% severe). Concurrently, Highway Links exhibited the highest fatal crash proportion among all road types at **6.03%** (63 fatal crashes out of 1,044 incidents), and fatal crashes across all corridors occurred at a higher average nominal speed limit (**48.4 km/h**) than minor crashes (**44.3 km/h**).

#### 3. Prototype / Intelligence Layer: describe the implemented Excel analysis and summary dashboard
The intelligence layer is operationalized via a comprehensive 12-sheet Excel analytical workbook (`outputs/road_safety_analysis.xlsx`), complemented by a 6-slide executive presentation (`outputs/business_insights_presentation.pptx`):
- **Executive Summary Dashboard (Sheet 1)**: Formatted KPI cards displaying Total Incidents (5,976), Severe Crash Share (31.4%), and Random Forest Accuracy (63.4%), backed by dynamic Excel formulas (`COUNTIF`, `SUM`, `AVERAGE`), severity distributions, and cross-tab navigation.
- **Native PivotTables & PivotCharts (Sheet 2)**: Generated using Microsoft Excel COM automation (`win32com.client`), featuring native dynamic PivotTables and stacked column PivotCharts for *Road Classification vs. Severity* and *Weather Condition vs. Severity*.
- **Domain Analytical Sheets (Sheets 3 to 10)**: Dedicated drill-down worksheets (`Accident Hotspots`, `Time Analysis`, `Weather Analysis`, `Junction Analysis`, `Road Type Analysis`, `Traffic Analysis`, `Data Quality`, `Cleaned Data`) containing formatted data tables, conditional formatting, percentage calculations, and embedded high-resolution figures.
- **Model Performance Sheet (Sheet 11)**: Interactive confusion matrix, class-wise precision/recall/F1 metrics, and feature importance rankings.
- **Pivot-Ready Data Sheet (Sheet 12)**: Flat tabular repository with Excel AutoFilter enabled across all 23 attributes for ad-hoc user analysis.

#### 4. Recommendations: two practical recommendations justified by the findings
- **Recommendation 1 (Dynamic Fog Early Warning & Variable Speed Reduction)**:
  - *Empirical Justification*: Dense Fog elevates the fatal crash share to **8.10%** and the severe crash proportion to **45.48%** (compared to 3.18% fatal and 28.79% severe under Clear skies).
  - *Intervention*: Deploy automated visibility sensors linked to dynamic Variable Message Signs (VMS) along high-speed corridors (East Lake Road, Outer Loop - Sector A, and Highway Links). When visibility drops below 1.0 km, mandate an automated temporary speed limit reduction to 30 km/h, activate overhead hazard beacons, and illuminate high-intensity roadside fog delineators.
- **Recommendation 2 (Dedicated Motorcycle Lanes & Night Corridor Speed Enforcement)**:
  - *Empirical Justification*: Two-wheelers suffer a **5.06% fatal crash share** (35.71% severe share across 1,602 incidents), and nighttime hours exhibit the highest fatal crash proportion (**4.71%**), with fatal collisions averaging higher nominal speed limits (48.4 km/h vs. 44.3 km/h).
  - *Intervention*: Construct physical curb-separated motorcycle lanes along high-volume urban arterials (such as Central Market Junction and University Gate Road), and deploy automated speed-enforcement radar cameras paired with targeted police visibility patrols between 21:00 and 05:00 on East Lake Road and Highway Links.

---

### FINAL SUBMISSION

- **GitHub Repository Link**: https://github.com/Shankar-v27/DSA-SAT2.git
- **Overall Result / Conclusion**:
  - The CH15 road safety pipeline successfully cleaned, integrated, and evaluated 5,976 synthetic incident records across 5 multi-source tables, demonstrating that collision frequency hotspots (Central Market Junction) diverge sharply from high-lethality corridors (East Lake Road) and that dense fog and motorcycle transit significantly compound crash severity.
  - A Scikit-Learn Random Forest pipeline achieved 63.38% test accuracy with balanced cost-sensitive weighting on heavy class imbalance, delivering a fully operational 12-sheet Excel decision dashboard with native PivotTables and PivotCharts to support evidence-based municipal traffic engineering.
