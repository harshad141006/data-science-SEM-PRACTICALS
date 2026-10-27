# CH15 – Road Safety Insight: Understand Accidents. Discover Risk Patterns. Support Safer Roads.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/Assessment-Data%20Science%20Skill--Based-green.svg)]()

A comprehensive, production-grade Data Science and Machine Learning solution for road traffic accident analysis, hotspot detection, severity classification, and decision intelligence.

---

## 1. Project Overview & Problem Statement

Urban road networks face escalating risks of traffic collisions resulting in personal injury, fatalities, and economic loss. Effectively mitigating these hazards requires:
- Rigorous data hygiene and integration across heterogeneous sources (incidents, road network geometry, weather observations, traffic volumes, and vehicle types).
- Empirical discovery of collision hotspots and environmental risk factors.
- Supervised machine learning pipelines to classify injury severity and understand feature associations.
- Interactive decision intelligence tools for urban transport planners and traffic police.

This repository provides an end-to-end implementation covering data cleaning, exploratory data analysis, machine learning modeling with Scikit-learn, an interactive Streamlit dashboard with Plotly, an automated 11-sheet Excel reporting workbook, an executive PowerPoint presentation, and a fully executable Jupyter notebook.

---

## 2. Educational Synthetic Data Notice & Disclaimer

> **IMPORTANT DISCLAIMER:**  
> All incident records, locations, and measurements in this dataset are **computer-generated synthetic data** designed for educational skill assessments. They are **not real-world accident records** and must not be cited as actual municipal safety statistics. Location names (e.g., *Central Market Junction*, *East Lake Road*) are fictional urban labels.  
> 
> Furthermore, this dataset contains **accident incident records only** and does not provide baseline exposure denominators (e.g., total vehicle-kilometers travelled, lane-mile counts, or non-accident traffic volumes). Consequently, all findings indicate **relative crash associations and proportions within recorded collisions**, not absolute crash rates or real-world physical causation.

---

## 3. Dataset Architecture & Raw Profiles

The solution ingests five raw CSV files located in `data/raw/`:

| Dataset | Raw Rows | Key Columns | Purpose & Planted Quality Issues |
|---|---|---|---|
| `accidents.csv` | 6,040 | `incident_id`, `date`, `time`, `location`, `severity` | Primary incident table. Contains 40 duplicate incident records, 151 missing locations, and 24 invalid severity labels (`Unknown`, `Critical`, `0`, `5`). |
| `roads.csv` | 6,000 | `incident_id`, `road_type`, `junction_type`, `speed_limit_kmph` | Road infrastructure context. Contains 115 inconsistent road labels (whitespace/casing) and 18 invalid speed limits (negative, zero, 240–1000 km/h). |
| `weather.csv` | 6,000 | `incident_id`, `rainfall_mm`, `visibility_km`, `weather_condition` | Environmental conditions. Contains 300 entirely missing weather observation rows plus scattered missing rainfall/visibility readings. |
| `traffic.csv` | 6,000 | `incident_id`, `traffic_volume_vph`, `peak_flag` | Traffic flow conditions. Contains 24 extreme traffic volume outliers (negative, zero, and 60,000–130,000 vph). |
| `vehicles.csv` | 6,000 | `incident_id`, `vehicle_category`, `vehicle_count` | Vehicle involvement. 1 representative vehicle category per incident. |

---

## 4. Project Directory Structure

```text
v:/Project/DS SAT 2/
├── data/
│   ├── raw/                                  # Raw synthetic CSV files & profile
│   │   ├── accidents.csv
│   │   ├── dataset_profile.csv
│   │   ├── README.md
│   │   ├── roads.csv
│   │   ├── traffic.csv
│   │   ├── vehicles.csv
│   │   └── weather.csv
│   └── processed/                            # Cleaned data and audit artifacts
│       ├── cleaned_accidents.csv             # 5,976 cleaned integrated records
│       └── data_quality_report.csv           # Granular column quality audit
├── models/
│   └── road_safety_model.joblib              # Fitted Scikit-learn Pipeline artifact
├── notebooks/
│   └── road_safety_analysis.ipynb            # Fully executed 14-section notebook
├── outputs/
│   ├── charts/                               # 14 publication-quality 300-DPI charts
│   ├── business_insights_presentation.pptx   # 6-slide executive widescreen deck
│   ├── key_insights.json                     # Empirically derived findings
│   ├── model_metrics.json                    # Out-of-sample ML evaluation metrics
│   └── road_safety_analysis.xlsx             # 11-sheet formatted Excel workbook
├── src/
│   ├── __init__.py
│   ├── build_notebook.py                     # Automated notebook generator & executor
│   ├── data_processing.py                    # Quality assessment & cleaning pipeline
│   ├── eda.py                                # Visual analytics & insight generator
│   ├── excel_reporting.py                    # 11-sheet Excel report generator
│   ├── feature_engineering.py                # Leak-free feature transformer
│   ├── generate_presentation.py              # PowerPoint presentation generator
│   ├── model_training.py                     # ML pipeline training & evaluation
│   └── utils.py                              # Path management & JSON serialization
├── app.py                                    # Streamlit interactive dashboard
├── requirements.txt                          # Pinned dependency requirements
├── README.md                                 # Master technical documentation
└── .gitignore                                # Version control exclusions
```

---

## 5. Data Cleaning & Integration Pipeline

The reproducible data processing pipeline (`src/data_processing.py`) performs the following operations:

1. **Deduplication**: Identifies and removes 40 duplicate incident records in `accidents.csv` based on `incident_id`, yielding 6,000 unique incident records.
2. **Target Label Validation**: Validates `severity` against valid classes (`Minor`, `Serious`, `Fatal`). Filters 24 invalid records (`Unknown`: 8, `Critical`: 6, `0`: 5, `5`: 5) to establish a clean modeling cohort of 5,976 records.
3. **Location Preservation**: Imputes 149 missing locations in the cleaned cohort with `'Unknown Location'`, preventing arbitrary row deletion.
4. **Categorical Normalization**: Trims whitespace and normalizes casing across 115 inconsistent `road_type` values, reducing them to standard categories (*Urban Arterial*, *Collector Road*, *Service Road*, *Highway Link*, *Residential Street*).
5. **Speed Limit Imputation**: Identifies 18 speed limit outliers (`<= 0` or `> 120` km/h) and imputes them with the median speed limit of the corresponding road classification.
6. **Weather Imputation**: Resolves 300 blank weather rows using mode imputation (`'Clear'`) and condition-specific medians for `rainfall_mm` and `visibility_km`.
7. **Traffic Volume Correction**: Imputes 24 invalid traffic volume outliers (`<= 0` or `> 10,000` vph) using medians grouped by `road_type` and `peak_flag`.
8. **Cardinality Verification**: Executes 1-to-1 inner joins on `incident_id` across all five datasets, confirming exactly 5,976 records with zero join multiplication.

---

## 6. Key Empirical Findings (EDA Highlights)

Derived automatically in `outputs/key_insights.json` and visualized in `outputs/charts/`:

1. **Class Imbalance**: Across 5,976 cleaned records, **Minor** accidents represent 68.61% (4,100), **Serious** 27.53% (1,645), and **Fatal-severity crashes** 3.87% (231). Severe accidents (Serious + Fatal) make up 31.39% (1,876).
2. **Corridor Divergence**: **Central Market Junction** recorded the highest overall crash volume (445 total incidents, 147 severe). Conversely, **East Lake Road** exhibited the highest fatal crash count (24 fatal-severity crashes, 7.50% fatal crash share of recorded accidents across 320 crashes), followed by **Outer Loop - Sector A** (19 fatal crashes, 6.93%).
3. **Temporal Peak vs Night Fatal Share**: Crash frequency peaks sharply during commuter rush hours: 18:00 (608 incidents) and 08:00 (547 incidents). However, **Night hours (21:00–04:59)**, while accounting for only 9.6% of accidents (573 crashes), recorded the highest fatal crash proportion at **4.71%**, nearly double the Afternoon share (2.70%).
4. **Adverse Weather Risk**: In dense **Fog** conditions, the fatal crash share surged to **8.10%** and the combined severe crash share reached **45.48%** (compared to 28.62% under Clear skies).
5. **Highway Links & Speed**: **Highway Links** exhibited a 6.03% fatal crash share (vs 2.68% on Collector Roads). Fatal crashes occurred on road segments with a mean nominal speed limit of 48.4 km/h (vs 44.3 km/h for minor crashes on the cleaned dataset).
6. **Vulnerable Road Users**: **Motorcycles** recorded a 5.06% fatal crash share and 30.65% serious injury share (35.71% total severe), the highest among motorized vehicle categories.
7. **Exposure-Adjusted Risk Distinction**: All percentages indicate proportions among recorded accident incidents, distinct from per-vehicle or per-km exposure risk.

---

## 7. Machine Learning Methodology & Evaluation

The supervised classification model predicts accident severity (`Minor`, `Serious`, `Fatal`) using Scikit-learn:

### Pipeline Design
- **Stratified Partitioning**: 80% training set ($N=4,780$), 20% holdout test set ($N=1,196$) with `random_state=42`, stratified on `severity`.
- **Leakage Prevention**: All preprocessing (`ColumnTransformer`) is fitted strictly on the training partition.
- **Numerical Pipeline**: `SimpleImputer(strategy='median')` + `StandardScaler()`.
- **Categorical Pipeline**: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Classifier**: `RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=2, class_weight='balanced', random_state=42)`. Balanced class weighting prevents mode collapse into the dominant Minor class.

### Out-of-Sample Test Metrics (from single unified run in `outputs/model_metrics.json`)

| Metric | Score | Interpretation |
|---|---|---|
| **Overall Accuracy** | **63.38%** | Global classification accuracy on untouched test data |
| **Macro F1-Score** | **0.3670** | Unweighted mean F1-score across all 3 classes |
| **Weighted F1-Score** | **0.6027** | Support-weighted average F1-score |
| **Minor F1-Score** | **0.7643** | Precision: 0.7086 \| Recall: 0.8295 (Support: 821) |
| **Serious F1-Score** | **0.2752** | Precision: 0.3472 \| Recall: 0.2280 (Support: 329) |
| **Fatal F1-Score** | **0.0615** | Precision: 0.1053 \| Recall: 0.0435 (Support: 46) |

### Confusion Matrix & Automated Validation Checks
- **Matrix Layout** (`Minor`, `Serious`, `Fatal`):
  * True Minor: `[681, 132, 8]` (Row sum = 821, matches true Minor support)
  * True Serious: `[245, 75, 9]` (Row sum = 329, matches true Serious support)
  * True Fatal: `[35, 9, 2]` (Row sum = 46, matches true Fatal support)
- **Column Sums (Predicted Counts)**:
  * Predicted Minor: $681 + 245 + 35 = 961$
  * Predicted Serious: $132 + 75 + 9 = 216$
  * Predicted Fatal: $8 + 9 + 2 = 19$
- **Automated Validation**: Automated assertion checks verify that row sums equal true test supports, column sums equal predicted counts, the total sum equals 1,196, and the diagonal trace matches accuracy (63.38%).

### Honest Assessment of Weak Minority Performance
- **Class Imbalance & Rarity**: Fatal crashes constitute only 3.85% of test instances (46 cases). Without vehicle kinematics, crash angle, or occupant telematics, separating rare fatal events from serious or minor events using synthetic environmental features remains challenging.
- **Model Associations vs Causality**: Feature importance (led by visibility, traffic volume, speed limit, rainfall, and hour) reflects statistical splits within decision trees, not real-world crash causality.

---

## 8. Interactive Road Safety Dashboard (`app.py`)

The Streamlit dashboard provides an intuitive, responsive analytics portal with Plotly visualizations:

### Dashboard Tabs
1. **Executive Overview**: Real-time KPI scorecards, severity proportion donut chart, top hotspot volume ranking, and empirical findings expander.
2. **Hotspot Analysis**: Sortable corridor leaderboard with conditional gradients and stacked severity bar charts.
3. **Temporal Trends**: 24-hour accident curves with commuter peak annotations, day-of-week volume, and peak vs off-peak comparisons.
4. **Weather & Infrastructure**: Severity distribution by weather condition, road classification cross-tabs, junction geometries, and speed limit box plots.
5. **Traffic & Vehicles**: Primary vehicle vulnerability breakdowns and traffic volume histograms.
6. **ML Severity Predictor**: Live scenario simulation tool. Users specify corridor, road type, junction, weather, speed limit, time of day, and vehicle details to obtain predicted severity and class probabilities.

### How to Launch Dashboard
```bash
streamlit run app.py
```
Open browser at `http://localhost:8501`.

---

## 9. Excel Analysis Workbook (`outputs/road_safety_analysis.xlsx`)

The generated multi-sheet Excel workbook contains 11 dedicated worksheets:

1. **Summary Dashboard**: Headline KPI blocks (`COUNTA`, `COUNTIF`, formulas), key findings, and embedded Column Chart.
2. **Cleaned Data**: Full 5,976 cleaned records with frozen header panes, auto-filters, and formatted numbers.
3. **Data Quality**: Ingestion audit summary and granular column-by-column quality report.
4. **Accident Hotspots**: Corridor summary table with Excel formulas and embedded horizontal Bar Chart.
5. **Time Analysis**: 24-hour table, time period summary, and embedded Line Chart.
6. **Weather Analysis**: Weather condition metrics, mean rainfall/visibility, and embedded Column Chart.
7. **Junction Analysis**: Junction type volume and severe rates.
8. **Road Type Analysis**: Road type crash volume, fatal crash shares, and embedded Column Chart.
9. **Traffic Analysis**: Flow volume statistics across road types and peak flags.
10. **Model Performance**: Classification benchmarks, class-wise metrics, confusion matrix, and feature importances.
11. **Pivot-Ready Data**: Structured flat table formatted for Excel Pivot Tables, accompanied by clear user instructions.

> **PivotTable Architecture Disclosure:**  
> Worksheets 4 through 9 are pre-calculated grouped summary tables and standard Excel charts generated using Python. Worksheet 11 (`Pivot-Ready Data`) provides the full flat structured dataset alongside step-by-step instructions for constructing native, interactive Microsoft Excel PivotTables and PivotCharts.

---

## 10. Executive Presentation (`outputs/business_insights_presentation.pptx`)

A widescreen (16:9) 6-slide PowerPoint presentation formatted with a corporate palette:
- **Slide 1**: Executive Title & Scope
- **Slide 2**: Data Ingestion, Quality Assessment & Cleaning Pipeline
- **Slide 3**: Corridor Hotspot Analysis & Spatial Concentration
- **Slide 4**: Multi-Dimensional Risk Factors (Temporal, Weather, Modes)
- **Slide 5**: Machine Learning Severity Classifier, Evaluation & Limitations
- **Slide 6**: Evidence-Based Strategic Recommendations & Roadmap

---

## 11. How to Reproduce the Full Project

### Prerequisites
- Python 3.10+
- Virtual environment recommended

### Installation
```bash
# Clone or navigate to project directory
cd "v:/Project/DS SAT 2"

# Install dependencies
pip install -r requirements.txt
```

### Execution Steps
```bash
# 1. Run Data Quality Assessment & Cleaning Pipeline
python -m src.data_processing

# 2. Run Exploratory Data Analysis & Generate Charts
python -m src.eda

# 3. Train & Evaluate ML Model (Outputs Model Artifact & Metrics)
python -m src.model_training

# 4. Generate 11-Sheet Excel Workbook
python -m src.excel_reporting

# 5. Generate Executive PowerPoint Presentation
python -m src.generate_presentation

# 6. Build and Execute Jupyter Notebook
python -m src.build_notebook

# 7. Launch Interactive Dashboard
streamlit run app.py
```

---

## 12. Strategic Recommendations for Safer Roads

1. **Corridor-Specific Engineering Audits**: Prioritize urgent geometric, lighting, and barrier inspections at high-fatality corridors (**East Lake Road** and **Outer Loop Sector A**), distinguishing them from high-volume minor collision sites (**Central Market Junction**).
2. **Night-Time Enforcement & Illumination**: Enhance speed camera enforcement and street illumination between 21:00 and 04:00 to address the elevated 4.71% night fatality rate.
3. **Dynamic Fog Warning Systems**: Deploy Variable Message Signs (VMS) on Highway Links and Flyovers to enforce reduced speed limits during dense Fog and heavy rain.
4. **Vulnerable Road User Segregation**: Introduce physical lane separation for motorcycles and bicycles along high-speed corridors.
5. **Exposure Denominator Collection**: Partner with municipal traffic management centers to collect baseline non-crash traffic flows and vehicle-kilometers travelled (VKT) for comprehensive exposure-adjusted risk modeling.

---
*CH15 Road Safety Insight Project • Data Science Assessment Deliverable*
