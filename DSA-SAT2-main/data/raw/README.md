# CH15 – Road Safety Insight: Synthetic Dataset Pack

## Important
All records in this package are **synthetic and computer-generated for a classroom assessment**. They are not real accident records and must not be presented as real-world safety statistics. Location names are illustrative labels.

## Files and join key
Use `incident_id` as the common join key across all five files.

| File | Rows (raw) | Purpose | Main columns |
|---|---:|---|---|
| `accidents.csv` | 6,040 | Accident date, time, location and target label | `incident_id`, `date`, `time`, `location`, `severity` |
| `roads.csv` | 6,000 | Road context | `incident_id`, `road_type`, `junction_type`, `speed_limit_kmph` |
| `weather.csv` | 6,000 | Conditions at the incident | `incident_id`, `rainfall_mm`, `visibility_km`, `weather_condition` |
| `traffic.csv` | 6,000 | Approximate traffic volume at the incident | `incident_id`, `traffic_volume_vph`, `peak_flag` |
| `vehicles.csv` | 6,000 | Representative vehicle category and number involved | `incident_id`, `vehicle_category`, `vehicle_count` |

There are 6,000 generated unique incident IDs before the deliberate quality issues. `accidents.csv` contains 6,040 raw rows because duplicate incident rows have been intentionally inserted. The other four files have one row per unique incident, so deduplicate `accidents.csv` by `incident_id` before merging to avoid multiplying rows.

## Data dictionary
- `incident_id`: synthetic unique incident key, format `RSA######`.
- `date`: date in `YYYY-MM-DD` format (synthetic dates during 2024–2025).
- `time`: local time in `HH:MM` format.
- `location`: illustrative fictional urban location; some values are intentionally missing.
- `severity`: target class for classification: `Minor`, `Serious`, or `Fatal`; a small number of deliberately invalid values are included.
- `road_type`: road category; case and whitespace inconsistencies are intentionally included.
- `junction_type`: type of junction or road segment.
- `speed_limit_kmph`: nominal speed limit in km/h; a few invalid outliers are included for data-quality practice.
- `rainfall_mm`: synthetic rainfall reading in millimetres; blank values represent missing observations.
- `visibility_km`: synthetic visibility in kilometres; blank values represent missing observations.
- `weather_condition`: `Clear`, `Cloudy`, `Light Rain`, `Heavy Rain`, or `Fog`; missing observations are deliberately included.
- `traffic_volume_vph`: approximate synthetic traffic volume in vehicles per hour; a few extreme invalid values are included for outlier handling.
- `peak_flag`: `Peak` or `Off-Peak`, derived from the generated incident time.
- `vehicle_category`: representative category involved in the incident.
- `vehicle_count`: number of vehicles involved (synthetic integer count).

## Intentionally planted data-quality conditions
- Missing `location` values in `accidents.csv`: 151 rows.
- Duplicate incident rows in `accidents.csv`: 40 extra rows beyond unique IDs.
- Invalid `severity` entries in `accidents.csv`: 24 rows.
- Inconsistent road-type labels in `roads.csv`: 115 rows.
- Invalid speed-limit values in `roads.csv`: values include negative/zero values and implausibly high values (e.g. 999).
- Entirely missing weather-observation rows in `weather.csv`: 300 rows, plus scattered missing individual fields.
- Traffic-volume outliers in `traffic.csv`: values include negative/zero values and implausibly large readings (e.g. 99,999).

## Suggested analysis workflow
1. Load all five CSVs and inspect row counts, dtypes, missingness and duplicate incident IDs.
2. Remove duplicate incident IDs from `accidents.csv` and validate `severity` against `Minor`, `Serious`, and `Fatal`.
3. Impute or flag missing locations/weather measurements; standardize `road_type` by trimming whitespace and normalizing case.
4. Investigate speed-limit and traffic-volume outliers. Correct impossible values only when justified; otherwise treat as missing or exclude from affected analyses.
5. Merge the five files on `incident_id`. Verify the merge retains one row per incident.
6. Create features such as `hour`, `day_of_week`, `month`, and a severe-incident flag (e.g. Serious/Fatal), if needed.
7. Conduct EDA: location hotspots, time-of-day counts, weather vs severity, junction/road-type comparisons, and traffic-volume patterns.
8. For severity prediction, use a train/test split and evaluate Accuracy, macro/weighted F1-score, precision, recall, and a confusion matrix. Fit preprocessing only on training data to avoid leakage.

## Interpretation caution
This package includes **accident records only**, not exposure denominators such as vehicle-kilometres travelled, road length, or traffic observations for non-accident periods. Therefore, report counts and associations among the generated incident records; do not claim to estimate real-world accident risk rates or causation.
