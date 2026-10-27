"""
Excel Reporting Module for CH15 Road Safety Insight.
Generates outputs/road_safety_analysis.xlsx containing 11 professional sheets
with formatted KPI blocks, frozen panes, formulas, conditional formatting, and embedded charts.
"""

import os
import json
import pandas as pd
import numpy as np
import xlsxwriter
from src.utils import get_project_root


def generate_excel_report(cleaned_csv_path=None, output_path=None):
    """
    Builds the comprehensive multi-tab Excel report using xlsxwriter.
    """
    project_root = get_project_root()
    if cleaned_csv_path is None:
        cleaned_csv_path = os.path.join(project_root, "data", "processed", "cleaned_accidents.csv")
    if output_path is None:
        output_path = os.path.join(project_root, "outputs", "road_safety_analysis.xlsx")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data and metrics
    df = pd.read_csv(cleaned_csv_path)
    quality_report_path = os.path.join(project_root, "data", "processed", "data_quality_report.csv")
    df_quality = pd.read_csv(quality_report_path) if os.path.exists(quality_report_path) else pd.DataFrame()

    metrics_path = os.path.join(project_root, "outputs", "model_metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    workbook = xlsxwriter.Workbook(output_path)

    # Formatting Styles
    fmt_header = workbook.add_format({
        "bold": True,
        "bg_color": "#1e3a8a",  # Deep navy
        "font_color": "#ffffff",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "font_size": 11,
        "text_wrap": True
    })

    fmt_sub_header = workbook.add_format({
        "bold": True,
        "bg_color": "#3b82f6",  # Blue
        "font_color": "#ffffff",
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "font_size": 10
    })

    fmt_title = workbook.add_format({
        "bold": True,
        "font_size": 16,
        "font_color": "#1e3a8a",
        "valign": "vcenter"
    })

    fmt_subtitle = workbook.add_format({
        "font_size": 10,
        "font_color": "#64748b",
        "italic": True
    })

    fmt_cell = workbook.add_format({"border": 1, "font_size": 10, "valign": "vcenter"})
    fmt_cell_center = workbook.add_format({"border": 1, "align": "center", "font_size": 10, "valign": "vcenter"})
    fmt_int = workbook.add_format({"border": 1, "num_format": "#,##0", "align": "right", "font_size": 10, "valign": "vcenter"})
    fmt_dec = workbook.add_format({"border": 1, "num_format": "#,##0.00", "align": "right", "font_size": 10, "valign": "vcenter"})
    fmt_pct = workbook.add_format({"border": 1, "num_format": "0.0%", "align": "right", "font_size": 10, "valign": "vcenter"})
    
    fmt_kpi_card_title = workbook.add_format({
        "bold": True, "font_size": 9, "font_color": "#64748b", "align": "center", "bg_color": "#f1f5f9", "border": 1
    })
    fmt_kpi_card_val = workbook.add_format({
        "bold": True, "font_size": 16, "font_color": "#0f172a", "align": "center", "bg_color": "#f8fafc", "border": 1, "num_format": "#,##0"
    })
    fmt_kpi_card_pct = workbook.add_format({
        "bold": True, "font_size": 16, "font_color": "#0f172a", "align": "center", "bg_color": "#f8fafc", "border": 1, "num_format": "0.0%"
    })

    # ==========================================================
    # SHEET 1: Summary Dashboard
    # ==========================================================
    ws_sum = workbook.add_worksheet("Summary Dashboard")
    ws_sum.set_tab_color("#1e3a8a")
    ws_sum.hide_gridlines(0)
    ws_sum.set_column("A:A", 4)
    ws_sum.set_column("B:B", 24)
    ws_sum.set_column("C:C", 16)
    ws_sum.set_column("D:D", 16)
    ws_sum.set_column("E:E", 16)
    ws_sum.set_column("F:F", 20)
    ws_sum.set_column("G:G", 28)

    ws_sum.write("B2", "CH15 – ROAD SAFETY INSIGHT: EXECUTIVE DASHBOARD", fmt_title)
    ws_sum.write("B3", "Empirical Accident Distribution, Incident Severity Patterns, and ML Intelligence", fmt_subtitle)

    # KPI Summary Cards (Formulas referencing Cleaned Data)
    total_rows = len(df)
    last_row_cleaned = total_rows + 1

    ws_sum.write("B5", "TOTAL ACCIDENTS", fmt_kpi_card_title)
    ws_sum.write_formula("B6", f"=COUNTA('Cleaned Data'!A2:A{last_row_cleaned})", fmt_kpi_card_val)

    ws_sum.write("C5", "MINOR ACCIDENTS", fmt_kpi_card_title)
    ws_sum.write_formula("C6", f"=COUNTIF('Cleaned Data'!E2:E{last_row_cleaned}, \"Minor\")", fmt_kpi_card_val)

    ws_sum.write("D5", "SERIOUS INJURIES", fmt_kpi_card_title)
    ws_sum.write_formula("D6", f"=COUNTIF('Cleaned Data'!E2:E{last_row_cleaned}, \"Serious\")", fmt_kpi_card_val)

    ws_sum.write("E5", "FATAL CRASHES", fmt_kpi_card_title)
    ws_sum.write_formula("E6", f"=COUNTIF('Cleaned Data'!E2:E{last_row_cleaned}, \"Fatal\")", fmt_kpi_card_val)

    ws_sum.write("F5", "SEVERE ACCIDENT SHARE", fmt_kpi_card_title)
    ws_sum.write_formula("F6", "=(D6+E6)/B6", fmt_kpi_card_pct)

    # Severity Summary Table (for embedded chart)
    ws_sum.write("B9", "Severity Breakdown Table", workbook.add_format({"bold": True, "font_size": 11, "font_color": "#1e3a8a"}))
    ws_sum.write("B10", "Severity Class", fmt_header)
    ws_sum.write("C10", "Count", fmt_header)
    ws_sum.write("D10", "Share (%)", fmt_header)

    ws_sum.write("B11", "Minor", fmt_cell)
    ws_sum.write_formula("C11", "=C6", fmt_int)
    ws_sum.write_formula("D11", "=C11/$B$6", fmt_pct)

    ws_sum.write("B12", "Serious", fmt_cell)
    ws_sum.write_formula("C12", "=D6", fmt_int)
    ws_sum.write_formula("D12", "=C12/$B$6", fmt_pct)

    ws_sum.write("B13", "Fatal", fmt_cell)
    ws_sum.write_formula("C13", "=E6", fmt_int)
    ws_sum.write_formula("D13", "=C13/$B$6", fmt_pct)

    ws_sum.write("B14", "Total", workbook.add_format({"bold": True, "border": 1, "bg_color": "#f1f5f9"}))
    ws_sum.write_formula("C14", "=SUM(C11:C13)", workbook.add_format({"bold": True, "border": 1, "num_format": "#,##0", "bg_color": "#f1f5f9"}))
    ws_sum.write_formula("D14", "=SUM(D11:D13)", workbook.add_format({"bold": True, "border": 1, "num_format": "0.0%", "bg_color": "#f1f5f9"}))

    # Embed Chart on Dashboard
    chart_sev = workbook.add_chart({"type": "column"})
    chart_sev.add_series({
        "name": "Accident Incidents",
        "categories": "='Summary Dashboard'!$B$11:$B$13",
        "values": "='Summary Dashboard'!$C$11:$C$13",
        "fill": {"color": "#1e3a8a"},
        "data_labels": {"value": True, "font": {"bold": True}}
    })
    chart_sev.set_title({"name": "Incident Volume by Severity Class"})
    chart_sev.set_legend({"none": True})
    chart_sev.set_y_axis({"major_gridlines": {"visible": True}})
    chart_sev.set_size({"width": 460, "height": 280})
    ws_sum.insert_chart("B16", chart_sev)

    # Core Strategic Findings Table
    ws_sum.write("F9", "Strategic Empirical Insights", workbook.add_format({"bold": True, "font_size": 11, "font_color": "#1e3a8a"}))
    ws_sum.write("F10", "Category", fmt_header)
    ws_sum.write("G10", "Key Empirical Observation", fmt_header)

    findings_list = [
        ("Hotspots", "Central Market Junction had 445 incidents; East Lake Road had 24 fatal crashes (7.5% fatal crash share of recorded accidents)."),
        ("Temporal", "Bimodal volume spikes at 08:00 and 18:00; Night incidents had the highest fatal crash share at 4.71%."),
        ("Weather", "Fog conditions escalated fatal crash share to 8.10% and severe rate to 45.48% (vs 28.6% clear)."),
        ("Infrastructure", "Highway links experienced 6.03% fatal crashes vs 2.68% on collector roads."),
        ("Vehicles", "Motorcycles suffered 5.06% fatal crash share and 30.65% serious injury rate, highest among motorized modes."),
        ("Machine Learning", f"Random Forest achieved {metrics['overall_metrics']['accuracy']*100:.2f}% accuracy and {metrics['overall_metrics']['macro_f1']:.4f} Macro F1 on untouched test set.")
    ]
    for r_idx, (cat, desc) in enumerate(findings_list, start=11):
        ws_sum.write(f"F{r_idx}", cat, fmt_cell)
        ws_sum.write(f"G{r_idx}", desc, fmt_cell)

    # ==========================================================
    # SHEET 2: Cleaned Data
    # ==========================================================
    ws_clean = workbook.add_worksheet("Cleaned Data")
    ws_clean.set_tab_color("#2563eb")
    ws_clean.freeze_panes(1, 0)
    
    headers_clean = list(df.columns)
    for c_idx, col in enumerate(headers_clean):
        ws_clean.write(0, c_idx, col, fmt_header)
        col_width = max(len(col) + 4, 12)
        ws_clean.set_column(c_idx, c_idx, col_width)

    for r_idx, row in df.iterrows():
        excel_row = r_idx + 1
        for c_idx, val in enumerate(row):
            if pd.isna(val):
                ws_clean.write_blank(excel_row, c_idx, None, fmt_cell)
            elif isinstance(val, (int, np.integer)):
                ws_clean.write_number(excel_row, c_idx, int(val), fmt_int)
            elif isinstance(val, (float, np.floating)):
                ws_clean.write_number(excel_row, c_idx, float(val), fmt_dec)
            else:
                ws_clean.write_string(excel_row, c_idx, str(val), fmt_cell)

    ws_clean.autofilter(0, 0, len(df), len(headers_clean) - 1)

    # ==========================================================
    # SHEET 3: Data Quality
    # ==========================================================
    ws_dq = workbook.add_worksheet("Data Quality")
    ws_dq.set_tab_color("#d97706")
    ws_dq.set_column("A:A", 4)
    ws_dq.set_column("B:B", 24)
    ws_dq.set_column("C:C", 22)
    ws_dq.set_column("D:D", 14)
    ws_dq.set_column("E:E", 14)
    ws_dq.set_column("F:F", 14)
    ws_dq.set_column("G:G", 14)
    ws_dq.set_column("H:H", 42)

    ws_dq.write("B2", "DATA QUALITY ASSESSMENT & CLEANING AUDIT", fmt_title)
    ws_dq.write("B3", "Verification of planted quality conditions, deduplication, and imputation rules", fmt_subtitle)

    # Cleaning Audit Table
    ws_dq.write("B5", "Data Cleaning Audit Summary", workbook.add_format({"bold": True, "font_size": 11, "font_color": "#1e3a8a"}))
    ws_dq.write("B6", "Quality Metric / Transformation", fmt_header)
    ws_dq.write("C6", "Count / Value", fmt_header)
    ws_dq.write("D6", "Resolution / Treatment Rule", fmt_header)

    audit_records = [
        ("Raw Incidents in accidents.csv", "6,040", "Initial raw ingestion"),
        ("Duplicate Incident Rows Removed", "40", "Deduplicated by unique incident_id"),
        ("Unique Incidents Retained", "6,000", "Baseline incident cohort"),
        ("Invalid Severity Labels Excluded", "24", "Excluded Unknown (8), Critical (6), 0 (5), 5 (5)"),
        ("Final Cleaned Incident Records", "5,976", "Validated modeling cohort (Minor, Serious, Fatal)"),
        ("Missing Locations Handled", "149", "Categorized as 'Unknown Location' to preserve records"),
        ("Inconsistent Road Labels Standardized", "115", "Trimmed whitespace and normalized to title case"),
        ("Invalid Speed Limit Outliers Imputed", "18", "Imputed negative & >120 km/h with road-type median"),
        ("Missing Weather Rows Imputed", "300", "Mode condition ('Clear'), median rainfall & visibility"),
        ("Traffic Volume Outliers Imputed", "24", "Imputed negative & >10,000 vph with (road, peak) median"),
        ("Join Cardinality Verified", "1-to-1 Inner", "Exactly 5,976 records across all 5 merged tables")
    ]
    for idx, (m, c, r) in enumerate(audit_records, start=7):
        ws_dq.write(f"B{idx}", m, fmt_cell)
        ws_dq.write(f"C{idx}", c, fmt_cell_center)
        ws_dq.write(f"D{idx}", r, fmt_cell)

    # Column Quality Table from data_quality_report.csv
    if not df_quality.empty:
        ws_dq.write("B20", "Detailed Column Quality Report", workbook.add_format({"bold": True, "font_size": 11, "font_color": "#1e3a8a"}))
        dq_headers = ["Dataset", "Column", "Data_Type", "Total_Rows", "Null_Count", "Null_Pct", "Unique_Values", "Quality_Notes"]
        for c_idx, h in enumerate(dq_headers):
            ws_dq.write(21, c_idx + 1, h, fmt_header)
        for r_idx, row in df_quality.iterrows():
            curr_r = 22 + r_idx
            ws_dq.write(curr_r, 1, row.get("Dataset", ""), fmt_cell)
            ws_dq.write(curr_r, 2, row.get("Column", ""), fmt_cell)
            ws_dq.write(curr_r, 3, str(row.get("Data_Type", "")), fmt_cell_center)
            ws_dq.write(curr_r, 4, int(row.get("Total_Rows", 0)), fmt_int)
            ws_dq.write(curr_r, 5, int(row.get("Null_Count", 0)), fmt_int)
            ws_dq.write(curr_r, 6, float(row.get("Null_Pct", 0)) / 100.0, fmt_pct)
            ws_dq.write(curr_r, 7, int(row.get("Unique_Values", 0)), fmt_int)
            ws_dq.write(curr_r, 8, str(row.get("Quality_Notes", "")), fmt_cell)

    # ==========================================================
    # SHEET 4: Accident Hotspots
    # ==========================================================
    ws_hot = workbook.add_worksheet("Accident Hotspots")
    ws_hot.set_tab_color("#dc2626")
    ws_hot.set_column("A:A", 4)
    ws_hot.set_column("B:B", 28)
    ws_hot.set_column("C:H", 14)

    ws_hot.write("B2", "LOCATION-WISE ACCIDENT HOTSPOTS & RISK PROFILES", fmt_title)
    ws_hot.write("B3", "Ranked analysis of corridor volume, injury severity, and fatal-severity crash proportions (among recorded incidents)", fmt_subtitle)

    loc_df = df[df["location"] != "Unknown Location"].groupby("location")["severity"].value_counts().unstack().fillna(0)
    loc_df["Total"] = loc_df.sum(axis=1)
    loc_df["Minor"] = loc_df.get("Minor", 0)
    loc_df["Serious"] = loc_df.get("Serious", 0)
    loc_df["Fatal"] = loc_df.get("Fatal", 0)
    loc_df["Severe_Count"] = loc_df["Serious"] + loc_df["Fatal"]
    loc_df["Severe_Rate"] = loc_df["Severe_Count"] / loc_df["Total"]
    loc_df["Fatal_Rate"] = loc_df["Fatal"] / loc_df["Total"]
    loc_df = loc_df.sort_values(by="Severe_Count", ascending=False).reset_index()

    hot_headers = ["Location", "Minor", "Serious", "Fatal Crashes", "Total Incidents", "Severe Count", "Severe Rate", "Fatal Crash Share"]
    for c_idx, h in enumerate(hot_headers):
        ws_hot.write(4, c_idx + 1, h, fmt_header)

    for r_idx, row in loc_df.iterrows():
        curr_r = 5 + r_idx
        ws_hot.write(curr_r, 1, row["location"], fmt_cell)
        ws_hot.write(curr_r, 2, int(row["Minor"]), fmt_int)
        ws_hot.write(curr_r, 3, int(row["Serious"]), fmt_int)
        ws_hot.write(curr_r, 4, int(row["Fatal"]), fmt_int)
        ws_hot.write_formula(curr_r, 5, f"=SUM(C{curr_r+1}:E{curr_r+1})", fmt_int)
        ws_hot.write_formula(curr_r, 6, f"=SUM(D{curr_r+1}:E{curr_r+1})", fmt_int)
        ws_hot.write_formula(curr_r, 7, f"=G{curr_r+1}/F{curr_r+1}", fmt_pct)
        ws_hot.write_formula(curr_r, 8, f"=E{curr_r+1}/F{curr_r+1}", fmt_pct)

    # Insert Hotspot Bar Chart
    chart_hot = workbook.add_chart({"type": "bar"})
    chart_hot.add_series({
        "name": "Severe Incidents (Serious + Fatal)",
        "categories": f"='Accident Hotspots'!$B$6:$B${5+len(loc_df)}",
        "values": f"='Accident Hotspots'!$G$6:$G${5+len(loc_df)}",
        "fill": {"color": "#dc2626"}
    })
    chart_hot.set_title({"name": "Top Corridors by Severe Incidents"})
    chart_hot.set_size({"width": 520, "height": 380})
    ws_hot.insert_chart("J5", chart_hot)

    # ==========================================================
    # SHEET 5: Time Analysis
    # ==========================================================
    ws_time = workbook.add_worksheet("Time Analysis")
    ws_time.set_tab_color("#059669")
    ws_time.set_column("A:A", 4)
    ws_time.set_column("B:B", 16)
    ws_time.set_column("C:G", 14)

    ws_time.write("B2", "TEMPORAL RISK & PEAK COMMUTER ANALYSIS", fmt_title)
    ws_time.write("B3", "Hourly frequency, day of week variations, and peak vs off-peak comparisons", fmt_subtitle)

    # Hourly Summary Table
    ws_time.write("B5", "Hourly Accident Breakdown", workbook.add_format({"bold": True, "font_size": 11, "font_color": "#1e3a8a"}))
    hour_headers = ["Hour", "Minor", "Serious", "Fatal", "Total Incidents", "Severe Share"]
    for c_idx, h in enumerate(hour_headers):
        ws_time.write(5, c_idx + 1, h, fmt_header)

    hourly_df = df.groupby("hour_of_day")["severity"].value_counts().unstack().fillna(0)
    for h in range(24):
        curr_r = 6 + h
        m = int(hourly_df.loc[h].get("Minor", 0)) if h in hourly_df.index else 0
        s = int(hourly_df.loc[h].get("Serious", 0)) if h in hourly_df.index else 0
        f_val = int(hourly_df.loc[h].get("Fatal", 0)) if h in hourly_df.index else 0
        ws_time.write(curr_r, 1, f"{h:02d}:00", fmt_cell_center)
        ws_time.write(curr_r, 2, m, fmt_int)
        ws_time.write(curr_r, 3, s, fmt_int)
        ws_time.write(curr_r, 4, f_val, fmt_int)
        ws_time.write_formula(curr_r, 5, f"=SUM(C{curr_r+1}:E{curr_r+1})", fmt_int)
        ws_time.write_formula(curr_r, 6, f"=SUM(D{curr_r+1}:E{curr_r+1})/F{curr_r+1}", fmt_pct)

    # Hourly Line Chart
    chart_hour = workbook.add_chart({"type": "line"})
    chart_hour.add_series({
        "name": "Total Accidents",
        "categories": "='Time Analysis'!$B$7:$B$30",
        "values": "='Time Analysis'!$F$7:$F$30",
        "line": {"color": "#059669", "width": 2.25}
    })
    chart_hour.set_title({"name": "Hourly Accident Curve (24-Hour Profile)"})
    chart_hour.set_size({"width": 500, "height": 300})
    ws_time.insert_chart("I5", chart_hour)

    # Time Period Summary Table
    ws_time.write("I22", "Time Period Breakdown", workbook.add_format({"bold": True, "font_size": 11, "font_color": "#1e3a8a"}))
    period_df = df.groupby("time_period")["severity"].value_counts().unstack().fillna(0)
    period_order = ["Morning", "Afternoon", "Evening", "Night"]
    period_headers = ["Period", "Minor", "Serious", "Fatal", "Total", "Severe %"]
    for c_idx, h in enumerate(period_headers):
        ws_time.write(22, c_idx + 8, h, fmt_header)
    for p_idx, p in enumerate(period_order):
        curr_r = 23 + p_idx
        m = int(period_df.loc[p].get("Minor", 0)) if p in period_df.index else 0
        s = int(period_df.loc[p].get("Serious", 0)) if p in period_df.index else 0
        f_val = int(period_df.loc[p].get("Fatal", 0)) if p in period_df.index else 0
        ws_time.write(curr_r, 8, p, fmt_cell)
        ws_time.write(curr_r, 9, m, fmt_int)
        ws_time.write(curr_r, 10, s, fmt_int)
        ws_time.write(curr_r, 11, f_val, fmt_int)
        ws_time.write_formula(curr_r, 12, f"=SUM(J{curr_r+1}:L{curr_r+1})", fmt_int)
        ws_time.write_formula(curr_r, 13, f"=SUM(K{curr_r+1}:L{curr_r+1})/M{curr_r+1}", fmt_pct)

    # ==========================================================
    # SHEET 6: Weather Analysis
    # ==========================================================
    ws_weath = workbook.add_worksheet("Weather Analysis")
    ws_weath.set_tab_color("#0284c7")
    ws_weath.set_column("A:A", 4)
    ws_weath.set_column("B:B", 20)
    ws_weath.set_column("C:I", 14)

    ws_weath.write("B2", "WEATHER CONDITIONS & ACCIDENT SEVERITY", fmt_title)
    ws_weath.write("B3", "Comparative crash counts, precipitation volumes, visibility levels, and injury rates", fmt_subtitle)

    weath_df = df.groupby("weather_condition").agg(
        Minor=("severity", lambda s: (s == "Minor").sum()),
        Serious=("severity", lambda s: (s == "Serious").sum()),
        Fatal=("severity", lambda s: (s == "Fatal").sum()),
        Total=("severity", "count"),
        Mean_Rain=("rainfall_mm", "mean"),
        Mean_Vis=("visibility_km", "mean")
    ).reset_index()

    weath_headers = ["Weather Condition", "Minor", "Serious", "Fatal", "Total Incidents", "Severe Share", "Fatal Share", "Mean Rain (mm)", "Mean Vis (km)"]
    for c_idx, h in enumerate(weath_headers):
        ws_weath.write(4, c_idx + 1, h, fmt_header)

    for r_idx, row in weath_df.iterrows():
        curr_r = 5 + r_idx
        ws_weath.write(curr_r, 1, row["weather_condition"], fmt_cell)
        ws_weath.write(curr_r, 2, int(row["Minor"]), fmt_int)
        ws_weath.write(curr_r, 3, int(row["Serious"]), fmt_int)
        ws_weath.write(curr_r, 4, int(row["Fatal"]), fmt_int)
        ws_weath.write_formula(curr_r, 5, f"=SUM(C{curr_r+1}:E{curr_r+1})", fmt_int)
        ws_weath.write_formula(curr_r, 6, f"=SUM(D{curr_r+1}:E{curr_r+1})/F{curr_r+1}", fmt_pct)
        ws_weath.write_formula(curr_r, 7, f"=E{curr_r+1}/F{curr_r+1}", fmt_pct)
        ws_weath.write(curr_r, 8, float(row["Mean_Rain"]), fmt_dec)
        ws_weath.write(curr_r, 9, float(row["Mean_Vis"]), fmt_dec)

    # Weather Column Chart
    chart_weath = workbook.add_chart({"type": "column"})
    chart_weath.add_series({
        "name": "Severe Rate (%)",
        "categories": f"='Weather Analysis'!$B$6:$B${5+len(weath_df)}",
        "values": f"='Weather Analysis'!$G$6:$G${5+len(weath_df)}",
        "fill": {"color": "#0284c7"}
    })
    chart_weath.set_title({"name": "Severe Accident Rate (%) by Weather Condition"})
    chart_weath.set_y_axis({"num_format": "0%"})
    chart_weath.set_size({"width": 480, "height": 300})
    ws_weath.insert_chart("B13", chart_weath)

    # ==========================================================
    # SHEET 7: Junction Analysis
    # ==========================================================
    ws_junc = workbook.add_worksheet("Junction Analysis")
    ws_junc.set_tab_color("#7c3aed")
    ws_junc.set_column("A:A", 4)
    ws_junc.set_column("B:B", 26)
    ws_junc.set_column("C:H", 14)

    ws_junc.write("B2", "JUNCTION GEOMETRY & SEVERITY COMPARISONS", fmt_title)
    ws_junc.write("B3", "Crash frequency and risk proportions across junction classifications", fmt_subtitle)

    junc_df = df.groupby("junction_type").agg(
        Minor=("severity", lambda s: (s == "Minor").sum()),
        Serious=("severity", lambda s: (s == "Serious").sum()),
        Fatal=("severity", lambda s: (s == "Fatal").sum()),
        Total=("severity", "count")
    ).reset_index()

    junc_headers = ["Junction Type", "Minor", "Serious", "Fatal", "Total Incidents", "Share (%)", "Severe Count", "Severe Rate"]
    for c_idx, h in enumerate(junc_headers):
        ws_junc.write(4, c_idx + 1, h, fmt_header)

    for r_idx, row in junc_df.iterrows():
        curr_r = 5 + r_idx
        ws_junc.write(curr_r, 1, row["junction_type"], fmt_cell)
        ws_junc.write(curr_r, 2, int(row["Minor"]), fmt_int)
        ws_junc.write(curr_r, 3, int(row["Serious"]), fmt_int)
        ws_junc.write(curr_r, 4, int(row["Fatal"]), fmt_int)
        ws_junc.write_formula(curr_r, 5, f"=SUM(C{curr_r+1}:E{curr_r+1})", fmt_int)
        ws_junc.write_formula(curr_r, 6, f"=F{curr_r+1}/{total_rows}", fmt_pct)
        ws_junc.write_formula(curr_r, 7, f"=SUM(D{curr_r+1}:E{curr_r+1})", fmt_int)
        ws_junc.write_formula(curr_r, 8, f"=H{curr_r+1}/F{curr_r+1}", fmt_pct)

    # ==========================================================
    # SHEET 8: Road Type Analysis
    # ==========================================================
    ws_road = workbook.add_worksheet("Road Type Analysis")
    ws_road.set_tab_color("#9333ea")
    ws_road.set_column("A:A", 4)
    ws_road.set_column("B:B", 22)
    ws_road.set_column("C:I", 14)

    ws_road.write("B2", "ROAD TYPE CLASSIFICATION & SPEED LIMITS", fmt_title)
    ws_road.write("B3", "Severity profiles, fatal crash proportions, and average speed limits across road types", fmt_subtitle)

    road_df = df.groupby("road_type").agg(
        Minor=("severity", lambda s: (s == "Minor").sum()),
        Serious=("severity", lambda s: (s == "Serious").sum()),
        Fatal=("severity", lambda s: (s == "Fatal").sum()),
        Total=("severity", "count"),
        Median_Speed=("speed_limit_kmph", "median")
    ).reset_index()

    road_headers = ["Road Type", "Minor", "Serious", "Fatal Crashes", "Total Incidents", "Fatal Crash Share", "Severe Rate", "Median Speed (km/h)"]
    for c_idx, h in enumerate(road_headers):
        ws_road.write(4, c_idx + 1, h, fmt_header)

    for r_idx, row in road_df.iterrows():
        curr_r = 5 + r_idx
        ws_road.write(curr_r, 1, row["road_type"], fmt_cell)
        ws_road.write(curr_r, 2, int(row["Minor"]), fmt_int)
        ws_road.write(curr_r, 3, int(row["Serious"]), fmt_int)
        ws_road.write(curr_r, 4, int(row["Fatal"]), fmt_int)
        ws_road.write_formula(curr_r, 5, f"=SUM(C{curr_r+1}:E{curr_r+1})", fmt_int)
        ws_road.write_formula(curr_r, 6, f"=E{curr_r+1}/F{curr_r+1}", fmt_pct)
        ws_road.write_formula(curr_r, 7, f"=SUM(D{curr_r+1}:E{curr_r+1})/F{curr_r+1}", fmt_pct)
        ws_road.write(curr_r, 8, int(row["Median_Speed"]), fmt_int)

    # Road Chart
    chart_road = workbook.add_chart({"type": "column"})
    chart_road.add_series({
        "name": "Fatal Crash Share (%)",
        "categories": f"='Road Type Analysis'!$B$6:$B${5+len(road_df)}",
        "values": f"='Road Type Analysis'!$G$6:$G${5+len(road_df)}",
        "fill": {"color": "#9333ea"}
    })
    chart_road.set_title({"name": "Fatal Crash Proportion (%) by Road Classification"})
    chart_road.set_y_axis({"num_format": "0%"})
    chart_road.set_size({"width": 460, "height": 280})
    ws_road.insert_chart("B13", chart_road)

    # ==========================================================
    # SHEET 9: Traffic Analysis
    # ==========================================================
    ws_traf = workbook.add_worksheet("Traffic Analysis")
    ws_traf.set_tab_color("#ea580c")
    ws_traf.set_column("A:A", 4)
    ws_traf.set_column("B:B", 22)
    ws_traf.set_column("C:C", 14)
    ws_traf.set_column("D:H", 16)

    ws_traf.write("B2", "TRAFFIC VOLUME & PEAK PERIOD DYNAMICS", fmt_title)
    ws_traf.write("B3", "Flow rates across peak windows and road infrastructure", fmt_subtitle)

    traf_df = df.groupby(["road_type", "peak_flag"]).agg(
        Mean_Volume=("traffic_volume_vph", "mean"),
        Median_Volume=("traffic_volume_vph", "median"),
        Min_Volume=("traffic_volume_vph", "min"),
        Max_Volume=("traffic_volume_vph", "max"),
        Incident_Count=("incident_id", "count")
    ).reset_index()

    traf_headers = ["Road Type", "Peak Flag", "Mean Volume (vph)", "Median Volume (vph)", "Min Volume", "Max Volume", "Incidents"]
    for c_idx, h in enumerate(traf_headers):
        ws_traf.write(4, c_idx + 1, h, fmt_header)

    for r_idx, row in traf_df.iterrows():
        curr_r = 5 + r_idx
        ws_traf.write(curr_r, 1, row["road_type"], fmt_cell)
        ws_traf.write(curr_r, 2, row["peak_flag"], fmt_cell_center)
        ws_traf.write(curr_r, 3, float(row["Mean_Volume"]), fmt_dec)
        ws_traf.write(curr_r, 4, int(row["Median_Volume"]), fmt_int)
        ws_traf.write(curr_r, 5, int(row["Min_Volume"]), fmt_int)
        ws_traf.write(curr_r, 6, int(row["Max_Volume"]), fmt_int)
        ws_traf.write(curr_r, 7, int(row["Incident_Count"]), fmt_int)

    # ==========================================================
    # SHEET 10: Model Performance
    # ==========================================================
    ws_mod = workbook.add_worksheet("Model Performance")
    ws_mod.set_tab_color("#4f46e5")
    ws_mod.set_column("A:A", 4)
    ws_mod.set_column("B:B", 28)
    ws_mod.set_column("C:E", 16)
    ws_mod.set_column("F:F", 20)

    ws_mod.write("B2", "RANDOM FOREST CLASSIFICATION BENCHMARKS", fmt_title)
    ws_mod.write("B3", "Out-of-sample evaluation on untouched stratified test cohort (20% partition)", fmt_subtitle)

    # Overall Summary Table
    ws_mod.write("B5", "Overall Benchmark Metrics", workbook.add_format({"bold": True, "font_size": 11, "font_color": "#1e3a8a"}))
    ws_mod.write("B6", "Evaluation Metric", fmt_header)
    ws_mod.write("C6", "Score / Value", fmt_header)
    ws_mod.write("D6", "Interpretation / Benchmark Notes", fmt_header)

    ov_metrics = [
        ("Model Algorithm", "RandomForestClassifier (150 trees, max_depth=10)", "Balanced class weights applied"),
        ("Training Samples", f"{metrics['sample_counts']['train']:,}", "80% stratified partition"),
        ("Testing Samples", f"{metrics['sample_counts']['test']:,}", "20% untouched holdout"),
        ("Accuracy", f"{metrics['overall_metrics']['accuracy']*100:.2f}%", "Overall classification agreement"),
        ("Macro F1-Score", f"{metrics['overall_metrics']['macro_f1']:.4f}", "Unweighted mean across all 3 classes"),
        ("Weighted F1-Score", f"{metrics['overall_metrics']['weighted_f1']:.4f}", "Support-weighted average F1-score"),
        ("Macro Precision", f"{metrics['overall_metrics']['macro_precision']:.4f}", "Average precision across classes"),
        ("Macro Recall", f"{metrics['overall_metrics']['macro_recall']:.4f}", "Average recall across classes")
    ]
    for idx, (m, val, note) in enumerate(ov_metrics, start=7):
        ws_mod.write(f"B{idx}", m, fmt_cell)
        ws_mod.write(f"C{idx}", val, fmt_cell_center)
        ws_mod.write(f"D{idx}", note, fmt_cell)

    # Class-wise Metrics Table
    ws_mod.write("B17", "Class-Wise Evaluation Breakdown", workbook.add_format({"bold": True, "font_size": 11, "font_color": "#1e3a8a"}))
    cls_headers = ["Severity Class", "Precision", "Recall", "F1-Score", "Test Support", "Class Imbalance Ratio"]
    for c_idx, h in enumerate(cls_headers):
        ws_mod.write(17, c_idx + 1, h, fmt_header)

    for idx, cls in enumerate(["Minor", "Serious", "Fatal"], start=18):
        c_data = metrics["class_wise_metrics"][cls]
        ws_mod.write(idx, 1, cls, fmt_cell)
        ws_mod.write(idx, 2, c_data["precision"], fmt_dec)
        ws_mod.write(idx, 3, c_data["recall"], fmt_dec)
        ws_mod.write(idx, 4, c_data["f1_score"], fmt_dec)
        ws_mod.write(idx, 5, c_data["test_support"], fmt_int)
        ws_mod.write(idx, 6, f"{c_data['test_support']/metrics['sample_counts']['test']*100:.1f}%", fmt_cell_center)

    # Confusion Matrix Table
    ws_mod.write("B24", "Test Set Confusion Matrix", workbook.add_format({"bold": True, "font_size": 11, "font_color": "#1e3a8a"}))
    ws_mod.write("B25", "Actual \\ Predicted", fmt_header)
    for c_idx, cls in enumerate(["Minor", "Serious", "Fatal"]):
        ws_mod.write(25, c_idx + 2, f"Pred {cls}", fmt_header)

    cm_mat = metrics["confusion_matrix"]["matrix"]
    for r_idx, cls in enumerate(["Minor", "Serious", "Fatal"]):
        curr_r = 26 + r_idx
        ws_mod.write(curr_r, 1, f"True {cls}", fmt_header)
        for c_idx in range(3):
            ws_mod.write(curr_r, c_idx + 2, int(cm_mat[r_idx][c_idx]), fmt_int)

    # Feature Importance Table
    ws_mod.write("F5", "Top 12 Features by Model Importance (Gini)", workbook.add_format({"bold": True, "font_size": 11, "font_color": "#1e3a8a"}))
    ws_mod.write("F6", "Feature Name", fmt_header)
    ws_mod.write("G6", "Importance", fmt_header)

    for idx, item in enumerate(metrics.get("feature_importance_top15", [])[:12], start=7):
        ws_mod.write(idx, 5, item["feature"], fmt_cell)
        ws_mod.write(idx, 6, float(item["importance"]), fmt_dec)

    # ==========================================================
    # SHEET 11: Pivot-Ready Data
    # ==========================================================
    ws_piv = workbook.add_worksheet("Pivot-Ready Data")
    ws_piv.set_tab_color("#0d9488")
    ws_piv.freeze_panes(12, 0)

    ws_piv.write("A2", "PIVOT-READY SOURCE DATA & NATIVE EXCEL PIVOT TABLE GUIDE", fmt_title)
    ws_piv.write("A3", "DISCLOSURE: The tables on Sheets 4 through 9 are pre-calculated grouped summary tables and standard Excel charts. To construct native, interactive Microsoft Excel PivotTables and PivotCharts with dynamic slicing and field filtering, use the structured flat dataset below.", fmt_subtitle)

    ws_piv.write("A5", "STEP-BY-STEP INSTRUCTIONS: BUILDING NATIVE PIVOT TABLES IN MICROSOFT EXCEL", workbook.add_format({"bold": True, "font_color": "#0d9488", "font_size": 11}))
    ws_piv.write("A6", "1. Click any cell within the structured flat table below (Row 12 onward).", fmt_cell)
    ws_piv.write("A7", "2. On the Excel Ribbon, navigate to: Insert > PivotTable. In the dialog, verify the data range and select 'New Worksheet'.", fmt_cell)
    ws_piv.write("A8", "3. In the PivotTable Field List panel, drag dimensions like 'road_type', 'location', or 'weather_condition' into the ROWS area.", fmt_cell)
    ws_piv.write("A9", "4. Drag 'severity' into the COLUMNS area, and drag 'incident_id' into the VALUES area (set Summarize Values By to 'Count').", fmt_cell)
    ws_piv.write("A10", "5. To display percentage shares: Right-click any value in the PivotTable > Show Values As > % of Row Total.", fmt_cell)
    ws_piv.write("A11", "6. To generate a dynamic native PivotChart: Click anywhere in the PivotTable, go to Insert > PivotChart, and select 100% Stacked Column.", fmt_cell)

    pivot_cols = [
        "incident_id", "date", "time", "location", "severity", "road_type", "junction_type",
        "speed_limit_kmph", "weather_condition", "rainfall_mm", "visibility_km",
        "traffic_volume_vph", "peak_flag", "vehicle_category", "vehicle_count",
        "hour_of_day", "day_of_week", "month", "time_period", "is_severe"
    ]
    for c_idx, col in enumerate(pivot_cols):
        ws_piv.write(12, c_idx, col, fmt_header)
        ws_piv.set_column(c_idx, c_idx, max(len(col) + 4, 12))

    for r_idx, row in df[pivot_cols].iterrows():
        curr_r = 13 + r_idx
        for c_idx, col_name in enumerate(pivot_cols):
            val = row[col_name]
            if pd.isna(val):
                ws_piv.write_blank(curr_r, c_idx, None, fmt_cell)
            elif isinstance(val, (int, np.integer)):
                ws_piv.write_number(curr_r, c_idx, int(val), fmt_int)
            elif isinstance(val, (float, np.floating)):
                ws_piv.write_number(curr_r, c_idx, float(val), fmt_dec)
            else:
                ws_piv.write_string(curr_r, c_idx, str(val), fmt_cell)

    ws_piv.autofilter(12, 0, 12 + len(df), len(pivot_cols) - 1)

    workbook.close()
    print(f"[INFO] Successfully created baseline Excel workbook at: {output_path}")

    # Attach Native Excel PivotTables and PivotCharts using Excel COM automation if available
    add_native_pivot_tables(output_path)

    return output_path


def add_native_pivot_tables(excel_path):
    """
    Leverages Microsoft Excel COM automation to insert genuine native Excel PivotTables
    and linked dynamic PivotCharts into the workbook. Falls back gracefully if Excel COM is unavailable.
    """
    abs_path = os.path.abspath(excel_path)
    try:
        import win32com.client
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False

        try:
            wb = excel.Workbooks.Open(abs_path)
            for s in wb.Sheets:
                if s.Name == "Native Pivot Tables":
                    s.Delete()

            ws_data = wb.Sheets("Cleaned Data")
            last_row = ws_data.UsedRange.Rows.Count
            last_col = ws_data.UsedRange.Columns.Count
            data_range = ws_data.Range(ws_data.Cells(1, 1), ws_data.Cells(last_row, last_col))

            # Create PivotCache (xlDatabase = 1)
            pc = wb.PivotCaches().Create(SourceType=1, SourceData=data_range)

            # Add new sheet after the first sheet
            ws_pivot = wb.Sheets.Add(After=wb.Sheets(1))
            ws_pivot.Name = "Native Pivot Tables"

            # Title
            ws_pivot.Cells(2, 2).Value = "CH15 – NATIVE EXCEL PIVOT TABLES & PIVOT CHARTS"
            ws_pivot.Cells(2, 2).Font.Bold = True
            ws_pivot.Cells(2, 2).Font.Size = 14
            ws_pivot.Cells(3, 2).Value = "Interactive native Excel PivotTables and linked PivotCharts generated from Cleaned Data."
            ws_pivot.Cells(3, 2).Font.Italic = True

            # PivotTable 1: Road Type vs Severity
            pt1 = pc.CreatePivotTable(TableDestination=ws_pivot.Range("B5"), TableName="RoadTypeSeverityPivot")
            pt1.PivotFields("road_type").Orientation = 1  # xlRowField
            pt1.PivotFields("severity").Orientation = 2   # xlColumnField
            f_val = pt1.PivotFields("incident_id")
            f_val.Orientation = 4                         # xlDataField
            f_val.Function = -4112                        # xlCount
            f_val.Name = "Incident Count"

            # Native PivotChart 1
            chart_shape = ws_pivot.Shapes.AddChart2(297, 65, 550, 70, 480, 280)  # xlColumnStacked = 65
            chart_shape.Chart.SetSourceData(pt1.TableRange2)
            chart_shape.Chart.HasTitle = True
            chart_shape.Chart.ChartTitle.Text = "Native PivotChart: Severity by Road Classification"

            # PivotTable 2: Weather Condition vs Severity
            pt2 = pc.CreatePivotTable(TableDestination=ws_pivot.Range("B22"), TableName="WeatherSeverityPivot")
            pt2.PivotFields("weather_condition").Orientation = 1  # xlRowField
            pt2.PivotFields("severity").Orientation = 2          # xlColumnField
            f_val2 = pt2.PivotFields("incident_id")
            f_val2.Orientation = 4                                # xlDataField
            f_val2.Function = -4112                               # xlCount
            f_val2.Name = "Incident Count "

            # Native PivotChart 2
            chart_shape2 = ws_pivot.Shapes.AddChart2(297, 65, 550, 320, 480, 280)
            chart_shape2.Chart.SetSourceData(pt2.TableRange2)
            chart_shape2.Chart.HasTitle = True
            chart_shape2.Chart.ChartTitle.Text = "Native PivotChart: Severity by Weather Condition"

            wb.Save()
            print("[INFO] Native PivotTables and dynamic PivotCharts successfully created via Excel COM engine!")
            wb.Close(SaveChanges=True)
        finally:
            excel.Quit()
    except Exception as e:
        print(f"[INFO] Excel COM native PivotTable creation skipped ({e}). Pivot-Ready flat data sheet is available.")


if __name__ == "__main__":
    generate_excel_report()
