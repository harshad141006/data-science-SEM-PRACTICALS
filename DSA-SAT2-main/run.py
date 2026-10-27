"""
Master Execution Pipeline for CH15 - Road Safety Insight.
Orchestrates end-to-end data processing, EDA, ML training, Excel workbook generation,
PowerPoint presentation generation, and Jupyter notebook execution.
"""

import os
import sys
import time

def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("  CH15 ROAD SAFETY INSIGHT - END-TO-END EXECUTION PIPELINE")
    print("=" * 70)

    # 1. Data Processing & Ingestion
    print("\n[STEP 1/6] Running Data Quality Assessment & Cleaning...")
    from src.data_processing import assess_data_quality, clean_and_integrate_data
    assess_data_quality()
    df_clean, audit = clean_and_integrate_data()
    print(f"  -> Cleaned cohort: {len(df_clean)} records across {df_clean.shape[1]} columns.")

    # 2. Exploratory Data Analysis & Chart Generation
    print("\n[STEP 2/6] Running Exploratory Data Analysis & Visualizations...")
    from src.eda import run_eda
    run_eda()
    print("  -> Generated 12 publication charts and key insights JSON.")

    # 3. Model Training & Evaluation
    print("\n[STEP 3/6] Running Machine Learning Pipeline (Random Forest)...")
    from src.model_training import train_and_evaluate_model
    train_and_evaluate_model()
    print("  -> Trained Pipeline, evaluated on held-out test split, persisted metrics and charts.")

    # 4. Multi-Sheet Excel Workbook with Native PivotTables
    print("\n[STEP 4/6] Generating 12-Sheet Excel Reporting Workbook...")
    from src.excel_reporting import generate_excel_report
    generate_excel_report()
    print("  -> Generated outputs/road_safety_analysis.xlsx with native PivotTables and charts.")

    # 5. Executive Presentation Deck
    print("\n[STEP 5/6] Generating Executive PowerPoint Presentation...")
    from src.generate_presentation import create_presentation
    create_presentation()
    print("  -> Generated outputs/business_insights_presentation.pptx (6 widescreen slides).")

    # 6. Jupyter Notebook Build and Execution
    print("\n[STEP 6/6] Building and Executing Jupyter Notebook...")
    from src.build_notebook import build_and_execute_notebook
    build_and_execute_notebook()
    print("  -> Fully populated notebooks/road_safety_analysis.ipynb with executed cell outputs.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  PIPELINE EXECUTION COMPLETE in {elapsed:.1f} seconds")
    print("  All deliverables verified and synchronized:")
    print("   - Cleaned Dataset:     data/processed/cleaned_accidents.csv")
    print("   - Quality Report:      data/processed/data_quality_report.csv")
    print("   - ML Model Artifact:   models/road_safety_model.joblib")
    print("   - Model Metrics:       outputs/model_metrics.json")
    print("   - Visual Charts:       outputs/charts/ (14 charts)")
    print("   - Excel Workbook:      outputs/road_safety_analysis.xlsx (12 sheets)")
    print("   - Executive Deck:      outputs/business_insights_presentation.pptx")
    print("   - Jupyter Notebook:    notebooks/road_safety_analysis.ipynb")
    print("   - Submission Sheet:    outputs/assessment_sheet_answers.md")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_pipeline()
