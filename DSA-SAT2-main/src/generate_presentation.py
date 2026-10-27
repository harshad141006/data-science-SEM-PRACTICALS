"""
Presentation Generator Module for CH15 Road Safety Insight.
Generates outputs/business_insights_presentation.pptx with 6 polished,
professional widescreen slides embedding actual charts, empirical findings, and model metrics.
"""

import os
import json
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

from src.utils import get_project_root


# Corporate Color Palette
COLOR_NAVY = RGBColor(15, 23, 42)       # Slate 900
COLOR_BLUE = RGBColor(30, 58, 138)      # Blue 900
COLOR_ACCENT = RGBColor(2, 132, 199)    # Sky 600
COLOR_CARD_BG = RGBColor(241, 245, 249) # Slate 100
COLOR_TEXT_MAIN = RGBColor(30, 41, 59)  # Slate 800
COLOR_TEXT_MUTED = RGBColor(100, 116, 139) # Slate 500
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_ORANGE = RGBColor(234, 88, 12)    # Orange 600
COLOR_PURPLE = RGBColor(147, 51, 234)   # Purple 600


def create_presentation(output_path=None):
    """
    Creates an executive 6-slide widescreen presentation.
    """
    project_root = get_project_root()
    if output_path is None:
        output_path = os.path.join(project_root, "outputs", "business_insights_presentation.pptx")

    charts_dir = os.path.join(project_root, "outputs", "charts")
    metrics_path = os.path.join(project_root, "outputs", "model_metrics.json")
    insights_path = os.path.join(project_root, "outputs", "key_insights.json")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(insights_path, "r", encoding="utf-8") as f:
        insights = json.load(f)

    prs = Presentation()
    # 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text):
        """Adds a standardized clean corporate header banner."""
        # Category pill
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(10), Inches(0.35))
        tf_c = cat_box.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = category_text.upper()
        p_c.font.size = Pt(10)
        p_c.font.bold = True
        p_c.font.color.rgb = COLOR_ACCENT

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11), Inches(0.6))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_NAVY

    def add_card(slide, left, top, width, height, title, body_bullets, bg_color=COLOR_CARD_BG):
        """Adds a clean structured card with bullets."""
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = RGBColor(226, 232, 240)
        shape.line.width = Pt(1)

        tx_box = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.15), width - Inches(0.3), height - Inches(0.3))
        tf = tx_box.text_frame
        tf.word_wrap = True

        p_head = tf.paragraphs[0]
        p_head.text = title
        p_head.font.size = Pt(13)
        p_head.font.bold = True
        p_head.font.color.rgb = COLOR_NAVY
        p_head.space_after = Pt(8)

        for b in body_bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(10.5)
            p.font.color.rgb = COLOR_TEXT_MAIN
            p.space_after = Pt(4)

    # ==========================================================
    # SLIDE 1: Title Slide (Dark Executive Aesthetic)
    # ==========================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_NAVY
    bg1.line.color.rgb = COLOR_NAVY

    # Decorative accent bar
    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(2.0), Inches(0.2), Inches(3.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_ACCENT
    bar.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.5), Inches(2.0), Inches(10.5), Inches(3.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "DATA SCIENCE SKILL-BASED ASSESSMENT | CH15"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_ACCENT
    p0.space_after = Pt(10)

    p1 = tf1.add_paragraph()
    p1.text = "Road Safety Insight"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE

    p2 = tf1.add_paragraph()
    p2.text = "Understand Accidents • Discover Risk Patterns • Support Safer Roads"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(148, 163, 184)
    p2.space_after = Pt(24)

    p3 = tf1.add_paragraph()
    p3.text = "Executive Briefing: Multi-Source Data Quality Audit, Empirical EDA, ML Severity Prediction & Decision Platform"
    p3.font.size = Pt(12)
    p3.font.color.rgb = RGBColor(203, 213, 225)

    # Disclaimer note at bottom
    tb_disc = s1.shapes.add_textbox(Inches(1.5), Inches(6.2), Inches(10.5), Inches(0.6))
    p_disc = tb_disc.text_frame.paragraphs[0]
    p_disc.text = "Notice: All records reflect computer-generated educational synthetic data. Location names and numbers are illustrative."
    p_disc.font.size = Pt(9.5)
    p_disc.font.italic = True
    p_disc.font.color.rgb = RGBColor(100, 116, 139)

    # ==========================================================
    # SLIDE 2: Dataset Profile & Data Quality Remediation
    # ==========================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Data Ingestion, Quality Assessment & Cleaning Pipeline", "Phase 1 & Phase 2 Architecture")

    add_card(
        s2, Inches(0.8), Inches(1.5), Inches(3.8), Inches(5.3),
        "Planted Quality Issues Detected",
        [
            "Raw Ingestion: 6,040 accident records; 6,000 records each in roads, weather, traffic, and vehicles.",
            "40 Duplicate Incidents: Placed in accidents.csv; deduplicated to 6,000 unique incident IDs.",
            "24 Invalid Severities: Unknown (8), Critical (6), 0 (5), 5 (5); excluded from modeling.",
            "115 Inconsistent Road Types: Case and whitespace variations standardized into 5 classes.",
            "18 Speed Limit Outliers: Negative (-20..-1) and extreme values (>200 km/h) imputed by road-type median.",
            "300 Blank Weather Rows: Mode imputation for conditions, median for rainfall and visibility.",
            "24 Traffic Volume Outliers: Imputed via road-type and peak-flag grouped medians."
        ]
    )

    add_card(
        s2, Inches(4.8), Inches(1.5), Inches(3.8), Inches(5.3),
        "Integration & Cleaning Verification",
        [
            "Common Key: incident_id validated across all 5 tables (100% key intersection).",
            "1-to-1 Cardinality: Verified join integrity; zero unintended row multiplication or cartesian join expansion.",
            "Final Clean Cohort: Exactly 5,976 validated, integrated incidents preserved.",
            "149 Missing Locations: Categorized explicitly as 'Unknown Location' to preserve valid accident records.",
            "Temporal Feature Engineering: Derived hour, day of week, month, peak flag, and weekend indicators.",
            "Audit Trail: Full reconciliation exported to data_quality_report.csv and Excel audit sheet."
        ]
    )

    add_card(
        s2, Inches(8.8), Inches(1.5), Inches(3.7), Inches(5.3),
        "Cleaned Cohort Profile (N = 5,976)",
        [
            "Minor Severity: 4,100 incidents (68.61% baseline share).",
            "Serious Severity: 1,645 incidents (27.53% baseline share).",
            "Fatal Severity: 231 incidents (3.87% critical minority).",
            "Severe Cohort (Serious + Fatal): 1,876 incidents (31.39%).",
            "Road Types: Urban Arterial (26.6%), Collector (21.9%), Service (19.9%), Highway Link (17.6%), Residential (14.4%).",
            "Weather Conditions: Clear (50.8%), Cloudy (24.9%), Light Rain (14.1%), Fog (5.4%), Heavy Rain (5.2%).",
            "Primary Vehicles: Cars (39.9%), Motorcycles (26.9%), Auto-rickshaws (9.7%), Trucks (8.1%)."
        ]
    )

    # ==========================================================
    # SLIDE 3: Exploratory Findings & Spatial Hotspots
    # ==========================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Corridor Hotspot Analysis & Severity Concentration", "Phase 3 Exploratory Data Analysis")

    # Embed Chart 2: Top Hotspots
    c2_img = os.path.join(charts_dir, "02_top_accident_hotspots.png")
    if os.path.exists(c2_img):
        s3.shapes.add_picture(c2_img, Inches(0.8), Inches(1.5), width=Inches(6.8))

    add_card(
        s3, Inches(7.8), Inches(1.5), Inches(4.7), Inches(5.3),
        "Spatial Hotspot Insights",
        [
            "Volume Leader: Central Market Junction recorded the highest overall accident volume (445 incidents; 147 severe crashes).",
            "Fatal Corridor Leader: East Lake Road recorded the highest fatal crash count across all corridors (24 fatal-severity crashes, 7.50% fatal crash share across 320 crashes).",
            "Secondary Lethality Corridors: Outer Loop - Sector A exhibited 19 fatal crashes (6.93%), and School Zone Avenue recorded 17 fatal crashes (5.57%).",
            "Lowest Incident Site: Hillview Street recorded the fewest incidents (153 total) and lowest severe share (24.8%).",
            "Key Policy Takeaway: High crash frequency does NOT equal highest fatal crash share. Municipal road safety teams must deploy dual-track interventions: congestion calming at high-volume sites and physical crash cushions at high-severity corridors. (Distinguish observed counts from exposure-adjusted risk)."
        ]
    )

    # ==========================================================
    # SLIDE 4: Temporal, Weather & Vehicle Safety Risks
    # ==========================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Multi-Dimensional Risk Factors: Time, Weather & Modes", "Phase 3 Empirical Patterns")

    # Left: Temporal & Weather
    add_card(
        s4, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.3),
        "Temporal & Weather Risk Patterns",
        [
            "Rush-Hour Spikes: Accidents peak sharply during morning (08:00, 547 incidents) and evening (18:00, 608 incidents) commutes.",
            "Peak Congestion Share: Designated peak hours account for 3,624 incidents (60.6% of all recorded accidents).",
            "Night Fatal Crash Share: Night hours (21:00-04:59) represent only 9.6% of volume (573 crashes) but exhibit the HIGHEST fatal crash share at 4.71%, nearly double the Afternoon share (2.70%).",
            "Fog Multiplier Effect: Dense Fog (323 incidents) escalates fatal crash share to 8.10% and combined severe crash share to 45.48% (vs 28.6% under Clear skies).",
            "Precipitation Impact: Light Rain elevated severe crash share to 39.31%, highlighting slick surface risks."
        ]
    )

    # Right: Infrastructure & Vehicles
    add_card(
        s4, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3),
        "Infrastructure & Road User Vulnerabilities",
        [
            "Highway Link Hazard: Highway Links exhibited a 6.03% fatal crash share, more than 2.2x higher than Collector Roads (2.68%).",
            "Speed Limit Association: Fatal crashes occurred on road segments with a mean nominal speed limit of 48.4 km/h, compared to 44.3 km/h for minor crashes.",
            "Junction Geometries: Flyovers and Midblock segments recorded the highest fatal crash shares (5.03% and 4.88%), whereas Roundabouts had the lowest (2.47%).",
            "Vulnerable Road Users: Motorcycles recorded 5.06% fatal and 30.65% serious injury shares (35.71% total severe), highest among motorized transport.",
            "Passenger Cars: Recorded the lowest fatal crash share among motorized vehicles at 3.48% (71.8% minor)."
        ]
    )

    # ==========================================================
    # SLIDE 5: Machine Learning Severity Classifier & Metrics
    # ==========================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Supervised Machine Learning Model & Evaluation", "Phase 5 Predictive Analytics")

    # Embed Confusion Matrix
    cm_img = os.path.join(charts_dir, "13_confusion_matrix.png")
    if os.path.exists(cm_img):
        s5.shapes.add_picture(cm_img, Inches(0.8), Inches(1.5), width=Inches(5.0))

    add_card(
        s5, Inches(6.0), Inches(1.5), Inches(6.5), Inches(5.3),
        "Model Methodology, Metrics & Limitations",
        [
            "Pipeline Architecture: Scikit-learn Pipeline with ColumnTransformer (imputation + scaling + OneHotEncoder) and RandomForestClassifier (150 trees, max_depth=10).",
            "Leakage Prevention: Preprocessor fitted strictly on training partition (4,780 rows). Evaluated on untouched 20% stratified test set (1,196 rows).",
            f"Overall Accuracy: {metrics['overall_metrics']['accuracy']*100:.2f}% on untouched holdout test data.",
            f"Macro F1-Score: {metrics['overall_metrics']['macro_f1']:.4f} | Weighted F1-Score: {metrics['overall_metrics']['weighted_f1']:.4f}.",
            f"Class-Wise Performance: Minor (F1: {metrics['class_wise_metrics']['Minor']['f1_score']:.2f}, Recall: {metrics['class_wise_metrics']['Minor']['recall']*100:.1f}%), Serious (F1: {metrics['class_wise_metrics']['Serious']['f1_score']:.2f}, Recall: {metrics['class_wise_metrics']['Serious']['recall']*100:.1f}%), Fatal (F1: {metrics['class_wise_metrics']['Fatal']['f1_score']:.2f}, Recall: {metrics['class_wise_metrics']['Fatal']['recall']*100:.1f}%).",
            "Severe Class Imbalance: Extreme rarity of Fatal crashes (3.9%) and Serious injuries (27.5%) causes lower minority recall; balanced class weights prevented majority-class collapse.",
            "Feature Association: Visibility, traffic volume, and speed limit are top associative features. These describe model correlations, NOT physical crash causality."
        ]
    )

    # ==========================================================
    # SLIDE 6: Evidence-Based Recommendations & Roadmap
    # ==========================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Strategic Safety Recommendations & Roadmap", "Phase 8 Decision Intelligence")

    add_card(
        s6, Inches(0.8), Inches(1.5), Inches(5.7), Inches(5.3),
        "Targeted Action Recommendations",
        [
            "1. Corridor-Specific Engineering Audits: Conduct urgent geometric and barrier safety audits along East Lake Road and Outer Loop Sector A to address high fatal crash counts.",
            "2. Night-Time Speed Enforcement: Deploy automated radar speed enforcement and upgrade retroreflective street lighting between 21:00 and 04:00 to address the elevated 4.71% night fatal crash share.",
            "3. Dynamic Weather Warning Systems: Install Variable Message Signs (VMS) on Highway Links and Flyovers to enforce dynamic speed reductions during Fog and heavy rain.",
            "4. Dedicated Two-Wheeler Corridors: Implement physical barrier separation and motorcycle-safe crash barriers along Urban Arterials and collector links.",
            "5. Congestion Management at Market Sites: Optimize traffic signal phases and pedestrian grade separation at Central Market Junction to suppress collision frequency."
        ]
    )

    add_card(
        s6, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3),
        "Implementation Roadmap & Limitations",
        [
            "Phase 1 (Months 1-3): Rapid deployment of high-visibility enforcement and fog warning signage at top 5 severe corridors.",
            "Phase 2 (Months 4-6): Midblock junction redesign, flyover barrier reinforcement, and streetlight retrofitting.",
            "Phase 3 (Months 7-12): Integration of continuous sensor feeds into the deployed Streamlit Intelligence Platform for real-time risk scoring.",
            "Critical Analytical Limitation: The synthetic dataset lacks exposure denominators (vehicle-km travelled, total traffic volumes for non-accident periods).",
            "Exposure Caution: Reported metrics represent incident proportions, not per-exposure accident probabilities. Future phases must incorporate telematics exposure data."
        ]
    )

    prs.save(output_path)
    print(f"[INFO] Successfully created PowerPoint presentation at: {output_path}")
    return output_path


if __name__ == "__main__":
    create_presentation()
