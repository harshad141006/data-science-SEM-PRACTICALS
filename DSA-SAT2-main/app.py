"""
Streamlit Web Application for CH15 Road Safety Insight.
Provides interactive visual analytics, hotspot exploration, temporal analysis,
weather/road safety intelligence, and a live Machine Learning severity prediction tool.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Road Safety Insight | CH15 Analytics Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look and feel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-subtitle {
        font-size: 12px;
        color: #38bdf8;
        margin-top: 4px;
    }
    .badge-minor {
        background-color: #0284c7;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-serious {
        background-color: #ea580c;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-fatal {
        background-color: #9333ea;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .disclaimer-box {
        background-color: #fef3c7;
        color: #92400e;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 13px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Color tokens
COLOR_MAP = {"Minor": "#0284c7", "Serious": "#ea580c", "Fatal": "#9333ea"}


@st.cache_data
def load_cleaned_data():
    """Loads cleaned integrated accident dataset."""
    path = os.path.join(os.path.dirname(__file__), "data", "processed", "cleaned_accidents.csv")
    if not os.path.exists(path):
        from src.data_processing import clean_and_integrate_data
        df, _ = clean_and_integrate_data()
        return df
    return pd.read_csv(path)


@st.cache_resource
def load_ml_model():
    """Loads fitted scikit-learn preprocessing and Random Forest pipeline."""
    path = os.path.join(os.path.dirname(__file__), "models", "road_safety_model.joblib")
    if not os.path.exists(path):
        from src.model_training import train_and_evaluate_model
        pipeline, _ = train_and_evaluate_model()
        return pipeline
    return joblib.load(path)


@st.cache_data
def load_metrics_and_insights():
    """Loads precomputed model metrics and derived key insights."""
    base_dir = os.path.dirname(__file__)
    metrics_path = os.path.join(base_dir, "outputs", "model_metrics.json")
    insights_path = os.path.join(base_dir, "outputs", "key_insights.json")
    
    metrics = None
    insights = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    if os.path.exists(insights_path):
        with open(insights_path, "r", encoding="utf-8") as f:
            insights = json.load(f)
    return metrics, insights


# Load data and assets
df_raw = load_cleaned_data()
model_pipeline = load_ml_model()
model_metrics, key_insights = load_metrics_and_insights()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
st.sidebar.title("Road Safety Filters")
st.sidebar.caption("CH15 Analytics & Decision Intelligence")

# Filter: Date range
min_date = pd.to_datetime(df_raw["date"]).min().date()
max_date = pd.to_datetime(df_raw["date"]).max().date()
selected_dates = st.sidebar.date_input("Incident Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Filter: Severity
severity_options = ["All"] + list(df_raw["severity"].unique())
selected_severity = st.sidebar.selectbox("Severity Classification", severity_options)

# Filter: Road Type
road_options = ["All"] + sorted(list(df_raw["road_type"].unique()))
selected_road = st.sidebar.selectbox("Road Classification", road_options)

# Filter: Weather
weather_options = ["All"] + sorted(list(df_raw["weather_condition"].unique()))
selected_weather = st.sidebar.selectbox("Weather Condition", weather_options)

# Filter: Time Period
period_options = ["All", "Morning", "Afternoon", "Evening", "Night"]
selected_period = st.sidebar.selectbox("Time of Day Period", period_options)

# Filter: Location
loc_list = sorted([l for l in df_raw["location"].unique() if l != "Unknown Location"])
location_options = ["All Locations"] + loc_list
selected_location = st.sidebar.selectbox("Location Hotspot", location_options)

# Apply filters
df = df_raw.copy()
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_d, end_d = selected_dates
    df = df[(pd.to_datetime(df["date"]).dt.date >= start_d) & (pd.to_datetime(df["date"]).dt.date <= end_d)]
if selected_severity != "All":
    df = df[df["severity"] == selected_severity]
if selected_road != "All":
    df = df[df["road_type"] == selected_road]
if selected_weather != "All":
    df = df[df["weather_condition"] == selected_weather]
if selected_period != "All":
    df = df[df["time_period"] == selected_period]
if selected_location != "All Locations":
    df = df[df["location"] == selected_location]

# ----------------- HEADER & DISCLAIMER -----------------
st.title("🚦 CH15 – Road Safety Insight Dashboard")
st.subheader("Understand Accidents • Discover Risk Patterns • Support Safer Roads")

st.markdown("""
<div class="disclaimer-box">
    <strong>Educational Synthetic Data Notice:</strong> All incident records in this platform are computer-generated synthetic data developed for classroom evaluation. Fictional urban locations, dates, and metric measurements do not represent real-world municipal casualty statistics. Observed figures represent crash counts and severity proportions among recorded collisions, not exposure-adjusted (per-km or per-vehicle) risk.
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tabs = st.tabs([
    "📊 Executive Overview",
    "📍 Hotspot Analysis",
    "⏰ Temporal Trends",
    "🌧️ Weather & Infrastructure",
    "🚗 Traffic & Vehicles",
    "🤖 ML Severity Predictor"
])

# ----------------- TAB 1: OVERVIEW -----------------
with tabs[0]:
    st.markdown("### Executive Overview & KPI Scorecard")
    if len(df) == 0:
        st.warning("No incident records match the current sidebar filter criteria. Please adjust your filters.")
    else:
        total_incidents = len(df)
        fatal_count = (df["severity"] == "Fatal").sum()
        serious_count = (df["severity"] == "Serious").sum()
        minor_count = (df["severity"] == "Minor").sum()
        severe_count = fatal_count + serious_count
        severe_pct = (severe_count / total_incidents * 100) if total_incidents else 0.0

        # Top location by severe count
        valid_locs = df[df["location"] != "Unknown Location"]
        if not valid_locs.empty:
            loc_sev = valid_locs.groupby("location")["is_severe"].sum().sort_values(ascending=False)
            top_loc_name = loc_sev.index[0]
            top_loc_val = int(loc_sev.iloc[0])
        else:
            top_loc_name = "N/A"
            top_loc_val = 0

        # KPI Row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Incidents</div>
                <div class="metric-value">{total_incidents:,}</div>
                <div class="metric-subtitle">Filtered records</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Minor Collisions</div>
                <div class="metric-value">{minor_count:,}</div>
                <div class="metric-subtitle">{minor_count/total_incidents*100:.1f}% of total</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Serious Injuries</div>
                <div class="metric-value">{serious_count:,}</div>
                <div class="metric-subtitle">{serious_count/total_incidents*100:.1f}% of total</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Fatal Crashes</div>
                <div class="metric-value">{fatal_count:,}</div>
                <div class="metric-subtitle">{fatal_count/total_incidents*100:.1f}% fatal crash share</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Top Hotspot</div>
                <div class="metric-value" style="font-size: 18px; line-height: 1.2;">{top_loc_name[:18]}</div>
                <div class="metric-subtitle">{top_loc_val} severe incidents</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Overview Charts
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown("##### Severity Breakdown")
            sev_df = df["severity"].value_counts().reset_index()
            sev_df.columns = ["Severity", "Count"]
            fig_pie = px.pie(
                sev_df,
                values="Count",
                names="Severity",
                color="Severity",
                color_discrete_map=COLOR_MAP,
                hole=0.45,
                title="Proportion of Accidents by Severity"
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_right:
            st.markdown("##### High-Risk Corridors (Top 8 Locations by Volume)")
            top8 = valid_locs["location"].value_counts().head(8).reset_index()
            top8.columns = ["Location", "Total Accidents"]
            fig_bar = px.bar(
                top8,
                x="Total Accidents",
                y="Location",
                orientation="h",
                color="Total Accidents",
                color_continuous_scale="Blues",
                title="Top 8 Accident Hotspots"
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        # Strategic Findings Expander
        if key_insights and "findings" in key_insights:
            with st.expander("📌 View Empirically Derived Safety Insights & Hotspot Findings", expanded=False):
                for f in key_insights["findings"]:
                    st.markdown(f"**[{f['category']}] {f['title']}**")
                    st.write(f['description'])
                    st.markdown("---")

# ----------------- TAB 2: HOTSPOTS -----------------
with tabs[1]:
    st.markdown("### Location Risk Analysis & Corridor Profiling")
    if len(df) == 0:
        st.warning("No records found for current filters.")
    else:
        valid_locs = df[df["location"] != "Unknown Location"]
        hotspot_summary = valid_locs.groupby("location")["severity"].value_counts().unstack().fillna(0)
        hotspot_summary["Total"] = hotspot_summary.sum(axis=1)
        hotspot_summary["Fatal"] = hotspot_summary.get("Fatal", 0)
        hotspot_summary["Serious"] = hotspot_summary.get("Serious", 0)
        hotspot_summary["Minor"] = hotspot_summary.get("Minor", 0)
        hotspot_summary["Severe_Count"] = hotspot_summary["Serious"] + hotspot_summary["Fatal"]
        hotspot_summary["Severe_Rate_%"] = (hotspot_summary["Severe_Count"] / hotspot_summary["Total"] * 100).round(2)
        hotspot_summary["Fatal_Rate_%"] = (hotspot_summary["Fatal"] / hotspot_summary["Total"] * 100).round(2)
        hotspot_summary = hotspot_summary.sort_values(by="Severe_Count", ascending=False).reset_index()

        c_table, c_chart = st.columns([1.1, 1.3])
        with c_table:
            st.markdown("##### Sortable Corridor Hotspot Leaderboard")
            st.dataframe(
                hotspot_summary.style.format({
                    "Total": "{:,.0f}",
                    "Minor": "{:,.0f}",
                    "Serious": "{:,.0f}",
                    "Fatal": "{:,.0f}",
                    "Severe_Count": "{:,.0f}",
                    "Severe_Rate_%": "{:.1f}%",
                    "Fatal_Rate_%": "{:.1f}%"
                }).background_gradient(subset=["Severe_Count", "Fatal_Rate_%"], cmap="Reds"),
                height=480,
                use_container_width=True
            )

        with c_chart:
            st.markdown("##### Severe vs Minor Accidents by Location")
            fig_hot = px.bar(
                hotspot_summary.head(15),
                x=["Minor", "Serious", "Fatal"],
                y="location",
                orientation="h",
                color_discrete_map=COLOR_MAP,
                title="Severity Composition at Top 15 Locations"
            )
            fig_hot.update_layout(
                barmode="stack",
                yaxis=dict(autorange="reversed"),
                legend_title="Severity",
                height=480,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_hot, use_container_width=True)

# ----------------- TAB 3: TEMPORAL -----------------
with tabs[2]:
    st.markdown("### Temporal Risk & Peak Commuter Patterns")
    if len(df) == 0:
        st.warning("No records found for current filters.")
    else:
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            st.markdown("##### Accidents by Hour of the Day (24h)")
            hourly_agg = df.groupby(["hour_of_day", "severity"]).size().reset_index(name="Count")
            fig_hour = px.line(
                hourly_agg,
                x="hour_of_day",
                y="Count",
                color="severity",
                color_discrete_map=COLOR_MAP,
                markers=True,
                title="Hourly Incident Volume by Severity"
            )
            fig_hour.add_vrect(x0=7, x1=10, fillcolor="orange", opacity=0.15, line_width=0, annotation_text="Morning Peak", annotation_position="top left")
            fig_hour.add_vrect(x0=17, x1=20, fillcolor="red", opacity=0.15, line_width=0, annotation_text="Evening Peak", annotation_position="top left")
            fig_hour.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=2), height=400)
            st.plotly_chart(fig_hour, use_container_width=True)

        with c_t2:
            st.markdown("##### Day of Week Accident Volume")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_agg = df.groupby(["day_of_week", "severity"]).size().reset_index(name="Count")
            fig_day = px.bar(
                day_agg,
                x="day_of_week",
                y="Count",
                color="severity",
                category_orders={"day_of_week": day_order},
                color_discrete_map=COLOR_MAP,
                title="Accidents by Day of Week and Severity"
            )
            fig_day.update_layout(barmode="stack", height=400)
            st.plotly_chart(fig_day, use_container_width=True)

        st.markdown("##### Peak vs Off-Peak Traffic Comparison")
        peak_comp = pd.crosstab(df["peak_flag"], df["severity"], normalize="index").reset_index()
        fig_peak = px.bar(
            peak_comp,
            x="peak_flag",
            y=["Minor", "Serious", "Fatal"],
            color_discrete_map=COLOR_MAP,
            title="Relative Severity Share (%) in Peak vs Off-Peak Windows"
        )
        fig_peak.update_layout(barmode="stack", yaxis=dict(tickformat=".0%"), height=350)
        st.plotly_chart(fig_peak, use_container_width=True)

# ----------------- TAB 4: WEATHER & ROADS -----------------
with tabs[3]:
    st.markdown("### Weather Impacts & Road Infrastructure Hazards")
    if len(df) == 0:
        st.warning("No records found for current filters.")
    else:
        c_w1, c_w2 = st.columns(2)
        with c_w1:
            st.markdown("##### Severity Share by Weather Condition")
            weath_crosstab = pd.crosstab(df["weather_condition"], df["severity"], normalize="index").reset_index()
            fig_weath = px.bar(
                weath_crosstab,
                x="weather_condition",
                y=["Minor", "Serious", "Fatal"],
                color_discrete_map=COLOR_MAP,
                title="Severity Distribution by Weather Condition"
            )
            fig_weath.update_layout(barmode="stack", yaxis=dict(tickformat=".0%"), height=400)
            st.plotly_chart(fig_weath, use_container_width=True)

        with c_w2:
            st.markdown("##### Road Type vs Accident Severity")
            road_crosstab = pd.crosstab(df["road_type"], df["severity"], normalize="index").reset_index()
            fig_road = px.bar(
                road_crosstab,
                x="road_type",
                y=["Minor", "Serious", "Fatal"],
                color_discrete_map=COLOR_MAP,
                title="Severity Distribution Across Road Classifications"
            )
            fig_road.update_layout(barmode="stack", yaxis=dict(tickformat=".0%"), height=400)
            st.plotly_chart(fig_road, use_container_width=True)

        st.markdown("##### Junction Geometry & Nominal Speed Limits")
        c_j1, c_j2 = st.columns(2)
        with c_j1:
            junc_crosstab = pd.crosstab(df["junction_type"], df["severity"], normalize="index").reset_index()
            fig_junc = px.bar(
                junc_crosstab,
                x="junction_type",
                y=["Minor", "Serious", "Fatal"],
                color_discrete_map=COLOR_MAP,
                title="Accident Severity by Junction Geometry"
            )
            fig_junc.update_layout(barmode="stack", yaxis=dict(tickformat=".0%"), height=380)
            st.plotly_chart(fig_junc, use_container_width=True)

        with c_j2:
            fig_speed = px.box(
                df,
                x="severity",
                y="speed_limit_kmph",
                color="severity",
                category_orders={"severity": ["Minor", "Serious", "Fatal"]},
                color_discrete_map=COLOR_MAP,
                title="Speed Limit Distribution by Severity Level"
            )
            fig_speed.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_speed, use_container_width=True)

# ----------------- TAB 5: TRAFFIC & VEHICLES -----------------
with tabs[4]:
    st.markdown("### Traffic Flow Dynamics & Vehicle Vulnerability")
    if len(df) == 0:
        st.warning("No records found for current filters.")
    else:
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.markdown("##### Primary Vehicle Involved vs Accident Severity")
            veh_crosstab = pd.crosstab(df["vehicle_category"], df["severity"], normalize="index").reset_index()
            fig_veh = px.bar(
                veh_crosstab,
                x="vehicle_category",
                y=["Minor", "Serious", "Fatal"],
                color_discrete_map=COLOR_MAP,
                title="Severity Distribution by Vehicle Category"
            )
            fig_veh.update_layout(barmode="stack", yaxis=dict(tickformat=".0%"), height=420)
            st.plotly_chart(fig_veh, use_container_width=True)

        with c_v2:
            st.markdown("##### Traffic Volume (vph) Distribution Across Peak Windows")
            fig_vol = px.histogram(
                df,
                x="traffic_volume_vph",
                color="peak_flag",
                barmode="overlay",
                nbins=40,
                color_discrete_map={"Peak": "#ea580c", "Off-Peak": "#0284c7"},
                title="Traffic Volume Frequency Distribution"
            )
            fig_vol.update_layout(height=420)
            st.plotly_chart(fig_vol, use_container_width=True)

# ----------------- TAB 6: ML PREDICTOR -----------------
with tabs[5]:
    st.markdown("### Machine Learning Severity Classifier & Risk Simulator")
    st.markdown("""
    This intelligence module utilizes the fitted **Random Forest Pipeline** (`road_safety_model.joblib`) 
    trained on multi-table engineered features (speed limit, weather, temporal windows, road geometry, and vehicle categories).
    """)

    # Model Evaluation Metrics Card
    if model_metrics:
        st.markdown("##### Model Architecture & Test Evaluation Performance")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Overall Accuracy", f"{model_metrics['overall_metrics']['accuracy']*100:.1f}%")
        m_col2.metric("Macro F1-Score", f"{model_metrics['overall_metrics']['macro_f1']:.4f}")
        m_col3.metric("Weighted F1-Score", f"{model_metrics['overall_metrics']['weighted_f1']:.4f}")
        m_col4.metric("Test Sample Size", f"{model_metrics['sample_counts']['test']:,}")

        with st.expander("🔍 View Detailed Class-Wise Evaluation & Confusion Matrix", expanded=False):
            cw_df = pd.DataFrame(model_metrics["class_wise_metrics"]).T.reset_index()
            cw_df.columns = ["Class", "Precision", "Recall", "F1-Score", "Test Support"]
            st.dataframe(cw_df.style.format({
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
                "Test Support": "{:,.0f}"
            }), use_container_width=True)

            st.caption(f"**Imbalance Analysis:** {model_metrics['performance_assessment']['class_imbalance_analysis']}")
            st.caption(f"**Causality Notice:** {model_metrics['performance_assessment']['causality_disclaimer']}")

    st.markdown("---")
    st.markdown("#### Live Accident Severity Scenario Predictor")

    with st.form("ml_prediction_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            inp_location = st.selectbox("Accident Location", loc_list, index=0)
            inp_road_type = st.selectbox("Road Type", sorted(df_raw["road_type"].unique()), index=1)
            inp_junction_type = st.selectbox("Junction Type", sorted(df_raw["junction_type"].unique()), index=0)
            inp_speed_limit = st.slider("Nominal Speed Limit (km/h)", 20, 80, 50, step=5)
            inp_veh_category = st.selectbox("Primary Vehicle", sorted(df_raw["vehicle_category"].unique()), index=1)

        with col_f2:
            inp_weather = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Light Rain", "Heavy Rain", "Fog"], index=4)
            inp_rainfall = st.slider("Rainfall (mm)", 0.0, 45.0, 5.0, step=0.5)
            inp_visibility = st.slider("Visibility (km)", 0.1, 10.0, 2.0, step=0.1)
            inp_traffic_vol = st.slider("Traffic Volume (vph)", 200, 6000, 2800, step=100)
            inp_veh_count = st.slider("Vehicles Involved", 1, 4, 2)

        with col_f3:
            inp_hour = st.slider("Hour of Incident (0-23)", 0, 23, 18)
            inp_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
            inp_month = st.slider("Month of Year", 1, 12, 10)
            inp_peak = "Peak" if inp_hour in [7, 8, 9, 10, 17, 18, 19, 20] else "Off-Peak"
            st.info(f"Derived Peak Flag: **{inp_peak}**")
            
            # Time period derived
            if 5 <= inp_hour < 12:
                inp_time_period = "Morning"
            elif 12 <= inp_hour < 17:
                inp_time_period = "Afternoon"
            elif 17 <= inp_hour < 21:
                inp_time_period = "Evening"
            else:
                inp_time_period = "Night"
            st.info(f"Derived Time Period: **{inp_time_period}**")

        predict_button = st.form_submit_button("Run Severity Classification", use_container_width=True)

    if predict_button:
        # Construct input dataframe
        scenario_df = pd.DataFrame([{
            "hour_of_day": inp_hour,
            "month": inp_month,
            "is_weekend": 1 if inp_day in ["Saturday", "Sunday"] else 0,
            "speed_limit_kmph": inp_speed_limit,
            "rainfall_mm": inp_rainfall,
            "visibility_km": inp_visibility,
            "traffic_volume_vph": inp_traffic_vol,
            "vehicle_count": inp_veh_count,
            "day_of_week": inp_day,
            "time_period": inp_time_period,
            "location": inp_location,
            "road_type": inp_road_type,
            "junction_type": inp_junction_type,
            "weather_condition": inp_weather,
            "peak_flag": inp_peak,
            "vehicle_category": inp_veh_category
        }])

        prediction = model_pipeline.predict(scenario_df)[0]
        probabilities = model_pipeline.predict_proba(scenario_df)[0]
        prob_dict = dict(zip(model_pipeline.classes_, probabilities))

        st.markdown("#### Scenario Risk Evaluation Results")
        res_col1, res_col2 = st.columns([1, 1.5])

        with res_col1:
            badge_class = f"badge-{prediction.lower()}"
            st.markdown(f"""
            <div style="background: #1e293b; padding: 24px; border-radius: 12px; border: 2px solid {COLOR_MAP.get(prediction, '#38bdf8')}; text-align: center;">
                <div style="font-size: 14px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Classified Severity Level</div>
                <div style="font-size: 32px; font-weight: 800; color: {COLOR_MAP.get(prediction, '#ffffff')}; margin: 12px 0;">{prediction.upper()}</div>
                <div style="font-size: 14px; color: #cbd5e1;">Confidence: <strong>{prob_dict[prediction]*100:.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("##### Predicted Class Probability Distribution")
            prob_df = pd.DataFrame({
                "Severity": list(prob_dict.keys()),
                "Probability": list(prob_dict.values())
            })
            fig_prob = px.bar(
                prob_df,
                x="Probability",
                y="Severity",
                orientation="h",
                color="Severity",
                color_discrete_map=COLOR_MAP,
                range_x=[0, 1]
            )
            fig_prob.update_layout(xaxis=dict(tickformat=".0%"), height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_prob, use_container_width=True)

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 12px; padding: 12px 0;">
    CH15 Road Safety Insight Project • Data Science Assessment • Powered by Python, Streamlit, Scikit-learn, Plotly & Pandas
</div>
""", unsafe_allow_html=True)
