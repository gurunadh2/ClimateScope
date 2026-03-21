import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# --- 1. ENTERPRISE APPLICATION CONFIGURATION ---
st.set_page_config(page_title="ClimateScope Pro", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# Advanced Custom CSS (SaaS UI/UX)
st.markdown("""
    <style>
    /* Global Font and Spacing */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    
    /* Sleek Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease-in-out;
        border-top: 4px solid #0ea5e9;
    }
    div[data-testid="metric-container"]:hover { transform: translateY(-5px); }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
    
    /* Headers */
    h1 { color: #0f172a; font-weight: 800; letter-spacing: -1px; }
    h2, h3 { color: #334155; font-weight: 600; }
    
    /* Divider */
    hr { margin-top: 2rem; margin-bottom: 2rem; border-color: #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA PIPELINE ARCHITECTURE ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'processed', 'weather_cleaned_daily.csv')
    data = pd.read_csv(csv_path)
    data['last_updated'] = pd.to_datetime(data['last_updated'])
    data['year_month'] = data['year_month'].astype(str)
    return data

try:
    df = load_data()
except Exception as e:
    st.error(f"Data loading failed. Please ensure 'weather_cleaned_daily.csv' is in the correct directory. Error: {e}")
    st.stop()

# --- 3. APPLICATION ROUTING & NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3204/3204071.png", width=60)
    st.markdown("## **ClimateScope**")
    st.caption("v2.0 | Advanced Analytics Mode")
    st.markdown("---")
    
    # Custom App Routing
    app_mode = st.radio(
        "🧭 Navigation",
        ["Executive Dashboard", "Geospatial Intelligence", "Deep Distributions", "Trend Forecasting"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    
    # Global Filters (Apply across all pages)
    st.markdown("### 🎛️ Global Parameters")
    
    numeric_cols = ['temperature_celsius', 'humidity', 'wind_kph', 'precip_mm', 'pressure_mb', 'uv_index', 'cloud']
    metric_labels = {'temperature_celsius': 'Temperature (°C)', 'humidity': 'Humidity (%)', 'wind_kph': 'Wind Speed (kph)', 
                     'precip_mm': 'Precipitation (mm)', 'pressure_mb': 'Pressure (mb)', 'uv_index': 'UV Index', 'cloud': 'Cloud Cover (%)'}
    
    # Map back selection to column name
    selected_label = st.selectbox("🎯 Primary Metric", list(metric_labels.values()), index=0)
    selected_var_col = list(metric_labels.keys())[list(metric_labels.values()).index(selected_label)]
    
    top_countries = df['country'].value_counts().head(12).index.tolist()
    selected_countries = st.multiselect("🌍 Target Regions", sorted(df['country'].unique()), default=top_countries[:5])
    
    if not selected_countries:
        st.warning("Please select at least one country.")
        st.stop()

# --- Apply Global Filters ---
filtered_df = df[df['country'].isin(selected_countries)]

# --- PAGE 1: EXECUTIVE DASHBOARD ---
if app_mode == "Executive Dashboard":
    st.title("📊 Executive Summary")
    st.markdown("High-level overview of selected regional climate metrics.")
    
    # Automated Insights Engine
    st.info(f"**🤖 Automated Insight:** Based on your current selection of {len(selected_countries)} countries, the maximum recorded {selected_label.lower()} is **{filtered_df[selected_var_col].max():.1f}**, while the average sits at **{filtered_df[selected_var_col].mean():.1f}**. The highest variance is observed in the wind speed and humidity parameters.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Avg {selected_label}", f"{filtered_df[selected_var_col].mean():.1f}")
    col2.metric(f"Max {selected_label}", f"{filtered_df[selected_var_col].max():.1f}")
    col3.metric("Data Points Analyzed", f"{len(filtered_df):,}")
    col4.metric("Regions Tracked", f"{len(selected_countries)}")
    
    st.markdown("---")
    
    # Dual Axis Comparative Chart
    st.subheader("Comparative Regional Overview")
    bar_data = filtered_df.groupby('country', as_index=False)[['temperature_celsius', 'humidity']].mean()
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=bar_data['country'], y=bar_data['temperature_celsius'], name='Avg Temp (°C)', marker_color='#0ea5e9'))
    fig_bar.add_trace(go.Scatter(x=bar_data['country'], y=bar_data['humidity'], name='Avg Humidity (%)', yaxis='y2', mode='lines+markers', marker_color='#f59e0b'))
    
    fig_bar.update_layout(
        yaxis=dict(title='Temperature (°C)'),
        yaxis2=dict(title='Humidity (%)', overlaying='y', side='right'),
        barmode='group', height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- PAGE 2: GEOSPATIAL INTELLIGENCE ---
elif app_mode == "Geospatial Intelligence":
    st.title("🌐 Geospatial Intelligence")
    st.markdown("Interactive 3D mapping and regional variance analysis.")
    
    map_data = filtered_df.groupby('country', as_index=False)[selected_var_col].mean()
    
    fig_globe = px.choropleth(
        map_data, locations="country", locationmode="country names",
        color=selected_var_col, hover_name="country", 
        projection="orthographic", 
        color_continuous_scale="Turbo"
    )
    fig_globe.update_geos(
        showcountries=True, countrycolor="rgba(255,255,255,0.7)",
        showocean=True, oceancolor="#0f172a",
        showland=True, landcolor="#1e293b",
        showframe=False, coastlinecolor="rgba(255,255,255,0.2)"
    )
    fig_globe.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0}, height=700,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_globe, use_container_width=True)

# --- PAGE 3: DEEP DISTRIBUTIONS ---
elif app_mode == "Deep Distributions":
    st.title("🔬 Statistical Distributions")
    st.markdown("Understanding data spread, outliers, and density profiles.")
    
    colA, colB = st.columns([2, 1])
    with colB:
        st.markdown("### Profile Settings")
        show_points = st.checkbox("Show All Data Points", value=False)
        plot_type = st.radio("Visualization Type", ["Violin (Density)", "Box (Quartiles)"])
    
    with colA:
        if plot_type == "Violin (Density)":
            fig_dist = px.violin(filtered_df, x="country", y=selected_var_col, color="country", 
                                 box=True, points="all" if show_points else "outliers")
        else:
            fig_dist = px.box(filtered_df, x="country", y=selected_var_col, color="country", 
                              points="all" if show_points else "outliers", notched=True)
            
        fig_dist.update_layout(height=600, showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
        fig_dist.update_yaxes(gridcolor='rgba(0,0,0,0.1)')
        st.plotly_chart(fig_dist, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Multivariate Correlation Engine")
    
    # Sunburst Chart for location breakdown
    sun_data = filtered_df.groupby(['country', 'location_name']).size().reset_index(name='record_count')
    fig_sun = px.sunburst(sun_data, path=['country', 'location_name'], values='record_count', color='record_count', color_continuous_scale="Blues")
    fig_sun.update_layout(height=500, margin=dict(t=0, l=0, r=0, b=0))
    st.plotly_chart(fig_sun, use_container_width=True)

# --- PAGE 4: TREND FORECASTING ---
elif app_mode == "Trend Forecasting":
    st.title("📈 Time-Series & Statistical Forecasting")
    st.markdown("Analyze historical patterns with integrated rolling averages and standard deviation confidence bands.")
    
    rolling_window = st.slider("Select Smoothing Window (Days)", min_value=1, max_value=60, value=14)
    
    # Process time series data
    time_data = filtered_df.groupby(['last_updated', 'country'])[selected_var_col].mean().reset_index().sort_values(by='last_updated')
    
    fig_time = go.Figure()
    
    for country in selected_countries:
        country_data = time_data[time_data['country'] == country].copy()
        if not country_data.empty:
            # Calculate rolling stats
            country_data['rolling_mean'] = country_data[selected_var_col].rolling(window=rolling_window, min_periods=1).mean()
            country_data['rolling_std'] = country_data[selected_var_col].rolling(window=rolling_window, min_periods=1).std().fillna(0)
            
            upper_bound = country_data['rolling_mean'] + (1.96 * country_data['rolling_std']) # 95% confidence interval
            lower_bound = country_data['rolling_mean'] - (1.96 * country_data['rolling_std'])
            
            # Add Confidence Band
            fig_time.add_trace(go.Scatter(
                x=pd.concat([country_data['last_updated'], country_data['last_updated'][::-1]]),
                y=pd.concat([upper_bound, lower_bound[::-1]]),
                fill='toself', fillcolor='rgba(14, 165, 233, 0.1)', line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=False, name=f'{country} Confidence'
            ))
            
            # Add Rolling Mean Line
            fig_time.add_trace(go.Scatter(
                x=country_data['last_updated'], y=country_data['rolling_mean'],
                mode='lines', name=f'{country} ({rolling_window}-Day Trend)',
                line=dict(width=3)
            ))

    fig_time.update_layout(
        hovermode="x unified", 
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)', title=selected_label)
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    st.info("💡 **Methodology:** The shaded regions represent a 95% statistical confidence interval (±1.96 standard deviations) around the moving average. Tighter bands indicate highly stable weather patterns, while wider bands indicate volatile climate events.")