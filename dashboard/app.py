import streamlit as st
import textwrap
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# --- 1. ENTERPRISE CONFIGURATION ---
st.set_page_config(page_title="ClimateScope AI", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# --- ENTHUSIASTIC SaaS UI (Dark/Neon Theme) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    
    /* Glowing Title */
    .super-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(to right, #00f2fe, #4facfe, #00f2fe);
        background-size: 200% auto;
        color: transparent;
        -webkit-background-clip: text;
        animation: gradient 3s linear infinite;
        margin-bottom: 0px;
    }
    @keyframes gradient { 0% {background-position: 0% center;} 100% {background-position: 200% center;} }
    
    /* Premium Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(4px);
        border-top: 4px solid #38bdf8;
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px 0 rgba(56, 189, 248, 0.2);
        border-top: 4px solid #a855f7;
    }
    div[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 1.1rem !important;}
    div[data-testid="metric-container"] value { color: #f8fafc !important; font-weight: 700 !important;}
    
    hr { border-color: rgba(255,255,255,0.1); }
    </style>
    <h1 class="super-title">🌍 ClimateScope AI</h1>
    <p style='text-align: center; font-size: 1.2rem; color: #94a3b8; margin-bottom: 2rem;'>
    Next-Generation Global Climate Intelligence Platform
    </p>
    """, unsafe_allow_html=True)

# --- 2. ROBUST DATA PIPELINE ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, '..', 'data', 'processed', 'weather_cleaned_daily.csv')
    
    try:
        df = pd.read_csv(path)
    except:
        st.error("⚠️ Data file not found. Ensure 'weather_cleaned_daily.csv' is in your data folder.")
        st.stop()
        
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    df = df[(df['temperature_celsius'] >= -50) & (df['temperature_celsius'] <= 60)]

    df['year'] = df['last_updated'].dt.year
    df['month'] = df['last_updated'].dt.month
    df['year_month'] = df['last_updated'].dt.to_period('M').astype(str)

    # Data Normalization
    for col in ['temperature_celsius', 'wind_kph', 'precip_mm']:
        df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Compound Risk Score
    df['risk_score'] = ((df['temperature_celsius_norm'] * 0.4) + 
                        (df['wind_kph_norm'] * 0.3) + 
                        (df['precip_mm_norm'] * 0.3)) * 100
    return df

df = load_data()

# --- 3. DYNAMIC SIDEBAR ---
with st.sidebar:
    st.markdown("### 🎛️ Command Center")
    app_mode = st.radio(
        "Navigation",
        ["Executive Insights", 
         "Climate Change & Risk", 
         "AI Pattern Detection", 
         "Seasonal Decomposition", 
         "Scenario Simulator", 
         "Extreme Climate Analytics",
         "About the Platform"] # <-- ADDED THIS LINE
    )
    st.markdown("---")
    target_metric = st.selectbox("🎯 Target Analysis Metric", ['temperature_celsius', 'humidity', 'wind_kph', 'precip_mm', 'uv_index'])
    
    requested_defaults = ['India', 'United States of America', 'China', 'Germany', 'Brazil']
    valid_defaults = [c for c in requested_defaults if c in df['country'].unique()]
    selected_countries = st.multiselect("🌍 Target Regions", sorted(df['country'].unique()), default=valid_defaults)

    # NEW: Advanced Date Range Filter
    st.markdown("### 📅 Temporal Filter")
    min_date = df['last_updated'].min().date()
    max_date = df['last_updated'].max().date()
    date_range = st.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    

# Apply Global Filters (Country + Date)
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[
        (df['country'].isin(selected_countries)) & 
        (df['last_updated'].dt.date >= start_date) & 
        (df['last_updated'].dt.date <= end_date)
    ]
else:
    filtered_df = df[df['country'].isin(selected_countries)]

if filtered_df.empty:
    st.warning("⚠️ No data matches your filter criteria. Please broaden your date or country selection.")
    st.stop()

# --- PLOTLY THEME ---
PLOTLY_THEME = "plotly_dark"

# ==========================================

# MODULE 1: EXECUTIVE INSIGHTS

if app_mode == "Executive Insights":
    
    # 1. ADVANCED ENTERPRISE CSS & ANIMATIONS
    st.markdown("""
        <style>
        /* Flowing Liquid Gradient for the Main Project Title */
        .project-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 900;
            line-height: 1.2;
            background: linear-gradient(-45deg, #00f2fe, #4facfe, #818cf8, #c084fc, #00f2fe);
            background-size: 300% 300%;
            animation: flowGradient 6s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        
        .project-subtitle {
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            color: #e2e8f0;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 40px;
            animation: fadeIn 2s ease-in forwards;
        }

        /* Keyframe Animations */
        @keyframes flowGradient { 
            0% {background-position: 0% 50%;} 
            50% {background-position: 100% 50%;} 
            100% {background-position: 0% 50%;} 
        }
        @keyframes fadeInUp { 
            from { opacity: 0; transform: translateY(30px); } 
            to { opacity: 1; transform: translateY(0); } 
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 15px rgba(14, 165, 233, 0.2); }
            50% { box-shadow: 0 0 30px rgba(14, 165, 233, 0.6); }
            100% { box-shadow: 0 0 15px rgba(14, 165, 233, 0.2); }
        }

        /* Custom Animated KPI Grid */
        .kpi-wrapper {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 40px;
        }
        .kpi-card {
            flex: 1;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 16px;
            padding: 25px 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            border-top: 4px solid #38bdf8;
            opacity: 0; /* Start hidden for animation */
            animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        .kpi-card:hover {
            transform: translateY(-10px) scale(1.02);
            border-top: 4px solid #c084fc;
            animation: pulseGlow 2s infinite;
        }
        
        /* Staggered Animation Delays */
        .delay-1 { animation-delay: 0.1s; }
        .delay-2 { animation-delay: 0.1s; }
        .delay-3 { animation-delay: 0.1s; }
        .delay-4 { animation-delay: 0.1s; }

        .kpi-value { font-size: 2.5rem; font-weight: 800; color: #f8fafc; margin-bottom: 5px; }
        .kpi-label { font-size: 1rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;}
        </style>
        
        <h1 class="project-title">ClimateScope</h1>
        <h2 class="project-subtitle">Visualizing Global Weather Trends and Extreme Events</h2>
    """, unsafe_allow_html=True)

    # 2. DATA CALCULATIONS FOR KPIs
    avg_metric = f"{filtered_df[target_metric].mean():.2f}"
    max_metric = f"{filtered_df[target_metric].max():.2f}"
    avg_risk = f"{filtered_df['risk_score'].mean():.1f}"
    data_points = f"{len(filtered_df):,}"
    metric_name = target_metric.split('_')[0].title()

    # 3. RENDER CUSTOM ANIMATED KPI CARDS
    st.markdown(f"""
        <div class="kpi-wrapper">
            <div class="kpi-card delay-1">
                <div class="kpi-value">{avg_metric}</div>
                <div class="kpi-label">Average {metric_name}</div>
            </div>
            <div class="kpi-card delay-2">
                <div class="kpi-value">{max_metric}</div>
                <div class="kpi-label">Recorded Peak</div>
            </div>
            <div class="kpi-card delay-3">
                <div class="kpi-value">{avg_risk} <span style="font-size:1.2rem; color:#64748b;">/ 100</span></div>
                <div class="kpi-label">Systemic Risk Score</div>
            </div>
            <div class="kpi-card delay-4">
                <div class="kpi-value">{data_points}</div>
                <div class="kpi-label">Data Points Analyzed</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Chart 1: Regional Trends (Full Width) ---
    st.markdown("### 📈 Regional Metric Trends (Dual-Axis)")
    trend_data = filtered_df.groupby('country')[['temperature_celsius', 'precip_mm']].mean().reset_index()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(x=trend_data['country'], y=trend_data['precip_mm'], name='Avg Precip (mm)', marker_color='rgba(56, 189, 248, 0.8)'))
    fig_trend.add_trace(go.Scatter(x=trend_data['country'], y=trend_data['temperature_celsius'], name='Avg Temp (°C)', yaxis='y2', mode='lines+markers', marker_color='#c084fc', line=dict(width=4)))
    
    fig_trend.update_layout(template=PLOTLY_THEME, yaxis2=dict(overlaying='y', side='right'), margin=dict(t=30, b=0), height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.05rem; font-style: italic; margin-top: -10px;'>💡 <b>Insight:</b> This dual-axis chart compares average precipitation (blue bars) against temperature (purple line). It allows executives to instantly identify if warmer regions concurrently experience heavier rainfall over the selected timeframe.</p>", unsafe_allow_html=True)

    st.divider()

    # --- Chart 2: Smart Correlation Engine (Full Width) ---
    st.markdown("### 🔗 Dynamic Correlation Engine")
    
    corr_cols = ['temperature_celsius', 'humidity', 'wind_kph', 'precip_mm', 'uv_index', 'pressure_mb']
    corr_matrix = filtered_df[corr_cols].corr()
    
    corr_pairs = corr_matrix.unstack().dropna()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) < corr_pairs.index.get_level_values(1)]
    top_corrs = corr_pairs.reindex(corr_pairs.abs().sort_values(ascending=False).index).head(3)
    
    # Animated Insight Box
    st.markdown("<div style='background: rgba(14, 165, 233, 0.05); border-left: 4px solid #0ea5e9; padding: 20px; margin-bottom: 20px; border-radius: 8px; animation: fadeInUp 1s ease forwards;'>", unsafe_allow_html=True)
    st.markdown("#### 🔥 Top Algorithmic Discoveries")
    
    for (var1, var2), val in top_corrs.items():
        v1_clean = var1.replace('_', ' ').title().replace('Celsius', '°C').replace('Kph', 'kph')
        v2_clean = var2.replace('_', ' ').title().replace('Celsius', '°C').replace('Kph', 'kph')
        
        strength = "Strong Positive 📈" if val > 0.5 else "Strong Negative 📉" if val < -0.5 else "Moderate 📊"
        color = "#ef4444" if val > 0 else "#38bdf8" 
        
        st.markdown(f"- **{v1_clean}** and **{v2_clean}** have a correlation of <span style='color:{color}; font-weight:bold; font-size:1.1rem;'>{val:.2f}</span>  *( {strength} )*", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto", zmin=-1, zmax=1)
    fig_corr.update_layout(template=PLOTLY_THEME, margin=dict(t=10, b=0, l=0, r=0), height=550, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.05rem; font-style: italic; margin-top: -10px;'>💡 <b>Insight:</b> Use the Date and Country filters in the sidebar to see how these mathematical relationships change across different seasons and geographies.</p>", unsafe_allow_html=True)
# ==========================================

# MODULE 2: CLIMATE CHANGE & RISK (ANIMATED GLOBE)

elif app_mode == "Climate Change & Risk":
    st.subheader("🌐 Global Risk Simulation (Animated Timeline)")
    st.markdown("Press the **Play** button below to watch extreme weather risk evolve over time across the globe.")

    # Prepare data for animation
    anim_data = df.groupby(['year_month', 'country'])['risk_score'].mean().reset_index().sort_values('year_month')
    
    fig_globe = px.choropleth(
        anim_data, locations="country", locationmode="country names",
        color="risk_score", animation_frame="year_month",
        projection="orthographic", color_continuous_scale="Plasma",
        range_color=[0, anim_data['risk_score'].max()]
    )
    
    # FIXED: Changed 'coastlinescolor' to 'coastlinecolor'
    fig_globe.update_geos(
        showocean=True, oceancolor="rgba(10, 15, 36, 1)",
        showland=True, landcolor="#1e293b",
        showframe=False, coastlinecolor="rgba(255,255,255,0.2)", 
        projection_rotation=dict(lon=20, lat=20, roll=0)
    )
    fig_globe.update_layout(template=PLOTLY_THEME, height=750, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_globe, use_container_width=True)

# ==========================================

# MODULE 3: AI PATTERN DETECTION

elif app_mode == "AI Pattern Detection":
    st.subheader("🧠 Machine Learning Climate Clusters")
    st.markdown("Using **K-Means Clustering** to sort regions with identical micro-climates, and **PCA** to visualize 5-dimensional data in 3D space.")
    
    features = ['temperature_celsius', 'humidity', 'wind_kph', 'precip_mm', 'pressure_mb']
    ml_data = df.groupby('country')[features].mean().dropna()
    scaled = StandardScaler().fit_transform(ml_data)

    # Stylized Control Box for the Slider
    st.markdown("<div style='background: rgba(15, 23, 42, 0.6); border: 1px solid #a855f7; padding: 20px; border-radius: 12px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    n_clusters = st.slider("🧮 Select K-Means Clusters (Buckets)", 2, 8, 4)
    st.markdown("</div>", unsafe_allow_html=True)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    ml_data['Cluster'] = kmeans.fit_predict(scaled).astype(str)

    # 3D PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled)
    ml_data['PCA_X'] = pca_result[:, 0]
    ml_data['PCA_Y'] = pca_result[:, 1]
    ml_data['PCA_Z'] = pca_result[:, 2]

    # --- GRAPH 1: 3D PCA (Full Width) ---
    st.markdown("### 1. 3D Dimensionality Reduction (PCA)")
    fig_pca = px.scatter_3d(ml_data.reset_index(), x='PCA_X', y='PCA_Y', z='PCA_Z', color='Cluster', hover_name='country', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pca.update_layout(template=PLOTLY_THEME, height=700, margin=dict(t=0, b=0, l=0, r=0)) # Height increased to 700 for a massive 3D view
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # --- EXPLANATION BOX 1 ---
    st.markdown("""
    <div style='background: rgba(15, 23, 42, 0.6); border-left: 4px solid #38bdf8; padding: 15px; border-radius: 8px; margin-top: -10px;'>
        <p style='color: #e2e8f0; font-size: 1.05rem; margin-bottom: 8px; font-weight: 600;'>💡 How to read this 3D Map:</p>
        <ul style='color: #94a3b8; font-size: 0.95rem; margin-top: 0; line-height: 1.6;'>
            <li><b>The Dots:</b> Every single dot represents a specific country.</li>
            <li><b>The Colors:</b> Dots of the same color belong to the same AI-assigned climate bucket.</li>
            <li><b>The Distance:</b> Countries floating close together have nearly identical overall climates across all 5 variables, regardless of actual geography.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()

    # --- GRAPH 2: Cluster Profile Radar (Full Width) ---
    st.markdown("### 2. Cluster Profile (Radar)")
    cluster_avg = ml_data.groupby('Cluster')[features].mean()
    # Scale for radar readability
    radar_scaled = StandardScaler().fit_transform(cluster_avg)
    
    fig_radar = go.Figure()
    for i, cluster in enumerate(cluster_avg.index):
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_scaled[i], theta=features, fill='toself', name=f'Cluster {cluster}'
        ))
    fig_radar.update_layout(template=PLOTLY_THEME, height=650, polar=dict(radialaxis=dict(visible=False)), margin=dict(t=40, b=40, l=40, r=40)) # Height increased to 650
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # --- EXPLANATION BOX 2 ---
    st.markdown("""
    <div style='background: rgba(15, 23, 42, 0.6); border-left: 4px solid #a855f7; padding: 15px; border-radius: 8px; margin-top: -10px;'>
        <p style='color: #e2e8f0; font-size: 1.05rem; margin-bottom: 8px; font-weight: 600;'>💡 How to read this Radar:</p>
        <ul style='color: #94a3b8; font-size: 0.95rem; margin-top: 0; line-height: 1.6;'>
            <li><b>The Decoder:</b> This explains <i>why</i> the AI grouped those countries together.</li>
            <li><b>The Shape:</b> Each colored polygon shows the "average weather footprint" of that cluster.</li>
            <li><b>The Spikes:</b> If a shape spikes heavily outward toward "wind_kph", it means that specific cluster is defined by intense winds.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==========================================

# MODULE 4: SEASONAL DECOMPOSITION

elif app_mode == "Seasonal Decomposition":
    st.subheader("📈 Statistical Time-Series Decomposition")
    st.markdown("Deconstruct complex weather patterns into three underlying mathematical forces: Core Trends, Repeating Seasonal Cycles, and Unpredictable Noise.")
    
    # Allow multiple countries for comparative decomposition
    target_countries = st.multiselect(
        "Select Regions for Comparative Decomposition", 
        selected_countries, 
        default=selected_countries[:2] if len(selected_countries) >= 2 else selected_countries
    )

    if not target_countries:
        st.warning("Please select at least one region to analyze.")
        st.stop()

    # Calculate decomposition for all selected countries
    decomp_results = {}
    for country in target_countries:
        ts = filtered_df[filtered_df['country'] == country].sort_values('last_updated')
        ts_daily = ts.groupby(ts['last_updated'].dt.date)[target_metric].mean().reset_index()
        ts_daily.set_index('last_updated', inplace=True)

        if len(ts_daily) > 14:
            # We use extrapolate_trend to handle NaN edges cleanly for plotting
            result = seasonal_decompose(ts_daily[target_metric].dropna(), period=7, extrapolate_trend='freq')
            decomp_results[country] = result

    if not decomp_results:
        st.warning("Not enough continuous daily data points found to establish a statistical season.")
        st.stop()

    # Use Plotly's premium color palette for high contrast
    colors = px.colors.qualitative.Pastel

    # --- GRAPH 1: The Underlying Macro Trend ---
    st.markdown("### 1. The Underlying Macro Trend")
    fig_trend = go.Figure()
    for i, (country, res) in enumerate(decomp_results.items()):
        fig_trend.add_trace(go.Scatter(x=res.trend.index, y=res.trend, mode='lines', name=country, line=dict(width=3, color=colors[i % len(colors)])))

    fig_trend.update_layout(template=PLOTLY_THEME, height=500, hovermode="x unified", margin=dict(t=20, b=0, l=0, r=0))
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("""
    <div style='background: rgba(15, 23, 42, 0.6); border-left: 4px solid #0ea5e9; padding: 15px; border-radius: 8px; margin-top: -10px; margin-bottom: 30px;'>
        <p style='color: #94a3b8; font-size: 1.05rem; margin: 0; font-style: italic;'>💡 <b>Insight:</b> The 'Trend' line strips away all daily weather noise and seasonal fluctuations to reveal the true, long-term trajectory. Use this to compare which regions are consistently warming up or cooling down.</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- GRAPH 2: Cyclical Seasonality Patterns ---
    st.markdown("### 2. Cyclical Seasonality Patterns")
    fig_season = go.Figure()
    for i, (country, res) in enumerate(decomp_results.items()):
        fig_season.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, mode='lines', name=country, line=dict(width=2, color=colors[i % len(colors)]), opacity=0.8))

    fig_season.update_layout(template=PLOTLY_THEME, height=450, hovermode="x unified", margin=dict(t=20, b=0, l=0, r=0))
    st.plotly_chart(fig_season, use_container_width=True)

    st.markdown("""
    <div style='background: rgba(15, 23, 42, 0.6); border-left: 4px solid #a855f7; padding: 15px; border-radius: 8px; margin-top: -10px; margin-bottom: 30px;'>
        <p style='color: #94a3b8; font-size: 1.05rem; margin: 0; font-style: italic;'>💡 <b>Insight:</b> This isolates the predictable, repeating cycles. Taller, more aggressive waves indicate regions with extreme seasonal swings, while flatter waves belong to highly stable, unvarying climates (like the tropics).</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- GRAPH 3: Residual Volatility (Scatter Plot) ---
    st.markdown("### 3. Residual Volatility (Unpredictable Anomalies)")
    fig_resid = go.Figure()
    for i, (country, res) in enumerate(decomp_results.items()):
        fig_resid.add_trace(go.Scatter(x=res.resid.index, y=res.resid, mode='markers', name=country, marker=dict(size=7, opacity=0.7, color=colors[i % len(colors)])))

    # Add a zero baseline to show deviations clearly
    fig_resid.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig_resid.update_layout(template=PLOTLY_THEME, height=450, hovermode="closest", margin=dict(t=20, b=0, l=0, r=0))
    st.plotly_chart(fig_resid, use_container_width=True)

    st.markdown("""
    <div style='background: rgba(15, 23, 42, 0.6); border-left: 4px solid #f43f5e; padding: 15px; border-radius: 8px; margin-top: -10px;'>
        <p style='color: #94a3b8; font-size: 1.05rem; margin: 0; font-style: italic;'>💡 <b>Insight:</b> 'Residuals' represent chaotic weather events that cannot be explained by the macro trend or normal seasons. Dots drifting far away from the center zero-line pinpoint sudden freaks of nature, like a flash flood or instant heatwave.</p>
    </div>
    """, unsafe_allow_html=True)
# ==========================================

# MODULE 5: SCENARIO SIMULATOR

elif app_mode == "Scenario Simulator":
    st.subheader("🔬 \"What-If\" Climate Simulator")
    st.markdown("Model the compounding impacts of theoretical temperature shifts on global weather patterns and systemic risk.")

    # Main Control Slider
    st.markdown("<div style='background: rgba(15, 23, 42, 0.6); border: 1px solid #38bdf8; padding: 20px; border-radius: 12px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    temp_shift = st.slider("🌡️ Simulate Global Temperature Increase (°C)", 0.0, 5.0, 1.5, step=0.1)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Mathematical Simulation Engine
    sim = filtered_df.copy()
    sim['Baseline Temp'] = sim['temperature_celsius']
    sim['Projected Temp'] = sim['temperature_celsius'] + temp_shift
    sim['simulated_risk'] = sim['risk_score'] * (1 + (temp_shift * 0.08))
    sim.loc[sim['simulated_risk'] > 100, 'simulated_risk'] = 100 # Cap risk at 100
    
    st.divider()

    # --- CHART 1: Distribution Shift ---
    st.markdown("### 1. Global Temperature Shift (Distribution)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=sim['Baseline Temp'], name='Current Baseline', opacity=0.75, marker_color='#0ea5e9')) # Vibrant Cyan
    fig_hist.add_trace(go.Histogram(x=sim['Projected Temp'], name=f'+{temp_shift}°C Projection', opacity=0.75, marker_color='#f43f5e')) # Vibrant Rose/Red
    fig_hist.update_layout(template=PLOTLY_THEME, barmode='overlay', height=550, margin=dict(t=20, b=0, l=0, r=0))
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.05rem; font-style: italic; margin-top: -10px;'>💡 <b>Insight:</b> This overlaid histogram visualizes the entire global dataset shifting to the right. Notice how the 'tail' of the red projected data creates entirely new extremes that never existed in the baseline.</p>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True) # Deep breathing room

    # --- CHART 2: Regional Extremes Forecast ---
    st.markdown("### 2. Regional Heatwave Projections")
    sim_melt = pd.melt(sim, id_vars=['country'], value_vars=['Baseline Temp', 'Projected Temp'], var_name='Scenario', value_name='Temperature')
    fig_box = px.box(sim_melt, x='country', y='Temperature', color='Scenario', color_discrete_sequence=['#3b82f6', '#ef4444'], points="outliers")
    fig_box.update_layout(template=PLOTLY_THEME, height=550, margin=dict(t=20, b=0, l=0, r=0))
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.05rem; font-style: italic; margin-top: -10px;'>💡 <b>Insight:</b> The box plot reveals how specific regions will physically experience the shift. The dots extending above the solid boxes represent severe, sudden heatwave events that break normal boundaries.</p>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True) # Deep breathing room

    # --- CHART 3: Risk Escalation ---
    st.markdown("### 3. Compounding Systemic Risk")
    impact_data = pd.DataFrame({
        'Current Risk': filtered_df.groupby('country')['risk_score'].mean(),
        'Projected Risk': sim.groupby('country')['simulated_risk'].mean()
    }).reset_index()
    
    fig_impact = px.bar(impact_data, x='country', y=['Current Risk', 'Projected Risk'], barmode='group', color_discrete_sequence=['#10b981', '#f59e0b']) # Emerald Green and Amber
    fig_impact.update_layout(template=PLOTLY_THEME, height=550, margin=dict(t=20, b=0, l=0, r=0))
    st.plotly_chart(fig_impact, use_container_width=True)
    
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.05rem; font-style: italic; margin-top: -10px;'>💡 <b>Insight:</b> This compares current climate stability to the simulated reality. Because weather variables compound, even a small temperature rise exponentially escalates a region's overall Extreme Risk Score.</p>", unsafe_allow_html=True)

# ==========================================

# MODULE 6: EXTREME CLIMATE ANALYTICS

elif app_mode == "Extreme Climate Analytics":
    st.subheader("🌪️ Extreme Event Profiling")
    st.markdown("Analyze the absolute boundaries of global weather. This module isolates the highest peaks and lowest drops across key climate variables, allowing you to identify regions experiencing the most severe weather anomalies.")
    
    metrics = {
        "Temperature (°C)": "temperature_celsius",
        "Humidity (%)": "humidity",
        "Wind Speed (kph)": "wind_kph",
        "Precipitation (mm)": "precip_mm"
    }
    
    # NEW: Dynamic Polished Descriptions for the Presentation
    metric_descriptions = {
        "Temperature (°C)": "🌡️ **Temperature Profiling:** Tracks the absolute hottest heatwaves and most severe cold snaps. Maximums indicate severe drought and heatstroke risk, while extreme minimums highlight infrastructural freezing vulnerabilities.",
        "Humidity (%)": "💧 **Atmospheric Moisture:** High extremes (especially combined with heat) create dangerous 'wet-bulb' conditions where the human body cannot cool itself. Conversely, extreme low humidity accelerates dehydration and catastrophic wildfire risks.",
        "Wind Speed (kph)": "💨 **Wind Volatility:** Maximums highlight regions battered by severe gales, hurricanes, or typhoons. Minimum wind speeds indicate stagnant air masses, which can trap urban air pollution and create smog events.",
        "Precipitation (mm)": "☔ **Precipitation & Drought:** Tracks peak rainfall and flash flood risks. <br><span style='color:#94a3b8; font-size:0.9rem;'><i><b>Data Note on 0.0 mm Minimums:</b> Because this dataset aggregates readings from specific metropolitan weather stations rather than entire national landmasses, a 0.0 mm minimum accurately reflects that the primary population center experienced a completely dry period, even if remote regions received isolated rain.</i></span>"
    }
    
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_ext_metric = st.radio("Select Profile Metric:", list(metrics.keys()), horizontal=True)
    with filter_col2:
        time_resolution = st.radio("Select Time Resolution:", ["Daily Records", "Monthly Averages"], horizontal=True)
        
    # Inject the dynamic description box right below the selectors
    st.markdown(f"""
    <div style='background: rgba(15, 23, 42, 0.6); border-left: 4px solid #38bdf8; padding: 15px; border-radius: 8px; margin-bottom: 25px; margin-top: 10px;'>
        <p style='color: #e2e8f0; font-size: 1.05rem; margin: 0; line-height: 1.5;'>{metric_descriptions[selected_ext_metric]}</p>
    </div>
    """, unsafe_allow_html=True)
        
    col = metrics[selected_ext_metric]
    
    if time_resolution == "Monthly Averages":
        analysis_df = df.groupby(['country', 'year_month'])[col].mean().reset_index()
    else:
        analysis_df = df[['country', col]]
    
    grouped = analysis_df.groupby('country')[col]
    stats = pd.DataFrame({
        'Country': grouped.mean().index,
        'Max': grouped.max().values,
        'Min': grouped.min().values
    })
    
    highest = stats.loc[stats['Max'].idxmax()]
    lowest = stats.loc[stats['Min'].idxmin()]
    
    resolution_text = "daily record" if time_resolution == "Daily Records" else "sustained monthly average"
    
    st.info(f"**🤖 Deep Insight:** The absolute most extreme {resolution_text} for {selected_ext_metric.lower()} belongs to **{highest['Country']}** at **{highest['Max']:.2f}**. Conversely, **{lowest['Country']}** represents the lowest boundary at **{lowest['Min']:.2f}**.")
    
    top10_max = stats.sort_values(by='Max', ascending=False).head(10)
    top10_min = stats.sort_values(by='Min', ascending=True).head(10)
    
    colA, colB = st.columns(2)
    
    with colA:
        fig_max = px.bar(top10_max, x='Max', y='Country', color='Max', orientation='h', 
                         color_continuous_scale="Turbo", title=f"Top 10 Highest ({time_resolution})",
                         text_auto=".2f") 
        fig_max.update_layout(template=PLOTLY_THEME, height=450, yaxis={'categoryorder':'total ascending'}, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_max, use_container_width=True)
        
    with colB:
        color_target = 'Min' if top10_min['Min'].nunique() > 1 else None
        
        fig_min = px.bar(top10_min, x='Min', y='Country', color=color_target, orientation='h', 
                         color_continuous_scale="Tealgrn", title=f"Top 10 Lowest ({time_resolution})",
                         text_auto=".2f")
        
        if top10_min['Min'].max() == 0:
            fig_min.update_xaxes(range=[0, 1])
            
        fig_min.update_layout(template=PLOTLY_THEME, height=450, yaxis={'categoryorder':'total descending'}, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_min, use_container_width=True)

# ==========================================


# MODULE 7: ABOUT THE PLATFORM
# ==========================================
elif app_mode == "About the Platform":
    st.markdown("""
<style>
.about-header { text-align: center; font-size: 3rem; font-weight: 900; background: linear-gradient(-45deg, #00f2fe, #4facfe, #818cf8, #c084fc, #00f2fe); background-size: 300% 300%; animation: flowGradient 6s ease infinite; -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }
.about-subtitle { text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 50px; letter-spacing: 1px; }
.glass-card-wide { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 16px; padding: 40px; backdrop-filter: blur(10px); margin-bottom: 40px; transition: all 0.4s ease; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5); }
.glass-card-wide:hover { transform: translateY(-5px); border-color: rgba(192, 132, 252, 0.6); box-shadow: 0 20px 40px -10px rgba(192, 132, 252, 0.2); }
.social-btn { display: inline-block; padding: 12px 30px; margin: 10px; border-radius: 30px; color: #fff !important; text-decoration: none !important; font-weight: 600; font-size: 1.05rem; transition: all 0.3s ease; }
.github-btn { background: rgba(30, 41, 59, 0.8); border: 1px solid #475569; }
.github-btn:hover { background: #334155; transform: scale(1.05); border-color: #94a3b8; }
.linkedin-btn { background: rgba(14, 165, 233, 0.8); border: 1px solid #0284c7; }
.linkedin-btn:hover { background: #0ea5e9; transform: scale(1.05); border-color: #7dd3fc; }
.tech-badge { display: inline-block; background: rgba(56, 189, 248, 0.1); color: #38bdf8; padding: 10px 20px; border-radius: 30px; font-weight: 600; font-size: 1rem; margin: 8px; border: 1px solid rgba(56, 189, 248, 0.3); transition: all 0.3s ease; }
.tech-badge:hover { background: rgba(56, 189, 248, 0.2); transform: translateY(-2px); border-color: #38bdf8;}
</style>
<h1 class="about-header">Architecting Climate Intelligence</h1>
<p class="about-subtitle">Bridging the gap between raw global meteorological data and actionable systemic risk analysis.</p>
<div class="glass-card-wide" style="border-top: 4px solid #38bdf8;">
<h2 style="color: #f8fafc; margin-top:0; font-size: 2rem;">🌍 Executive Mission</h2>
<p style="color: #cbd5e1; line-height: 1.8; font-size: 1.15rem;"><b>ClimateScope</b> was engineered to transform how organizations, researchers, and policymakers interact with planetary weather data. Traditional dashboards rely on static, backward-looking reports. ClimateScope introduces a dynamic, forward-looking paradigm.</p>
<p style="color: #cbd5e1; line-height: 1.8; font-size: 1.15rem;">By leveraging unsupervised machine learning (K-Means Clustering) and real-time dimensionality reduction (PCA), this platform finds hidden micro-climates across borders. It processes over <b>120,000+ localized data points</b>, allowing decision-makers to simulate extreme temperature shifts and profile global climate anomalies with zero latency.</p>
</div>
<div class="glass-card-wide" style="border-top: 4px solid #10b981;">
<h2 style="color: #f8fafc; margin-top:0; font-size: 2rem;">⚙️ System Architecture & Tech Stack</h2>
<p style="color: #cbd5e1; line-height: 1.8; font-size: 1.15rem; margin-bottom: 25px;">Built for high performance and seamless interactivity, ClimateScope utilizes a modern Python-based data science stack. The architecture handles dynamic grouping, advanced statistical decomposition, and interactive rendering entirely in the cloud.</p>
<div style="display: flex; flex-wrap: wrap; gap: 10px;">
<span class="tech-badge">🐍 Python 3.10+</span>
<span class="tech-badge">☁️ Streamlit Cloud</span>
<span class="tech-badge">📊 Plotly Enterprise Graphic Engine</span>
<span class="tech-badge">🤖 Scikit-Learn (ML)</span>
<span class="tech-badge">📈 Statsmodels (Time Series)</span>
<span class="tech-badge">🧮 Pandas & NumPy</span>
</div>
</div>
<div class="glass-card-wide" style="border-top: 4px solid #c084fc; text-align: center;">
<div style="width: 150px; height: 150px; background: linear-gradient(135deg, #38bdf8, #c084fc); border-radius: 50%; margin: 0 auto 25px auto; display: flex; align-items: center; justify-content: center; font-size: 4.5rem; box-shadow: 0 0 30px rgba(192, 132, 252, 0.4); border: 4px solid rgba(255,255,255,0.1);">👨‍💻</div>
<h1 style="color: #f8fafc; margin-bottom: 5px; font-size: 2.5rem;">GURU</h1>
<p style="color: #38bdf8; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; font-size: 1.1rem; margin-top: 0;">Lead Data Scientist & Platform Architect</p>
<p style="color: #94a3b8; font-size: 1.1rem; line-height: 1.8; max-width: 800px; margin: 25px auto;">Specializing in full-stack data science, machine learning architecture, and high-performance interactive visualizations. GURU designed and deployed the ClimateScope AI engine from initial raw data ingestion, through feature engineering, down to the final Enterprise cloud deployment and custom UI/UX design.</p>
<div style="margin-top: 35px;">
<a href="https://github.com/gurunadh2" target="_blank" class="social-btn github-btn"><img src="https://img.icons8.com/ios-glyphs/30/ffffff/github.png" style="vertical-align: middle; margin-right: 8px; width: 22px;"/> GitHub</a>
<a href="https://www.linkedin.com/in/gurunadh-nandigama-7655222b8?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3Bsdsn1AUGS4SJTOq8MvVWpA%3D%3D" target="_blank" class="social-btn linkedin-btn"><img src="https://img.icons8.com/ios-filled/50/ffffff/linkedin.png" style="vertical-align: middle; margin-right: 8px; width: 22px;"/> LinkedIn</a>
</div>
</div>
""", unsafe_allow_html=True)