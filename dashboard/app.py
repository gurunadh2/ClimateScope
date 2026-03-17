import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. ENTERPRISE PAGE CONFIGURATION ---
st.set_page_config(page_title="ClimateScope Enterprise UI", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Clean Enterprise UI Spacing and Typography */
    .stTabs [data-baseweb="tab-list"] { gap: 30px; border-bottom: 2px solid #e9ecef; }
    .stTabs [data-baseweb="tab"] { height: 55px; white-space: pre-wrap; font-size: 16px; font-weight: 600; color: #495057; }
    .stTabs [aria-selected="true"] { color: #1c83e1 !important; border-bottom: 3px solid #1c83e1 !important; }
    div[data-testid="metric-container"] { background-color: #ffffff; border: 1px solid #dee2e6; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 6px solid #1c83e1; }
    .plot-container { margin-bottom: 3rem; } /* Adds breathing room between stacked charts */
    </style>
    """, unsafe_allow_html=True)

st.title("🌍 ClimateScope: Executive Intelligence")
st.markdown("Explore macroeconomic climate trends with dynamic filtering and hierarchical data models.")
st.divider()

# --- 2. DATA PIPELINE ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'processed', 'weather_cleaned_daily.csv')
    data = pd.read_csv(csv_path)
    
    data['last_updated'] = pd.to_datetime(data['last_updated'])
    data['year_month'] = data['year_month'].astype(str)
    return data

df = load_data()

# --- 3. DYNAMIC ENTERPRISE SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Master Control Panel")
    st.markdown("Configure your analytical view.")
    
    # Primary & Secondary Metrics with robust tooltips
    numeric_cols = ['temperature_celsius', 'humidity', 'wind_kph', 'precip_mm', 'pressure_mb', 'uv_index', 'cloud']
    selected_var_col = st.selectbox("🎯 Primary Metric", numeric_cols, index=0, help="This metric will drive the main globe and trend lines.")
    
    secondary_options = [col for col in numeric_cols if col != selected_var_col]
    secondary_var_col = st.selectbox("⚖️ Secondary Metric", secondary_options, index=0, help="Used for cross-variable scatter plots.")
    
    st.markdown("---")
    
    # Multi-Select Regions
    st.subheader("🌍 Geography Focus")
    top_countries = df['country'].value_counts().head(10).index.tolist()
    selected_countries = st.multiselect("Target Countries", sorted(df['country'].unique()), default=top_countries[:4])
    
    # Advanced Filters Expander (Clean UI)
    with st.expander("🛠️ Advanced Dynamic Filters"):
        st.markdown("Layer additional parameters to narrow your dataset.")
        
        # Date Slider
        min_date, max_date = df['last_updated'].min().date(), df['last_updated'].max().date()
        date_selection = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        
        # Additional dynamic sliders
        wind_range = st.slider("Wind Speed (kph)", float(df['wind_kph'].min()), float(df['wind_kph'].max()), (0.0, 100.0))
        hum_range = st.slider("Humidity (%)", float(df['humidity'].min()), float(df['humidity'].max()), (0.0, 100.0))
        
        remove_outliers = st.checkbox("🛡️ Strip 1% Extreme Outliers", value=False)

# --- Apply Engine ---
if len(date_selection) == 2:
    start_date, end_date = pd.to_datetime(date_selection[0]), pd.to_datetime(date_selection[1])
    filtered_df = df[
        (df['country'].isin(selected_countries)) & 
        (df['last_updated'] >= start_date) & 
        (df['last_updated'] <= end_date) &
        (df['wind_kph'] >= wind_range[0]) & (df['wind_kph'] <= wind_range[1]) &
        (df['humidity'] >= hum_range[0]) & (df['humidity'] <= hum_range[1])
    ]
else:
    filtered_df = df[df['country'].isin(selected_countries)]

if remove_outliers and not filtered_df.empty:
    q_hi = filtered_df[selected_var_col].quantile(0.99)
    q_low = filtered_df[selected_var_col].quantile(0.01)
    filtered_df = filtered_df[(filtered_df[selected_var_col] <= q_hi) & (filtered_df[selected_var_col] >= q_low)]

# --- 4. EXECUTIVE KPIs ---
if filtered_df.empty:
    st.error("⚠️ Dataset is empty based on current filter combination. Please broaden your criteria.")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Mean {selected_var_col.split('_')[0].title()}", f"{filtered_df[selected_var_col].mean():.2f}")
    col2.metric(f"Max {selected_var_col.split('_')[0].title()}", f"{filtered_df[selected_var_col].max():.2f}")
    col3.metric(f"Mean {secondary_var_col.split('_')[0].title()}", f"{filtered_df[secondary_var_col].mean():.2f}")
    col4.metric("Active Data Rows", f"{len(filtered_df):,}")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # --- 5. ENTERPRISE VISUALIZATION TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 3D Macro Geography", 
        "📈 Temporal Trends", 
        "📊 Data Distributions", 
        "🔗 Multivariate Analysis"
    ])

    # TAB 1: 3D FILLED GLOBE (Choropleth)
    with tab1:
        st.markdown(f"### Global Distribution of {selected_var_col.title().replace('_', ' ')}")
        map_data = filtered_df.groupby('country', as_index=False)[selected_var_col].mean()
        
        # Enterprise 3D Filled Globe
        fig_globe = px.choropleth(
            map_data, locations="country", locationmode="country names",
            color=selected_var_col, hover_name="country", 
            projection="orthographic", # Forces the 3D globe view
            color_continuous_scale="Viridis"
        )
        
        # UI Enhancements: Adding ocean, borders, and smooth rotation
        fig_globe.update_geos(
            showcountries=True, countrycolor="rgba(255,255,255,0.5)",
            showocean=True, oceancolor="rgba(14, 30, 64, 0.05)",
            showland=True, landcolor="rgba(210, 210, 210, 0.2)",
            framecolor="rgba(0,0,0,0)", coastlinescolor="rgba(0,0,0,0.1)"
        )
        fig_globe.update_layout(margin={"r":0,"t":20,"l":0,"b":0}, height=600)
        st.plotly_chart(fig_globe, use_container_width=True)

    # TAB 2: TEMPORAL TRENDS
    with tab2:
        st.markdown("### Time-Series with Moving Averages")
        rolling_window = st.slider("Smoothing Window (Days)", min_value=1, max_value=30, value=7, help="Increases to smooth out daily noise.")
        
        time_data = filtered_df.groupby(['last_updated', 'country'])[selected_var_col].mean().reset_index().sort_values(by='last_updated')
        time_data['Trendline'] = time_data.groupby('country')[selected_var_col].transform(lambda x: x.rolling(rolling_window, 1).mean())
        
        fig_time = px.line(time_data, x='last_updated', y='Trendline', color='country')
        fig_time.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=500)
        st.plotly_chart(fig_time, use_container_width=True)

    # TAB 3: DEMYSTIFIED DISTRIBUTIONS
    with tab3:
        st.markdown("### Statistical Profiling")
        st.info("💡 **How to read this:** This tab shows how 'spread out' your data is. A tightly packed box means the weather is very consistent. A tall, stretched out shape means the weather fluctuates wildly.")
        
        # Enterprise UX Toggle
        chart_type = st.radio("Select View Complexity:", ["Standard Box Plot (Easier to read)", "Violin Plot (Shows data density)"], horizontal=True)
        
        if "Box" in chart_type:
            fig_dist = px.box(filtered_df, x="country", y=selected_var_col, color="country", points="outliers", notched=True)
        else:
            fig_dist = px.violin(filtered_df, x="country", y=selected_var_col, color="country", box=True, points="all")
            
        fig_dist.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    # TAB 4: MULTIVARIATE ANALYSIS (Full Width, Breathing Room)
    with tab4:
        st.markdown("### Cross-Variable Interactions")
        st.info("Analyzing how multiple climate factors impact each other globally.")
        
        # Chart 1: Full Width Scatter
        st.markdown(f"**Correlation: {selected_var_col} vs {secondary_var_col}**")
        fig_scatter = px.scatter(
            filtered_df.sample(n=min(3000, len(filtered_df))), 
            x=selected_var_col, y=secondary_var_col, 
            color="country", size="wind_kph" if selected_var_col != "wind_kph" and secondary_var_col != "wind_kph" else None,
            marginal_x="histogram", marginal_y="histogram", opacity=0.7, trendline="ols"
        )
        fig_scatter.update_layout(height=600) # Makes it nice and tall
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True) # Massive breathing room
        st.divider()
        
        # Chart 2: Full Width Sunburst
        st.markdown("**Hierarchical Data Volume (Country ➔ City)**")
        sun_data = filtered_df.groupby(['country', 'location_name']).size().reset_index(name='record_count')
        fig_sun = px.sunburst(sun_data, path=['country', 'location_name'], values='record_count', color='record_count', color_continuous_scale="Blues")
        fig_sun.update_layout(height=600, margin=dict(t=20, l=0, r=0, b=20))
        st.plotly_chart(fig_sun, use_container_width=True)