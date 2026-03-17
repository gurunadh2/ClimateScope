import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. ENTERPRISE PAGE CONFIGURATION ---
st.set_page_config(page_title="ClimateScope Enterprise", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* Professional BI Tool CSS Enhancements */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    div[data-testid="metric-container"] { background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 15px; border-radius: 8px; border-left: 5px solid #1c83e1; box-shadow: 1px 1px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🌍 ClimateScope: Enterprise Analytics")
st.markdown("Advanced global climate intelligence, statistical distributions, and hierarchical data modeling.")

# --- 2. DATA PIPELINE ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'processed', 'weather_cleaned_daily.csv')
    data = pd.read_csv(csv_path)
    
    # Advanced Date Parsing for Time Sliders
    data['last_updated'] = pd.to_datetime(data['last_updated'])
    data['year_month'] = data['year_month'].astype(str)
    
    # Create a mock 'Continent' column for hierarchical charting if it doesn't exist
    # (Assuming basic text matching for demonstration; adjust as needed)
    if 'continent' not in data.columns:
        data['continent'] = 'Global' 
    return data

df = load_data()

# --- 3. POWER BI STYLE SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3204/3204071.png", width=50) # Generic analytic icon
    st.header("Global Master Filters")
    
    # 1. Primary & Secondary Metrics
    numeric_cols = ['temperature_celsius', 'humidity', 'wind_kph', 'precip_mm', 'pressure_mb', 'uv_index', 'cloud']
    selected_var_col = st.selectbox("🎯 Primary Analysis Metric", numeric_cols, index=0)
    secondary_var_col = st.selectbox("⚖️ Secondary Comparison Metric", numeric_cols, index=1)
    
    st.markdown("---")
    
    # 2. Advanced Date Range Picker
    min_date = df['last_updated'].min().date()
    max_date = df['last_updated'].max().date()
    date_selection = st.date_input("📅 Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    
    # 3. Multi-Select Regions
    top_countries = df['country'].value_counts().head(10).index.tolist()
    selected_countries = st.multiselect("🌍 Target Countries", sorted(df['country'].unique()), default=top_countries[:4])
    
    # 4. Outlier Removal Toggle (Senior Level Feature)
    remove_outliers = st.checkbox("🛡️ Filter Extreme Outliers (99th Percentile)", value=False)

# --- Apply Filters ---
if len(date_selection) == 2:
    start_date, end_date = pd.to_datetime(date_selection[0]), pd.to_datetime(date_selection[1])
    filtered_df = df[
        (df['country'].isin(selected_countries)) & 
        (df['last_updated'] >= start_date) & 
        (df['last_updated'] <= end_date)
    ]
else:
    filtered_df = df[df['country'].isin(selected_countries)]

if remove_outliers and not filtered_df.empty:
    q_hi = filtered_df[selected_var_col].quantile(0.99)
    q_low = filtered_df[selected_var_col].quantile(0.01)
    filtered_df = filtered_df[(filtered_df[selected_var_col] < q_hi) & (filtered_df[selected_var_col] > q_low)]

# --- 4. DYNAMIC KPI DASHBOARD ---
if filtered_df.empty:
    st.warning("⚠️ No data matches your filter criteria. Please broaden your date or country selection.")
else:
    # Top Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Mean {selected_var_col.split('_')[0].title()}", f"{filtered_df[selected_var_col].mean():.2f}")
    col2.metric(f"Max {selected_var_col.split('_')[0].title()}", f"{filtered_df[selected_var_col].max():.2f}")
    col3.metric(f"Mean {secondary_var_col.split('_')[0].title()}", f"{filtered_df[secondary_var_col].mean():.2f}")
    col4.metric("Active Data Points", f"{len(filtered_df):,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 5. ENTERPRISE VISUALIZATION TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Macro Geography", 
        "📈 Time-Series & Forecasting", 
        "📊 Statistical Distributions", 
        "🔗 Multivariate Analysis"
    ])

    # TAB 1: ADVANCED GEOGRAPHY
    with tab1:
        st.markdown("### Global Density & Distribution")
        map_col1, map_col2 = st.columns([3, 1])
        
        with map_col1:
            # High-end Bubble Map using Scatter Geo
            fig_geo = px.scatter_geo(
                filtered_df.groupby('country', as_index=False).mean(numeric_only=True), 
                locations="country", locationmode="country names",
                color=selected_var_col, size=secondary_var_col,
                hover_name="country", projection="orthographic",
                color_continuous_scale="Plasma", title=f"3D Globe: {selected_var_col} vs {secondary_var_col}"
            )
            fig_geo.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_geo, use_container_width=True)
            
        with map_col2:
            st.markdown("**Top 5 Regions by Metric**")
            top_bar = filtered_df.groupby('country')[selected_var_col].mean().sort_values(ascending=True).tail(5)
            fig_bar = px.bar(top_bar, orientation='h', color=top_bar.values, color_continuous_scale="Plasma")
            fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_bar, use_container_width=True)

    # TAB 2: ADVANCED TIME SERIES
    with tab2:
        st.markdown("### Temporal Analysis with Moving Averages")
        # Toggle for rolling average
        rolling_window = st.slider("Select Rolling Average Window (Days)", min_value=1, max_value=30, value=7)
        
        time_data = filtered_df.groupby(['last_updated', 'country'])[selected_var_col].mean().reset_index()
        time_data = time_data.sort_values(by='last_updated')
        time_data['Rolling_Avg'] = time_data.groupby('country')[selected_var_col].transform(lambda x: x.rolling(rolling_window, 1).mean())
        
        fig_time = px.line(time_data, x='last_updated', y='Rolling_Avg', color='country', 
                           title=f"{selected_var_col} - {rolling_window} Day Moving Average")
        fig_time.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_time, use_container_width=True)

    # TAB 3: STATISTICAL DISTRIBUTIONS
    with tab3:
        st.markdown("### Violin & Density Profiling")
        st.markdown("Violin plots combine box plots and density traces to show the exact shape of your data distribution.")
        
        fig_violin = px.violin(filtered_df, y=selected_var_col, x="country", color="country", 
                               box=True, points="all", hover_data=['location_name'])
        fig_violin.update_layout(showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)

    # TAB 4: MULTIVARIATE ANALYSIS
    with tab4:
        st.markdown("### Cross-Variable Interactions")
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown(f"**Dual-Axis Scatter: {selected_var_col} vs {secondary_var_col}**")
            
            # Enterprise Error Handling: Prevent the DuplicateError crash
            if selected_var_col == secondary_var_col:
                st.info("💡 Please select a different Secondary Metric in the sidebar to view cross-variable interactions.")
            else:
                # Marginal charts add histograms to the edges of the scatter plot
                fig_scatter = px.scatter(filtered_df.sample(n=min(2000, len(filtered_df))), 
                                         x=selected_var_col, y=secondary_var_col, 
                                         color="country", marginal_x="histogram", marginal_y="box",
                                         trendline="ols")
                st.plotly_chart(fig_scatter, use_container_width=True)
            
        with colB:
            st.markdown("**Hierarchical Sunburst**")
            st.markdown("Breakdown of data volume by Country and then specific Location/City.")
            # Sunburst requires categorical hierarchy
            sun_data = filtered_df.groupby(['country', 'location_name']).size().reset_index(name='count')
            fig_sun = px.sunburst(sun_data, path=['country', 'location_name'], values='count', color='count', color_continuous_scale="Teal")
            fig_sun.update_layout(margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_sun, use_container_width=True)