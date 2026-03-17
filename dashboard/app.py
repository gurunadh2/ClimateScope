import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. PAGE SETUP & CONFIGURATION ---
st.set_page_config(page_title="ClimateScope Ultimate", page_icon="🌍", layout="wide")

# Custom CSS for a hyper-professional look
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        border-left: 5px solid #1c83e1;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌍 ClimateScope: Advanced Global Analytics")
st.markdown("An interactive platform exploring distributions, correlations, and seasonal climate trends.")

# --- 2. DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'data', 'processed', 'weather_cleaned_daily.csv')
    data = pd.read_csv(csv_path)
    
    # Preprocessing for advanced filters
    data['last_updated'] = pd.to_datetime(data['last_updated'])
    data['year_month'] = data['year_month'].astype(str)
    return data

df = load_data()

# --- 3. ADVANCED SIDEBAR (Filters, Sliders, Region Selectors) ---
with st.sidebar:
    st.header("🎛️ Master Controls")
    
    # Variable Selector
    weather_vars = {
        'Temperature (°C)': 'temperature_celsius', 
        'Humidity (%)': 'humidity', 
        'Precipitation (mm)': 'precip_mm',
        'Wind Speed (kph)': 'wind_kph',
        'Cloud Cover': 'cloud'
    }
    selected_var_name = st.selectbox("📊 Primary Metric", list(weather_vars.keys()))
    selected_var_col = weather_vars[selected_var_name]

    st.markdown("---")
    
    # Region Selector (Multi-Select)
    all_countries = sorted(df['country'].unique().tolist())
    default_countries = ['United States of America', 'India', 'Australia', 'Brazil', 'United Kingdom']
    selected_countries = st.multiselect("🌍 Select Regions/Countries", all_countries, default=default_countries)
    
    # Sliders (Numeric Filtering)
    st.markdown("### 🎚️ Advanced Filters")
    
    min_temp, max_temp = float(df['temperature_celsius'].min()), float(df['temperature_celsius'].max())
    temp_range = st.slider("Temperature Range (°C)", min_temp, max_temp, (min_temp, max_temp))
    
    min_hum, max_hum = int(df['humidity'].min()), int(df['humidity'].max())
    hum_range = st.slider("Humidity Range (%)", min_hum, max_hum, (min_hum, max_hum))

    # Apply all filters to create the working dataframe
    filtered_df = df[
        (df['country'].isin(selected_countries)) & 
        (df['temperature_celsius'] >= temp_range[0]) & 
        (df['temperature_celsius'] <= temp_range[1]) &
        (df['humidity'] >= hum_range[0]) & 
        (df['humidity'] <= hum_range[1])
    ]

# --- 4. DYNAMIC EXECUTIVE KPIs ---
if filtered_df.empty:
    st.error("No data matches your current filter selection. Please adjust the sidebar controls.")
else:
    st.subheader("Filtered Dataset Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(f"Avg {selected_var_name}", f"{filtered_df[selected_var_col].mean():.1f}")
    kpi2.metric("Max Temp in Selection", f"{filtered_df['temperature_celsius'].max():.1f} °C")
    kpi3.metric("Max Wind in Selection", f"{filtered_df['wind_kph'].max():.1f} kph")
    kpi4.metric("Total Data Points", f"{len(filtered_df):,}")
    
    st.divider()

    # --- 5. THE ANALYTICS ENGINE (Tabs fulfilling all Mentor Requirements) ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Geographic Overview", 
        "📈 Trends & Seasonality", 
        "📊 Distributions", 
        "🔗 Correlations"
    ])

    # TAB 1: GEOGRAPHIC (Choropleth Map)
    with tab1:
        st.subheader("Global Metric Distribution")
        map_data = filtered_df.groupby('country', as_index=False)[selected_var_col].mean()
        fig_map = px.choropleth(map_data, locations="country", locationmode="country names",
                                color=selected_var_col, color_continuous_scale="Turbo",
                                hover_name="country")
        fig_map.update_geos(projection_type="natural earth", showcoastlines=True)
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_map, use_container_width=True)

    # TAB 2: TRENDS & SEASONALITY (Line Charts & Bar Charts)
    with tab2:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Long-Term Trends")
            trend_data = filtered_df.groupby(['year_month', 'country'])[selected_var_col].mean().reset_index()
            fig_trend = px.line(trend_data, x='year_month', y=selected_var_col, color='country', markers=True)
            fig_trend.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with colB:
            st.subheader("Seasonal Patterns (Monthly Averages)")
            seasonal_data = filtered_df.groupby(['month', 'country'])[selected_var_col].mean().reset_index()
            # Mapping month numbers to names for better readability
            month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
            seasonal_data['month_name'] = seasonal_data['month'].map(month_map)
            fig_season = px.bar(seasonal_data, x='month_name', y=selected_var_col, color='country', barmode='group')
            st.plotly_chart(fig_season, use_container_width=True)

    # TAB 3: DISTRIBUTIONS (Box Plots & Histograms)
    with tab3:
        st.markdown("Understand how the data is spread out and identify statistical outliers.")
        colC, colD = st.columns(2)
        with colC:
            st.subheader(f"Data Spread: {selected_var_name}")
            # Box plot is the industry standard for showing distributions and outliers
            fig_box = px.box(filtered_df, x='country', y=selected_var_col, color='country')
            st.plotly_chart(fig_box, use_container_width=True)
            
        with colD:
            st.subheader("Frequency Histogram")
            fig_hist = px.histogram(filtered_df, x=selected_var_col, nbins=30, opacity=0.7, 
                                    color_discrete_sequence=['#1c83e1'], marginal="violin")
            st.plotly_chart(fig_hist, use_container_width=True)

    # TAB 4: CORRELATIONS (Scatter Matrices & Heatmaps)
    with tab4:
        st.subheader("Variable Correlation Matrix")
        st.markdown("This heatmap reveals statistical relationships between all numerical weather variables. (Values closer to 1 or -1 indicate strong correlations).")
        
        # Select only numeric columns for the correlation matrix
        numeric_cols = ['temperature_celsius', 'humidity', 'wind_kph', 'precip_mm', 'pressure_mb', 'cloud', 'uv_index']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", origin='lower')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("Deep Dive: Temperature vs. Humidity")
        # Sampling data to keep browser fast while plotting individual scatter points
        sample_df = filtered_df.sample(n=min(2000, len(filtered_df)), random_state=42)
        fig_scatter = px.scatter(sample_df, x='temperature_celsius', y='humidity', color='country', 
                                 size='wind_kph', hover_data=['location_name'])
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # --- 6. RAW DATA EXPLORER ---
    with st.expander("📂 View Raw Data & Download"):
        st.dataframe(filtered_df.head(100), use_container_width=True)
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Filtered Data as CSV", data=csv, file_name='filtered_climate_data.csv', mime='text/csv')