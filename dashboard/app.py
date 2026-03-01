import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. PRO PAGE SETUP ---
# Set the page to wide mode and add an emoji icon to the browser tab
st.set_page_config(page_title="ClimateScope Pro", page_icon="🌍", layout="wide")

# Custom CSS to make the metric cards look cleaner
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌍 ClimateScope: Global Weather Analytics")
st.markdown("Explore global weather patterns, extreme events, and seasonal trends.")

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('../data/processed/weather_cleaned_daily.csv')
    # Ensure year_month is treated as a string/category for clean plotting
    df['year_month'] = df['year_month'].astype(str)
    return df

df = load_data()

# --- 3. PRO SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    weather_vars = {
        'Temperature (°C)': 'temperature_celsius', 
        'Humidity (%)': 'humidity', 
        'Precipitation (mm)': 'precip_mm',
        'Wind Speed (kph)': 'wind_kph'
    }
    selected_var_name = st.selectbox("1. Select Primary Metric", list(weather_vars.keys()))
    selected_var_col = weather_vars[selected_var_name]
    
    st.markdown("---")
    st.markdown("**Project Details**")
    st.caption("Data Source: Global Weather Repository")
    st.caption("Lead Data Scientist: GURU")

# --- 4. EXECUTIVE KPI CARDS ---
st.subheader("Historical Extremes (All-Time)")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

max_temp = df.loc[df['temperature_celsius'].idxmax()]
min_temp = df.loc[df['temperature_celsius'].idxmin()]
max_wind = df.loc[df['wind_kph'].idxmax()]
max_rain = df.loc[df['precip_mm'].idxmax()]

kpi1.metric("🔥 Hottest Recorded", f"{max_temp['temperature_celsius']} °C", f"{max_temp['country']}")
kpi2.metric("🧊 Coldest Recorded", f"{min_temp['temperature_celsius']} °C", f"{min_temp['country']}")
kpi3.metric("🌪️ Max Wind Gust", f"{max_wind['wind_kph']} kph", f"{max_wind['country']}")
kpi4.metric("🌧️ Max Daily Rain", f"{max_rain['precip_mm']} mm", f"{max_rain['country']}")

st.divider() # Adds a clean horizontal line

# --- 5. INTERACTIVE TABS ---
# This is a massive UI upgrade. It organizes the app beautifully.
tab1, tab2, tab3 = st.tabs(["🗺️ Global Map", "📈 Regional Trends", "📊 Variable Correlations"])

with tab1:
    st.subheader(f"Global Distribution of {selected_var_name}")
    map_data = df.groupby('country', as_index=False)[selected_var_col].mean()
    
    fig_map = px.choropleth(
        map_data, 
        locations="country", 
        locationmode="country names",
        color=selected_var_col,
        color_continuous_scale="RdYlBu_r" if "Temperature" in selected_var_name else "Blues",
        hover_name="country"
    )
    
    # Pro Styling: Clean margins, invisible background, sleek projection
    fig_map.update_geos(projection_type="natural earth", showcoastlines=True, coastlinecolor="rgba(255, 255, 255, 0.2)")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    st.subheader("Compare Regional Time-Series")
    # Allow multiple countries to be selected for comparison
    countries = sorted(df['country'].unique().tolist())
    default_countries = ['United States of America', 'India', 'Australia', 'Brazil']
    selected_countries = st.multiselect("Select Countries to Compare:", countries, default=default_countries)
    
    if selected_countries:
        trend_df = df[df['country'].isin(selected_countries)]
        trend_data = trend_df.groupby(['year_month', 'country'])[selected_var_col].mean().reset_index()
        
        fig_line = px.line(
            trend_data, 
            x='year_month', 
            y=selected_var_col, 
            color='country',
            markers=True
        )
        # Pro Styling: clean legend, transparent background
        fig_line.update_layout(
            margin={"r":10,"t":30,"l":10,"b":10}, 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("Please select at least one country to view trends.")

with tab3:
    st.subheader("Correlation Engine")
    st.markdown("Investigate how temperature interacts with humidity globally.")
    
    # Taking a sample so the browser doesn't crash rendering 122k points
    sample_df = df.sample(n=min(5000, len(df)), random_state=42) 
    
    fig_scatter = px.scatter(
        sample_df, 
        x='temperature_celsius', 
        y='humidity', 
        color='precip_mm',
        size='wind_kph',
        hover_data=['country', 'location_name'],
        color_continuous_scale="Viridis",
        opacity=0.7
    )
    fig_scatter.update_layout(margin={"r":10,"t":30,"l":10,"b":10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_scatter, use_container_width=True)