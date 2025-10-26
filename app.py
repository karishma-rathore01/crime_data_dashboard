import streamlit as st
import plotly.express as px
import pandas as pd
from model import load_data, train_predict

# Page setup
st.set_page_config(page_title="Crime Data Dashboard", layout="wide")
st.title("🛡️ Crime Data Dashboard with Prediction")

# Load dataset
df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")
crime_type = st.sidebar.selectbox("Select Crime Type", ["All"] + df["Type"].unique().tolist())
if crime_type != "All":
    df = df[df["Type"] == crime_type]

year_range = st.sidebar.slider("Select Year Range", int(df["Year"].min()), int(df["Year"].max()), 
                               (int(df["Year"].min()), int(df["Year"].max())))
df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

# Show dataset preview
st.subheader("📂 Crime Dataset")
st.dataframe(df.head())

# Crime counts by year + prediction
st.subheader("📈 Crime Trend & Predictions")
yearly, future_years, preds = train_predict(df)

# Line chart with actual + predictions
fig = px.line(yearly, x="Year", y="Crimes", markers=True, title="Crime Trend Over Years")
fig.add_scatter(x=future_years, y=preds, mode="lines+markers", name="Predicted", line=dict(dash="dot"))
st.plotly_chart(fig, use_container_width=True)

# Future predictions text
st.subheader("🔮 Future Crime Predictions")
for y, p in zip(future_years, preds):
    st.write(f"**{y}** → Predicted Crimes: {int(p)}")

# Crime type distribution
st.subheader("🍩 Crime Type Distribution")
fig_pie = px.pie(df, names="Type", title="Crime Breakdown by Type")
st.plotly_chart(fig_pie, use_container_width=True)

# Crime hotspot map
st.subheader("🗺️ Crime Hotspot Map")
if "Latitude" in df.columns and "Longitude" in df.columns:
    fig_map = px.scatter_mapbox(
        df, 
        lat="Latitude", lon="Longitude", 
        hover_name="Type", 
        hover_data=["Location", "Date"],
        color="Type",
        zoom=3, height=500
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.warning("⚠️ No location data available in dataset.")
