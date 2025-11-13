# Mapping code

import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point
import geopandas as gpd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from streamlit_folium import st_folium
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_india_boundary():
    india_gdf = gpd.read_file(BASE_DIR / "india.geojson")
    return india_gdf

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "all_data.csv")
    df.rename(columns={"diagnosis": "disease"}, inplace=True)
    return df

@st.cache_data
def process_clusters(df):
    le = LabelEncoder()
    df["disease_encoded"] = le.fit_transform(df["disease"])
    kmeans = KMeans(n_clusters=10, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[["latitude", "longitude", "disease_encoded"]])
    return df

def is_within_india(lat, lon, india_gdf):
    point = Point(lon, lat)
    return any(india_gdf.geometry.contains(point))

def get_cluster_info(df_filtered):
    cluster_info = {}
    for cluster in range(10):
        cluster_data = df_filtered[df_filtered["cluster"] == cluster]
        if cluster_data.empty:
            cluster_info[cluster] = {"name": f"Empty Cluster {cluster}", "details": "No data points"}
            continue

        top_diseases = cluster_data["disease"].mode().tolist()[:2]
        top_terrain = cluster_data["terrain"].mode()[0] if not cluster_data["terrain"].mode().empty else "Mixed"
        top_states = cluster_data["state"].mode().tolist()[:2]
        avg_rain = cluster_data["rainfall"].mean() if "rainfall" in cluster_data else 0
        avg_humidity = cluster_data["humidity"].mean() if "humidity" in cluster_data else 0

        name = f"{top_terrain} {', '.join(top_diseases)} Zone"
        if avg_rain > 500:
            name += " (Monsoon)"
        elif avg_rain < 100:
            name += " (Dry)"

        details = (
            f"Diseases: {', '.join(top_diseases)}<br>"
            f"Terrain: {top_terrain}<br>"
            f"States: {', '.join(top_states)}<br>"
            f"Avg Rainfall: {avg_rain:.1f} mm<br>"
            f"Avg Humidity: {avg_humidity:.1f}%"
        )

        cluster_info[cluster] = {"name": name, "details": details}
    return cluster_info

def render_disease_cluster_map():
    st.subheader("🗺️ Disease Cluster Map")

    with st.spinner("⏳ Processing data and generating map..."):
        india_gdf = load_india_boundary()
        df = load_data()
        df = process_clusters(df)

        df["within_india"] = df.apply(lambda row: is_within_india(row["latitude"], row["longitude"], india_gdf), axis=1)
        df_filtered = df[df["within_india"]].reset_index(drop=True)

        MAX_POINTS = 1000
        df_filtered = df_filtered.sample(min(len(df_filtered), MAX_POINTS), random_state=42)

        cluster_info = get_cluster_info(df_filtered)

        india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="OpenStreetMap")

        folium.GeoJson(
            india_gdf,
            name="India Boundary",
            style_function=lambda feature: {
                'fillColor': '#ffff00',
                'color': '#000000',
                'weight': 2,
                'fillOpacity': 0.1
            }
        ).add_to(india_map)

        marker_cluster = MarkerCluster().add_to(india_map)
        cluster_colors = {i: mcolors.to_hex(plt.cm.tab10(i / 10)) for i in range(10)}

        for _, row in df_filtered.iterrows():
            color = cluster_colors[row["cluster"]]
            popup_text = f"{row['disease']}<br>Cluster {row['cluster']}: {cluster_info[row['cluster']]['name']}"
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(popup_text, max_width=250)
            ).add_to(marker_cluster)

        legend_html = '''
        <div style="position: fixed;
             bottom: 50px; right: 50px;
             background-color: rgba(255, 255, 255, 0.9);
             border: 2px solid grey;
             padding: 10px;
             z-index: 9999;
             font-size: 12px;
             font-family: Arial;">
             <b>Cluster Legend</b><br>
        '''
        for i in range(10):
            color = cluster_colors[i]
            legend_html += f'''
             <div style="margin-bottom: 5px;">
                 <span style="color: {color}; font-size: 18px;">●</span>
                 Cluster {i}: {cluster_info[i]['name']}<br>
                 <span style="font-size: 10px; color: #555;">{cluster_info[i]['details']}</span>
             </div>
        '''
        legend_html += '</div>'
        india_map.get_root().html.add_child(folium.Element(legend_html))

        st_folium(india_map, width=1200, height=700)
