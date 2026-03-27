"""
UrbanSAR - Folium Map View Component

Interactive map with:
- Building polygons color-coded by vulnerability tier
- Click popups with building details
- Vulnerability heatmap overlay
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

import folium
from folium.plugins import HeatMap
import geopandas as gpd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MAP_CENTER_LAT, MAP_CENTER_LON, MAP_ZOOM, VULNERABILITY_TIERS


def create_base_map(
    center_lat: float = MAP_CENTER_LAT,
    center_lon: float = MAP_CENTER_LON,
    zoom: int = MAP_ZOOM,
) -> folium.Map:
    """
    Create base Folium map centered on the study area (Rotterdam).

    Args:
        center_lat: Map center latitude
        center_lon: Map center longitude
        zoom: Initial zoom level

    Returns:
        Folium Map object
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB dark_matter",
        attr="CartoDB",
    )
    return m


def add_building_overlays(
    m: folium.Map,
    buildings_gdf: gpd.GeoDataFrame,
) -> folium.Map:
    """
    Add building polygons to map, color-coded by vulnerability tier.

    Args:
        m: Folium Map
        buildings_gdf: GeoDataFrame with columns:
            - geometry (polygon)
            - predicted_height_m
            - vulnerability_tier
            - tier_color
            - building_id (optional)

    Returns:
        Updated Folium Map
    """
    for _, row in buildings_gdf.iterrows():
        if row.geometry is None:
            continue

        # Get properties
        height = row.get("predicted_height_m", "N/A")
        tier = row.get("vulnerability_tier", "Unknown")
        color = row.get("tier_color", "#888888")
        bldg_id = row.get("building_id", "Unknown")
        ref_height = row.get("height_m", "N/A")

        # Format height display
        height_str = f"{height:.1f}m" if isinstance(height, (int, float)) else str(height)
        ref_str = f"{ref_height:.1f}m" if isinstance(ref_height, (int, float)) else str(ref_height)

        est_floors = max(1, int(round(height / 3.0))) if isinstance(height, (int, float)) else "N/A"

        # Popup HTML
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 200px;">
            <h4 style="margin: 0 0 8px 0; color: {color};">
                {tier}
            </h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td><b>Building ID:</b></td><td>{bldg_id}</td></tr>
                <tr><td><b>Predicted Height:</b></td><td>{height_str}</td></tr>
                <tr><td><b>Reference Height:</b></td><td>{ref_str}</td></tr>
                <tr><td><b>Est. Floors:</b></td><td>{est_floors}</td></tr>
                <tr><td><b>Risk Tier:</b></td><td style="color:{color}; font-weight:bold;">{tier}</td></tr>
            </table>
            <p style="font-size: 10px; color: #999; margin-top: 8px;">
                Height-based proxy classification
            </p>
        </div>
        """

        # Convert geometry to GeoJSON coordinates
        geojson = row.geometry.__geo_interface__

        folium.GeoJson(
            geojson,
            style_function=lambda feature, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 2,
                "fillOpacity": 0.5,
            },
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{bldg_id}: {height_str} ({tier})",
        ).add_to(m)

    return m


def add_vulnerability_heatmap(
    m: folium.Map,
    buildings_gdf: gpd.GeoDataFrame,
) -> folium.Map:
    """
    Add a heatmap overlay showing vulnerability concentration.

    Higher weight = higher vulnerability (Critical Risk buildings get highest weight).

    Args:
        m: Folium Map
        buildings_gdf: GeoDataFrame with vulnerability data

    Returns:
        Updated Folium Map
    """
    heat_data = []

    # Weight by inverse of height (lower buildings = higher vulnerability)
    for _, row in buildings_gdf.iterrows():
        if row.geometry is None:
            continue

        centroid = row.geometry.centroid
        height = row.get("predicted_height_m", 10)

        if isinstance(height, (int, float)) and height > 0:
            # Inverse weight: lower buildings = more vulnerability = higher heat
            weight = max(0.1, 1.0 / (height / 10.0))
            heat_data.append([centroid.y, centroid.x, weight])

    if heat_data:
        HeatMap(
            heat_data,
            name="Vulnerability Heatmap",
            min_opacity=0.3,
            radius=25,
            blur=15,
            gradient={
                "0.2": "#44BB44",
                "0.5": "#FFAA00",
                "0.8": "#FF4444",
                "1.0": "#CC0000",
            },
        ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m


def create_legend_html() -> str:
    """Create HTML for the map legend."""
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px; left: 30px;
        background: rgba(0,0,0,0.8);
        padding: 15px 20px;
        border-radius: 8px;
        z-index: 9999;
        font-family: Arial, sans-serif;
        color: white;
        font-size: 13px;
    ">
        <b style="font-size: 14px;">Vulnerability Tiers</b><br><br>
    """

    for tier_name, info in VULNERABILITY_TIERS.items():
        legend_html += f"""
        <div style="margin-bottom: 5px;">
            <span style="
                display: inline-block;
                width: 14px; height: 14px;
                background: {info['color']};
                border-radius: 3px;
                margin-right: 8px;
                vertical-align: middle;
            "></span>
            {tier_name}
        </div>
        """

    legend_html += """
        <div style="margin-top: 10px; font-size: 10px; color: #aaa;">
            Height-based proxy heuristic
        </div>
    </div>
    """
    return legend_html


def build_map(buildings_gdf: gpd.GeoDataFrame) -> folium.Map:
    """
    Build the complete interactive map.

    Args:
        buildings_gdf: GeoDataFrame with predictions and vulnerability data

    Returns:
        Complete Folium Map ready for Streamlit
    """
    # Auto-center on data if available
    if len(buildings_gdf) > 0 and buildings_gdf.geometry.notna().any():
        bounds = buildings_gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
    else:
        center_lat = MAP_CENTER_LAT
        center_lon = MAP_CENTER_LON

    m = create_base_map(center_lat, center_lon)
    m = add_building_overlays(m, buildings_gdf)
    m = add_vulnerability_heatmap(m, buildings_gdf)

    # Add legend
    m.get_root().html.add_child(folium.Element(create_legend_html()))

    return m
