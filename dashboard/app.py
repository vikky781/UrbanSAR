"""
UrbanSAR - Smart City Dashboard

Interactive Streamlit dashboard for building height estimation
and disaster vulnerability mapping.

Pages:
    1. Map View - Interactive map with building overlays
    2. Height Analysis - Predicted vs. reference comparisons
    3. Vulnerability Overview - Tier distribution and heatmaps
    4. Metrics - MAE, RMSE, R² with visual cards

Usage:
    streamlit run dashboard/app.py

Falls back to cached JSON data if the trained model is unavailable.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium

from config import STREAMLIT_PAGE_TITLE, STREAMLIT_LAYOUT, MODEL_DIR, FALLBACK_DIR
from dashboard.map_view import build_map
from dashboard.charts import (
    height_comparison_bar, height_scatter,
    vulnerability_distribution, error_distribution,
    metrics_summary,
)
from models.vulnerability import classify_vulnerability, batch_classify, get_tier_summary
from utils.fallback import load_fallback, get_latest_fallback
from utils.geo_utils import create_demo_footprints, assign_heights_to_footprints


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    page_icon="🏙️",
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(124, 58, 237, 0.3);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2.2em;
        font-weight: bold;
        color: #7C3AED;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85em;
        color: #9CA3AF;
    }
    .tier-critical { color: #FF4444; font-weight: bold; }
    .tier-moderate { color: #FFAA00; font-weight: bold; }
    .tier-safe { color: #44BB44; font-weight: bold; }
    .header-gradient {
        background: linear-gradient(90deg, #7C3AED, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 1.1em;
        margin-top: -10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """
    Load prediction data. Tries in order:
        1. Real model predictions from fallback JSON
        2. Demo data for dashboard testing
    """
    # Try loading real predictions
    predictions_data = load_fallback("predictions")
    if predictions_data is not None:
        return predictions_data, "real"

    # Try latest fallback
    latest = get_latest_fallback()
    if latest is not None:
        return latest, "fallback"

    # Generate demo data
    demo_gdf = create_demo_footprints(n_buildings=15)
    predictions = demo_gdf["height_m"].tolist()
    # Simulate "reference" heights with slight offset
    rng = np.random.RandomState(42)
    references = [max(3, h + rng.normal(0, 2)) for h in predictions]

    demo_data = {
        "predictions": predictions,
        "references": references,
        "building_ids": demo_gdf["building_id"].tolist(),
        "footprints": demo_gdf,
    }
    return demo_data, "demo"


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="header-gradient">UrbanSAR</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Multi-Sensor Fusion Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🗺️ Map View", "📊 Height Analysis", "⚠️ Vulnerability", "📈 Metrics"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Data Source:**")

    data, data_source = load_data()

    if data_source == "demo":
        st.warning("📌 Using demo data. Run inference to load real predictions.")
    elif data_source == "fallback":
        st.info("📌 Using cached fallback data.")
    else:
        st.success("✅ Using real model predictions.")

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 11px; color: #666;">
        Team Binary Blackhole<br>
        Cosmix Abhisarga 2026<br>
        Track 3: Multi-Sensor Fusion<br>
        IIIT Sri City
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# EXTRACT DATA
# ============================================================
predictions = data.get("predictions", [])
references = data.get("references", predictions)
building_ids = data.get("building_ids", [f"Bldg_{i+1}" for i in range(len(predictions))])

# Classify all buildings
classifications = batch_classify(predictions)

# Compute metrics
errors = [p - r for p, r in zip(predictions, references)]
abs_errors = [abs(e) for e in errors]
mae = np.mean(abs_errors) if abs_errors else 0
rmse = np.sqrt(np.mean([e**2 for e in errors])) if errors else 0
ss_res = sum(e**2 for e in errors)
ss_tot = sum((r - np.mean(references))**2 for r in references) if references else 1
r2 = 1 - ss_res / (ss_tot + 1e-10)
metrics = {
    "MAE": mae, "RMSE": rmse, "R2": r2,
    "Num_Samples": len(predictions),
}


# ============================================================
# PAGES
# ============================================================
if page == "🗺️ Map View":
    st.markdown("## 🗺️ Interactive Building Map")
    st.markdown("Buildings color-coded by vulnerability tier. Click any building for details.")

    # Prepare GeoDataFrame
    if "footprints" in data and data["footprints"] is not None:
        gdf = data["footprints"]
        # Add vulnerability data if not present
        if "vulnerability_tier" not in gdf.columns:
            gdf = gdf.copy()
            gdf["predicted_height_m"] = predictions[:len(gdf)]
            vulns = batch_classify(gdf["predicted_height_m"].tolist())
            gdf["vulnerability_tier"] = [v["tier"] for v in vulns]
            gdf["tier_color"] = [v["color"] for v in vulns]
    else:
        gdf = create_demo_footprints(len(predictions))
        gdf["predicted_height_m"] = predictions[:len(gdf)]
        vulns = batch_classify(gdf["predicted_height_m"].tolist())
        gdf["vulnerability_tier"] = [v["tier"] for v in vulns]
        gdf["tier_color"] = [v["color"] for v in vulns]

    m = build_map(gdf)
    st_folium(m, width=None, height=600, use_container_width=True)

    # Quick stats below map
    col1, col2, col3 = st.columns(3)
    tier_summary = get_tier_summary(classifications)
    for col, (tier, info) in zip([col1, col2, col3], tier_summary.items()):
        with col:
            color = info["color"]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {color};">{info['count']}</div>
                <div class="metric-label">{tier}<br>{info['percentage']}%</div>
            </div>
            """, unsafe_allow_html=True)


elif page == "📊 Height Analysis":
    st.markdown("## 📊 Building Height Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig_bar = height_comparison_bar(predictions, references, building_ids)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_scatter = height_scatter(predictions, references)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Comparison table
    st.markdown("### Detailed Comparison")
    comparison_df = pd.DataFrame({
        "Building": building_ids,
        "Predicted (m)": [round(p, 2) for p in predictions],
        "Reference (m)": [round(r, 2) for r in references],
        "Error (m)": [round(e, 2) for e in errors],
        "Abs Error (m)": [round(ae, 2) for ae in abs_errors],
        "Tier": [c["tier"] for c in classifications],
    })
    st.dataframe(comparison_df, use_container_width=True, height=400)


elif page == "⚠️ Vulnerability":
    st.markdown("## ⚠️ Disaster Vulnerability Overview")

    st.info(
        "⚠️ **Proxy Heuristic:** Classification is based on building height alone. "
        "Actual vulnerability depends on construction type, materials, and local conditions."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_vuln = vulnerability_distribution(classifications)
        st.plotly_chart(fig_vuln, use_container_width=True)

    with col2:
        st.markdown("### Tier Breakdown")
        tier_summary = get_tier_summary(classifications)
        for tier_name, info in tier_summary.items():
            color = info["color"]
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: {color}; font-size: 1.3em; font-weight: bold;">{tier_name}</span>
                    <span style="font-size: 2em; font-weight: bold; color: {color};">{info['count']}</span>
                </div>
                <div style="color: #9CA3AF; font-size: 0.85em; margin-top: 5px;">
                    {info['percentage']}% of analyzed buildings
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Zone query
    st.markdown("### 🔍 Zone Query")
    selected_tier = st.selectbox("Filter by tier:", ["All"] + list(tier_summary.keys()))
    if selected_tier != "All":
        filtered = [
            {"id": bid, "height": p, "tier": c["tier"]}
            for bid, p, c in zip(building_ids, predictions, classifications)
            if c["tier"] == selected_tier
        ]
        st.dataframe(pd.DataFrame(filtered), use_container_width=True)


elif page == "📈 Metrics":
    st.markdown("## 📈 Model Performance Metrics")

    # Metric cards
    fmt = metrics_summary(metrics)
    cols = st.columns(4)
    for col, (key, info) in zip(cols, fmt.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{info['value']}</div>
                <div class="metric-label">{info['label']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Error distribution
    fig_err = error_distribution(errors)
    st.plotly_chart(fig_err, use_container_width=True)

    # Training info
    st.markdown("### Training Details")
    model_path = MODEL_DIR / "best_model.pth"
    if model_path.exists():
        import torch
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        st.json({
            "Best Epoch": ckpt.get("epoch", "N/A"),
            "Validation MAE": ckpt.get("val_mae", "N/A"),
            "Validation RMSE": ckpt.get("val_rmse", "N/A"),
            "Validation R²": ckpt.get("val_r2", "N/A"),
        })
    else:
        st.warning("No trained model found. Train the model first.")
