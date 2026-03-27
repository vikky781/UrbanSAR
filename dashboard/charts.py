"""
UrbanSAR - Plotly Charts Component

Charts for the Streamlit dashboard:
- Predicted vs. Reference height bar chart
- Scatter plot with regression line
- Metrics cards
- Vulnerability tier distribution
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import VULNERABILITY_TIERS


def height_comparison_bar(
    predictions: List[float],
    references: List[float],
    building_ids: Optional[List[str]] = None,
) -> go.Figure:
    """
    Side-by-side bar chart comparing predicted vs. reference heights.

    Args:
        predictions: Predicted heights
        references: Reference heights
        building_ids: Building labels

    Returns:
        Plotly Figure
    """
    if building_ids is None:
        building_ids = [f"Bldg {i+1}" for i in range(len(predictions))]

    fig = go.Figure(data=[
        go.Bar(
            name="Predicted",
            x=building_ids,
            y=predictions,
            marker_color="#7C3AED",
            opacity=0.85,
        ),
        go.Bar(
            name="Reference",
            x=building_ids,
            y=references,
            marker_color="#06B6D4",
            opacity=0.85,
        ),
    ])

    fig.update_layout(
        barmode="group",
        title="Predicted vs. Reference Building Heights",
        xaxis_title="Building",
        yaxis_title="Height (meters)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=450,
    )

    return fig


def height_scatter(
    predictions: List[float],
    references: List[float],
) -> go.Figure:
    """
    Scatter plot of predicted vs. reference heights with ideal line.

    Args:
        predictions: Predicted heights
        references: Reference heights

    Returns:
        Plotly Figure
    """
    preds = np.array(predictions)
    refs = np.array(references)

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=refs, y=preds,
        mode="markers",
        marker=dict(
            size=10,
            color=preds,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Height (m)"),
        ),
        name="Buildings",
        hovertemplate="Ref: %{x:.1f}m<br>Pred: %{y:.1f}m<extra></extra>",
    ))

    # Perfect prediction line
    max_val = max(max(preds, default=1), max(refs, default=1)) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(dash="dash", color="rgba(255,255,255,0.4)"),
        name="Perfect Prediction",
    ))

    fig.update_layout(
        title="Prediction Accuracy",
        xaxis_title="Reference Height (m)",
        yaxis_title="Predicted Height (m)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(range=[0, max_val]),
        yaxis=dict(range=[0, max_val]),
        height=450,
    )

    return fig


def vulnerability_distribution(classifications: List[Dict]) -> go.Figure:
    """
    Pie/donut chart showing vulnerability tier distribution.

    Args:
        classifications: List of classification dicts with 'tier' key

    Returns:
        Plotly Figure
    """
    tier_counts = {}
    tier_colors = {}

    for c in classifications:
        tier = c["tier"]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        tier_colors[tier] = c.get("color", VULNERABILITY_TIERS.get(tier, {}).get("color", "#888"))

    labels = list(tier_counts.keys())
    values = list(tier_counts.values())
    colors = [tier_colors[t] for t in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(size=14, color="white"),
    )])

    fig.update_layout(
        title="Vulnerability Distribution",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=400,
        showlegend=True,
    )

    return fig


def error_distribution(errors: List[float]) -> go.Figure:
    """
    Histogram of prediction errors.

    Args:
        errors: List of (predicted - reference) values

    Returns:
        Plotly Figure
    """
    fig = go.Figure(data=[go.Histogram(
        x=errors,
        nbinsx=20,
        marker_color="#7C3AED",
        opacity=0.8,
    )])

    fig.add_vline(
        x=0, line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        annotation_text="Zero Error",
    )

    fig.update_layout(
        title="Prediction Error Distribution",
        xaxis_title="Error (meters)",
        yaxis_title="Count",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=350,
    )

    return fig


def metrics_summary(metrics: Dict[str, float]) -> Dict:
    """
    Format metrics for display as cards in Streamlit.

    Args:
        metrics: Dict with MAE, RMSE, R2, etc.

    Returns:
        Dict ready for Streamlit metric display
    """
    return {
        "MAE": {"value": f"{metrics.get('MAE', 0):.2f}m", "label": "Mean Absolute Error"},
        "RMSE": {"value": f"{metrics.get('RMSE', 0):.2f}m", "label": "Root Mean Sq Error"},
        "R²": {"value": f"{metrics.get('R2', 0):.3f}", "label": "R² Score"},
        "Samples": {"value": str(metrics.get("Num_Samples", 0)), "label": "Buildings Analyzed"},
    }
