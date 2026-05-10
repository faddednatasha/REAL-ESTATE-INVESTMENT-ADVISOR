import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")


# ─── Colour Palette ────────────────────────────────────────────────────────────
PALETTE = px.colors.sequential.Plasma
ACCENT  = "#F72585"
BG      = "#0D1117"
CARD_BG = "#161B22"
TEXT    = "#E6EDF3"


def price_distribution(df: pd.DataFrame):
    """Histogram of Price_in_Lakhs."""
    fig = px.histogram(
        df, x="Price_in_Lakhs", nbins=60,
        title="Distribution of Property Prices (₹ Lakhs)",
        color_discrete_sequence=[ACCENT]
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, title_font_size=16
    )
    return fig


def size_distribution(df: pd.DataFrame):
    """Histogram of Size_in_SqFt."""
    fig = px.histogram(
        df, x="Size_in_SqFt", nbins=60,
        title="Distribution of Property Sizes (SqFt)",
        color_discrete_sequence=["#4CC9F0"]
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, title_font_size=16
    )
    return fig


def price_per_sqft_by_type(df: pd.DataFrame):
    """Box plot – Price/SqFt by Property Type."""
    fig = px.box(
        df, x="Property_Type", y="Price_per_SqFt",
        title="Price per SqFt by Property Type",
        color="Property_Type",
        color_discrete_sequence=PALETTE
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, showlegend=False, title_font_size=16
    )
    return fig


def size_vs_price_scatter(df: pd.DataFrame):
    """Scatter – Size vs Price coloured by BHK."""
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Size_in_SqFt", y="Price_in_Lakhs",
        color="BHK", opacity=0.6,
        title="Property Size vs Price (coloured by BHK)",
        color_continuous_scale="Plasma"
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, title_font_size=16
    )
    return fig


def avg_price_per_sqft_by_state(df: pd.DataFrame):
    """Bar – Average Price/SqFt by State."""
    agg = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(
        agg, x="State", y="Price_per_SqFt",
        title="Avg Price per SqFt by State",
        color="Price_per_SqFt", color_continuous_scale="Plasma"
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, xaxis_tickangle=-45, title_font_size=16
    )
    return fig


def avg_price_by_city(df: pd.DataFrame, top_n: int = 20):
    """Bar – Average Property Price by City (top N)."""
    agg = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(top_n).reset_index()
    fig = px.bar(
        agg, x="City", y="Price_in_Lakhs",
        title=f"Average Property Price – Top {top_n} Cities (₹ Lakhs)",
        color="Price_in_Lakhs", color_continuous_scale="Magma"
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, xaxis_tickangle=-45, title_font_size=16
    )
    return fig


def bhk_distribution_by_city(df: pd.DataFrame, top_n: int = 10):
    """Grouped bar – BHK distribution for top cities."""
    top_cities = df["City"].value_counts().head(top_n).index
    sub = df[df["City"].isin(top_cities)]
    agg = sub.groupby(["City", "BHK"]).size().reset_index(name="Count")
    fig = px.bar(
        agg, x="City", y="Count", color="BHK",
        barmode="group", title=f"BHK Distribution – Top {top_n} Cities",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, xaxis_tickangle=-45, title_font_size=16
    )
    return fig


def correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap of numeric columns."""
    num_cols = [
        "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
        "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
        "Floor_No", "Total_Floors"
    ]
    corr = df[num_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="Plasma",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}"
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, title_font_size=16, height=500
    )
    return fig


def schools_vs_price(df: pd.DataFrame):
    """Line – Avg Price per SqFt by number of Nearby Schools."""
    agg = df.groupby("Nearby_Schools")["Price_per_SqFt"].mean().reset_index()
    fig = px.line(
        agg, x="Nearby_Schools", y="Price_per_SqFt",
        markers=True, title="Avg Price/SqFt vs Nearby Schools",
        color_discrete_sequence=[ACCENT]
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, title_font_size=16
    )
    return fig


def hospitals_vs_price(df: pd.DataFrame):
    """Line – Avg Price per SqFt by number of Nearby Hospitals."""
    agg = df.groupby("Nearby_Hospitals")["Price_per_SqFt"].mean().reset_index()
    fig = px.line(
        agg, x="Nearby_Hospitals", y="Price_per_SqFt",
        markers=True, title="Avg Price/SqFt vs Nearby Hospitals",
        color_discrete_sequence=["#4CC9F0"]
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, title_font_size=16
    )
    return fig


def price_by_furnished_status(df: pd.DataFrame):
    """Violin – Price distribution by Furnished Status."""
    fig = px.violin(
        df, x="Furnished_Status", y="Price_in_Lakhs",
        color="Furnished_Status", box=True,
        title="Price Distribution by Furnished Status",
        color_discrete_sequence=PALETTE
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, showlegend=False, title_font_size=16
    )
    return fig


def price_by_facing(df: pd.DataFrame):
    """Bar – Avg Price/SqFt by Facing Direction."""
    agg = df.groupby("Facing")["Price_per_SqFt"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(
        agg, x="Facing", y="Price_per_SqFt",
        title="Avg Price/SqFt by Property Facing Direction",
        color="Facing", color_discrete_sequence=PALETTE
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, showlegend=False, title_font_size=16
    )
    return fig


def owner_type_distribution(df: pd.DataFrame):
    """Pie – Count by Owner Type."""
    agg = df["Owner_Type"].value_counts().reset_index()
    agg.columns = ["Owner_Type", "Count"]
    fig = px.pie(
        agg, names="Owner_Type", values="Count",
        title="Properties by Owner Type",
        color_discrete_sequence=PALETTE, hole=0.4
    )
    fig.update_layout(
        paper_bgcolor=CARD_BG, font_color=TEXT, title_font_size=16
    )
    return fig


def availability_status_distribution(df: pd.DataFrame):
    """Pie – Availability Status."""
    agg = df["Availability_Status"].value_counts().reset_index()
    agg.columns = ["Status", "Count"]
    fig = px.pie(
        agg, names="Status", values="Count",
        title="Properties by Availability Status",
        color_discrete_sequence=["#4CC9F0", ACCENT], hole=0.4
    )
    fig.update_layout(
        paper_bgcolor=CARD_BG, font_color=TEXT, title_font_size=16
    )
    return fig


def parking_vs_price(df: pd.DataFrame):
    """Box – Price by Parking Space availability."""
    fig = px.box(
        df, x="Parking_Space", y="Price_in_Lakhs",
        color="Parking_Space",
        title="Property Price vs Parking Availability",
        color_discrete_sequence=[ACCENT, "#4CC9F0"]
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, showlegend=False, title_font_size=16
    )
    return fig


def transport_vs_price(df: pd.DataFrame):
    """Bar – Avg Price/SqFt by Public Transport Accessibility."""
    agg = df.groupby("Public_Transport_Accessibility")["Price_per_SqFt"].mean().reset_index()
    fig = px.bar(
        agg, x="Public_Transport_Accessibility", y="Price_per_SqFt",
        title="Avg Price/SqFt by Public Transport Access",
        color="Public_Transport_Accessibility",
        color_discrete_sequence=PALETTE
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, showlegend=False, title_font_size=16
    )
    return fig


def good_investment_ratio_by_city(df: pd.DataFrame, top_n: int = 15):
    """Bar – % Good Investment by City."""
    if "Good_Investment" not in df.columns:
        return None
    agg = df.groupby("City")["Good_Investment"].mean().sort_values(ascending=False).head(top_n).reset_index()
    agg["Pct"] = (agg["Good_Investment"] * 100).round(1)
    fig = px.bar(
        agg, x="City", y="Pct",
        title=f"% 'Good Investment' Properties – Top {top_n} Cities",
        color="Pct", color_continuous_scale="RdYlGn",
        labels={"Pct": "Good Investment (%)"}
    )
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font_color=TEXT, xaxis_tickangle=-45, title_font_size=16
    )
    return fig
