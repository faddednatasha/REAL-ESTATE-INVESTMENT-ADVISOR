import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)


# ─── Load Data ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data...")
def load_raw_data():
    if not os.path.exists("processed_india_housing_prices.csv"):
        st.error("processed_india_housing_prices.csv not found. Place it in the same folder as app.py")
        st.stop()
    return pd.read_csv("processed_india_housing_prices.csv")


@st.cache_data(show_spinner="Processing data...")
def load_processed_data():
    from data_preprocessing import (
        load_data, handle_missing_and_duplicates,
        engineer_features, encode_categoricals
    )
    df = load_data("india_housing_prices.csv")
    df = handle_missing_and_duplicates(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)
    return df, encoders


def load_models():
    if not os.path.exists("models/classifier.pkl"):
        return None, None, None
    with open("models/classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("models/regressor.pkl", "rb") as f:
        reg = pickle.load(f)
    with open("models/encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    return clf, reg, enc


raw_df = load_raw_data()
proc_df, encoders = load_processed_data()
clf_model, reg_model, saved_enc = load_models()
if saved_enc:
    encoders = saved_enc

# ─── Sidebar Navigation ───────────────────────────────────────────────────────

st.sidebar.title("🏠 Real Estate Advisor")
st.sidebar.markdown("AI-powered investment analysis for Indian real estate.")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Investment Predictor",
    "EDA Dashboard",
    "Property Explorer"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**  \n250,000 properties · 20 states · 42 cities")


# ─── Helper: build input for model ───────────────────────────────────────────

def build_input_row(state, city, prop_type, bhk, size, price, year_built,
                    furnished, floor_no, total_floors, schools, hospitals,
                    transport, parking, security, availability, facing,
                    owner_type, amenities_list):

    age            = 2025 - year_built
    amenity_count  = len(amenities_list)
    infra_score    = (min(schools, 10) / 10 + min(hospitals, 10) / 10) / 2
    floor_ratio    = floor_no / max(total_floors, 1)
    has_parking    = 1 if parking    == "Yes"             else 0
    has_security   = 1 if security   == "Yes"             else 0
    is_ready       = 1 if availability == "Ready_to_Move" else 0
    high_transport = 1 if transport  == "High"            else 0

    def enc(col, val):
        le = encoders.get(col)
        if le is None:
            return 0
        try:
            return int(le.transform([val])[0])
        except Exception:
            return 0

    row = {
        "BHK": bhk, "Size_in_SqFt": size, "Price_in_Lakhs": price,
        "Age_of_Property": age, "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals, "Floor_No": floor_no,
        "Total_Floors": total_floors, "Amenity_Count": amenity_count,
        "Infra_Score": infra_score, "Floor_Ratio": floor_ratio,
        "Has_Parking": has_parking, "Has_Security": has_security,
        "Is_Ready": is_ready, "High_Transport": high_transport,
        "State_enc": enc("State", state), "City_enc": enc("City", city),
        "Property_Type_enc": enc("Property_Type", prop_type),
        "Furnished_Status_enc": enc("Furnished_Status", furnished),
        "Facing_enc": enc("Facing", facing),
        "Owner_Type_enc": enc("Owner_Type", owner_type),
    }
    return pd.DataFrame([row])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("Real Estate Investment Advisor")
    st.markdown("This app helps investors evaluate Indian properties using machine learning — classifying whether a property is a **Good Investment** and predicting its **price after 5 years**.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Properties", f"{len(raw_df):,}")
    col2.metric("Avg Price", f"₹{raw_df['Price_in_Lakhs'].mean():.1f} L")
    col3.metric("Median Size", f"{int(raw_df['Size_in_SqFt'].median()):,} sqft")
    col4.metric("Cities Covered", raw_df["City"].nunique())

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Property Type Breakdown")
        counts = raw_df["Property_Type"].value_counts().reset_index()
        counts.columns = ["Type", "Count"]
        fig = px.pie(counts, names="Type", values="Count", hole=0.4)
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Average Price by State")
        agg = raw_df.groupby("State")["Price_in_Lakhs"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(agg, x="State", y="Price_in_Lakhs", labels={"Price_in_Lakhs": "Avg Price (₹ L)"})
        fig.update_layout(xaxis_tickangle=-45, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Sample")
    st.dataframe(raw_df.head(10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – INVESTMENT PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Investment Predictor":
    st.title("Investment Predictor")
    st.markdown("Fill in the property details below to get a **Good/Bad Investment** verdict and a **5-year price forecast**.")

    if clf_model is None:
        st.warning("Models not found. Run `python train_models.py` first.")
        st.stop()

    st.markdown("---")
    st.subheader("Property Details")

    c1, c2, c3 = st.columns(3)
    with c1:
        state        = st.selectbox("State",         sorted(raw_df["State"].unique()))
        bhk          = st.selectbox("BHK",           [1, 2, 3, 4, 5], index=2)
        floor_no     = st.number_input("Floor No",   0, 30, 5)
        schools      = st.slider("Nearby Schools",   1, 10, 5)
        parking      = st.selectbox("Parking",       ["No", "Yes"])

    with c2:
        city         = st.selectbox("City",          sorted(raw_df["City"].unique()))
        size         = st.number_input("Size (SqFt)", 500, 5000, 1500, step=50)
        total_floors = st.number_input("Total Floors", 1, 30, 10)
        hospitals    = st.slider("Nearby Hospitals", 1, 10, 5)
        security     = st.selectbox("Security",      ["No", "Yes"])

    with c3:
        prop_type    = st.selectbox("Property Type", sorted(raw_df["Property_Type"].unique()))
        price        = st.number_input("Price (₹ Lakhs)", 10.0, 500.0, 150.0, step=5.0)
        year_built   = st.selectbox("Year Built",    list(range(1990, 2024))[::-1])
        transport    = st.selectbox("Transport Access", ["Low", "Medium", "High"])
        availability = st.selectbox("Availability",  ["Ready_to_Move", "Under_Construction"])

    c4, c5 = st.columns(2)
    with c4:
        furnished  = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
        facing     = st.selectbox("Facing",   ["North", "South", "East", "West"])
    with c5:
        owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
        amenities  = st.multiselect("Amenities", ["Gym", "Pool", "Clubhouse", "Garden", "Playground"], default=["Gym", "Pool"])

    st.markdown("---")

    if st.button("Analyse Property", use_container_width=True):
        input_df = build_input_row(
            state, city, prop_type, bhk, size, price, year_built,
            furnished, floor_no, total_floors, schools, hospitals,
            transport, parking, security, availability, facing,
            owner_type, amenities
        )

        clf_pred = clf_model.predict(input_df)[0]
        clf_prob = clf_model.predict_proba(input_df)[0]
        reg_pred = reg_model.predict(input_df)[0]
        appreciation = ((reg_pred - price) / price) * 100

        st.subheader("Results")
        r1, r2 = st.columns(2)

        with r1:
            if clf_pred == 1:
                st.success(f"✅ Good Investment  —  {clf_prob[1]*100:.1f}% confidence")
            else:
                st.error(f"❌ Not Recommended  —  {clf_prob[0]*100:.1f}% confidence")

        with r2:
            st.info(f"📈 Estimated Price After 5 Years: **₹{reg_pred:.1f} L**  (+{appreciation:.1f}%)")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(clf_prob[1] * 100, 1),
            title={"text": "Investment Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "green" if clf_pred == 1 else "red"},
                "steps": [
                    {"range": [0,  50], "color": "#ffe0e0"},
                    {"range": [50, 100], "color": "#e0ffe0"},
                ]
            },
            number={"suffix": "%"}
        ))
        fig.update_layout(height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Input Summary")
        summary = pd.DataFrame({
            "Feature": ["State", "City", "Property Type", "BHK", "Size (SqFt)",
                        "Price (₹ L)", "Year Built", "Furnished", "Transport",
                        "Parking", "Security", "Availability"],
            "Value": [state, city, prop_type, bhk, f"{size:,}", f"₹{price:.1f}L",
                      year_built, furnished, transport, parking, security, availability]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "EDA Dashboard":
    st.title("EDA Dashboard")
    st.markdown("Exploratory analysis of 250,000 Indian real estate listings.")
    st.markdown("---")

    @st.cache_data
    def get_sample(n=20000):
        s = raw_df.sample(n, random_state=42)
        if "Good_Investment" in proc_df.columns:
            s = s.merge(proc_df[["ID", "Good_Investment"]], on="ID", how="left")
        return s

    df = get_sample()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Price & Size", "Location", "Correlations", "Categories", "Investment"
    ])

    with tab1:
        st.subheader("Price Distribution")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="Price_in_Lakhs", nbins=50, title="Price (₹ Lakhs)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df, x="Size_in_SqFt", nbins=50, title="Size (SqFt)")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.box(df, x="Property_Type", y="Price_per_SqFt",
                         title="Price/SqFt by Property Type", color="Property_Type")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            s = df.sample(3000, random_state=1)
            fig = px.scatter(s, x="Size_in_SqFt", y="Price_in_Lakhs", color="BHK",
                             title="Size vs Price", opacity=0.5,
                             color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Location Analysis")
        c1, c2 = st.columns(2)
        with c1:
            agg = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(agg, x="State", y="Price_per_SqFt", title="Avg Price/SqFt by State")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            agg = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(15).reset_index()
            fig = px.bar(agg, x="City", y="Price_in_Lakhs", title="Avg Price – Top 15 Cities")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        top_cities = df["City"].value_counts().head(10).index
        sub = df[df["City"].isin(top_cities)]
        agg = sub.groupby(["City", "BHK"]).size().reset_index(name="Count")
        fig = px.bar(agg, x="City", y="Count", color="BHK", barmode="group",
                     title="BHK Distribution – Top 10 Cities", color_continuous_scale="Viridis")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Correlation Heatmap")
        num_cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
                    "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
                    "Floor_No", "Total_Floors"]
        corr = df[num_cols].corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale="RdBu", zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}"
        ))
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            agg = df.groupby("Nearby_Schools")["Price_per_SqFt"].mean().reset_index()
            fig = px.line(agg, x="Nearby_Schools", y="Price_per_SqFt",
                          markers=True, title="Schools vs Avg Price/SqFt")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            agg = df.groupby("Nearby_Hospitals")["Price_per_SqFt"].mean().reset_index()
            fig = px.line(agg, x="Nearby_Hospitals", y="Price_per_SqFt",
                          markers=True, title="Hospitals vs Avg Price/SqFt")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Category Analysis")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.violin(df, x="Furnished_Status", y="Price_in_Lakhs",
                            color="Furnished_Status", box=True,
                            title="Price by Furnished Status")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            agg = df.groupby("Facing")["Price_per_SqFt"].mean().sort_values(ascending=False).reset_index()
            fig = px.bar(agg, x="Facing", y="Price_per_SqFt",
                         title="Avg Price/SqFt by Facing Direction", color="Facing")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.box(df, x="Parking_Space", y="Price_in_Lakhs",
                         color="Parking_Space", title="Price vs Parking Availability")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            agg = df.groupby("Public_Transport_Accessibility")["Price_per_SqFt"].mean().reset_index()
            fig = px.bar(agg, x="Public_Transport_Accessibility", y="Price_per_SqFt",
                         title="Price/SqFt by Transport Access",
                         color="Public_Transport_Accessibility")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            counts = df["Owner_Type"].value_counts().reset_index()
            counts.columns = ["Owner_Type", "Count"]
            fig = px.pie(counts, names="Owner_Type", values="Count",
                         title="Properties by Owner Type", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with c6:
            counts = df["Availability_Status"].value_counts().reset_index()
            counts.columns = ["Status", "Count"]
            fig = px.pie(counts, names="Status", values="Count",
                         title="Availability Status", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        if "Good_Investment" not in df.columns:
            st.info("Good Investment column not found in the sample.")
        else:
            st.subheader("Investment Analysis")
            overall = df["Good_Investment"].mean() * 100
            st.metric("Overall Good Investment Rate", f"{overall:.1f}%")

            c1, c2 = st.columns(2)
            with c1:
                agg = df.groupby("City")["Good_Investment"].mean().sort_values(ascending=False).head(15).reset_index()
                agg["Pct"] = (agg["Good_Investment"] * 100).round(1)
                fig = px.bar(agg, x="City", y="Pct", title="% Good Investment by City",
                             color="Pct", color_continuous_scale="RdYlGn",
                             labels={"Pct": "Good Investment (%)"})
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                agg = df.groupby("Property_Type")["Good_Investment"].mean().reset_index()
                agg["Pct"] = (agg["Good_Investment"] * 100).round(1)
                fig = px.bar(agg, x="Property_Type", y="Pct",
                             title="Good Investment % by Property Type",
                             color="Property_Type")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – PROPERTY EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Property Explorer":
    st.title("Property Explorer")
    st.markdown("Filter and browse the dataset.")
    st.markdown("---")

    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_states   = st.multiselect("State", sorted(raw_df["State"].unique()))
            bhk_filter   = st.multiselect("BHK",   sorted(raw_df["BHK"].unique()))
        with c2:
            sel_cities   = st.multiselect("City",  sorted(raw_df["City"].unique()))
            avail_filter = st.multiselect("Availability", raw_df["Availability_Status"].unique().tolist())
        with c3:
            sel_types    = st.multiselect("Property Type", sorted(raw_df["Property_Type"].unique()))
            price_range  = st.slider("Price Range (₹ Lakhs)",
                                     float(raw_df["Price_in_Lakhs"].min()),
                                     float(raw_df["Price_in_Lakhs"].max()),
                                     (50.0, 400.0))

    filtered = raw_df.copy()
    if sel_states:   filtered = filtered[filtered["State"].isin(sel_states)]
    if sel_cities:   filtered = filtered[filtered["City"].isin(sel_cities)]
    if sel_types:    filtered = filtered[filtered["Property_Type"].isin(sel_types)]
    if bhk_filter:   filtered = filtered[filtered["BHK"].isin(bhk_filter)]
    if avail_filter: filtered = filtered[filtered["Availability_Status"].isin(avail_filter)]
    filtered = filtered[
        (filtered["Price_in_Lakhs"] >= price_range[0]) &
        (filtered["Price_in_Lakhs"] <= price_range[1])
    ]

    st.markdown(f"**{len(filtered):,} properties match your filters**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Price",   f"₹{filtered['Price_in_Lakhs'].mean():.1f} L" if len(filtered) else "—")
    c2.metric("Median Size", f"{int(filtered['Size_in_SqFt'].median()):,} sqft" if len(filtered) else "—")
    c3.metric("Avg BHK",     f"{filtered['BHK'].mean():.1f}" if len(filtered) else "—")

    st.markdown("---")

    display_cols = ["State", "City", "Property_Type", "BHK", "Size_in_SqFt",
                    "Price_in_Lakhs", "Price_per_SqFt", "Furnished_Status",
                    "Availability_Status", "Year_Built", "Nearby_Schools",
                    "Nearby_Hospitals", "Public_Transport_Accessibility"]

    st.dataframe(filtered[display_cols].head(500).reset_index(drop=True),
                 use_container_width=True, height=400)

    if len(filtered) > 0:
        csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download Filtered Data (CSV)", data=csv,
                           file_name="filtered_properties.csv", mime="text/csv")