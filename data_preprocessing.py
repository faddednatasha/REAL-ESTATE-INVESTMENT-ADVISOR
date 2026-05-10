
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw dataset."""
    df = pd.read_csv(filepath)
    return df


def handle_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and fill any missing values."""
    df = df.drop_duplicates()
    # Fill numeric nulls with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    # Fill categorical nulls with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new meaningful features for modelling."""
    df = df.copy()

    # 1. Amenity count – how many amenities does the property have
    df["Amenity_Count"] = df["Amenities"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) else 0
    )

    # 2. School + Hospital density combined score (normalised 0-1)
    df["Infra_Score"] = (
        (df["Nearby_Schools"] / df["Nearby_Schools"].max()) +
        (df["Nearby_Hospitals"] / df["Nearby_Hospitals"].max())
    ) / 2

    # 3. Recalculate Price per SqFt properly (in Lakhs per SqFt)
    df["Calc_PricePerSqFt"] = df["Price_in_Lakhs"] / df["Size_in_SqFt"]

    # 4. Floor ratio
    df["Floor_Ratio"] = np.where(
        df["Total_Floors"] > 0,
        df["Floor_No"] / df["Total_Floors"],
        0
    )

    # 5. Binary helpers
    df["Has_Parking"] = (df["Parking_Space"] == "Yes").astype(int)
    df["Has_Security"] = (df["Security"] == "Yes").astype(int)
    df["Is_Ready"] = (df["Availability_Status"] == "Ready_to_Move").astype(int)
    df["High_Transport"] = (df["Public_Transport_Accessibility"] == "High").astype(int)

    # 6. Future Price after 5 years (8% compound growth)
    df["Future_Price_5yr"] = df["Price_in_Lakhs"] * ((1 + 0.08) ** 5)

    # 7. Good Investment label (multi-factor approach)
    median_price_per_sqft = df["Calc_PricePerSqFt"].median()
    df["Good_Investment"] = (
        (df["Calc_PricePerSqFt"] <= median_price_per_sqft) &
        (df["BHK"] >= 2) &
        (df["Infra_Score"] >= 0.4) &
        (df["Is_Ready"] == 1)
    ).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame):
    """Label-encode all categorical columns and return df + encoder dict."""
    df = df.copy()
    encoders = {}
    cat_cols = [
        "State", "City", "Property_Type", "Furnished_Status",
        "Public_Transport_Accessibility", "Facing",
        "Owner_Type", "Availability_Status"
    ]
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def scale_features(X_train, X_test):
    """Standard scale numeric features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def get_feature_columns():
    """Return the list of feature columns used in models."""
    return [
        "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Age_of_Property",
        "Nearby_Schools", "Nearby_Hospitals", "Floor_No", "Total_Floors",
        "Amenity_Count", "Infra_Score", "Floor_Ratio",
        "Has_Parking", "Has_Security", "Is_Ready", "High_Transport",
        "State_enc", "City_enc", "Property_Type_enc",
        "Furnished_Status_enc", "Facing_enc", "Owner_Type_enc"
    ]
