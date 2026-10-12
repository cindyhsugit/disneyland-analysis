import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from visuals import create_visuals

# Define paths (matching your new folder structure)
DATA_DIR = "dataset"


def load_data():
    print("📥 Loading Disneyland datasets...")

    # Load each CSV using pandas (think of a DataFrame like a powerful, queryable array of structs)
    wait_times_df = pd.read_csv(os.path.join(DATA_DIR, "disney_wait_times.csv"))

    raw = pd.read_csv(
        os.path.join(DATA_DIR, "open-meteo-33.85N117.91W51m.csv"), skiprows=3
    )

    weather_df2 = pd.DataFrame(
        {
            "dateTime": pd.to_datetime(raw["time"]),
            "rainStatus": (raw["precipitation (mm)"].fillna(0) > 0).astype(int),
            "feelsLikeF": raw["apparent_temperature (°F)"].round().astype("Int64"),
        }
    )
    weather_df2.to_csv(os.path.join(DATA_DIR, "hourlyWeather_2024.csv"), index=False)

    print(f"   - Loaded {len(wait_times_df)} wait time records")
    print(f"   - Loaded {len(weather_df2)} hourly weather 2024 records")

    return (
        wait_times_df,
        weather_df2,
    )


def clean_2024(waits, weather_2024):
    print("🧹 Cleaning 2024-2025 data...\n")

    # 1. Timestamps (keep local clock time, drop the -07:00/-08:00 offset)
    waits["Local Time"] = pd.to_datetime(waits["Local Time"].str[:19])
    weather_2024["dateTime"] = pd.to_datetime(weather_2024["dateTime"])

    # 2. Drop zero waits (ride closed or no wait recorded)
    before = len(waits)
    waits = waits[waits["Wait Time"] > 0].copy()
    print(f"   - Removed {before - len(waits)} zero-wait rows, {len(waits)} remain")

    # 3. Add an hour column so waits can be merged with hourly weather
    waits["hour_start"] = waits["Local Time"].dt.floor("h")

    # 4. Quick audit
    print("\n🔍 Missing values (waits):")
    print(waits.isnull().sum())
    print(
        f"\n📅 Waits range: {waits['Local Time'].min()} to {waits['Local Time'].max()}"
    )

    return waits, weather_2024


def build_park_schedule_2024():
    waits = pd.read_csv(os.path.join(DATA_DIR, "disney_wait_times.csv"))
    waits["Local Time"] = pd.to_datetime(waits["Local Time"].str[:19])
    op = waits[waits["Wait Time"] > 0].copy()
    op["date"] = op["Local Time"].dt.date
    daily = (
        op.groupby("date")
        .agg(
            openTime=("Local Time", lambda s: s.min().strftime("%H:%M:%S")),
            closeTime=("Local Time", lambda s: s.max().strftime("%H:%M:%S")),
            avgWait=("Wait Time", "mean"),
        )
        .reset_index()
    )
    daily["parkID"] = 336894
    daily["crowdLevel"] = pd.qcut(daily["avgWait"], 10, labels=False) + 1
    daily[["date", "parkID", "openTime", "closeTime", "crowdLevel"]].to_csv(
        os.path.join(DATA_DIR, "parkSchedules_2024.csv"), index=False
    )


def train_model(merged_data):

    start_time = time.time()

    # 1. Ensure datetime is parsed and extract numeric Hour
    merged_data["Local Time"] = pd.to_datetime(merged_data["Local Time"])
    merged_data["hour_start"] = merged_data["hour_start"].dt.hour
    merged_data["feelsLikeF"] = merged_data["feelsLikeF"].astype(float)

    # 2. Define Target Variable: 1 if wait time > 60 minutes, else 0
    print("Engineering target and feature variables...")
    merged_data["is_high_wait"] = (merged_data["Wait Time"] > 60).astype(int)

    # Select features (Temporal, Environmental, and Categorical Ride names)
    X = merged_data[["hour_start", "feelsLikeF", "rainStatus", "Ride"]]
    y = merged_data["is_high_wait"]

    # 3. Stratified Train/Test Split (80% train, 20% test)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # 4. Build Scikit-Learn Preprocessing & Modeling Pipeline
    # Routes numerical features to StandardScaler and categorical features to OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["hour_start", "feelsLikeF", "rainStatus"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Ride"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # 5. Train Model
    print("Training Logistic Regression model...")
    model.fit(X_train, y_train)

    # 6. Evaluate Performance
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    print(f"y_pred : {y_pred}")
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"y_prob : {y_prob}")
    results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    print(f"results : {results}")

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy Score: {acc * 100:.2f}%")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # 7. Generate & Save Confusion Matrix Visual to charts/
    os.makedirs("charts", exist_ok=True)
    print("\nGenerating confusion matrix visualization...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal (<=60m)", "High Wait (>60m)"],
        yticklabels=["Normal (<=60m)", "High Wait (>60m)"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Disneyland High-Wait Confusion Matrix")
    plt.tight_layout()

    output_path = os.path.join("charts", "confusion_matrix.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved successfully to {output_path}!")

    end_time = time.time()
    train_duration = end_time - start_time
    print(
        f"Training completed in {train_duration:.2f} seconds ({train_duration / 60:.2f} minutes)."
    )
    # Return the trained pipeline object back to main
    return model


def save_model(model, filename="disney_wait_model.pkl", directory="models"):
    """Saves the trained machine learning pipeline to disk using joblib."""
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, filename)
    joblib.dump(model, model_path)
    print(f"Model pipeline saved successfully to {model_path}!")
    return model_path


if __name__ == "__main__":

    build_park_schedule_2024()

    waits, weather2 = load_data()

    # Quick sanity check: Print the first few rows of waits
    print("\n⏳ Sample waits Data:")
    print(waits.head(3))

    # Quick sanity check: Print the first few rows of weather
    print("\n☀️ Sample weather 2024 Data:")
    print(weather2.head(3))

    # Quick sanity check: Print the busiest rides on average
    print("\n⏱️ Top 5 Busiest Rides:")
    avg_waits = (
        waits.groupby("Ride")["Wait Time"].mean().sort_values(ascending=False).head(5)
    )
    print(avg_waits)

    waits, weather2 = clean_2024(waits, weather2)
    print("\n✅ Milestone 2 cleaning complete!")

    merged = waits.merge(
        weather2, left_on="hour_start", right_on="dateTime", how="left"
    )
    assert len(merged) == len(waits), "merge changed the row count"
    assert merged["feelsLikeF"].notna().all(), "some waits have no weather"
    assert (merged["Wait Time"] > 0).all(), "zero waits slipped through"
    print("\n✅ Milestone 3 merging complete!")

    create_visuals(merged)
    print("\n✅ Milestone 4 visualization complete!")

    print(merged.head())
    print(merged.info())

    trained_model = train_model(merged)
    print("\n✅ Milestone 5: Artifact Export & Visuals complete!")

    # ready to be loaded later for inference
    save_model(trained_model)
