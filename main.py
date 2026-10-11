import os
import pandas as pd
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
