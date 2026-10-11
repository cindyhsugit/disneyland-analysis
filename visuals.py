import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ACCENT = "#1F4E79"   # navy: the main message
HILITE = "#E4572E"   # orange: the one thing to notice
MUTED = "#C9CED6"    # gray: everything else
TEXT = "#2B2B2B"
SOURCE = "Source: Queue-Times wait data, Disney California Adventure"


def hour_label(h):
    return f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"


def style():
    sns.set_theme(style="white", rc={
        "font.family": "DejaVu Sans",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.titleweight": "bold",
    })


def finish(fig, ax_list, title, subtitle, footer):
    for ax in ax_list:
        sns.despine(ax=ax, left=True)
    fig.text(0.015, 0.97, title, fontsize=17, fontweight="bold", color=TEXT, va="top")
    fig.text(0.015, 0.905, subtitle, fontsize=11.5, color="#6B7280", va="top")
    fig.text(0.015, 0.012, footer, fontsize=8.5, color="#9CA3AF")


def create_visuals(merged, out_dir="charts"):
    """merged = cleaned waits + weather (columns: Ride, Wait Time, Local Time, rainStatus, feelsLikeF)."""
    print("📊 Creating visuals...")
    os.makedirs(out_dir, exist_ok=True)
    style()

    df = merged.copy()
    df["hour"] = df["Local Time"].dt.hour
    start = df["Local Time"].min().strftime("%Y-%m-%d")
    end = df["Local Time"].max().strftime("%Y-%m-%d")
    footer = f"{SOURCE}, {start} to {end}"
    suffix = f"{start}_to_{end}"

    # ---------- 1. Busiest rides ----------
    rides = df.groupby("Ride")["Wait Time"].mean().sort_values(ascending=False).head(10)
    rides = rides.iloc[::-1]  # biggest bar on top
    rides.index = [textwrap.fill(n, 26) for n in rides.index]
    colors = [ACCENT if i >= len(rides) - 3 else MUTED for i in range(len(rides))]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.27, right=0.94)
    ax.barh(rides.index, rides.values, color=colors, height=0.68)
    for y, v in enumerate(rides.values):
        ax.text(v + 0.8, y, f"{v:.0f} min", va="center", fontsize=11, color=TEXT)
    ax.set_xlim(0, rides.max() * 1.15)
    ax.set_xticks([]); ax.tick_params(axis="y", length=0, labelsize=11)
    ax.spines["bottom"].set_visible(False)
    top = rides.index[-1].replace("\n", " ")
    finish(fig, [ax], f"{top} has the longest line in the park",
           "Average wait by ride, top 10 (minutes)", footer)
    fig.savefig(os.path.join(out_dir, f"1_busiest_rides_{suffix}.png"), dpi=200)
    plt.close(fig)

    # ---------- 2. Park busy-ness through the day ----------
    by_hour = df.groupby("hour")["Wait Time"].mean()
    peak_h, peak_v = by_hour.idxmax(), by_hour.max()
    first_v = by_hour.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.07, right=0.96)
    ax.fill_between(by_hour.index, by_hour.values, color=ACCENT, alpha=0.12)
    ax.plot(by_hour.index, by_hour.values, color=ACCENT, linewidth=3)
    ax.scatter([peak_h], [peak_v], color=HILITE, s=110, zorder=5)
    ax.annotate(f"Busiest: {peak_v:.0f} min at {hour_label(peak_h)}",
                (peak_h, peak_v), xytext=(0, 18), textcoords="offset points",
                ha="center", fontsize=12, fontweight="bold", color=HILITE)
    ax.set_xticks(by_hour.index)
    ax.set_xticklabels([hour_label(h) for h in by_hour.index], fontsize=10)
    ax.set_ylim(0, peak_v * 1.25)
    ax.yaxis.grid(True, color="#EEF0F3"); ax.set_axisbelow(True)
    ax.set_ylabel("Average wait across all rides (min)", fontsize=10.5)
    ax.tick_params(axis="both", length=0)
    finish(fig, [ax], f"The park gets about {peak_v / first_v:.1f}x busier from morning to late afternoon",
           "Average wait across all open rides, by hour of day", footer)
    fig.savefig(os.path.join(out_dir, f"2_busy_by_hour_{suffix}.png"), dpi=200)
    plt.close(fig)

    # ---------- 3. Does weather matter? ----------
    rain = df.groupby("rainStatus")["Wait Time"].mean()
    rain.index = ["No rain" if i == 0 else "Rain" for i in rain.index]
    bands = pd.cut(df["feelsLikeF"].astype(float), [0, 60, 70, 80, 90, 130],
                   labels=["Under 60°", "60-70°", "70-80°", "80-90°", "90°+"])
    temp = df.groupby(bands, observed=True)["Wait Time"].mean()
    ymax = max(rain.max(), temp.max()) * 1.25
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [2, 5]})
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.06, right=0.97, wspace=0.12)
    for ax, series, subtitle in ((a1, rain, "Rain"), (a2, temp, "Feels-like temperature")):
        ax.bar(series.index.astype(str), series.values, color=ACCENT, width=0.6)
        for x, v in enumerate(series.values):
            ax.text(x, v + ymax * 0.02, f"{v:.0f}", ha="center", fontsize=12, fontweight="bold", color=TEXT)
        ax.set_ylim(0, ymax)
        ax.set_title(subtitle, fontsize=11.5, color="#6B7280", loc="left", fontweight="normal")
        ax.set_yticks([]); ax.tick_params(axis="x", length=0, labelsize=10.5)
    finish(fig, [a1, a2], "Weather barely changes how long you wait",
           "Average wait in minutes, by rain and by temperature", footer)
    fig.savefig(os.path.join(out_dir, f"3_weather_effect_{suffix}.png"), dpi=200)
    plt.close(fig)

    print(f"   - Saved 3 charts to {out_dir}/")
