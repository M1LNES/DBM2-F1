import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_driver_weather_heatmap():
    """Heatmap showing driver performance in different weather conditions."""
    conn = duckdb.connect('../race_database.db')

    query = """
    WITH race_weather AS (
        SELECT DISTINCT
            wd.race_id,
            CASE
                WHEN AVG(wd.rainfall::INTEGER) > 0.3 THEN 'Rain'
                ELSE 'Dry'
            END as rain_condition,
            AVG(wd.track_temp) as avg_track_temp,
            CASE
                WHEN AVG(wd.track_temp) < 30 THEN 'Cold Track'
                WHEN AVG(wd.track_temp) BETWEEN 30 AND 40 THEN 'Medium Track'
                ELSE 'Hot Track'
            END as temp_condition
        FROM weather_data wd
        WHERE wd.track_temp IS NOT NULL
        GROUP BY wd.race_id
    ),
    driver_weather_performance AS (
        SELECT
            rr.driver,
            rw.rain_condition,
            rw.temp_condition,
            AVG(rr.position) as avg_position,
            COUNT(DISTINCT rr.race_id) as races_count
        FROM race_results rr
        JOIN race_weather rw ON rr.race_id = rw.race_id
        WHERE rr.position IS NOT NULL
            AND rr.deleted = FALSE
            AND rr.lap_number = (
                SELECT MAX(lap_number)
                FROM race_results rr2
                WHERE rr2.race_id = rr.race_id
                    AND rr2.driver = rr.driver
            )
        GROUP BY rr.driver, rw.rain_condition, rw.temp_condition
        HAVING COUNT(DISTINCT rr.race_id) >= 1
    )
    SELECT
        driver,
        rain_condition || ' + ' || temp_condition as weather_condition,
        avg_position,
        races_count
    FROM driver_weather_performance
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Create pivot table for heatmap - ALL drivers
    pivot_df = df.pivot_table(
        values='avg_position',
        index='driver',
        columns='weather_condition',
        aggfunc='mean'
    )

    # Reorder columns for better readability
    desired_order = [
        'Dry + Cold Track', 'Dry + Medium Track', 'Dry + Hot Track',
        'Rain + Cold Track', 'Rain + Medium Track', 'Rain + Hot Track'
    ]
    pivot_df = pivot_df.reindex(columns=[col for col in desired_order if col in pivot_df.columns])

    # Sort by average position across all conditions
    pivot_df['avg_all'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('avg_all')
    pivot_df = pivot_df.drop('avg_all', axis=1)

    # Dynamic figure height based on number of drivers
    num_drivers = len(pivot_df)
    fig_height = max(18, num_drivers * 0.4)

    fig, ax = plt.subplots(figsize=(14, fig_height))
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn_r',
                center=10, vmin=1, vmax=20, linewidths=0.5,
                cbar_kws={'label': 'Average Final Position'}, ax=ax)

    ax.set_title(f'Driver Performance Heatmap by Weather Conditions (All {num_drivers} Drivers)',
                 fontsize=14, pad=20)
    ax.set_xlabel('Weather Conditions', fontsize=11)
    ax.set_ylabel('Driver', fontsize=11)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'driver_weather_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Graf uložen: {plots_dir / 'driver_weather_heatmap.png'}")
    print(f"Počet zobrazených jezdců: {num_drivers}")

    plt.show()
    plt.close()

    print("\n=== Weather Heatmap Data (All Drivers) ===")
    print(pivot_df)


def plot_driver_comparison_rain_vs_dry():
    """Direct comparison: Driver performance in Rain vs Dry conditions."""
    conn = duckdb.connect('../race_database.db')

    query = """
    WITH race_weather AS (
        SELECT DISTINCT
            wd.race_id,
            CASE
                WHEN AVG(wd.rainfall::INTEGER) > 0.3 THEN TRUE
                ELSE FALSE
            END as is_rain
        FROM weather_data wd
        GROUP BY wd.race_id
    ),
    driver_performance AS (
        SELECT
            rr.driver,
            rw.is_rain,
            AVG(rr.position) as avg_position,
            COUNT(DISTINCT rr.race_id) as races_count
        FROM race_results rr
        JOIN race_weather rw ON rr.race_id = rw.race_id
        WHERE rr.position IS NOT NULL
            AND rr.deleted = FALSE
            AND rr.lap_number = (
                SELECT MAX(lap_number)
                FROM race_results rr2
                WHERE rr2.race_id = rr.race_id
                    AND rr2.driver = rr.driver
            )
        GROUP BY rr.driver, rw.is_rain
        HAVING COUNT(DISTINCT rr.race_id) >= 2
    )
    SELECT
        driver,
        is_rain,
        avg_position,
        races_count
    FROM driver_performance
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Get drivers who have data for both rain and dry conditions
    drivers_both = df.groupby('driver')['is_rain'].nunique()
    drivers_both = drivers_both[drivers_both == 2].index

    df_both = df[df['driver'].isin(drivers_both)]

    # Create pivot for scatter plot
    pivot_df = df_both.pivot_table(
        values='avg_position',
        index='driver',
        columns='is_rain',
        aggfunc='mean'
    )
    pivot_df.columns = ['Dry Conditions', 'Rain Conditions']

    # Calculate difference (positive = better in rain)
    pivot_df['difference'] = pivot_df['Dry Conditions'] - pivot_df['Rain Conditions']
    pivot_df = pivot_df.sort_values('difference', ascending=False)

    # Use ALL drivers
    all_drivers = pivot_df

    fig, ax = plt.subplots(figsize=(16, 12))

    # Scatter plot
    colors = ['#4682B4' if diff > 0 else '#FF8C00' for diff in all_drivers['difference']]
    ax.scatter(all_drivers['Dry Conditions'], all_drivers['Rain Conditions'],
               s=150, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)

    # Add diagonal line (equal performance)
    lim_min = min(all_drivers[['Dry Conditions', 'Rain Conditions']].min())
    lim_max = max(all_drivers[['Dry Conditions', 'Rain Conditions']].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'k--', alpha=0.5, linewidth=2, label='Equal Performance')

    # Add labels for each point
    for idx, row in all_drivers.iterrows():
        ax.annotate(idx, (row['Dry Conditions'], row['Rain Conditions']),
                   fontsize=7, alpha=0.7, ha='center', va='center')

    ax.set_xlabel('Average Position in Dry Conditions (lower is better)', fontsize=11)
    ax.set_ylabel('Average Position in Rain Conditions (lower is better)', fontsize=11)
    ax.set_title(f'Driver Performance: Dry vs Rain Conditions (All {len(all_drivers)} Drivers)',
                 fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add text explanation
    ax.text(0.02, 0.98, 'Blue = Better in Rain (Rain Masters)\nOrange = Better in Dry',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'driver_comparison_rain_dry.png', dpi=300, bbox_inches='tight')
    print(f"Graf uložen: {plots_dir / 'driver_comparison_rain_dry.png'}")
    print(f"Počet zobrazených jezdců: {len(all_drivers)}")

    plt.show()
    plt.close()

    print("\n=== Rain vs Dry Performance (All Drivers) ===")
    print(all_drivers[['Dry Conditions', 'Rain Conditions', 'difference']].to_string())


def plot_driver_comparison_track_temperature():
    """Direct comparison: Driver performance in Cold vs Hot track temperatures."""
    conn = duckdb.connect('../race_database.db')

    query = """
    WITH race_weather AS (
        SELECT DISTINCT
            wd.race_id,
            AVG(wd.track_temp) as avg_track_temp,
            CASE
                WHEN AVG(wd.track_temp) < 35 THEN FALSE
                ELSE TRUE
            END as is_hot_track
        FROM weather_data wd
        WHERE wd.track_temp IS NOT NULL
        GROUP BY wd.race_id
    ),
    driver_performance AS (
        SELECT
            rr.driver,
            rw.is_hot_track,
            AVG(rr.position) as avg_position,
            COUNT(DISTINCT rr.race_id) as races_count
        FROM race_results rr
        JOIN race_weather rw ON rr.race_id = rw.race_id
        WHERE rr.position IS NOT NULL
            AND rr.deleted = FALSE
            AND rr.lap_number = (
                SELECT MAX(lap_number)
                FROM race_results rr2
                WHERE rr2.race_id = rr.race_id
                    AND rr2.driver = rr.driver
            )
        GROUP BY rr.driver, rw.is_hot_track
        HAVING COUNT(DISTINCT rr.race_id) >= 2
    )
    SELECT
        driver,
        is_hot_track,
        avg_position,
        races_count
    FROM driver_performance
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Get drivers who have data for both cold and hot track conditions
    drivers_both = df.groupby('driver')['is_hot_track'].nunique()
    drivers_both = drivers_both[drivers_both == 2].index

    df_both = df[df['driver'].isin(drivers_both)]

    # Create pivot for scatter plot
    pivot_df = df_both.pivot_table(
        values='avg_position',
        index='driver',
        columns='is_hot_track',
        aggfunc='mean'
    )
    pivot_df.columns = ['Cold Track (<35°C)', 'Hot Track (≥35°C)']

    # Calculate difference (positive = better on hot track)
    pivot_df['difference'] = pivot_df['Cold Track (<35°C)'] - pivot_df['Hot Track (≥35°C)']
    pivot_df = pivot_df.sort_values('difference', ascending=False)

    # Use ALL drivers
    all_drivers = pivot_df

    fig, ax = plt.subplots(figsize=(16, 12))

    # Scatter plot
    colors = ['#DC143C' if diff > 0 else '#1E90FF' for diff in all_drivers['difference']]
    ax.scatter(all_drivers['Cold Track (<35°C)'], all_drivers['Hot Track (≥35°C)'],
               s=150, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)

    # Add diagonal line (equal performance)
    lim_min = min(all_drivers[['Cold Track (<35°C)', 'Hot Track (≥35°C)']].min())
    lim_max = max(all_drivers[['Cold Track (<35°C)', 'Hot Track (≥35°C)']].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'k--', alpha=0.5, linewidth=2, label='Equal Performance')

    # Add labels for each point
    for idx, row in all_drivers.iterrows():
        ax.annotate(idx, (row['Cold Track (<35°C)'], row['Hot Track (≥35°C)']),
                   fontsize=7, alpha=0.7, ha='center', va='center')

    ax.set_xlabel('Average Position on Cold Track (lower is better)', fontsize=11)
    ax.set_ylabel('Average Position on Hot Track (lower is better)', fontsize=11)
    ax.set_title(f'Driver Performance: Cold vs Hot Track Temperature (All {len(all_drivers)} Drivers)',
                 fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add text explanation
    ax.text(0.02, 0.98, 'Red = Better on Hot Track\nBlue = Better on Cold Track',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'driver_comparison_track_temp.png', dpi=300, bbox_inches='tight')
    print(f"Graf uložen: {plots_dir / 'driver_comparison_track_temp.png'}")
    print(f"Počet zobrazených jezdců: {len(all_drivers)}")

    plt.show()
    plt.close()

    print("\n=== Cold vs Hot Track Performance (All Drivers) ===")
    print(all_drivers[['Cold Track (<35°C)', 'Hot Track (≥35°C)', 'difference']].to_string())


if __name__ == "__main__":
    print("=== 1. Driver Performance Heatmap by Weather (All Drivers) ===")
    plot_driver_weather_heatmap()

    print("\n=== 2. Rain vs Dry Conditions Comparison (All Drivers) ===")
    plot_driver_comparison_rain_vs_dry()

    print("\n=== 3. Cold vs Hot Track Temperature Comparison (All Drivers) ===")
    plot_driver_comparison_track_temperature()
