import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def plot_driver_heatmap_conditions():
    """Heatmap showing driver performance across different conditions."""
    conn = duckdb.connect('../race_database.db')

    query = """
    SELECT
        rr.driver,
        CASE
            WHEN r.night_race = TRUE AND r.city_circuit = TRUE THEN 'Night + City'
            WHEN r.night_race = TRUE AND r.city_circuit = FALSE THEN 'Night + Regular'
            WHEN r.night_race = FALSE AND r.city_circuit = TRUE THEN 'Day + City'
            ELSE 'Day + Regular'
        END as condition_type,
        AVG(rr.position) as avg_position
    FROM race_results rr
    JOIN races r ON rr.race_id = r.race_id
    WHERE rr.position IS NOT NULL
        AND rr.deleted = FALSE
        AND rr.lap_number = (
            SELECT MAX(lap_number)
            FROM race_results rr2
            WHERE rr2.race_id = rr.race_id
                AND rr2.driver = rr.driver
        )
    GROUP BY rr.driver, condition_type
    HAVING COUNT(DISTINCT r.race_id) >= 1
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Create pivot table for heatmap - ALL drivers
    pivot_df = df.pivot_table(
        values='avg_position',
        index='driver',
        columns='condition_type',
        aggfunc='mean'
    )

    # Sort by average position across all conditions
    pivot_df['avg_all'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('avg_all')
    pivot_df = pivot_df.drop('avg_all', axis=1)

    # Dynamic figure height based on number of drivers
    num_drivers = len(pivot_df)
    fig_height = max(18, num_drivers * 0.4)

    fig, ax = plt.subplots(figsize=(12, fig_height))
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn_r',
                center=10, vmin=1, vmax=20, linewidths=0.5,
                cbar_kws={'label': 'Average Final Position'}, ax=ax)

    ax.set_title(f'Driver Performance Heatmap by Race Conditions (All {num_drivers} Drivers)',
                 fontsize=14, pad=20)
    ax.set_xlabel('Race Conditions', fontsize=11)
    ax.set_ylabel('Driver', fontsize=11)

    plt.tight_layout()

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'driver_performance_heatmap.png', dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)

    return pivot_df


def plot_driver_comparison_day_vs_night():
    """Direct comparison: Driver performance in Day vs Night races."""
    conn = duckdb.connect('../race_database.db')

    query = """
    WITH driver_performance AS (
        SELECT
            rr.driver,
            r.night_race,
            AVG(rr.position) as avg_position,
            COUNT(DISTINCT r.race_id) as races_count
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        WHERE rr.position IS NOT NULL
            AND rr.deleted = FALSE
            AND rr.lap_number = (
                SELECT MAX(lap_number)
                FROM race_results rr2
                WHERE rr2.race_id = rr.race_id
                    AND rr2.driver = rr.driver
            )
        GROUP BY rr.driver, r.night_race
        HAVING COUNT(DISTINCT r.race_id) >= 2
    )
    SELECT
        driver,
        night_race,
        avg_position,
        races_count
    FROM driver_performance
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Get drivers who have data for both day and night races
    drivers_both = df.groupby('driver')['night_race'].nunique()
    drivers_both = drivers_both[drivers_both == 2].index

    df_both = df[df['driver'].isin(drivers_both)].copy()

    # Create pivot for scatter plot
    pivot_df = df_both.pivot_table(
        values='avg_position',
        index='driver',
        columns='night_race',
        aggfunc='mean'
    )
    # Ensure columns are named consistently
    pivot_df.rename(columns={False: 'Day Race', True: 'Night Race'}, inplace=True)

    # Drop drivers with missing data just in case
    pivot_df = pivot_df.dropna(subset=['Day Race', 'Night Race'])

    # Calculate difference (positive = better in night races)
    pivot_df['difference'] = pivot_df['Day Race'] - pivot_df['Night Race']
    pivot_df = pivot_df.sort_values('difference', ascending=False)

    all_drivers = pivot_df

    fig, ax = plt.subplots(figsize=(16, 12))

    # Scatter plot
    colors = ['#4169E1' if diff > 0 else '#FFD700' for diff in all_drivers['difference']]
    ax.scatter(all_drivers['Day Race'], all_drivers['Night Race'],
               s=150, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)

    # Add diagonal line (equal performance)
    lim_min = min(all_drivers[['Day Race', 'Night Race']].min())
    lim_max = max(all_drivers[['Day Race', 'Night Race']].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'k--', alpha=0.5, linewidth=2, label='Equal Performance')

    # Add labels for each point
    for idx, row in all_drivers.iterrows():
        ax.annotate(idx, (row['Day Race'], row['Night Race']),
                   fontsize=7, alpha=0.7, ha='center', va='center')

    ax.set_xlabel('Average Position in Day Races (lower is better)', fontsize=11)
    ax.set_ylabel('Average Position in Night Races (lower is better)', fontsize=11)
    ax.set_title(f'Driver Performance: Day vs Night Races (All {len(all_drivers)} Drivers)',
                 fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add text explanation
    ax.text(0.02, 0.98, 'Blue = Better in Night Races\nGold = Better in Day Races',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'driver_comparison_day_night.png', dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)

    return all_drivers


def plot_driver_comparison_city_vs_regular():
    """Direct comparison: Driver performance in City vs Regular circuits."""
    conn = duckdb.connect('../race_database.db')

    query = """
    WITH driver_performance AS (
        SELECT
            rr.driver,
            r.city_circuit,
            AVG(rr.position) as avg_position,
            COUNT(DISTINCT r.race_id) as races_count
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        WHERE rr.position IS NOT NULL
            AND rr.deleted = FALSE
            AND rr.lap_number = (
                SELECT MAX(lap_number)
                FROM race_results rr2
                WHERE rr2.race_id = rr.race_id
                    AND rr2.driver = rr.driver
            )
        GROUP BY rr.driver, r.city_circuit
        HAVING COUNT(DISTINCT r.race_id) >= 2
    )
    SELECT
        driver,
        city_circuit,
        avg_position,
        races_count
    FROM driver_performance
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Get drivers who have data for both city and regular circuits
    drivers_both = df.groupby('driver')['city_circuit'].nunique()
    drivers_both = drivers_both[drivers_both == 2].index

    df_both = df[df['driver'].isin(drivers_both)].copy()

    # Create pivot for scatter plot
    pivot_df = df_both.pivot_table(
        values='avg_position',
        index='driver',
        columns='city_circuit',
        aggfunc='mean'
    )
    # Ensure columns are named consistently
    pivot_df.rename(columns={False: 'Regular Circuit', True: 'City Circuit'}, inplace=True)

    # Drop drivers with missing data just in case
    pivot_df = pivot_df.dropna(subset=['Regular Circuit', 'City Circuit'])

    # Calculate difference (positive = better in city circuits)
    pivot_df['difference'] = pivot_df['Regular Circuit'] - pivot_df['City Circuit']
    pivot_df = pivot_df.sort_values('difference', ascending=False)

    all_drivers = pivot_df

    fig, ax = plt.subplots(figsize=(16, 12))

    # Scatter plot
    colors = ['#FF6347' if diff > 0 else '#32CD32' for diff in all_drivers['difference']]
    ax.scatter(all_drivers['Regular Circuit'], all_drivers['City Circuit'],
               s=150, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)

    # Add diagonal line (equal performance)
    lim_min = min(all_drivers[['Regular Circuit', 'City Circuit']].min())
    lim_max = max(all_drivers[['Regular Circuit', 'City Circuit']].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'k--', alpha=0.5, linewidth=2, label='Equal Performance')

    # Add labels for each point
    for idx, row in all_drivers.iterrows():
        ax.annotate(idx, (row['Regular Circuit'], row['City Circuit']),
                   fontsize=7, alpha=0.7, ha='center', va='center')

    ax.set_xlabel('Average Position in Regular Circuits (lower is better)', fontsize=11)
    ax.set_ylabel('Average Position in City Circuits (lower is better)', fontsize=11)
    ax.set_title(f'Driver Performance: City vs Regular Circuits (All {len(all_drivers)} Drivers)',
                 fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add text explanation
    ax.text(0.02, 0.98, 'Red = Better in City Circuits\nGreen = Better in Regular Circuits',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'driver_comparison_city_regular.png', dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(fig)

    return all_drivers


if __name__ == "__main__":
    # Run plots and collect returned DataFrames if needed
    heatmap_df = plot_driver_heatmap_conditions()
    day_night_df = plot_driver_comparison_day_vs_night()
    city_regular_df = plot_driver_comparison_city_vs_regular()
