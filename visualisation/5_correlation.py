import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def plot_weather_performance_correlation():
    """Correlation between weather conditions and race performance."""
    conn = duckdb.connect('../race_database.db')

    query = """
    WITH race_final AS (
        SELECT
            race_id, driver,
            position as final_position,
            AVG(speed_fl) as avg_speed,
            AVG(lap_time_seconds) as avg_lap_time
        FROM race_results
        WHERE lap_number = (SELECT MAX(lap_number) FROM race_results rr2
                           WHERE rr2.race_id = race_results.race_id
                           AND rr2.driver = race_results.driver)
        AND position IS NOT NULL
        GROUP BY race_id, driver, position
    ),
    avg_weather AS (
        SELECT
            race_id,
            AVG(air_temp) as avg_air_temp,
            AVG(track_temp) as avg_track_temp,
            AVG(humidity) as avg_humidity,
            AVG(wind_speed) as avg_wind_speed,
            AVG(pressure) as avg_pressure
        FROM weather_data
        GROUP BY race_id
    )
    SELECT
        rf.final_position,
        rf.avg_speed,
        rf.avg_lap_time,
        aw.avg_air_temp,
        aw.avg_track_temp,
        aw.avg_humidity,
        aw.avg_wind_speed,
        aw.avg_pressure
    FROM race_final rf
    JOIN avg_weather aw ON rf.race_id = aw.race_id
    WHERE rf.final_position IS NOT NULL
    """

    df = pd.read_sql(query, conn)
    conn.close()

    corr_matrix = df.corr()

    fig = plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Correlation coefficient'})

    plt.title('Correlation Matrix: Weather Conditions vs Performance', fontsize=14, pad=20)
    plt.tight_layout()

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'correlation_weather_performance.png', dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

    return corr_matrix


def plot_tyre_strategy_correlation():
    """Correlation between tyre strategy and race results."""
    conn = duckdb.connect('../race_database.db')

    query = """
    WITH race_stats AS (
        SELECT
            race_id, driver,
            MAX(position) as final_position,
            COUNT(DISTINCT compound) as compounds_used,
            MAX(stint) as pit_stops,
            AVG(tyre_life) as avg_tyre_life,
            AVG(lap_time_seconds) as avg_lap_time,
            COUNT(CASE WHEN fresh_tyre = TRUE THEN 1 END) as fresh_tyre_laps
        FROM race_results
        WHERE lap_number = (SELECT MAX(lap_number) FROM race_results rr2
                           WHERE rr2.race_id = race_results.race_id
                           AND rr2.driver = race_results.driver)
        GROUP BY race_id, driver
    )
    SELECT
        final_position,
        compounds_used,
        pit_stops,
        avg_tyre_life,
        avg_lap_time,
        fresh_tyre_laps
    FROM race_stats
    WHERE final_position IS NOT NULL
    """

    df = pd.read_sql(query, conn)
    conn.close()

    corr_matrix = df.corr()

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Correlation coefficient'})

    plt.title('Correlation Matrix: Tyre Strategy vs Performance', fontsize=14, pad=20)
    plt.tight_layout()

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'correlation_tyre_strategy.png', dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

    return corr_matrix


def plot_circuit_type_correlation():
    """Correlation between circuit characteristics and performance metrics."""
    conn = duckdb.connect('../race_database.db')

    query = """
    WITH race_stats AS (
        SELECT
            rr.race_id,
            r.city_circuit,
            r.night_race,
            AVG(rr.speed_fl) as avg_speed,
            AVG(rr.lap_time_seconds) as avg_lap_time,
            MAX(rr.stint) as avg_pit_stops
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        GROUP BY rr.race_id, rr.driver, r.city_circuit, r.night_race
    )
    SELECT
        CAST(city_circuit AS INTEGER) as city_circuit,
        CAST(night_race AS INTEGER) as night_race,
        avg_speed,
        avg_lap_time,
        avg_pit_stops
    FROM race_stats
    """

    df = pd.read_sql(query, conn)
    conn.close()

    corr_matrix = df.corr()

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Correlation coefficient'})

    plt.title('Correlation Matrix: Circuit Type vs Performance', fontsize=14, pad=20)
    plt.tight_layout()

    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'correlation_circuit_type.png', dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()

    return corr_matrix


if __name__ == "__main__":
    plot_weather_performance_correlation()
    plot_tyre_strategy_correlation()
    plot_circuit_type_correlation()
