import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    conn = duckdb.connect('race_database.db')

    query = """
    WITH race_weather AS (
        SELECT DISTINCT
            race_id,
            MAX(rainfall::INTEGER) as had_rain
        FROM weather_data
        GROUP BY race_id
    ),
    final_positions AS (
        SELECT
            race_id,
            driver,
            position
        FROM race_results
        WHERE deleted = FALSE
        QUALIFY ROW_NUMBER() OVER (PARTITION BY race_id, driver ORDER BY lap_number DESC) = 1
    ),
    driver_stats AS (
        SELECT
            fp.driver,
            AVG(CASE WHEN rw.had_rain = 1 THEN fp.position END) as avg_pos_rain,
            AVG(CASE WHEN rw.had_rain = 0 THEN fp.position END) as avg_pos_dry,
            COUNT(CASE WHEN rw.had_rain = 1 THEN 1 END) as rain_races,
            COUNT(CASE WHEN rw.had_rain = 0 THEN 1 END) as dry_races
        FROM final_positions fp
        JOIN race_weather rw ON fp.race_id = rw.race_id
        GROUP BY fp.driver
        HAVING rain_races >= 3 AND dry_races >= 5
    )
    SELECT
        driver,
        ROUND(avg_pos_rain, 2) as avg_pos_rain,
        ROUND(avg_pos_dry, 2) as avg_pos_dry,
        ROUND(avg_pos_dry - avg_pos_rain, 2) as rain_advantage,
        rain_races,
        dry_races
    FROM driver_stats
    ORDER BY rain_advantage DESC;
    """

    df = conn.execute(query).df()
    conn.close()

    print("Top 10 'rain masters' (největší zlepšení v dešti):")
    print(df.head(10).to_string(index=False))

    plt.figure(figsize=(12, 8))
    top_10 = df.head(10)

    # Barevné kódování (zelená = zlepšení, červená = zhoršení)
    colors = ['green' if x > 0 else 'red' for x in top_10['rain_advantage']]

    sns.barplot(data=top_10, y='driver', x='rain_advantage', palette=colors)
    plt.xlabel('Zlepšení pozice v dešti (kladné = lepší)', fontsize=12)
    plt.ylabel('Jezdec', fontsize=12)
    plt.title('Jezdci nejsilnější v dešti', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

    for i, row in top_10.iterrows():
        plt.text(row['rain_advantage'] + 0.1, i,
                 f"{row['rain_advantage']:.2f} ({row['rain_races']} závodů)",
                 va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()