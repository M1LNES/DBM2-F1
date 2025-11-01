import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    conn = duckdb.connect('race_database.db')

    query = """
    WITH race_laps AS (
        SELECT
            race_id,
            MAX(lap_number) as total_laps
        FROM race_results
        WHERE deleted = FALSE
        GROUP BY race_id
    ),
    final_stint_data AS (
        SELECT
            rr.driver,
            rr.compound,
            rr.tyre_life,
            rr.lap_time_seconds,
            rr.lap_number,
            rl.total_laps,
            rl.total_laps - rr.lap_number as laps_to_finish
        FROM race_results rr
        JOIN race_laps rl ON rr.race_id = rl.race_id
        WHERE rr.deleted = FALSE
            AND rr.lap_time_seconds IS NOT NULL
            AND rr.lap_time_seconds BETWEEN 60 AND 150
            AND rr.tyre_life IS NOT NULL
            AND rl.total_laps - rr.lap_number <= 10
    ),
    tyre_age_groups AS (
        SELECT
            CASE
                WHEN tyre_life <= 5 THEN '0-5 kol'
                WHEN tyre_life <= 15 THEN '6-15 kol'
                WHEN tyre_life <= 25 THEN '16-25 kol'
                ELSE '26+ kol'
            END as tyre_age_group,
            AVG(lap_time_seconds) as avg_lap_time,
            COUNT(*) as sample_size
        FROM final_stint_data
        GROUP BY tyre_age_group
    )
    SELECT * FROM tyre_age_groups
    ORDER BY
        CASE tyre_age_group
            WHEN '0-5 kol' THEN 1
            WHEN '6-15 kol' THEN 2
            WHEN '16-25 kol' THEN 3
            ELSE 4
        END;
    """

    df = conn.execute(query).df()
    conn.close()

    print("Průměrné časy kol v závěru závodu podle stáří pneumatik:")
    print(df.to_string(index=False))

    # Vizualizace
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Graf 1: Průměrné časy
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
    bars = ax1.bar(df['tyre_age_group'], df['avg_lap_time'], color=colors)
    ax1.set_xlabel('Stáří pneumatik', fontsize=12)
    ax1.set_ylabel('Průměrný čas kola (s)', fontsize=12)
    ax1.set_title('Degradace pneumatik v posledních 10 kolech', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, (idx, row) in enumerate(df.iterrows()):
        ax1.text(i, row['avg_lap_time'] + 0.3,
                 f"{row['avg_lap_time']:.2f}s",
                 ha='center', fontsize=10, fontweight='bold')

    # Graf 2: Relativní zpomalení
    baseline = df.iloc[0]['avg_lap_time']
    df['slowdown_pct'] = ((df['avg_lap_time'] - baseline) / baseline) * 100

    ax2.plot(df['tyre_age_group'], df['slowdown_pct'], marker='o', linewidth=2.5,
             markersize=10, color='#e74c3c')
    ax2.fill_between(range(len(df)), df['slowdown_pct'], alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Stáří pneumatik', fontsize=12)
    ax2.set_ylabel('Zpomalení oproti novým (%)', fontsize=12)
    ax2.set_title('Progresivní ztráta tempa', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

    for i, row in df.iterrows():
        ax2.text(i, row['slowdown_pct'] + 0.05,
                 f"+{row['slowdown_pct']:.2f}%",
                 ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()