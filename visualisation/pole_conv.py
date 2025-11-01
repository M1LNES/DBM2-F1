import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def main():
    # Připojení k databázi
    conn = duckdb.connect('race_database.db')

    # SQL dotaz pro analýzu P1 -> vítězství
    query = """
    WITH pole_positions AS (
        SELECT
            q.race_id,
            q.driver,
            r.circuit_id,
            c.name as circuit_name,
            c.location
        FROM qualifying q
        JOIN races r ON q.race_id = r.race_id
        JOIN circuits c ON r.circuit_id = c.circuit_id
        WHERE q.position = 1
    ),
    race_winners AS (
        SELECT
            race_id,
            driver as winner
        FROM race_results
        WHERE deleted = FALSE
        QUALIFY ROW_NUMBER() OVER (PARTITION BY race_id ORDER BY lap_number DESC, position) = 1
    ),
    conversions AS (
        SELECT
            pp.circuit_name,
            pp.location,
            COUNT(*) as pole_positions,
            SUM(CASE WHEN pp.driver = rw.winner THEN 1 ELSE 0 END) as wins_from_pole,
            ROUND(100.0 * SUM(CASE WHEN pp.driver = rw.winner THEN 1 ELSE 0 END) / COUNT(*), 1) as conversion_rate
        FROM pole_positions pp
        JOIN race_winners rw ON pp.race_id = rw.race_id
        GROUP BY pp.circuit_name, pp.location
        HAVING COUNT(*) >= 3
    )
    SELECT *
    FROM conversions
    ORDER BY conversion_rate DESC;
    """

    df = conn.execute(query).df()
    conn.close()

    print(df.to_string(index=False))

    # Graf top 15 tratí
    plt.figure(figsize=(12, 8))
    top_15 = df.head(15)

    sns.barplot(data=top_15, y='circuit_name', x='conversion_rate', palette='RdYlGn')
    plt.xlabel('Úspěšnost P1 → vítězství (%)', fontsize=12)
    plt.ylabel('Okruh', fontsize=12)
    plt.title('Tratě s nejvyšší šancí na vítězství z pole position', fontsize=14, fontweight='bold')
    plt.xlim(0, 100)

    # Přidání hodnot na grafy
    for i, row in top_15.iterrows():
        plt.text(row['conversion_rate'] + 1, i,
                 f"{row['conversion_rate']:.1f}% ({row['wins_from_pole']}/{row['pole_positions']})",
                 va='center', fontsize=9)

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()