import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    conn = duckdb.connect('race_database.db')

    query = """
    WITH position_changes_per_lap AS (
        SELECT
            rr.race_id,
            rr.lap_number,
            rr.driver,
            rr.position,
            LAG(rr.position) OVER (
                PARTITION BY rr.race_id, rr.driver 
                ORDER BY rr.lap_number
            ) as prev_position
        FROM race_results rr
        WHERE rr.deleted = FALSE
            AND rr.position IS NOT NULL
            AND rr.lap_number > 1
    ),
    overtakes_per_race AS (
        SELECT
            race_id,
            COUNT(*) as total_position_changes,
            SUM(CASE WHEN position < prev_position THEN 1 ELSE 0 END) as overtakes,
            SUM(CASE WHEN position > prev_position THEN 1 ELSE 0 END) as positions_lost
        FROM position_changes_per_lap
        WHERE prev_position IS NOT NULL
            AND ABS(position - prev_position) >= 1
        GROUP BY race_id
    ),
    race_info AS (
        SELECT
            r.race_id,
            r.year,
            r.race_name,
            c.name as circuit_name,
            c.country,
            COUNT(DISTINCT rr.lap_number) as total_laps
        FROM races r
        JOIN circuits c ON r.circuit_id = c.circuit_id
        JOIN race_results rr ON r.race_id = rr.race_id
        WHERE rr.deleted = FALSE
        GROUP BY r.race_id, r.year, r.race_name, c.name, c.country
    )
    SELECT
        ri.circuit_name,
        ri.country,
        COUNT(DISTINCT ri.race_id) as races,
        ROUND(AVG(opr.overtakes), 1) as avg_overtakes,
        ROUND(AVG(opr.overtakes::DOUBLE / ri.total_laps), 2) as overtakes_per_lap,
        MAX(opr.overtakes) as max_overtakes,
        MIN(opr.overtakes) as min_overtakes,
        ROUND(STDDEV(opr.overtakes), 1) as consistency
    FROM race_info ri
    JOIN overtakes_per_race opr ON ri.race_id = opr.race_id
    GROUP BY ri.circuit_name, ri.country
    HAVING COUNT(DISTINCT ri.race_id) >= 2
    ORDER BY avg_overtakes DESC;
    """

    df = conn.execute(query).df()
    conn.close()

    print("Circuit Overtaking Analysis:")
    print(df.to_string(index=False))

    # Vizualizace
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Graf 1: Top 15 nejvíce předjíždějících okruhů
    ax1 = fig.add_subplot(gs[0, :])
    top_15 = df.head(15)
    colors = plt.cm.RdYlGn(top_15['avg_overtakes'] / top_15['avg_overtakes'].max())

    bars = ax1.barh(range(len(top_15)), top_15['avg_overtakes'], color=colors, alpha=0.85)
    ax1.set_yticks(range(len(top_15)))
    ax1.set_yticklabels([f"{row['circuit_name']}\n({row['country']})"
                          for _, row in top_15.iterrows()], fontsize=9)
    ax1.set_xlabel('Průměrný počet předjetí za závod', fontsize=11)
    ax1.set_title('🏁 Top 15 Overtaking Circuits', fontsize=15, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(top_15.iterrows()):
        ax1.text(row['avg_overtakes'] + 2, i,
                f"{row['avg_overtakes']:.1f} ({row['races']} závodů)",
                va='center', fontsize=8, fontweight='bold')

    # Graf 2: Bottom 15 nejnudnějších okruhů
    ax2 = fig.add_subplot(gs[1, :])
    bottom_15 = df.tail(15).iloc[::-1]
    colors_b = plt.cm.RdYlGn_r(bottom_15['avg_overtakes'] / df['avg_overtakes'].max())

    bars = ax2.barh(range(len(bottom_15)), bottom_15['avg_overtakes'], color=colors_b, alpha=0.85)
    ax2.set_yticks(range(len(bottom_15)))
    ax2.set_yticklabels([f"{row['circuit_name']}\n({row['country']})"
                          for _, row in bottom_15.iterrows()], fontsize=9)
    ax2.set_xlabel('Průměrný počet předjetí za závod', fontsize=11)
    ax2.set_title('😴 Bottom 15 Processional Circuits', fontsize=15, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(bottom_15.iterrows()):
        ax2.text(row['avg_overtakes'] + 1, i,
                f"{row['avg_overtakes']:.1f}",
                va='center', fontsize=8)

    # Graf 3: Overtakes per lap (normalizováno na délku závodu)
    ax3 = fig.add_subplot(gs[2, 0])
    top_per_lap = df.nlargest(12, 'overtakes_per_lap')
    colors_pl = plt.cm.viridis(top_per_lap['overtakes_per_lap'] / top_per_lap['overtakes_per_lap'].max())

    bars = ax3.barh(range(len(top_per_lap)), top_per_lap['overtakes_per_lap'],
                    color=colors_pl, alpha=0.85)
    ax3.set_yticks(range(len(top_per_lap)))
    ax3.set_yticklabels([row['circuit_name'] for _, row in top_per_lap.iterrows()], fontsize=9)
    ax3.set_xlabel('Předjetí na kolo', fontsize=10)
    ax3.set_title('Nejvíce akce na kolo', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(top_per_lap.iterrows()):
        ax3.text(row['overtakes_per_lap'] + 0.02, i,
                f"{row['overtakes_per_lap']:.2f}",
                va='center', fontsize=8)

    # Graf 4: Konzistence (std. odchylka) - předvídatelnost závodu
    ax4 = fig.add_subplot(gs[2, 1])
    df_sorted_cons = df.nlargest(12, 'consistency')

    bars = ax4.barh(range(len(df_sorted_cons)), df_sorted_cons['consistency'],
                   color='#e74c3c', alpha=0.7)
    ax4.set_yticks(range(len(df_sorted_cons)))
    ax4.set_yticklabels([row['circuit_name'] for _, row in df_sorted_cons.iterrows()], fontsize=9)
    ax4.set_xlabel('Standardní odchylka', fontsize=10)
    ax4.set_title('Nejméně předvídatelné okruhy', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(df_sorted_cons.iterrows()):
        ax4.text(row['consistency'] + 1, i,
                f"±{row['consistency']:.1f}",
                va='center', fontsize=8)

    plt.tight_layout()
    plt.show()

    # Detailní statistiky
    print("\n" + "="*80)
    print("🏆 TOP 5 OVERTAKING PARADISES:")
    print("="*80)
    for _, row in df.head(5).iterrows():
        print(f"{row['circuit_name']:30s} | Průměr: {row['avg_overtakes']:6.1f} | "
              f"Na kolo: {row['overtakes_per_lap']:.2f} | "
              f"Max: {row['max_overtakes']:.0f} | Závodů: {row['races']}")

    print("\n" + "="*80)
    print("😴 TOP 5 PROCESSIONAL BORES:")
    print("="*80)
    for _, row in df.tail(5).iterrows():
        print(f"{row['circuit_name']:30s} | Průměr: {row['avg_overtakes']:6.1f} | "
              f"Na kolo: {row['overtakes_per_lap']:.2f} | "
              f"Min: {row['min_overtakes']:.0f} | Závodů: {row['races']}")

    print("\n" + "="*80)
    print("📊 CELKOVÁ STATISTIKA:")
    print("="*80)
    print(f"Nejatraktivnější okruh: {df.iloc[0]['circuit_name']} ({df.iloc[0]['avg_overtakes']:.1f} předjetí)")
    print(f"Nejnudnější okruh: {df.iloc[-1]['circuit_name']} ({df.iloc[-1]['avg_overtakes']:.1f} předjetí)")
    print(f"Průměr všech okruhů: {df['avg_overtakes'].mean():.1f} předjetí na závod")
    print(f"Medián: {df['avg_overtakes'].median():.1f}")

if __name__ == "__main__":
    main()