import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    conn = duckdb.connect('race_database.db')

    query = """
    WITH qualifying_positions AS (
        SELECT
            q.race_id,
            q.driver,
            q.position as quali_position
        FROM qualifying q
        WHERE q.position IS NOT NULL
    ),
    first_lap_positions AS (
        SELECT
            rr.race_id,
            rr.driver,
            rr.position as lap1_position
        FROM race_results rr
        WHERE rr.deleted = FALSE
            AND rr.lap_number = 1
            AND rr.position IS NOT NULL
    ),
    position_changes AS (
        SELECT
            qp.driver,
            qp.race_id,
            qp.quali_position,
            flp.lap1_position,
            qp.quali_position - flp.lap1_position as positions_gained
        FROM qualifying_positions qp
        JOIN first_lap_positions flp 
            ON qp.race_id = flp.race_id 
            AND qp.driver = flp.driver
    )
    SELECT
        driver,
        COUNT(*) as races,
        ROUND(AVG(positions_gained), 2) as avg_positions_gained,
        ROUND(STDDEV(positions_gained), 2) as consistency,
        SUM(CASE WHEN positions_gained > 0 THEN 1 ELSE 0 END) as races_gained,
        SUM(CASE WHEN positions_gained < 0 THEN 1 ELSE 0 END) as races_lost,
        SUM(CASE WHEN positions_gained = 0 THEN 1 ELSE 0 END) as races_same,
        MAX(positions_gained) as best_start,
        MIN(positions_gained) as worst_start
    FROM position_changes
    GROUP BY driver
    HAVING COUNT(*) >= 5
    ORDER BY avg_positions_gained DESC;
    """

    df = conn.execute(query).df()
    conn.close()

    print("First Lap Performance Analysis:")
    print(df.head(20).to_string(index=False))

    # Vizualizace
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Graf 1: Top gainers/losers
    ax1 = fig.add_subplot(gs[0, :])
    top_10 = pd.concat([df.head(10), df.tail(10)])
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in top_10['avg_positions_gained']]

    bars = ax1.barh(range(len(top_10)), top_10['avg_positions_gained'], color=colors, alpha=0.8)
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10['driver'], fontsize=9)
    ax1.set_xlabel('Průměrný zisk/ztráta pozic v 1. kole', fontsize=11)
    ax1.set_title('First Lap Heroes & Villains', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(top_10.iterrows()):
        value = row['avg_positions_gained']
        label = f"{value:+.2f} ({row['races']} závodů)"
        if value > 0:
            ax1.text(value + 0.05, i, label, va='center', fontsize=8, color='#27ae60')
        else:
            ax1.text(value - 0.05, i, label, va='center', ha='right', fontsize=8, color='#e74c3c')

    # Graf 2: Konzistence vs. průměr
    ax2 = fig.add_subplot(gs[1, 0])
    scatter_colors = ['#27ae60' if x > 0 else '#e74c3c' for x in df['avg_positions_gained']]
    ax2.scatter(df['avg_positions_gained'], df['consistency'],
               c=scatter_colors, s=df['races']*3, alpha=0.6)

    ax2.set_xlabel('Průměrný zisk pozic', fontsize=10)
    ax2.set_ylabel('Konzistence (std. odchylka)', fontsize=10)
    ax2.set_title('Konzistence vs. Agresivita', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # Anotace pro extrémní body
    for _, row in df.nlargest(3, 'avg_positions_gained').iterrows():
        ax2.annotate(row['driver'],
                    (row['avg_positions_gained'], row['consistency']),
                    fontsize=7, ha='left')

    # Graf 3: Success rate (% závodů se ziskem)
    ax3 = fig.add_subplot(gs[1, 1])
    df['gain_rate'] = (df['races_gained'] / df['races'] * 100)
    top_gainers = df.nlargest(15, 'gain_rate')

    bars = ax3.barh(range(len(top_gainers)), top_gainers['gain_rate'],
                   color='#3498db', alpha=0.8)
    ax3.set_yticks(range(len(top_gainers)))
    ax3.set_yticklabels(top_gainers['driver'], fontsize=9)
    ax3.set_xlabel('% závodů se ziskem pozic', fontsize=10)
    ax3.set_title('Most Consistent Starters', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(top_gainers.iterrows()):
        ax3.text(row['gain_rate'] + 1, i, f"{row['gain_rate']:.1f}%",
                va='center', fontsize=8)

    plt.tight_layout()
    plt.show()

    # Detailní statistiky
    print("\n" + "="*70)
    print("TOP 5 FIRST LAP GAINERS:")
    print("="*70)
    for _, row in df.head(5).iterrows():
        print(f"{row['driver']:20s} | Průměr: {row['avg_positions_gained']:+.2f} | "
              f"Nejlepší start: {row['best_start']:+.0f} | "
              f"% se ziskem: {row['races_gained']/row['races']*100:.1f}%")

    print("\n" + "="*70)
    print("TOP 5 FIRST LAP LOSERS:")
    print("="*70)
    for _, row in df.tail(5).iterrows():
        print(f"{row['driver']:20s} | Průměr: {row['avg_positions_gained']:+.2f} | "
              f"Nejhorší start: {row['worst_start']:+.0f} | "
              f"% se ztrátou: {row['races_lost']/row['races']*100:.1f}%")
if __name__ == "__main__":
    main()