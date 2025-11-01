import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def main():
    conn = duckdb.connect('race_database.db')

    query = """
    WITH tyre_performance AS (
        SELECT
            compound,
            tyre_life,
            lap_time_seconds,
            driver,
            race_id,
            team
        FROM race_results
        WHERE deleted = FALSE
            AND compound IS NOT NULL
            AND compound IN ('SOFT', 'MEDIUM', 'HARD')
            AND lap_time_seconds IS NOT NULL
            AND lap_time_seconds BETWEEN 70 AND 130
            AND tyre_life IS NOT NULL
            AND tyre_life <= 40
    ),
    compound_stats AS (
        SELECT
            compound,
            COUNT(*) as laps,
            ROUND(AVG(lap_time_seconds), 3) as avg_lap_time,
            ROUND(STDDEV(lap_time_seconds), 3) as consistency,
            ROUND(MIN(lap_time_seconds), 3) as fastest_lap,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lap_time_seconds), 3) as median_lap_time
        FROM tyre_performance
        GROUP BY compound
    ),
    degradation_profile AS (
        SELECT
            compound,
            CASE
                WHEN tyre_life <= 5 THEN '1-5'
                WHEN tyre_life <= 10 THEN '6-10'
                WHEN tyre_life <= 15 THEN '11-15'
                WHEN tyre_life <= 20 THEN '16-20'
                WHEN tyre_life <= 25 THEN '21-25'
                WHEN tyre_life <= 30 THEN '26-30'
                ELSE '31+'
            END as life_bracket,
            ROUND(AVG(lap_time_seconds), 3) as avg_time,
            COUNT(*) as samples
        FROM tyre_performance
        GROUP BY compound, life_bracket
    ),
    stint_analysis AS (
        SELECT
            compound,
            ROUND(AVG(CASE WHEN tyre_life <= 5 THEN lap_time_seconds END), 3) as first_5_laps,
            ROUND(AVG(CASE WHEN tyre_life BETWEEN 11 AND 15 THEN lap_time_seconds END), 3) as mid_stint,
            ROUND(AVG(CASE WHEN tyre_life >= 20 THEN lap_time_seconds END), 3) as end_stint
        FROM tyre_performance
        GROUP BY compound
    )
    SELECT
        cs.compound,
        cs.laps,
        cs.avg_lap_time,
        cs.median_lap_time,
        cs.consistency,
        cs.fastest_lap,
        sa.first_5_laps,
        sa.mid_stint,
        sa.end_stint,
        ROUND(sa.end_stint - sa.first_5_laps, 3) as degradation
    FROM compound_stats cs
    JOIN stint_analysis sa ON cs.compound = sa.compound
    ORDER BY cs.avg_lap_time;
    """

    df_summary = conn.execute(query).df()

    # Degradační profil pro vizualizaci
    query_degradation = """
    SELECT
        compound,
        CASE
            WHEN tyre_life <= 5 THEN 1
            WHEN tyre_life <= 10 THEN 2
            WHEN tyre_life <= 15 THEN 3
            WHEN tyre_life <= 20 THEN 4
            WHEN tyre_life <= 25 THEN 5
            WHEN tyre_life <= 30 THEN 6
            ELSE 7
        END as life_bracket,
        CASE
            WHEN tyre_life <= 5 THEN '1-5'
            WHEN tyre_life <= 10 THEN '6-10'
            WHEN tyre_life <= 15 THEN '11-15'
            WHEN tyre_life <= 20 THEN '16-20'
            WHEN tyre_life <= 25 THEN '21-25'
            WHEN tyre_life <= 30 THEN '26-30'
            ELSE '31+'
        END as life_label,
        ROUND(AVG(lap_time_seconds), 3) as avg_time,
        COUNT(*) as samples
    FROM race_results
    WHERE deleted = FALSE
        AND compound IN ('SOFT', 'MEDIUM', 'HARD')
        AND lap_time_seconds BETWEEN 70 AND 130
        AND tyre_life <= 35
    GROUP BY compound, life_bracket, life_label
    ORDER BY compound, life_bracket
    """

    df_degradation = conn.execute(query_degradation).df()
    conn.close()

    print("Compound Performance Summary:")
    print(df_summary.to_string(index=False))

    # Vizualizace
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Graf 1: Průměrné časy směsí
    ax1 = fig.add_subplot(gs[0, 0])
    compound_colors = {'SOFT': '#e74c3c', 'MEDIUM': '#f39c12', 'HARD': '#95a5a6'}
    colors = [compound_colors[c] for c in df_summary['compound']]

    bars = ax1.bar(df_summary['compound'], df_summary['avg_lap_time'], color=colors, alpha=0.8)
    ax1.set_ylabel('Průměrný čas kola (s)', fontsize=10)
    ax1.set_title('Rychlost směsí', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for i, (idx, row) in enumerate(df_summary.iterrows()):
        ax1.text(i, row['avg_lap_time'] + 0.1,
                 f"{row['avg_lap_time']:.2f}s",
                 ha='center', fontsize=9, fontweight='bold')

    # Graf 2: Degradace (rozdíl začátek vs. konec stintu)
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(df_summary['compound'], df_summary['degradation'], color=colors, alpha=0.8)
    ax2.set_ylabel('Zpomalení (s)', fontsize=10)
    ax2.set_title('Celková degradace', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for i, (idx, row) in enumerate(df_summary.iterrows()):
        ax2.text(i, row['degradation'] + 0.05,
                 f"+{row['degradation']:.2f}s",
                 ha='center', fontsize=9, fontweight='bold')

    # Graf 3: Konzistence (std. odchylka)
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(df_summary['compound'], df_summary['consistency'], color=colors, alpha=0.8)
    ax3.set_ylabel('Std. odchylka (s)', fontsize=10)
    ax3.set_title('Konzistence', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    for i, (idx, row) in enumerate(df_summary.iterrows()):
        ax3.text(i, row['consistency'] + 0.03,
                 f"{row['consistency']:.2f}s",
                 ha='center', fontsize=9)

    # Graf 4: Degradační křivky (hlavní graf!)
    ax4 = fig.add_subplot(gs[1, :])

    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        data = df_degradation[df_degradation['compound'] == compound].sort_values('life_bracket')

        ax4.plot(data['life_label'], data['avg_time'],
                 marker='o', linewidth=2.5, markersize=8,
                 label=compound, color=compound_colors[compound], alpha=0.85)

        # Trend line
        x_numeric = data['life_bracket'].values
        y = data['avg_time'].values
        z = np.polyfit(x_numeric, y, 2)
        p = np.poly1d(z)
        ax4.plot(data['life_label'], p(x_numeric),
                 linestyle='--', linewidth=1.5, alpha=0.4, color=compound_colors[compound])

    ax4.set_xlabel('Stáří pneumatik (kola)', fontsize=11)
    ax4.set_ylabel('Průměrný čas kola (s)', fontsize=11)
    ax4.set_title('📈 Degradační profil směsí pneumatik', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, loc='upper left')
    ax4.grid(True, alpha=0.3)

    # Graf 5: Stint breakdown (začátek/střed/konec)
    ax5 = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(len(df_summary))
    width = 0.25

    bars1 = ax5.bar(x_pos - width, df_summary['first_5_laps'], width,
                    label='První 5 kol', color='#2ecc71', alpha=0.8)
    bars2 = ax5.bar(x_pos, df_summary['mid_stint'], width,
                    label='Střed stintu', color='#f39c12', alpha=0.8)
    bars3 = ax5.bar(x_pos + width, df_summary['end_stint'], width,
                    label='Konec stintu', color='#e74c3c', alpha=0.8)

    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(df_summary['compound'])
    ax5.set_ylabel('Čas kola (s)', fontsize=10)
    ax5.set_title('Fáze stintu', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(axis='y', alpha=0.3)

    # Graf 6: Relativní výkonnost (% oproti SOFT baseline)
    ax6 = fig.add_subplot(gs[2, 1])
    soft_baseline = df_summary[df_summary['compound'] == 'SOFT']['avg_lap_time'].values[0]
    df_summary['relative_perf'] = ((df_summary['avg_lap_time'] - soft_baseline) / soft_baseline) * 100

    bars = ax6.barh(df_summary['compound'], df_summary['relative_perf'], color=colors, alpha=0.8)
    ax6.set_xlabel('% pomalejší než SOFT', fontsize=10)
    ax6.set_title('Relativní výkon', fontsize=12, fontweight='bold')
    ax6.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax6.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(df_summary.iterrows()):
        ax6.text(row['relative_perf'] + 0.02, i,
                 f"{row['relative_perf']:+.2f}%",
                 va='center', fontsize=9)

    # Graf 7: Sample size (validita dat)
    ax7 = fig.add_subplot(gs[2, 2])
    bars = ax7.bar(df_summary['compound'], df_summary['laps'], color=colors, alpha=0.8)
    ax7.set_ylabel('Počet kol', fontsize=10)
    ax7.set_title('Velikost vzorku', fontsize=12, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)

    for i, (idx, row) in enumerate(df_summary.iterrows()):
        ax7.text(i, row['laps'] + 50,
                 f"{row['laps']:,.0f}",
                 ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Detailní statistiky
    print("\n" + "=" * 90)
    print("🏁 COMPOUND PERFORMANCE BREAKDOWN:")
    print("=" * 90)
    for _, row in df_summary.iterrows():
        print(f"\n{row['compound']}:")
        print(f"  ├─ Průměrný čas: {row['avg_lap_time']:.3f}s (medián: {row['median_lap_time']:.3f}s)")
        print(f"  ├─ Nejrychlejší kolo: {row['fastest_lap']:.3f}s")
        print(f"  ├─ První 5 kol: {row['first_5_laps']:.3f}s → Konec: {row['end_stint']:.3f}s")
        print(f"  ├─ Degradace: +{row['degradation']:.3f}s ({row['degradation'] / row['first_5_laps'] * 100:.2f}%)")
        print(f"  ├─ Konzistence: ±{row['consistency']:.3f}s")
        print(f"  └─ Celkem kol: {row['laps']:,}")

    print("\n" + "=" * 90)
    print("💡 ZÁVĚR:")
    print("=" * 90)
    best_compound = df_summary.iloc[0]['compound']
    most_degrading = df_summary.loc[df_summary['degradation'].idxmax(), 'compound']
    most_consistent = df_summary.loc[df_summary['consistency'].idxmin(), 'compound']

    print(f"✅ Nejrychlejší směs: {best_compound}")
    print(f"⚠️ Nejvíc degradující: {most_degrading} (+{df_summary['degradation'].max():.3f}s)")
    print(f"🎯 Nejkonzistentnější: {most_consistent} (±{df_summary['consistency'].min():.3f}s)")

if __name__ == "__main__":
    main()