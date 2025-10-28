import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def analyze_overtaking_by_circuits():
    # Připojení k databázi
    conn = duckdb.connect('race_database.db')

    # SQL dotaz pro analýzu overtaků podle závodů
    query = """
    WITH position_changes AS (
        SELECT
            rr.race_id,
            rr.driver,
            rr.lap_number,
            rr.position,
            LAG(rr.position) OVER (PARTITION BY rr.race_id, rr.driver ORDER BY rr.lap_number) as prev_position
        FROM race_results rr
        WHERE rr.deleted = FALSE
        AND rr.position IS NOT NULL
        AND rr.lap_number IS NOT NULL
    ),
    overtakes_per_race AS (
        SELECT
            pc.race_id,
            COUNT(CASE WHEN pc.prev_position > pc.position THEN 1 END) as total_overtakes,
            COUNT(CASE WHEN pc.prev_position < pc.position THEN 1 END) as total_lost_positions,
            COUNT(DISTINCT pc.driver) as drivers_count,
            MAX(pc.lap_number) as total_laps
        FROM position_changes pc
        WHERE pc.prev_position IS NOT NULL
        GROUP BY pc.race_id
    ),
    race_overtake_stats AS (
        SELECT
            r.race_name,
            r.year,
            r.city_circuit,
            r.night_race,
            opr.total_overtakes,
            opr.total_lost_positions,
            opr.drivers_count,
            opr.total_laps,
            -- Normalizace podle počtu jezdců a kol
            ROUND(opr.total_overtakes * 1.0 / NULLIF(opr.drivers_count, 0), 2) as overtakes_per_driver,
            ROUND(opr.total_overtakes * 1.0 / NULLIF(opr.total_laps, 0), 2) as overtakes_per_lap,
            ROUND(opr.total_overtakes * 100.0 / NULLIF(opr.drivers_count * opr.total_laps, 0), 2) as overtake_percentage
        FROM races r
        JOIN overtakes_per_race opr ON r.race_id = opr.race_id
        WHERE opr.total_overtakes > 0
    ),
    circuit_averages AS (
        SELECT
            race_name,
            COUNT(*) as races_analyzed,
            AVG(total_overtakes) as avg_total_overtakes,
            AVG(overtakes_per_driver) as avg_overtakes_per_driver,
            AVG(overtakes_per_lap) as avg_overtakes_per_lap,
            AVG(overtake_percentage) as avg_overtake_percentage,
            MAX(total_overtakes) as max_overtakes,
            MIN(total_overtakes) as min_overtakes,
            STDDEV(total_overtakes) as overtake_variability,
            AVG(CASE WHEN city_circuit = TRUE THEN 1.0 ELSE 0.0 END) as is_city_circuit,
            AVG(CASE WHEN night_race = TRUE THEN 1.0 ELSE 0.0 END) as is_night_race
        FROM race_overtake_stats
        GROUP BY race_name
        HAVING COUNT(*) >= 2  -- Alespoň 2 závody pro spolehlivost
    )
    SELECT * FROM circuit_averages
    ORDER BY avg_total_overtakes DESC;
    """

    df = conn.execute(query).df()
    conn.close()

    if df.empty:
        print("Žádná data pro analýzu overtaků nebyla nalezena.")
        return

    # Vytvoření vizualizací
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Analýza předjíždění podle závodních okruhů 🏎️💨', fontsize=18, fontweight='bold')

    # Graf 1: Top 15 okruhů s nejvíce overtaky
    top_circuits = df.head(15).copy()
    colors = ['red' if x >= 0.5 else 'blue' for x in top_circuits['is_city_circuit']]

    bars = axes[0, 0].barh(range(len(top_circuits)), top_circuits['avg_total_overtakes'], color=colors, alpha=0.7)
    axes[0, 0].set_yticks(range(len(top_circuits)))
    axes[0, 0].set_yticklabels([name[:12] + '...' if len(name) > 12 else name
                                for name in top_circuits['race_name']], fontsize=10)
    axes[0, 0].set_xlabel('Průměrný počet overtaků')
    axes[0, 0].set_title('TOP 15: Okruhy s nejvíce předjížděním')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # Přidání hodnot na konec barů
    for i, (bar, value) in enumerate(zip(bars, top_circuits['avg_total_overtakes'])):
        axes[0, 0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f'{value:.1f}', ha='left', va='center', fontsize=9)

    # Graf 2: Bottom 10 okruhů s nejméně overtaky
    bottom_circuits = df.tail(10).copy()
    colors_bottom = ['red' if x >= 0.5 else 'blue' for x in bottom_circuits['is_city_circuit']]

    bars2 = axes[0, 1].barh(range(len(bottom_circuits)), bottom_circuits['avg_total_overtakes'],
                            color=colors_bottom, alpha=0.7)
    axes[0, 1].set_yticks(range(len(bottom_circuits)))
    axes[0, 1].set_yticklabels([name[:12] + '...' if len(name) > 12 else name
                                for name in bottom_circuits['race_name']], fontsize=10)
    axes[0, 1].set_xlabel('Průměrný počet overtaků')
    axes[0, 1].set_title('BOTTOM 10: Okruhy s nejméně předjížděním')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    for i, (bar, value) in enumerate(zip(bars2, bottom_circuits['avg_total_overtakes'])):
        axes[0, 1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                        f'{value:.1f}', ha='left', va='center', fontsize=9)

    # Graf 3: Městské vs standardní okruhy
    city_circuits = df[df['is_city_circuit'] >= 0.5]
    standard_circuits = df[df['is_city_circuit'] < 0.5]

    circuit_comparison = pd.DataFrame({
        'Typ okruhu': ['Městské', 'Standardní'],
        'Průměr overtaků': [city_circuits['avg_total_overtakes'].mean(),
                            standard_circuits['avg_total_overtakes'].mean()],
        'Počet okruhů': [len(city_circuits), len(standard_circuits)]
    })

    bars3 = axes[0, 2].bar(circuit_comparison['Typ okruhu'], circuit_comparison['Průměr overtaků'],
                           color=['red', 'blue'], alpha=0.7)
    axes[0, 2].set_ylabel('Průměrný počet overtaků')
    axes[0, 2].set_title('Městské vs Standardní okruhy')
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # Přidání hodnot a počtu okruhů
    for bar, value, count in zip(bars3, circuit_comparison['Průměr overtaků'], circuit_comparison['Počet okruhů']):
        axes[0, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{value:.1f}\n({count} okruhů)', ha='center', va='bottom', fontsize=10)

    # Graf 4: Scatter plot - variabilita vs průměr
    scatter_colors = ['red' if x >= 0.5 else 'blue' for x in df['is_city_circuit']]
    scatter = axes[1, 0].scatter(df['avg_total_overtakes'], df['overtake_variability'],
                                 c=scatter_colors, alpha=0.6, s=60)
    axes[1, 0].set_xlabel('Průměrný počet overtaků')
    axes[1, 0].set_ylabel('Variabilita (směrodatná odchylka)')
    axes[1, 0].set_title('Průměr vs Variabilita overtaků')
    axes[1, 0].grid(True, alpha=0.3)

    # Popisky pro extrémní hodnoty
    for i, row in df.iterrows():
        if row['avg_total_overtakes'] > df['avg_total_overtakes'].quantile(0.9) or \
                row['overtake_variability'] > df['overtake_variability'].quantile(0.9):
            axes[1, 0].annotate(row['race_name'][:8],
                                (row['avg_total_overtakes'], row['overtake_variability']),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Graf 5: Noční vs denní závody
    night_circuits = df[df['is_night_race'] >= 0.5]
    day_circuits = df[df['is_night_race'] < 0.5]

    if len(night_circuits) > 0:
        night_day_comparison = pd.DataFrame({
            'Typ závodu': ['Noční', 'Denní'],
            'Průměr overtaků': [night_circuits['avg_total_overtakes'].mean(),
                                day_circuits['avg_total_overtakes'].mean()],
            'Počet okruhů': [len(night_circuits), len(day_circuits)]
        })

        bars5 = axes[1, 1].bar(night_day_comparison['Typ závodu'], night_day_comparison['Průměr overtaků'],
                               color=['purple', 'orange'], alpha=0.7)
        axes[1, 1].set_ylabel('Průměrný počet overtaků')
        axes[1, 1].set_title('Noční vs Denní závody')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        for bar, value, count in zip(bars5, night_day_comparison['Průměr overtaků'],
                                     night_day_comparison['Počet okruhů']):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'{value:.1f}\n({count} okruhů)', ha='center', va='bottom', fontsize=10)
    else:
        axes[1, 1].text(0.5, 0.5, 'Nedostatek dat\no nočních závodech',
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Noční vs Denní závody')

    # Graf 6: Heatmapa top 15 okruhů podle různých metrik
    top_15 = df.head(15).copy()
    heatmap_data = top_15[['avg_total_overtakes', 'avg_overtakes_per_driver',
                           'avg_overtakes_per_lap', 'overtake_variability']].T

    # Normalizace pro lepší vizualizaci
    heatmap_normalized = (heatmap_data - heatmap_data.min(axis=1).values.reshape(-1, 1)) / \
                         (heatmap_data.max(axis=1) - heatmap_data.min(axis=1)).values.reshape(-1, 1)

    im = axes[1, 2].imshow(heatmap_normalized, cmap='YlOrRd', aspect='auto')
    axes[1, 2].set_xticks(range(len(top_15)))
    axes[1, 2].set_xticklabels([name[:8] for name in top_15['race_name']],
                               rotation=45, ha='right', fontsize=9)
    axes[1, 2].set_yticks(range(len(heatmap_data.index)))
    axes[1, 2].set_yticklabels(['Celkem', 'Na jezdce', 'Na kolo', 'Variabilita'], fontsize=10)
    axes[1, 2].set_title('Heatmapa metrik (TOP 15)')

    # Colorbar
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # Statistické shrnutí
    print("\n" + "=" * 70)
    print("🏎️ ANALÝZA PŘEDJÍŽDĚNÍ PODLE OKRUHŮ 🏎️")
    print("=" * 70)

    print(f"\n📊 CELKOVÉ STATISTIKY:")
    print(f"Analyzováno okruhů: {len(df)}")
    print(f"Průměrný počet overtaků na okruh: {df['avg_total_overtakes'].mean():.1f}")
    print(f"Nejvíce overtaků: {df['avg_total_overtakes'].max():.1f}")
    print(f"Nejméně overtaků: {df['avg_total_overtakes'].min():.1f}")

    print(f"\n🥇 TOP 5 OKRUHŮ PRO PŘEDJÍŽDĚNÍ:")
    for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
        circuit_type = "🏙️" if row['is_city_circuit'] >= 0.5 else "🏁"
        night_indicator = "🌙" if row['is_night_race'] >= 0.5 else "☀️"
        print(f"{i}. {row['race_name']} {circuit_type}{night_indicator}: {row['avg_total_overtakes']:.1f} overtaků "
              f"({row['races_analyzed']} závodů)")

    print(f"\n🐌 TOP 5 NEJOBTÍŽNĚJŠÍCH OKRUHŮ PRO PŘEDJÍŽDĚNÍ:")
    for i, (_, row) in enumerate(df.tail(5).iterrows(), 1):
        circuit_type = "🏙️" if row['is_city_circuit'] >= 0.5 else "🏁"
        night_indicator = "🌙" if row['is_night_race'] >= 0.5 else "☀️"
        print(f"{i}. {row['race_name']} {circuit_type}{night_indicator}: {row['avg_total_overtakes']:.1f} overtaků "
              f"({row['races_analyzed']} závodů)")

    # Analýza podle typu okruhu
    if len(city_circuits) > 0 and len(standard_circuits) > 0:
        print(f"\n🏙️ MĚSTSKÉ vs 🏁 STANDARDNÍ OKRUHY:")
        print(f"Městské okruhy průměr: {city_circuits['avg_total_overtakes'].mean():.1f} overtaků")
        print(f"Standardní okruhy průměr: {standard_circuits['avg_total_overtakes'].mean():.1f} overtaků")

        if city_circuits['avg_total_overtakes'].mean() > standard_circuits['avg_total_overtakes'].mean():
            advantage = city_circuits['avg_total_overtakes'].mean() - standard_circuits['avg_total_overtakes'].mean()
            print(f"✅ Městské okruhy mají více předjíždění o {advantage:.1f} overtaků")
        else:
            advantage = standard_circuits['avg_total_overtakes'].mean() - city_circuits['avg_total_overtakes'].mean()
            print(f"✅ Standardní okruhy mají více předjíždění o {advantage:.1f} overtaků")

    # Analýza variability
    print(f"\n📈 NEJPREDVÍDATELNĚJŠÍ OKRUHY (nízká variabilita):")
    most_consistent = df.nsmallest(3, 'overtake_variability')
    for i, (_, row) in enumerate(most_consistent.iterrows(), 1):
        print(f"{i}. {row['race_name']}: {row['avg_total_overtakes']:.1f} ± {row['overtake_variability']:.1f} overtaků")

    print(f"\n🎲 NEJMÉNĚ PŘEDVÍDATELNÉ OKRUHY (vysoká variabilita):")
    most_variable = df.nlargest(3, 'overtake_variability')
    for i, (_, row) in enumerate(most_variable.iterrows(), 1):
        print(f"{i}. {row['race_name']}: {row['avg_total_overtakes']:.1f} ± {row['overtake_variability']:.1f} overtaků")

    # Doporučení pro fanoušky
    print(f"\n🎯 DOPORUČENÍ PRO FANOUŠKY:")
    exciting_races = df[df['avg_total_overtakes'] > df['avg_total_overtakes'].quantile(0.8)]
    print(f"Pro vzrušující závody sledujte: {', '.join(exciting_races['race_name'].head(3).values)}")

    strategic_races = df[df['avg_total_overtakes'] < df['avg_total_overtakes'].quantile(0.3)]
    print(f"Pro strategické závody sledujte: {', '.join(strategic_races['race_name'].head(3).values)}")


if __name__ == "__main__":
    analyze_overtaking_by_circuits()
