import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_city_circuit_performance():
    # Připojení k databázi
    conn = duckdb.connect('race_database.db')

    # Opravený SQL dotaz
    query = """
    WITH final_positions AS (
        SELECT
            race_id,
            driver,
            position as final_position,
            ROW_NUMBER() OVER (PARTITION BY race_id, driver ORDER BY time DESC) as rn
        FROM race_results
        WHERE deleted = FALSE
        AND position IS NOT NULL
    ),
    grid_positions AS (
        SELECT
            race_id,
            driver,
            position as grid_position,
            ROW_NUMBER() OVER (PARTITION BY race_id, driver ORDER BY time ASC) as rn
        FROM race_results
        WHERE deleted = FALSE
        AND position IS NOT NULL
    ),
    position_changes AS (
        SELECT
            rr.race_id,
            rr.driver,
            rr.team,
            rr.position,
            rr.lap_time_seconds,
            LAG(rr.position) OVER (PARTITION BY rr.race_id, rr.driver ORDER BY rr.time) as prev_position,
            CASE 
                WHEN LAG(rr.position) OVER (PARTITION BY rr.race_id, rr.driver ORDER BY rr.time) > rr.position 
                THEN 1 ELSE 0 
            END as position_gained,
            CASE 
                WHEN LAG(rr.position) OVER (PARTITION BY rr.race_id, rr.driver ORDER BY rr.time) < rr.position 
                THEN 1 ELSE 0 
            END as position_lost
        FROM race_results rr
        WHERE rr.deleted = FALSE
        AND rr.position IS NOT NULL
        AND rr.lap_time_seconds IS NOT NULL
        AND rr.lap_time_seconds > 60
        AND rr.lap_time_seconds < 200
    ),
    driver_race_stats AS (
        SELECT
            pc.race_id,
            pc.driver,
            pc.team,
            AVG(pc.lap_time_seconds) as avg_lap_time,
            MIN(pc.lap_time_seconds) as fastest_lap,
            STDDEV(pc.lap_time_seconds) as lap_consistency,
            COUNT(*) as total_laps,
            SUM(CASE WHEN pc.position <= 5 THEN 1 ELSE 0 END) as laps_in_top5,
            SUM(CASE WHEN pc.position <= 10 THEN 1 ELSE 0 END) as laps_in_top10,
            AVG(pc.position) as avg_position_during_race,
            SUM(pc.position_gained) as overtakes_made,
            SUM(pc.position_lost) as positions_lost
        FROM position_changes pc
        GROUP BY pc.race_id, pc.driver, pc.team
    )
    SELECT
        r.race_name,
        r.year,
        r.city_circuit,
        r.night_race,
        drs.driver,
        drs.team,
        fp.final_position,
        gp.grid_position,
        (gp.grid_position - fp.final_position) as positions_gained,
        drs.avg_lap_time,
        drs.fastest_lap,
        drs.lap_consistency,
        drs.total_laps,
        drs.laps_in_top5,
        drs.laps_in_top10,
        (drs.laps_in_top5 * 100.0 / drs.total_laps) as top5_percentage,
        (drs.laps_in_top10 * 100.0 / drs.total_laps) as top10_percentage,
        drs.avg_position_during_race,
        drs.overtakes_made,
        drs.positions_lost,
        (drs.overtakes_made - drs.positions_lost) as net_overtakes
    FROM driver_race_stats drs
    JOIN races r ON drs.race_id = r.race_id
    LEFT JOIN final_positions fp ON drs.race_id = fp.race_id AND drs.driver = fp.driver AND fp.rn = 1
    LEFT JOIN grid_positions gp ON drs.race_id = gp.race_id AND drs.driver = gp.driver AND gp.rn = 1
    WHERE fp.final_position IS NOT NULL
    AND gp.grid_position IS NOT NULL
    ORDER BY r.race_name, fp.final_position;
    """

    df = conn.execute(query).df()
    conn.close()

    if df.empty:
        print("Žádná data nebyla nalezena.")
        return

    # Vytvoření vizualizací
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Analýza specializace na městské okruhy vs. standardní tratě', fontsize=16, fontweight='bold')

    # Graf 1: Porovnání průměrných pozic na městských vs. standardních tratích
    circuit_comparison = df.groupby(['driver', 'city_circuit'])['final_position'].mean().reset_index()
    circuit_pivot = circuit_comparison.pivot(index='driver', columns='city_circuit', values='final_position')
    circuit_pivot = circuit_pivot.dropna()

    if len(circuit_pivot) > 0:
        circuit_pivot['city_advantage'] = circuit_pivot[False] - circuit_pivot[True]
        city_specialists = circuit_pivot.nlargest(10, 'city_advantage')

        x_pos = np.arange(len(city_specialists))
        colors = ['darkgreen' if x > 2 else 'green' if x > 0 else 'red' for x in city_specialists['city_advantage']]

        axes[0, 0].barh(x_pos, city_specialists['city_advantage'], color=colors, alpha=0.7)
        axes[0, 0].set_yticks(x_pos)
        axes[0, 0].set_yticklabels(city_specialists.index, fontsize=8)
        axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Výhoda na městských okruzích (pozice)')
        axes[0, 0].set_title('Top 10: Specialisti na městské okruhy')
        axes[0, 0].grid(True, alpha=0.3)

    # Graf 2: Scatter plot - městské vs. standardní tratě
    if len(circuit_pivot) > 5:
        axes[0, 1].scatter(circuit_pivot[False], circuit_pivot[True],
                          s=100, alpha=0.7, c='orange')

        min_pos = min(circuit_pivot[False].min(), circuit_pivot[True].min())
        max_pos = max(circuit_pivot[False].max(), circuit_pivot[True].max())
        axes[0, 1].plot([min_pos, max_pos], [min_pos, max_pos], 'r--', alpha=0.5)

        axes[0, 1].set_xlabel('Průměrná pozice (standardní tratě)')
        axes[0, 1].set_ylabel('Průměrná pozice (městské okruhy)')
        axes[0, 1].set_title('Standardní vs. Městské tratě')
        axes[0, 1].grid(True, alpha=0.3)

        for driver in city_specialists.index[:3]:
            if driver in circuit_pivot.index:
                axes[0, 1].annotate(driver,
                                  (circuit_pivot.loc[driver, False],
                                   circuit_pivot.loc[driver, True]),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Graf 3: Overtaking analysis
    overtake_comparison = df.groupby(['driver', 'city_circuit'])['net_overtakes'].mean().reset_index()
    overtake_pivot = overtake_comparison.pivot(index='driver', columns='city_circuit', values='net_overtakes')
    overtake_pivot = overtake_pivot.dropna()

    if len(overtake_pivot) > 0 and len(overtake_pivot.columns) > 1:
        overtake_pivot['overtake_diff'] = overtake_pivot[True] - overtake_pivot[False]
        top_overtakers = overtake_pivot.nlargest(8, 'overtake_diff')

        x_pos = np.arange(len(top_overtakers))
        axes[0, 2].bar(x_pos, top_overtakers['overtake_diff'],
                      color=['green' if x > 0 else 'red' for x in top_overtakers['overtake_diff']], alpha=0.7)
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(top_overtakers.index, rotation=45, ha='right')
        axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 2].set_ylabel('Rozdíl v net overtacích')
        axes[0, 2].set_title('Overtaking: Městské vs. Standardní')
        axes[0, 2].grid(True, alpha=0.3)

    # Graf 4: Týmová analýza
    team_performance = df.groupby(['team', 'city_circuit'])['top10_percentage'].mean().reset_index()
    team_pivot = team_performance.pivot(index='team', columns='city_circuit', values='top10_percentage')
    team_pivot = team_pivot.dropna()

    if len(team_pivot) > 0 and len(team_pivot.columns) > 1:
        top_teams = team_pivot.head(8)
        x_pos = np.arange(len(top_teams))
        width = 0.35

        axes[1, 0].bar(x_pos - width/2, top_teams[False], width, label='Standardní', alpha=0.8)
        axes[1, 0].bar(x_pos + width/2, top_teams[True], width, label='Městské', alpha=0.8)
        axes[1, 0].set_xlabel('Tým')
        axes[1, 0].set_ylabel('% kol v top 10')
        axes[1, 0].set_title('Týmový výkon podle typu tratě')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(top_teams.index, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Graf 5: Konzistence
    consistency_comparison = df.groupby(['driver', 'city_circuit'])['lap_consistency'].mean().reset_index()
    consistency_pivot = consistency_comparison.pivot(index='driver', columns='city_circuit', values='lap_consistency')
    consistency_pivot = consistency_pivot.dropna()

    if len(consistency_pivot) > 5 and len(consistency_pivot.columns) > 1:
        consistency_pivot = consistency_pivot[(consistency_pivot[False] < 10) & (consistency_pivot[True] < 10)]

        if len(consistency_pivot) > 0:
            axes[1, 1].scatter(consistency_pivot[False], consistency_pivot[True],
                              s=100, alpha=0.7, c='blue')

            min_cons = min(consistency_pivot[False].min(), consistency_pivot[True].min())
            max_cons = max(consistency_pivot[False].max(), consistency_pivot[True].max())
            axes[1, 1].plot([min_cons, max_cons], [min_cons, max_cons], 'r--', alpha=0.5)

            axes[1, 1].set_xlabel('Konzistence (standardní tratě)')
            axes[1, 1].set_ylabel('Konzistence (městské okruhy)')
            axes[1, 1].set_title('Konzistence podle typu tratě')
            axes[1, 1].grid(True, alpha=0.3)

    # Graf 6: Heatmapa startovních pozic na městských okruzích
    city_data = df[df['city_circuit'] == True]
    if len(city_data) > 0:
        try:
            grid_bins = pd.cut(city_data['grid_position'], bins=range(1, 22, 3))
            final_bins = pd.cut(city_data['final_position'], bins=range(1, 22, 3))
            heatmap_data = pd.crosstab(final_bins, grid_bins, values=city_data['driver'], aggfunc='count')

            if not heatmap_data.empty:
                sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 2])
                axes[1, 2].set_xlabel('Startovní pozice (grid)')
                axes[1, 2].set_ylabel('Finální pozice')
                axes[1, 2].set_title('Městské okruhy: Grid → Finální pozice')
            else:
                axes[1, 2].text(0.5, 0.5, 'Nedostatek dat pro heatmapu',
                               ha='center', va='center', transform=axes[1, 2].transAxes)
        except:
            axes[1, 2].text(0.5, 0.5, 'Nedostatek dat pro heatmapu',
                           ha='center', va='center', transform=axes[1, 2].transAxes)

    plt.tight_layout()
    plt.show()

    # Statistické shrnutí
    print("\n=== ANALÝZA MĚSTSKÝCH OKRUHŮ ===")
    print(f"Celkem analyzovaných výsledků: {len(df)}")
    print(f"Výsledky na městských okruzích: {len(df[df['city_circuit'] == True])}")
    print(f"Výsledky na standardních tratích: {len(df[df['city_circuit'] == False])}")

    if len(circuit_pivot) > 0:
        print(f"\n=== TOP 5 SPECIALISTŮ NA MĚSTSKÉ OKRUHY ===")
        top_city_drivers = circuit_pivot.nlargest(5, 'city_advantage')
        for driver, row in top_city_drivers.iterrows():
            advantage = row['city_advantage']
            std_avg = row[False]
            city_avg = row[True]
            print(f"{driver}: {advantage:.2f} pozic lepší (Standardní: {std_avg:.1f}, Městské: {city_avg:.1f})")

    if len(overtake_pivot) > 0 and 'overtake_diff' in overtake_pivot.columns:
        print(f"\n=== TOP 3 OVERTAKEŘI NA MĚSTSKÝCH OKRUZÍCH ===")
        top_overtakers_city = overtake_pivot.nlargest(3, 'overtake_diff')
        for driver, row in top_overtakers_city.iterrows():
            if True in row.index and False in row.index:
                diff = row['overtake_diff']
                city_overtakes = row[True]
                std_overtakes = row[False]
                print(f"{driver}: +{diff:.1f} více net overtaků na městských (Městské: {city_overtakes:.1f}, Standardní: {std_overtakes:.1f})")

if __name__ == "__main__":
    analyze_city_circuit_performance()
