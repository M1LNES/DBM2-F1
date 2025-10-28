import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


def analyze_ver_ham_2021_battle():
    # Připojení k databázi
    conn = duckdb.connect('race_database.db')

    # Opravený SQL dotaz s explicitním řazením podle data závodu
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
    driver_race_performance AS (
        SELECT
            rr.race_id,
            rr.driver,
            rr.team,
            AVG(rr.lap_time_seconds) as avg_lap_time,
            MIN(rr.lap_time_seconds) as fastest_lap,
            STDDEV(rr.lap_time_seconds) as lap_consistency,
            COUNT(*) as total_laps,
            SUM(CASE WHEN rr.position <= 3 THEN 1 ELSE 0 END) as laps_in_podium,
            SUM(CASE WHEN rr.position <= 5 THEN 1 ELSE 0 END) as laps_in_top5,
            AVG(rr.position) as avg_position_during_race,
            MAX(rr.position) as worst_position,
            MIN(rr.position) as best_position
        FROM race_results rr
        WHERE rr.deleted = FALSE
        AND rr.position IS NOT NULL
        AND rr.lap_time_seconds IS NOT NULL
        AND rr.lap_time_seconds > 60
        AND rr.lap_time_seconds < 200
        GROUP BY rr.race_id, rr.driver, rr.team
    ),
    both_drivers_races AS (
        -- Pouze závody, kde startovali oba jezdci
        SELECT race_id
        FROM driver_race_performance 
        WHERE driver IN ('VER', 'HAM')
        GROUP BY race_id
        HAVING COUNT(DISTINCT driver) = 2
    )
    SELECT
        r.race_name,
        r.year,
        r.race_date,
        r.city_circuit,
        r.night_race,
        drp.driver,
        drp.team,
        fp.final_position,
        gp.grid_position,
        (gp.grid_position - fp.final_position) as positions_gained,
        drp.avg_lap_time,
        drp.fastest_lap,
        drp.lap_consistency,
        drp.total_laps,
        drp.laps_in_podium,
        drp.laps_in_top5,
        (drp.laps_in_podium * 100.0 / drp.total_laps) as podium_percentage,
        (drp.laps_in_top5 * 100.0 / drp.total_laps) as top5_percentage,
        drp.avg_position_during_race,
        drp.worst_position,
        drp.best_position,
        -- Bodování F1 2021
        CASE
            WHEN fp.final_position = 1 THEN 25
            WHEN fp.final_position = 2 THEN 18
            WHEN fp.final_position = 3 THEN 15
            WHEN fp.final_position = 4 THEN 12
            WHEN fp.final_position = 5 THEN 10
            WHEN fp.final_position = 6 THEN 8
            WHEN fp.final_position = 7 THEN 6
            WHEN fp.final_position = 8 THEN 4
            WHEN fp.final_position = 9 THEN 2
            WHEN fp.final_position = 10 THEN 1
            ELSE 0
        END as points_scored
    FROM driver_race_performance drp
    JOIN races r ON drp.race_id = r.race_id
    JOIN both_drivers_races bdr ON drp.race_id = bdr.race_id
    LEFT JOIN final_positions fp ON drp.race_id = fp.race_id AND drp.driver = fp.driver AND fp.rn = 1
    LEFT JOIN grid_positions gp ON drp.race_id = gp.race_id AND drp.driver = gp.driver AND gp.rn = 1
    WHERE r.year = 2021
    AND drp.driver IN ('VER', 'HAM')
    AND fp.final_position IS NOT NULL
    AND gp.grid_position IS NOT NULL
    ORDER BY r.race_date, drp.driver;
    """

    df = conn.execute(query).df()
    conn.close()

    if df.empty:
        print("Žádná data pro VER vs HAM 2021 nebyla nalezena.")
        return

    # Řazení podle data a rozdělení dat
    df = df.sort_values(['race_date', 'driver'])
    ver_data = df[df['driver'] == 'VER'].copy().reset_index(drop=True)
    ham_data = df[df['driver'] == 'HAM'].copy().reset_index(drop=True)

    # Kontrola, že máme stejný počet závodů
    if len(ver_data) != len(ham_data):
        print(f"Varování: Různý počet závodů - VER: {len(ver_data)}, HAM: {len(ham_data)}")
        min_races = min(len(ver_data), len(ham_data))
        ver_data = ver_data.head(min_races)
        ham_data = ham_data.head(min_races)

    print(f"Analyzujeme {len(ver_data)} závodů pro každého jezdce")

    # Vytvoření vizualizací
    plt.style.use('default')
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('VER vs HAM 2021: Epický souboj o titul 🏆', fontsize=18, fontweight='bold')

    # Výpočet kumulativních bodů
    ver_data['cumulative_points'] = ver_data['points_scored'].cumsum()
    ham_data['cumulative_points'] = ham_data['points_scored'].cumsum()

    race_numbers = range(1, len(ver_data) + 1)

    # Graf 1: Vývoj bodů během sezóny
    axes[0, 0].plot(race_numbers, ver_data['cumulative_points'], 'o-', color='blue',
                    linewidth=3, markersize=8, label='VER')
    axes[0, 0].plot(race_numbers, ham_data['cumulative_points'], 'o-', color='red',
                    linewidth=3, markersize=8, label='HAM')
    axes[0, 0].set_xlabel('Závod č.')
    axes[0, 0].set_ylabel('Kumulativní body')
    axes[0, 0].set_title('Vývoj mistrovského boje 2021')
    axes[0, 0].legend(fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)

    # Graf 2: Finální pozice závod po závodu
    axes[0, 1].bar([x - 0.2 for x in race_numbers], ver_data['final_position'],
                   width=0.4, color='blue', alpha=0.7, label='VER')
    axes[0, 1].bar([x + 0.2 for x in race_numbers], ham_data['final_position'],
                   width=0.4, color='red', alpha=0.7, label='HAM')
    axes[0, 1].set_xlabel('Závod č.')
    axes[0, 1].set_ylabel('Finální pozice')
    axes[0, 1].set_title('Finální pozice v jednotlivých závodech')
    axes[0, 1].invert_yaxis()
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Graf 3: Porovnání průměrných časů kol
    avg_times_comparison = []
    race_names_short = []
    for i in range(len(ver_data)):
        ver_time = ver_data.iloc[i]['avg_lap_time']
        ham_time = ham_data.iloc[i]['avg_lap_time']
        if pd.notna(ver_time) and pd.notna(ham_time):
            avg_times_comparison.append(ver_time - ham_time)
            race_names_short.append(ver_data.iloc[i]['race_name'][:6])

    if avg_times_comparison:
        colors = ['red' if x > 0 else 'blue' for x in avg_times_comparison]
        axes[0, 2].bar(range(len(avg_times_comparison)), avg_times_comparison, color=colors, alpha=0.7)
        axes[0, 2].set_xlabel('Závod')
        axes[0, 2].set_ylabel('Rozdíl avg. času kola (s)')
        axes[0, 2].set_title('VER vs HAM: Rychlost (červená = HAM rychlejší)')
        axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 2].grid(True, alpha=0.3)

    # Graf 4: Nejrychlejší kola
    fastest_laps_comparison = []
    for i in range(len(ver_data)):
        ver_fastest = ver_data.iloc[i]['fastest_lap']
        ham_fastest = ham_data.iloc[i]['fastest_lap']
        if pd.notna(ver_fastest) and pd.notna(ham_fastest):
            fastest_laps_comparison.append(ver_fastest - ham_fastest)

    if fastest_laps_comparison:
        colors = ['red' if x > 0 else 'blue' for x in fastest_laps_comparison]
        axes[1, 0].bar(range(len(fastest_laps_comparison)), fastest_laps_comparison, color=colors, alpha=0.7)
        axes[1, 0].set_xlabel('Závod')
        axes[1, 0].set_ylabel('Rozdíl nejrychlejšího kola (s)')
        axes[1, 0].set_title('Nejrychlejší kola: VER vs HAM')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)

    # Graf 5: Konzistence
    ver_avg_consistency = ver_data['lap_consistency'].mean()
    ham_avg_consistency = ham_data['lap_consistency'].mean()

    axes[1, 1].bar(['VER', 'HAM'], [ver_avg_consistency, ham_avg_consistency],
                   color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel('Průměrná variabilita (s)')
    axes[1, 1].set_title('Konzistence během sezóny')
    axes[1, 1].grid(True, alpha=0.3)

    # Graf 6: Pozice během závodu
    categories = ['Průměrná pozice', 'Nejhorší pozice']
    ver_positions = [ver_data['avg_position_during_race'].mean(), ver_data['worst_position'].mean()]
    ham_positions = [ham_data['avg_position_during_race'].mean(), ham_data['worst_position'].mean()]

    x_pos = np.arange(len(categories))
    width = 0.35
    axes[1, 2].bar(x_pos - width / 2, ver_positions, width, label='VER', color='blue', alpha=0.7)
    axes[1, 2].bar(x_pos + width / 2, ham_positions, width, label='HAM', color='red', alpha=0.7)
    axes[1, 2].set_ylabel('Pozice')
    axes[1, 2].set_title('Pozice během závodů')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(categories)
    axes[1, 2].invert_yaxis()
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Graf 7: Kvalifikace vs závod
    ver_grid_vs_final = ver_data['positions_gained'].mean()
    ham_grid_vs_final = ham_data['positions_gained'].mean()

    axes[2, 0].bar(['VER', 'HAM'], [ver_grid_vs_final, ham_grid_vs_final],
                   color=['blue', 'red'], alpha=0.7)
    axes[2, 0].set_ylabel('Průměrný zisk pozic')
    axes[2, 0].set_title('Grid → Finální pozice')
    axes[2, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2, 0].grid(True, alpha=0.3)

    # Graf 8: Celkové statistiky
    ver_stats = {
        'Vítězství': len(ver_data[ver_data['final_position'] == 1]),
        'Pódia': len(ver_data[ver_data['final_position'] <= 3]),
        'Body': ver_data['points_scored'].sum()
    }
    ham_stats = {
        'Vítězství': len(ham_data[ham_data['final_position'] == 1]),
        'Pódia': len(ham_data[ham_data['final_position'] <= 3]),
        'Body': ham_data['points_scored'].sum()
    }

    stats_categories = ['Vítězství', 'Pódia']  # Body vynecháváme kvůli škálování
    ver_values = [ver_stats['Vítězství'], ver_stats['Pódia']]
    ham_values = [ham_stats['Vítězství'], ham_stats['Pódia']]

    x_pos = np.arange(len(stats_categories))
    width = 0.35
    axes[2, 1].bar(x_pos - width / 2, ver_values, width, label='VER', color='blue', alpha=0.7)
    axes[2, 1].bar(x_pos + width / 2, ham_values, width, label='HAM', color='red', alpha=0.7)
    axes[2, 1].set_ylabel('Počet')
    axes[2, 1].set_title('Vítězství a pódia 2021')
    axes[2, 1].set_xticks(x_pos)
    axes[2, 1].set_xticklabels(stats_categories)
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    # Graf 9: Rozdíl bodů během sezóny
    points_difference = ver_data['cumulative_points'] - ham_data['cumulative_points']
    colors = ['blue' if x > 0 else 'red' for x in points_difference]

    axes[2, 2].bar(race_numbers, points_difference, color=colors, alpha=0.7)
    axes[2, 2].set_xlabel('Závod č.')
    axes[2, 2].set_ylabel('Rozdíl bodů (VER - HAM)')
    axes[2, 2].set_title('Vývoj rozdílu bodů')
    axes[2, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistické shrnutí
    print("\n" + "=" * 60)
    print("🏁 VER vs HAM 2021: ŠAMPIONSKÝ SOUBOJ 🏁")
    print("=" * 60)

    print(f"\n📊 CELKOVÉ STATISTIKY:")
    print(f"VER: {ver_stats['Body']} bodů | {ver_stats['Vítězství']} vítězství | {ver_stats['Pódia']} pódií")
    print(f"HAM: {ham_stats['Body']} bodů | {ham_stats['Vítězství']} vítězství | {ham_stats['Pódia']} pódií")

    winner = "VER" if ver_stats['Body'] > ham_stats['Body'] else "HAM"
    margin = abs(ver_stats['Body'] - ham_stats['Body'])
    print(f"\n🏆 MISTR SVĚTA 2021: {winner} (rozdíl {margin} bodů)")

    # Rychlostní analýza
    ver_avg_time = ver_data['avg_lap_time'].mean()
    ham_avg_time = ham_data['avg_lap_time'].mean()
    if pd.notna(ver_avg_time) and pd.notna(ham_avg_time):
        faster = "VER" if ver_avg_time < ham_avg_time else "HAM"
        time_diff = abs(ver_avg_time - ham_avg_time)
        print(f"\n⚡ RYCHLOST:")
        print(f"Průměrný čas kola: VER {ver_avg_time:.3f}s | HAM {ham_avg_time:.3f}s")
        print(f"Rychlejší: {faster} o {time_diff:.3f}s")

    # Nejdramatičtější závody
    print(f"\n🔥 NEJDRAMATIČTĚJŠÍ SOUBOJE:")
    position_diffs = []
    for i in range(len(ver_data)):
        ver_pos = ver_data.iloc[i]['final_position']
        ham_pos = ham_data.iloc[i]['final_position']
        race_name = ver_data.iloc[i]['race_name']
        diff = abs(ver_pos - ham_pos)
        position_diffs.append((race_name, diff, ver_pos, ham_pos))

    position_diffs.sort(key=lambda x: x[1])
    for i, (race, diff, ver_pos, ham_pos) in enumerate(position_diffs[:3]):
        print(f"{i + 1}. {race}: VER P{int(ver_pos)} vs HAM P{int(ham_pos)} (rozdíl {int(diff)} pozic)")


if __name__ == "__main__":
    analyze_ver_ham_2021_battle()
