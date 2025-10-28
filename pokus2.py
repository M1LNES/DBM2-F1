import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta


def analyze_weather_performance():
    # Připojení k databázi
    conn = duckdb.connect('race_database.db')

    # Opravený SQL dotaz
    query = """
    WITH race_weather_summary AS (
        SELECT 
            w.race_id,
            AVG(w.air_temp) as avg_air_temp,
            AVG(w.track_temp) as avg_track_temp,
            AVG(w.humidity) as avg_humidity,
            AVG(w.wind_speed) as avg_wind_speed,
            BOOL_OR(w.rainfall) as had_rain,
            MIN(w.air_temp) as min_air_temp,
            MAX(w.air_temp) as max_air_temp,
            MIN(w.track_temp) as min_track_temp,
            MAX(w.track_temp) as max_track_temp
        FROM weather_data w
        GROUP BY w.race_id
    ),
    final_positions AS (
        SELECT 
            race_id,
            driver,
            position as final_position,
            ROW_NUMBER() OVER (PARTITION BY race_id, driver ORDER BY time DESC) as rn
        FROM race_results
        WHERE deleted = FALSE 
        AND position IS NOT NULL
    ),
    driver_performance AS (
        SELECT 
            rr.race_id,
            rr.driver,
            rr.team,
            MIN(rr.position) as best_position,
            AVG(rr.lap_time_seconds) as avg_lap_time,
            MIN(rr.lap_time_seconds) as fastest_lap,
            COUNT(*) as total_laps,
            SUM(CASE WHEN rr.position <= 10 THEN 1 ELSE 0 END) as laps_in_top10
        FROM race_results rr
        WHERE rr.deleted = FALSE 
        AND rr.position IS NOT NULL
        AND rr.lap_time_seconds IS NOT NULL
        AND rr.lap_time_seconds > 60
        GROUP BY rr.race_id, rr.driver, rr.team
    )
    SELECT 
        r.race_name,
        r.year,
        r.night_race,
        r.city_circuit,
        dp.driver,
        dp.team,
        fp.final_position,
        dp.best_position,
        dp.avg_lap_time,
        dp.fastest_lap,
        dp.laps_in_top10,
        dp.total_laps,
        (dp.laps_in_top10 * 100.0 / dp.total_laps) as top10_percentage,
        ws.avg_air_temp,
        ws.avg_track_temp,
        ws.avg_humidity,
        ws.avg_wind_speed,
        ws.had_rain,
        ws.min_air_temp,
        ws.max_air_temp,
        ws.min_track_temp,
        ws.max_track_temp,
        (ws.max_track_temp - ws.min_track_temp) as track_temp_variation
    FROM driver_performance dp
    JOIN races r ON dp.race_id = r.race_id
    LEFT JOIN race_weather_summary ws ON dp.race_id = ws.race_id
    LEFT JOIN final_positions fp ON dp.race_id = fp.race_id AND dp.driver = fp.driver AND fp.rn = 1
    WHERE ws.avg_air_temp IS NOT NULL
    AND fp.final_position IS NOT NULL
    ORDER BY r.race_name, fp.final_position;
    """

    df = conn.execute(query).df()
    conn.close()

    if df.empty:
        print("Žádná data o počasí nebyla nalezena.")
        return

    # Vytvoření vizualizací
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Analýza vlivu počasí na výkon v F1', fontsize=16, fontweight='bold')

    # Graf 1: Výkon v dešti vs. suchu
    rain_comparison = df.groupby(['driver', 'had_rain'])['final_position'].mean().reset_index()
    rain_pivot = rain_comparison.pivot(index='driver', columns='had_rain', values='final_position')
    rain_pivot = rain_pivot.dropna()

    if len(rain_pivot) > 0:
        rain_pivot['difference'] = rain_pivot[True] - rain_pivot[False]
        top_rain_drivers = rain_pivot.nsmallest(10, 'difference')

        x_pos = np.arange(len(top_rain_drivers))
        axes[0, 0].barh(x_pos, top_rain_drivers['difference'],
                        color=['green' if x < 0 else 'red' for x in top_rain_drivers['difference']])
        axes[0, 0].set_yticks(x_pos)
        axes[0, 0].set_yticklabels(top_rain_drivers.index, fontsize=8)
        axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Rozdíl pozice (Déšť - Sucho)')
        axes[0, 0].set_title('Top 10: Specializace na deštivé podmínky')
        axes[0, 0].grid(True, alpha=0.3)

    # Graf 2: Teplota vs. rychlost kol
    df_temp = df.dropna(subset=['avg_track_temp', 'avg_lap_time'])
    if len(df_temp) > 0:
        temp_performance = df_temp.groupby(pd.cut(df_temp['avg_track_temp'], bins=8))['avg_lap_time'].mean()
        temp_bins = [f"{interval.left:.0f}-{interval.right:.0f}°C" for interval in temp_performance.index]

        axes[0, 1].plot(temp_bins, temp_performance.values, 'o-', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Teplota tratě (°C)')
        axes[0, 1].set_ylabel('Průměrný čas kola (s)')
        axes[0, 1].set_title('Vliv teploty tratě na rychlost')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

    # Graf 3: Vlhkost vs. konzistence
    # Graf 3: Vlhkost vs. konzistence
    df_humid = df.dropna(subset=['avg_humidity'])
    if len(df_humid) > 0:
        # Výpočet konzistence pro každého jezdce v každém závodě
        consistency_data = []
        for (race_name, driver), group in df.groupby(['race_name', 'driver']):
            if len(group) > 1:  # Potřebujeme více než jedno měření pro std
                consistency = group['avg_lap_time'].std()
                humidity = group['avg_humidity'].iloc[0]
                consistency_data.append({
                    'race_name': race_name,
                    'driver': driver,
                    'consistency': consistency,
                    'avg_humidity': humidity
                })

        if consistency_data:
            consistency_df = pd.DataFrame(consistency_data).dropna()

            if len(consistency_df) > 0:
                humidity_bins = pd.cut(consistency_df['avg_humidity'], bins=6)
                humidity_consistency = consistency_df.groupby(humidity_bins)['consistency'].mean()

                humidity_labels = [f"{interval.left:.0f}-{interval.right:.0f}%" for interval in
                                   humidity_consistency.index]
                axes[0, 2].bar(range(len(humidity_labels)), humidity_consistency.values, alpha=0.7, color='skyblue')
                axes[0, 2].set_xlabel('Vlhkost vzduchu (%)')
                axes[0, 2].set_ylabel('Průměrná variabilita času kola')
                axes[0, 2].set_title('Vliv vlhkosti na konzistenci')
                axes[0, 2].set_xticks(range(len(humidity_labels)))
                axes[0, 2].set_xticklabels(humidity_labels, rotation=45)
                axes[0, 2].grid(True, alpha=0.3)
            else:
                axes[0, 2].text(0.5, 0.5, 'Nedostatek dat pro analýzu konzistence',
                                ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Vliv vlhkosti na konzistenci')
        else:
            axes[0, 2].text(0.5, 0.5, 'Nedostatek dat pro analýzu konzistence',
                            ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Vliv vlhkosti na konzistenci')

    # Graf 4: Teamová analýza
    team_weather = df.groupby(['team', 'had_rain'])['top10_percentage'].mean().reset_index()
    team_pivot = team_weather.pivot(index='team', columns='had_rain', values='top10_percentage')
    team_pivot = team_pivot.dropna()

    if len(team_pivot) > 0:
        top_teams = team_pivot.head(8)
        x_pos = np.arange(len(top_teams))
        width = 0.35

        axes[1, 0].bar(x_pos - width / 2, top_teams[False], width, label='Sucho', alpha=0.8)
        if True in top_teams.columns:
            axes[1, 0].bar(x_pos + width / 2, top_teams[True], width, label='Déšť', alpha=0.8)
        axes[1, 0].set_xlabel('Tým')
        axes[1, 0].set_ylabel('% kol v top 10')
        axes[1, 0].set_title('Týmový výkon: Sucho vs. Déšť')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(top_teams.index, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Graf 5: Noční vs. denní závody
    night_performance = df.groupby(['driver', 'night_race'])['final_position'].mean().reset_index()
    night_pivot = night_performance.pivot(index='driver', columns='night_race', values='final_position')
    night_pivot = night_pivot.dropna()

    if len(night_pivot) > 0 and len(night_pivot.columns) > 1:
        night_pivot['night_advantage'] = night_pivot[False] - night_pivot[True]
        top_night_drivers = night_pivot.nlargest(8, 'night_advantage')

        axes[1, 1].scatter(top_night_drivers[False], top_night_drivers[True],
                           s=100, alpha=0.7, c='purple')

        min_pos = min(top_night_drivers[False].min(), top_night_drivers[True].min())
        max_pos = max(top_night_drivers[False].max(), top_night_drivers[True].max())
        axes[1, 1].plot([min_pos, max_pos], [min_pos, max_pos], 'r--', alpha=0.5)

        axes[1, 1].set_xlabel('Průměrná pozice (denní závody)')
        axes[1, 1].set_ylabel('Průměrná pozice (noční závody)')
        axes[1, 1].set_title('Denní vs. Noční závody')
        axes[1, 1].grid(True, alpha=0.3)

        for idx, driver in enumerate(top_night_drivers.index[:3]):
            axes[1, 1].annotate(driver,
                                (top_night_drivers.loc[driver, False],
                                 top_night_drivers.loc[driver, True]),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Graf 6: Heatmapa korelací
    correlation_cols = ['final_position', 'avg_air_temp', 'avg_track_temp',
                        'avg_humidity', 'avg_wind_speed', 'track_temp_variation']
    available_cols = [col for col in correlation_cols if col in df.columns]

    if len(available_cols) > 2:
        correlation_data = df[available_cols].corr()
        sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0,
                    square=True, ax=axes[1, 2])
        axes[1, 2].set_title('Korelace: Počasí vs. Výkon')

    plt.tight_layout()
    plt.show()

    # Statistické shrnutí
    print("\n=== ANALÝZA VLIVU POČASÍ ===")
    print(f"Celkem analyzovaných výsledků: {len(df)}")
    print(f"Závody s deštěm: {df['had_rain'].sum()}")
    print(f"Noční závody: {df['night_race'].sum()}")

    # Analýza deště
    if 'had_rain' in df.columns and df['had_rain'].any():
        rain_stats = df.groupby('had_rain')['avg_lap_time'].agg(['mean', 'std']).round(2)
        print(f"\n=== VLIV DEŠTĚ ===")
        if False in rain_stats.index:
            print(f"Průměrný čas kola (sucho): {rain_stats.loc[False, 'mean']:.2f}s")
        if True in rain_stats.index:
            print(f"Průměrný čas kola (déšť): {rain_stats.loc[True, 'mean']:.2f}s")

    # Teplotní analýza
    if 'avg_track_temp' in df.columns:
        temp_corr = df['avg_track_temp'].corr(df['avg_lap_time'])
        print(f"\n=== TEPLOTNÍ ANALÝZA ===")
        print(f"Korelace teploty tratě s časem kola: {temp_corr:.3f}")

        if not df['avg_lap_time'].isna().all():
            optimal_temp = df.loc[df['avg_lap_time'].idxmin(), 'avg_track_temp']
            print(f"Optimální teplota tratě: {optimal_temp:.1f}°C")

    # Rain masters
    if len(rain_pivot) > 0:
        print(f"\n=== TOP 3 'RAIN MASTERS' ===")
        best_rain_drivers = rain_pivot.nsmallest(3, 'difference')
        for driver, row in best_rain_drivers.iterrows():
            improvement = -row['difference']
            print(f"{driver}: {improvement:.2f} pozic lepší v dešti")


if __name__ == "__main__":
    analyze_weather_performance()
