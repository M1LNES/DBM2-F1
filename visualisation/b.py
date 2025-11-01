def main():
    import duckdb
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    conn = duckdb.connect('race_database.db')

    print("=== Testování hypotéz a generování vizualizací ===\n")

    # HYPOTÉZA 1: Vliv teploty na rychlost kol
    print("1. Vliv teploty na časy kol...")
    temp_query = """
    SELECT 
        rr.lap_time_seconds,
        AVG(wd.air_temp) as avg_temp
    FROM race_results rr
    JOIN weather_data wd ON rr.race_id = wd.race_id
    WHERE rr.lap_time_seconds IS NOT NULL 
        AND rr.lap_time_seconds > 0
        AND wd.air_temp IS NOT NULL
    GROUP BY rr.result_id, rr.lap_time_seconds
    HAVING AVG(wd.air_temp) BETWEEN 0 AND 50
    """
    temp_data = conn.execute(temp_query).df()

    plt.figure(figsize=(12, 6))
    plt.hexbin(temp_data['avg_temp'], temp_data['lap_time_seconds'],
               gridsize=30, cmap='YlOrRd', mincnt=1)
    plt.colorbar(label='Počet kol')

    z = np.polyfit(temp_data['avg_temp'], temp_data['lap_time_seconds'], 1)
    p = np.poly1d(z)
    plt.plot(temp_data['avg_temp'].sort_values(),
             p(temp_data['avg_temp'].sort_values()),
             "r--", linewidth=2, label=f'Trend: {z[0]:.3f}s/°C')

    plt.xlabel('Průměrná teplota vzduchu (°C)', fontsize=12)
    plt.ylabel('Čas kola (s)', fontsize=12)
    plt.title('Vliv teploty na rychlost kol', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hypothesis_temperature.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_temperature.png'\n")

    # HYPOTÉZA 2: Kvalifikace vs. výsledek v závodě
    print("2. Korelace kvalifikace vs. finální pozice...")
    quali_query = """
    SELECT 
        q.position as quali_position,
        rr.position as race_position
    FROM qualifying q
    JOIN race_results rr ON q.race_id = rr.race_id AND q.driver = rr.driver
    WHERE q.position IS NOT NULL 
        AND rr.position IS NOT NULL
        AND rr.lap_number = (
            SELECT MAX(lap_number) 
            FROM race_results rr2 
            WHERE rr2.race_id = rr.race_id AND rr2.driver = rr.driver
        )
    """
    quali_data = conn.execute(quali_query).df()

    plt.figure(figsize=(10, 10))
    plt.hexbin(quali_data['quali_position'], quali_data['race_position'],
               gridsize=20, cmap='Blues', mincnt=1)
    plt.colorbar(label='Počet jezdců')

    plt.plot([0, 20], [0, 20], 'r--', linewidth=2, label='Ideální shoda')

    corr = quali_data['quali_position'].corr(quali_data['race_position'])
    plt.text(15, 3, f'Korelace: {corr:.3f}', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('Pozice v kvalifikaci', fontsize=12)
    plt.ylabel('Finální pozice v závodě', fontsize=12)
    plt.title('Kvalifikace vs. finální výsledek', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hypothesis_qualifying.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_qualifying.png'\n")

    # HYPOTÉZA 3: Degradace pneumatik během závodu
    print("3. Degradace pneumatik během závodu...")
    deg_query = """
    SELECT 
        stint_lap,
        AVG(lap_time_seconds) as avg_lap_time,
        compound
    FROM (
        SELECT 
            rr.lap_time_seconds,
            rr.compound,
            ROW_NUMBER() OVER (
                PARTITION BY rr.race_id, rr.driver, rr.compound 
                ORDER BY rr.lap_number
            ) as stint_lap
        FROM race_results rr
        WHERE rr.lap_time_seconds IS NOT NULL 
            AND rr.lap_time_seconds > 0
            AND rr.compound IS NOT NULL
    )
    WHERE stint_lap <= 30
    GROUP BY stint_lap, compound
    ORDER BY compound, stint_lap
    """
    deg_data = conn.execute(deg_query).df()

    plt.figure(figsize=(12, 7))
    colors_map = {'SOFT': '#FF1E1E', 'MEDIUM': '#FFF200', 'HARD': '#FFFFFF'}

    for compound in deg_data['compound'].unique():
        subset = deg_data[deg_data['compound'] == compound]
        color = colors_map.get(compound, 'gray')
        plt.plot(subset['stint_lap'], subset['avg_lap_time'],
                 marker='o', label=compound, linewidth=2, markersize=4,
                 color=color, markeredgecolor='black')

    plt.xlabel('Kolo ve stintu', fontsize=12)
    plt.ylabel('Průměrný čas kola (s)', fontsize=12)
    plt.title('Degradace pneumatik během stintu', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hypothesis_tyre_degradation.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_tyre_degradation.png'\n")

    # HYPOTÉZA 4: Vliv nadmořské výšky
    print("4. Vliv nadmořské výšky okruhu...")
    altitude_query = """
    SELECT 
        c.alt as altitude,
        AVG(rr.lap_time_seconds) as avg_lap_time,
        c.name as circuit_name
    FROM race_results rr
    JOIN races r ON rr.race_id = r.race_id
    JOIN circuits c ON r.circuit_id = c.circuit_id
    WHERE rr.lap_time_seconds IS NOT NULL 
        AND rr.lap_time_seconds > 0
        AND c.alt IS NOT NULL
    GROUP BY c.circuit_id, c.alt, c.name
    """
    alt_data = conn.execute(altitude_query).df()

    plt.figure(figsize=(12, 7))
    plt.scatter(alt_data['altitude'], alt_data['avg_lap_time'],
                s=100, alpha=0.6, c=alt_data['altitude'], cmap='terrain')
    plt.colorbar(label='Nadmořská výška (m)')

    for idx, row in alt_data.iterrows():
        if row['altitude'] > 1000 or row['altitude'] < 100:
            plt.annotate(row['circuit_name'],
                         (row['altitude'], row['avg_lap_time']),
                         fontsize=8, alpha=0.7)

    plt.xlabel('Nadmořská výška okruhu (m)', fontsize=12)
    plt.ylabel('Průměrný čas kola (s)', fontsize=12)
    plt.title('Vliv nadmořské výšky na časy kol', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hypothesis_altitude.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_altitude.png'\n")

    conn.close()

    print("=== Všechny hypotézy vizualizovány ===")
    print("\nVýsledky:")
    print("  • hypothesis_temperature.png - Vliv teploty")
    print("  • hypothesis_qualifying.png - Kvalifikace vs. závod")
    print("  • hypothesis_tyre_degradation.png - Degradace pneumatik")
    print("  • hypothesis_altitude.png - Vliv nadmořské výšky")


if __name__ == "__main__":
    main()
