def main():
    import duckdb
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    conn = duckdb.connect('race_database.db')

    print("=== Další hypotézy a vizualizace ===\n")

    # HYPOTÉZA 5: Safety Car efekt na rozestup pole
    print("5. Vliv Safety Car na rozestupy v poli...")
    sc_query = """
    SELECT
        rr.lap_number,
        STDDEV(rr.lap_time_seconds) as time_spread,
        CASE WHEN rr.track_status IN ('4', '6', '7') THEN 'SC/VSC' ELSE 'Zelená' END as status
    FROM race_results rr
    WHERE rr.lap_time_seconds IS NOT NULL
        AND rr.lap_time_seconds > 0
        AND rr.lap_number <= 60
    GROUP BY rr.race_id, rr.lap_number, status
    """
    sc_data = conn.execute(sc_query).df()

    plt.figure(figsize=(14, 6))
    for status in sc_data['status'].unique():
        subset = sc_data[sc_data['status'] == status]
        grouped = subset.groupby('lap_number')['time_spread'].mean()
        color = 'yellow' if status == 'SC/VSC' else 'green'
        plt.plot(grouped.index, grouped.values, marker='o', label=status,
                linewidth=2, markersize=3, color=color, alpha=0.7)

    plt.xlabel('Číslo kola', fontsize=12)
    plt.ylabel('Rozptyl časů (standardní odchylka)', fontsize=12)
    plt.title('Vliv Safety Car na rozestupy v poli', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hypothesis_safety_car.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_safety_car.png'\n")

    # HYPOTÉZA 6: Vliv vlhkosti trati na počet chyb/DNF
    print("6. Vlhkost tratě vs. míra dokončení...")
    wet_query = """
    SELECT
        r.race_id,
        r.year,
        r.round,
        c.name as circuit,
        AVG(wd.track_temp) as avg_track_temp,
        AVG(wd.humidity) as avg_humidity,
        COUNT(DISTINCT rr.driver) as total_drivers,
        COUNT(DISTINCT CASE WHEN rr.status LIKE '%Finished%' THEN rr.driver END) as finished_drivers
    FROM races r
    JOIN race_results rr ON r.race_id = rr.race_id
    LEFT JOIN weather_data wd ON r.race_id = wd.race_id
    JOIN circuits c ON r.circuit_id = c.circuit_id
    WHERE wd.humidity IS NOT NULL
    GROUP BY r.race_id, r.year, r.round, c.name
    HAVING total_drivers > 0
    """
    wet_data = conn.execute(wet_query).df()
    wet_data['finish_rate'] = (wet_data['finished_drivers'] / wet_data['total_drivers']) * 100

    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(wet_data['avg_humidity'], wet_data['finish_rate'],
                         c=wet_data['avg_track_temp'], s=80, alpha=0.6,
                         cmap='coolwarm', edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Průměrná teplota trati (°C)')

    z = np.polyfit(wet_data['avg_humidity'], wet_data['finish_rate'], 1)
    p = np.poly1d(z)
    plt.plot(sorted(wet_data['avg_humidity']), p(sorted(wet_data['avg_humidity'])),
            "r--", linewidth=2, alpha=0.8, label=f'Trend: {z[0]:.2f}%/% vlhkosti')

    plt.xlabel('Průměrná vlhkost vzduchu (%)', fontsize=12)
    plt.ylabel('Míra dokončení závodu (%)', fontsize=12)
    plt.title('Vliv vlhkosti na míru dokončení závodu', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hypothesis_humidity_dnf.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_humidity_dnf.png'\n")

    # HYPOTÉZA 7: První kolo chaos - nejvíce incidentů
    print("7. Míra incidentů podle kola závodu...")
    incidents_query = """
    SELECT
        rr.lap_number,
        COUNT(*) as total_laps,
        COUNT(CASE WHEN rr.track_status NOT IN ('1', '2') THEN 1 END) as incidents,
        (COUNT(CASE WHEN rr.track_status NOT IN ('1', '2') THEN 1 END) * 100.0 / COUNT(*)) as incident_rate
    FROM race_results rr
    WHERE rr.lap_number <= 30
        AND rr.track_status IS NOT NULL
    GROUP BY rr.lap_number
    ORDER BY rr.lap_number
    """
    inc_data = conn.execute(incidents_query).df()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.bar(inc_data['lap_number'], inc_data['incidents'],
           color='crimson', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Číslo kola', fontsize=12)
    ax1.set_ylabel('Počet incidentů', fontsize=12)
    ax1.set_title('Absolutní počet incidentů podle kola', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.plot(inc_data['lap_number'], inc_data['incident_rate'],
            marker='o', linewidth=2.5, markersize=6, color='darkred')
    ax2.fill_between(inc_data['lap_number'], inc_data['incident_rate'],
                     alpha=0.3, color='red')
    ax2.set_xlabel('Číslo kola', fontsize=12)
    ax2.set_ylabel('Míra incidentů (%)', fontsize=12)
    ax2.set_title('Relativní míra incidentů (% z celkového počtu kol)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hypothesis_lap_incidents.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_lap_incidents.png'\n")

    # HYPOTÉZA 8: DRS zóny - efekt na předjíždění
    print("8. Aktivita DRS a předjíždění...")
    drs_query = """
    SELECT
        rr.lap_number,
        AVG(rr.drs) as avg_drs_active,
        COUNT(DISTINCT CASE 
            WHEN rr.position < LAG(rr.position) OVER (
                PARTITION BY rr.race_id, rr.driver ORDER BY rr.lap_number
            ) THEN rr.result_id 
        END) as overtakes
    FROM race_results rr
    WHERE rr.lap_number BETWEEN 5 AND 50
        AND rr.drs IS NOT NULL
    GROUP BY rr.race_id, rr.lap_number
    """
    drs_data = conn.execute(drs_query).df()
    drs_agg = drs_data.groupby('lap_number').agg({'avg_drs_active': 'mean', 'overtakes': 'sum'}).reset_index()

    fig, ax1 = plt.subplots(figsize=(14, 7))

    color1 = 'tab:blue'
    ax1.set_xlabel('Číslo kola', fontsize=12)
    ax1.set_ylabel('Průměrná aktivita DRS', fontsize=12, color=color1)
    line1 = ax1.plot(drs_agg['lap_number'], drs_agg['avg_drs_active'],
                    color=color1, linewidth=2.5, marker='o', markersize=4, label='DRS aktivita')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Počet předjetí', fontsize=12, color=color2)
    line2 = ax2.plot(drs_agg['lap_number'], drs_agg['overtakes'],
                    color=color2, linewidth=2.5, marker='s', markersize=4, label='Předjetí')
    ax2.tick_params(axis='y', labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=11)

    plt.title('Aktivita DRS vs. počet předjetí', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hypothesis_drs_overtakes.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_drs_overtakes.png'\n")

    conn.close()

    print("=== Další hypotézy vizualizovány ===")
    print("\nNové výsledky:")
    print("  • hypothesis_safety_car.png - Vliv Safety Car")
    print("  • hypothesis_humidity_dnf.png - Vlhkost a míra dokončení")
    print("  • hypothesis_lap_incidents.png - Incidenty podle kola")
    print("  • hypothesis_drs_overtakes.png - DRS a předjíždění")


if __name__ == "__main__":
    main()
