import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def analyze_grid_position_performance():
    # Připojení k databázi
    conn = duckdb.connect('race_database.db')

    # SQL dotaz pro získání dat o startovních pozicích a finálních výsledcích
    query = """
    WITH race_positions AS (
        SELECT 
            r.race_name,
            r.year,
            r.city_circuit,
            rr.driver,
            rr.team,
            -- Předpokládáme, že první záznam každého jezdce v závodě je grid pozice
            FIRST_VALUE(rr.position) OVER (
                PARTITION BY rr.race_id, rr.driver 
                ORDER BY rr.time
            ) as grid_position,
            -- Finální pozice (poslední platná pozice)
            LAST_VALUE(rr.position) OVER (
                PARTITION BY rr.race_id, rr.driver 
                ORDER BY rr.time 
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) as final_position
        FROM race_results rr
        JOIN races r ON rr.race_id = r.race_id
        WHERE rr.deleted = FALSE 
        AND rr.position IS NOT NULL
    ),
    unique_results AS (
        SELECT DISTINCT 
            race_name,
            year,
            city_circuit,
            driver,
            team,
            grid_position,
            final_position,
            (grid_position - final_position) as positions_gained
        FROM race_positions
        WHERE grid_position IS NOT NULL 
        AND final_position IS NOT NULL
        AND grid_position <= 20  -- Omezíme na reálné grid pozice
    )
    SELECT * FROM unique_results
    ORDER BY race_name, grid_position;
    """

    df = conn.execute(query).df()
    conn.close()

    if df.empty:
        print("Žádná data nebyla nalezena.")
        return

    # Vytvoření vizualizací
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analýza vlivu startovní pozice na výkon v F1', fontsize=16, fontweight='bold')

    # Graf 1: Průměrný zisk/ztráta pozic podle grid pozice
    avg_positions = df.groupby('grid_position')['positions_gained'].agg(['mean', 'count']).reset_index()
    avg_positions = avg_positions[avg_positions['count'] >= 5]  # Pouze pozice s dostatečným počtem dat

    axes[0, 0].bar(avg_positions['grid_position'], avg_positions['mean'],
                   color=['red' if x < 0 else 'green' for x in avg_positions['mean']], alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Startovní pozice (Grid)')
    axes[0, 0].set_ylabel('Průměrný zisk pozic (+) / Ztráta pozic (-)')
    axes[0, 0].set_title('Průměrný výkon podle startovní pozice')
    axes[0, 0].grid(True, alpha=0.3)

    # Graf 2: Box plot - distribuce zisků/ztrát podle grid pozice
    top_grid_positions = df[df['grid_position'] <= 10]  # Top 10 grid pozic
    if not top_grid_positions.empty:
        sns.boxplot(data=top_grid_positions, x='grid_position', y='positions_gained', ax=axes[0, 1])
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Startovní pozice (Grid)')
        axes[0, 1].set_ylabel('Zisk/Ztráta pozic')
        axes[0, 1].set_title('Distribuce výkonu podle grid pozice (Top 10)')

    # Graf 3: Porovnání městských okruhů vs. ostatních tratí
    circuit_comparison = df.groupby(['city_circuit', 'grid_position'])['positions_gained'].mean().reset_index()

    city_data = circuit_comparison[circuit_comparison['city_circuit'] == True]
    normal_data = circuit_comparison[circuit_comparison['city_circuit'] == False]

    if not city_data.empty and not normal_data.empty:
        axes[1, 0].plot(city_data['grid_position'], city_data['positions_gained'],
                        'o-', label='Městské okruhy', linewidth=2, markersize=6)
        axes[1, 0].plot(normal_data['grid_position'], normal_data['positions_gained'],
                        's-', label='Standardní tratě', linewidth=2, markersize=6)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Startovní pozice (Grid)')
        axes[1, 0].set_ylabel('Průměrný zisk/ztráta pozic')
        axes[1, 0].set_title('Městské okruhy vs. Standardní tratě')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Graf 4: Heatmapa - finální pozice vs. grid pozice
    position_matrix = df.pivot_table(values='driver', index='final_position',
                                     columns='grid_position', aggfunc='count', fill_value=0)

    if not position_matrix.empty:
        sns.heatmap(position_matrix.iloc[:15, :15], annot=True, fmt='d',
                    cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Startovní pozice (Grid)')
        axes[1, 1].set_ylabel('Finální pozice')
        axes[1, 1].set_title('Heatmapa: Grid → Finální pozice')

    plt.tight_layout()
    plt.show()

    # Statistické shrnutí
    print("\n=== ANALÝZA STARTOVNÍCH POZIC ===")
    print(f"Celkem analyzovaných výsledků: {len(df)}")
    print(f"Počet různých závodů: {df['race_name'].nunique()}")
    print(f"Počet různých jezdců: {df['driver'].nunique()}")

    print("\n=== TOP 3 NEJLEPŠÍ GRID POZICE (podle průměrného zisku) ===")
    best_positions = avg_positions.nlargest(3, 'mean')[['grid_position', 'mean', 'count']]
    for _, row in best_positions.iterrows():
        print(f"Pozice {int(row['grid_position'])}: +{row['mean']:.2f} pozic (z {int(row['count'])} případů)")

    print("\n=== TOP 3 NEJHORŠÍ GRID POZICE (podle průměrného zisku) ===")
    worst_positions = avg_positions.nsmallest(3, 'mean')[['grid_position', 'mean', 'count']]
    for _, row in worst_positions.iterrows():
        print(f"Pozice {int(row['grid_position'])}: {row['mean']:.2f} pozic (z {int(row['count'])} případů)")

    # Analýza slipstream efektu (pozice 2-4)
    slipstream_positions = df[df['grid_position'].isin([2, 3, 4])]
    if not slipstream_positions.empty:
        avg_slipstream = slipstream_positions['positions_gained'].mean()
        print(f"\n=== SLIPSTREAM ANALÝZA ===")
        print(f"Průměrný výkon z pozic 2-4 (slipstream zone): {avg_slipstream:.2f} pozic")

        pole_position = df[df['grid_position'] == 1]['positions_gained'].mean()
        print(f"Průměrný výkon z pole position: {pole_position:.2f} pozic")

        if avg_slipstream > pole_position:
            print("✓ Potvrzuje se hypotéza o výhodě slipstream efektu!")
        else:
            print("✗ Slipstream efekt se nepotvrzuje v datech.")


if __name__ == "__main__":
    analyze_grid_position_performance()
