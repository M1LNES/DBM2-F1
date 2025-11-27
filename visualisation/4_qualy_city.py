import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def correlation_whole():
    # Připojení k databázi
    conn = duckdb.connect('../race_database.db')

    # Získání dat: pozice z kvalifikace vs konečná pozice v závodě
    query = """
      WITH race_final_positions AS (
          SELECT
              race_id,
              driver,
              position as final_position
          FROM race_results
          WHERE lap_number = (
              SELECT MAX(lap_number)
              FROM race_results rr2
              WHERE rr2.race_id = race_results.race_id
              AND rr2.driver = race_results.driver
          )
          AND position IS NOT NULL
      )
      SELECT
          q.position as quali_position,
          rfp.final_position,
          COUNT(*) as count
      FROM qualifying q
      JOIN race_final_positions rfp
          ON q.race_id = rfp.race_id
          AND q.driver = rfp.driver
      WHERE q.position IS NOT NULL
      GROUP BY q.position, rfp.final_position
      ORDER BY q.position, rfp.final_position
      """

    df = pd.read_sql(query, conn)
    conn.close()

    # Vytvoření pivot tabulky pro heatmapu (normalizováno na procenta)
    pivot = df.pivot_table(
        index='quali_position',
        columns='final_position',
        values='count',
        fill_value=0
    )

    # Převod na procenta (pro každou startovní pozici)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Vytvoření heatmapy
    fig = plt.figure(figsize=(16, 12))
    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        cbar_kws={'label': 'Percentage of races (%)'},
        linewidths=0.5
    )

    plt.title('Position Transition Matrix: Qualifying → Race Result',
              fontsize=14, pad=20)
    plt.xlabel('Final Position in Race', fontsize=12)
    plt.ylabel('Starting Position (Qualifying)', fontsize=12)
    plt.tight_layout()

    # Uložení grafu
    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'qualy_transition_whole.png', dpi=300, bbox_inches='tight')
    print(f"Graf uložen: {plots_dir / 'qualy_transition_whole.png'}")

    plt.show()
    plt.close()

    # Výpis statistiky pro diagonálu (stejná pozice)
    print("\n=== Statistika: Procento závodů se stejnou pozicí ===")
    for pos in pivot_pct.index:
        if pos in pivot_pct.columns:
            pct = pivot_pct.loc[pos, pos]
            print(f"Pozice {int(pos)}: {pct:.1f}% závodů")


def plot_qualy_city_circuit_correlation():
    # Připojení k databázi
    conn = duckdb.connect('../race_database.db')

    # Získání dat: pozice z kvalifikace vs konečná pozice v závodě + city_circuit
    query = """
    WITH race_final_positions AS (
        SELECT
            race_id,
            driver,
            position as final_position
        FROM race_results
        WHERE lap_number = (
            SELECT MAX(lap_number)
            FROM race_results rr2
            WHERE rr2.race_id = race_results.race_id
            AND rr2.driver = race_results.driver
        )
        AND position IS NOT NULL
    )
    SELECT
        q.position as quali_position,
        rfp.final_position,
        r.city_circuit,
        COUNT(*) as count
    FROM qualifying q
    JOIN race_final_positions rfp
        ON q.race_id = rfp.race_id
        AND q.driver = rfp.driver
    JOIN races r ON q.race_id = r.race_id
    WHERE q.position IS NOT NULL
    GROUP BY q.position, rfp.final_position, r.city_circuit
    ORDER BY q.position, rfp.final_position
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Rozdělení dat na městské a neměstské okruhy
    df_city = df[df['city_circuit'] == True]
    df_non_city = df[df['city_circuit'] == False]

    # Vytvoření složky pro ukládání
    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    for data, title, filename in [
        (df_city, 'City Circuits', 'qualy_transition_city_circuit.png'),
        (df_non_city, 'Non-City Circuits', 'qualy_transition_non_city_circuit.png')
    ]:
        # Vytvoření pivot tabulky
        pivot = data.pivot_table(
            index='quali_position',
            columns='final_position',
            values='count',
            fill_value=0
        )

        # Převod na procenta
        if pivot.empty:
            pivot_pct = pivot.copy()
        else:
            pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

        # Vytvoření samostatného figure pro každý graf
        fig = plt.figure(figsize=(16, 12))

        # Heatmapa
        sns.heatmap(
            pivot_pct,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Percentage of races (%)'},
            linewidths=0.5
        )

        plt.title(f'Position Transition Matrix: Qualifying → Race Result\n({title})',
                  fontsize=14, pad=20)
        plt.xlabel('Final Position in Race', fontsize=12)
        plt.ylabel('Starting Position (Qualifying)', fontsize=12)

        # Statistika pro diagonálu
        print(f"\n=== {title}: Procento závodů se stejnou pozicí ===")
        for pos in pivot_pct.index:
            if pos in pivot_pct.columns:
                pct = pivot_pct.loc[pos, pos]
                print(f"Pozice {int(pos)}: {pct:.1f}% závodů")

        plt.tight_layout()

        # Uložení grafu
        fig.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Graf uložen: {plots_dir / filename}")

        plt.show()
        plt.close()


def plot_qualy_night_race_correlation():
    # Připojení k databázi
    conn = duckdb.connect('../race_database.db')

    # Získání dat: pozice z kvalifikace vs konečná pozice v závodě + night_race
    query = """
    WITH race_final_positions AS (
        SELECT
            race_id,
            driver,
            position as final_position
        FROM race_results
        WHERE lap_number = (
            SELECT MAX(lap_number)
            FROM race_results rr2
            WHERE rr2.race_id = race_results.race_id
            AND rr2.driver = race_results.driver
        )
        AND position IS NOT NULL
    )
    SELECT
        q.position as quali_position,
        rfp.final_position,
        r.night_race,
        COUNT(*) as count
    FROM qualifying q
    JOIN race_final_positions rfp
        ON q.race_id = rfp.race_id
        AND q.driver = rfp.driver
    JOIN races r ON q.race_id = r.race_id
    WHERE q.position IS NOT NULL
    GROUP BY q.position, rfp.final_position, r.night_race
    ORDER BY q.position, rfp.final_position
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Rozdělení dat na noční a denní závody
    df_night = df[df['night_race'] == True]
    df_day = df[df['night_race'] == False]

    # Vytvoření složky pro ukládání
    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    for data, title, filename in [
        (df_night, 'Night Races', 'qualy_transition_night_race.png'),
        (df_day, 'Day Races', 'qualy_transition_day_race.png')
    ]:
        # Vytvoření pivot tabulky
        pivot = data.pivot_table(
            index='quali_position',
            columns='final_position',
            values='count',
            fill_value=0
        )

        # Převod na procenta
        if pivot.empty:
            pivot_pct = pivot.copy()
        else:
            pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

        # Vytvoření samostatného figure pro každý graf
        fig = plt.figure(figsize=(16, 12))

        # Heatmapa
        sns.heatmap(
            pivot_pct,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Percentage of races (%)'},
            linewidths=0.5
        )

        plt.title(f'Position Transition Matrix: Qualifying → Race Result\n({title})',
                  fontsize=14, pad=20)
        plt.xlabel('Final Position in Race', fontsize=12)
        plt.ylabel('Starting Position (Qualifying)', fontsize=12)

        # Statistika pro diagonálu
        print(f"\n=== {title}: Procento závodů se stejnou pozicí ===")
        for pos in pivot_pct.index:
            if pos in pivot_pct.columns:
                pct = pivot_pct.loc[pos, pos]
                print(f"Pozice {int(pos)}: {pct:.1f}% závodů")

        plt.tight_layout()

        # Uložení grafu
        fig.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Graf uložen: {plots_dir / filename}")

        plt.show()
        plt.close()


if __name__ == "__main__":
    correlation_whole()
    plot_qualy_city_circuit_correlation()
    plot_qualy_night_race_correlation()
