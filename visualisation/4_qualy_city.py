import duckdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def correlation_whole():
    # Connect to the database
    conn = duckdb.connect('../race_database.db')

    # Get data: qualifying position vs final race position
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

    # Create pivot table for heatmap (normalized to percentages)
    pivot = df.pivot_table(
        index='quali_position',
        columns='final_position',
        values='count',
        fill_value=0
    )

    # Convert to percentages (per starting position)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Create heatmap
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

    # Save plot
    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / 'qualy_transition_whole.png', dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_qualy_city_circuit_correlation():
    # Connect to the database
    conn = duckdb.connect('../race_database.db')

    # Get data: qualifying position vs final position + city_circuit flag
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

    # Split data into city and non-city circuits
    df_city = df[df['city_circuit'] == True]
    df_non_city = df[df['city_circuit'] == False]

    # Create folder for saving
    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    for data, title, filename in [
        (df_city, 'City Circuits', 'qualy_transition_city_circuit.png'),
        (df_non_city, 'Non-City Circuits', 'qualy_transition_non_city_circuit.png')
    ]:
        # Create pivot table
        pivot = data.pivot_table(
            index='quali_position',
            columns='final_position',
            values='count',
            fill_value=0
        )

        # Convert to percentages
        if pivot.empty:
            pivot_pct = pivot.copy()
        else:
            pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

        # Create a separate figure for each plot
        fig = plt.figure(figsize=(16, 12))

        # Heatmap
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

        plt.tight_layout()

        # Save plot
        fig.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()


def plot_qualy_night_race_correlation():
    # Connect to the database
    conn = duckdb.connect('../race_database.db')

    # Get data: qualifying position vs final position + night_race flag
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

    # Split data into night and day races
    df_night = df[df['night_race'] == True]
    df_day = df[df['night_race'] == False]

    # Create folder for saving
    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    for data, title, filename in [
        (df_night, 'Night Races', 'qualy_transition_night_race.png'),
        (df_day, 'Day Races', 'qualy_transition_day_race.png')
    ]:
        # Create pivot table
        pivot = data.pivot_table(
            index='quali_position',
            columns='final_position',
            values='count',
            fill_value=0
        )

        # Convert to percentages
        if pivot.empty:
            pivot_pct = pivot.copy()
        else:
            pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

        # Create a separate figure for each plot
        fig = plt.figure(figsize=(16, 12))

        # Heatmap
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

        plt.tight_layout()

        # Save plot
        fig.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()


if __name__ == "__main__":
    correlation_whole()
    plot_qualy_city_circuit_correlation()
    plot_qualy_night_race_correlation()
