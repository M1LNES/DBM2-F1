# python
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class TireLapTimeAnalysis:
    def __init__(self, db_path='../race_database.db'):
        self.conn = duckdb.connect(db_path, read_only=True)

    def analyze_tire_lap_times(self):
        """Calculate average lap time for each tire compound."""
        query = """
        SELECT
            compound as tire_compound,
            COUNT(*) as total_laps,
            AVG(lap_time_seconds) as avg_lap_time_s,
            STDDEV(lap_time_seconds) as std_lap_time_s,
            MIN(lap_time_seconds) as min_lap_time_s,
            MAX(lap_time_seconds) as max_lap_time_s,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lap_time_seconds) as median_lap_time_s
        FROM race_results
        WHERE compound IN ('SOFT', 'MEDIUM', 'HARD')
          AND lap_time_seconds > 0
          AND lap_time_seconds IS NOT NULL
          AND deleted = FALSE
        GROUP BY compound
        ORDER BY
            CASE compound
                WHEN 'SOFT' THEN 1
                WHEN 'MEDIUM' THEN 2
                WHEN 'HARD' THEN 3
            END
        """
        df = self.conn.execute(query).df()
        df['se_lap_time_s'] = df['std_lap_time_s'] / (df['total_laps'] ** 0.5)
        return df

    def plot_tire_lap_times(self, df):
        """Visualize tire lap time statistics (bar chart only)."""
        fig, ax1 = plt.subplots(figsize=(10, 6))

        colors = {'SOFT': '#FF4444', 'MEDIUM': '#FFD700', 'HARD': '#FFFFFF'}
        edge_colors = {'SOFT': '#CC0000', 'MEDIUM': '#CCAA00', 'HARD': '#333333'}

        x_pos = range(len(df))
        ax1.bar(
            x_pos, df['avg_lap_time_s'],
            color=[colors[comp] for comp in df['tire_compound']],
            edgecolor=[edge_colors[comp] for comp in df['tire_compound']],
            linewidth=2,
            alpha=0.8
        )

        ax1.errorbar(
            x_pos, df['avg_lap_time_s'],
            yerr=df['se_lap_time_s'],
            fmt='none', ecolor='black', capsize=5, linewidth=2
        )

        for i, (_, row) in enumerate(df.iterrows()):
            ax1.text(
                i, row['avg_lap_time_s'] + row['se_lap_time_s'] + 0.5,
                f"{row['avg_lap_time_s']:.2f} ± {row['se_lap_time_s']:.3f}",
                ha='center', va='bottom', fontweight='bold', fontsize=11
            )

        ax1.set_xticks(list(x_pos))
        ax1.set_xticklabels(df['tire_compound'], fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average lap time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('Average lap time by tire compound', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('tire_lap_time_analysis.png', dpi=300)
        plt.show()

    def print_results(self, df):
        """Print tire lap time statistics to console."""
        print("=" * 100)
        print("🏎️  TIRE LAP TIME ANALYSIS — Average lap time per compound")
        print("=" * 100)
        print(f"{'Compound':<12} {'Laps':>8} {'Avg ± SE (s)':>25} {'Median':>12} {'Min':>10} {'Max':>10}")
        print("-" * 100)

        for idx, row in df.iterrows():
            print(f"{row['tire_compound']:<12} "
                  f"{row['total_laps']:>8.0f} "
                  f"{row['avg_lap_time_s']:>10.2f} ± {row['se_lap_time_s']:>6.3f} "
                  f"{row['median_lap_time_s']:>12.2f} "
                  f"{row['min_lap_time_s']:>10.2f} "
                  f"{row['max_lap_time_s']:>10.2f}")

        print("=" * 100)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    analyzer = TireLapTimeAnalysis()
    lap_time_df = analyzer.analyze_tire_lap_times()
    analyzer.print_results(lap_time_df)
    analyzer.plot_tire_lap_times(lap_time_df)
    analyzer.close()
