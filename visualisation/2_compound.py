import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class TireStintAnalysis:
    def __init__(self, db_path='../race_database.db'):
        self.conn = duckdb.connect(db_path, read_only=True)

    def analyze_tire_stints(self):
        """Calculate average stint length for each tire compound."""
        query = """
        WITH stint_changes AS (
            SELECT
                race_id,
                driver,
                compound,
                lap_number,
                tyre_life,
                LAG(compound) OVER (PARTITION BY race_id, driver ORDER BY lap_number) as prev_compound,
                LAG(tyre_life) OVER (PARTITION BY race_id, driver ORDER BY lap_number) as prev_tyre_life
            FROM race_results
            WHERE compound IS NOT NULL 
              AND tyre_life IS NOT NULL
              AND compound IN ('SOFT', 'MEDIUM', 'HARD')
        ),
        stints AS (
            SELECT
                race_id,
                driver,
                compound,
                lap_number,
                tyre_life,
                CASE 
                    WHEN compound != prev_compound OR tyre_life < prev_tyre_life OR prev_compound IS NULL 
                    THEN 1 
                    ELSE 0 
                END as is_new_stint
            FROM stint_changes
        ),
        stint_groups AS (
            SELECT
                *,
                SUM(is_new_stint) OVER (PARTITION BY race_id, driver ORDER BY lap_number) as stint_id
            FROM stints
        ),
        stint_lengths AS (
            SELECT
                race_id,
                driver,
                compound as tire_compound,
                stint_id,
                COUNT(*) as laps
            FROM stint_groups
            GROUP BY race_id, driver, compound, stint_id
        )
        SELECT
            tire_compound,
            COUNT(*) as total_stints,
            AVG(laps) as avg_laps,
            STDDEV(laps) as std_laps,
            MIN(laps) as min_laps,
            MAX(laps) as max_laps,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY laps) as median_laps
        FROM stint_lengths
        WHERE laps > 0
        GROUP BY tire_compound
        ORDER BY
            CASE tire_compound
                WHEN 'SOFT' THEN 1
                WHEN 'MEDIUM' THEN 2
                WHEN 'HARD' THEN 3
            END
        """
        df = self.conn.execute(query).df()
        df['se_laps'] = df['std_laps'] / (df['total_stints'] ** 0.5)
        return df

    def get_stint_distributions(self, compound):
        """Get all stint lengths for a specific compound."""
        query = f"""
        WITH stint_changes AS (
            SELECT
                race_id,
                driver,
                compound,
                lap_number,
                tyre_life,
                LAG(compound) OVER (PARTITION BY race_id, driver ORDER BY lap_number) as prev_compound,
                LAG(tyre_life) OVER (PARTITION BY race_id, driver ORDER BY lap_number) as prev_tyre_life
            FROM race_results
            WHERE compound = '{compound}'
              AND tyre_life IS NOT NULL
        ),
        stints AS (
            SELECT
                race_id,
                driver,
                compound,
                lap_number,
                tyre_life,
                CASE 
                    WHEN compound != prev_compound OR tyre_life < prev_tyre_life OR prev_compound IS NULL 
                    THEN 1 
                    ELSE 0 
                END as is_new_stint
            FROM stint_changes
        ),
        stint_groups AS (
            SELECT
                *,
                SUM(is_new_stint) OVER (PARTITION BY race_id, driver ORDER BY lap_number) as stint_id
            FROM stints
        )
        SELECT
            COUNT(*) as laps
        FROM stint_groups
        GROUP BY race_id, driver, stint_id
        HAVING COUNT(*) > 0
        """
        return self.conn.execute(query).df()['laps'].tolist()

    def plot_tire_stints(self, df, fname_avg='tire_stint_avg_laps.png', fname_dist='tire_stint_distribution.png'):
        """Visualize tire stint statistics into two separate files."""
        colors = {'SOFT': '#FF4444', 'MEDIUM': '#FFD700', 'HARD': '#FFFFFF'}
        edge_colors = {'SOFT': '#CC0000', 'MEDIUM': '#CCAA00', 'HARD': '#333333'}

        # 1) Sloupcový graf průměrné délky stintu
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        x_pos = range(len(df))
        ax1.bar(
            x_pos, df['avg_laps'],
            color=[colors[comp] for comp in df['tire_compound']],
            edgecolor=[edge_colors[comp] for comp in df['tire_compound']],
            linewidth=2,
            alpha=0.8
        )
        ax1.errorbar(
            x_pos, df['avg_laps'],
            yerr=df['se_laps'],
            fmt='none', ecolor='black', capsize=5, linewidth=2
        )
        for i, (_, row) in enumerate(df.iterrows()):
            ax1.text(
                i, row['avg_laps'] + row['se_laps'] + 0.5,
                f"{row['avg_laps']:.1f} ± {row['se_laps']:.2f}",
                ha='center', va='bottom', fontweight='bold', fontsize=11
            )
        ax1.set_xticks(list(x_pos))
        ax1.set_xticklabels(df['tire_compound'], fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average stint length (laps)', fontsize=12, fontweight='bold')
        ax1.set_title('Average tire stint length by compound', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname_avg, dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # 2) Boxplot distribuce délek stintů
        stint_distributions = []
        labels = []
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            laps = self.get_stint_distributions(compound)
            stint_distributions.append(laps)
            labels.append(compound)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        bp = ax2.boxplot(
            stint_distributions,
            labels=labels,
            patch_artist=True,
            notch=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='red', markersize=8)
        )
        for patch, compound in zip(bp['boxes'], labels):
            patch.set_facecolor(colors[compound])
            patch.set_edgecolor(edge_colors[compound])
            patch.set_linewidth(2)
            patch.set_alpha(0.8)
        ax2.set_ylabel('Stint length (laps)', fontsize=12, fontweight='bold')
        ax2.set_title('Tire stint length distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname_dist, dpi=300, bbox_inches='tight')
        plt.close(fig2)

    def print_results(self, df):
        """Print tire stint statistics to console."""
        print("=" * 90)
        print("🏎️  TIRE STINT ANALYSIS — Average laps per compound")
        print("=" * 90)
        print(f"{'Compound':<12} {'Stints':>8} {'Avg ± SE':>18} {'Median':>10} {'Min':>8} {'Max':>8}")
        print("-" * 90)

        for idx, row in df.iterrows():
            print(f"{row['tire_compound']:<12} "
                  f"{row['total_stints']:>8.0f} "
                  f"{row['avg_laps']:>8.2f} ± {row['se_laps']:>5.2f} "
                  f"{row['median_laps']:>10.1f} "
                  f"{row['min_laps']:>8.0f} "
                  f"{row['max_laps']:>8.0f}")

        print("=" * 90)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    analyzer = TireStintAnalysis()
    tire_df = analyzer.analyze_tire_stints()
    analyzer.print_results(tire_df)
    analyzer.plot_tire_stints(tire_df)
    analyzer.close()
