# python
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


class RaceAttractiveness:
    def __init__(self, db_path='../race_database.db'):
        self.conn = duckdb.connect(db_path, read_only=True)

    def calculate_lead_changes(self) -> pd.DataFrame:
        """Calculate number of lead changes in a race based on time-ordered 'lead events' (captures intra-lap changes)."""
        query = """
        WITH leader_events AS (
            SELECT DISTINCT race_id, time, driver
            FROM race_results
            WHERE position = 1
              AND deleted = FALSE
              AND time IS NOT NULL
        ),
        ordered_leaders AS (
            SELECT
                race_id,
                time,
                driver,
                LAG(driver) OVER (PARTITION BY race_id ORDER BY time) AS prev_leader
            FROM leader_events
        )
        SELECT
            r.race_id,
            r.year,
            r.race_name,
            c.name AS circuit_name,
            c.country,
            SUM(CASE WHEN ol.driver != ol.prev_leader AND ol.prev_leader IS NOT NULL THEN 1 ELSE 0 END) AS lead_changes
        FROM ordered_leaders ol
        JOIN races r ON ol.race_id = r.race_id
        JOIN circuits c ON r.circuit_id = c.circuit_id
        GROUP BY r.race_id, r.year, r.race_name, c.name, c.country
        ORDER BY lead_changes DESC
        """
        return self.conn.execute(query).df()

    def calculate_top5_volatility(self) -> pd.DataFrame:
        """Calculate volatility of the top 5 positions."""
        query = """
        WITH top5_positions AS (
            SELECT
                race_id,
                driver,
                lap_number,
                position,
                LAG(position) OVER (PARTITION BY race_id, driver ORDER BY lap_number) as prev_position
            FROM race_results
            WHERE position <= 5
                AND deleted = FALSE
                AND lap_number IS NOT NULL
                AND pit_out_time IS NULL
        ),
        position_changes AS (
            SELECT
                race_id,
                driver,
                lap_number,
                CASE
                    WHEN prev_position IS NOT NULL AND position != prev_position
                    THEN ABS(position - prev_position)
                    ELSE 0
                END as pos_change
            FROM top5_positions
        )
        SELECT
            r.race_id,
            r.year,
            r.race_name,
            c.name as circuit_name,
            c.country,
            AVG(pc.pos_change) as avg_top5_position_change,
            COUNT(CASE WHEN pc.pos_change > 0 THEN 1 END) as top5_changes_count
        FROM position_changes pc
        JOIN races r ON pc.race_id = r.race_id
        JOIN circuits c ON r.circuit_id = c.circuit_id
        GROUP BY r.race_id, r.year, r.race_name, c.name, c.country
        ORDER BY avg_top5_position_change DESC
        """
        return self.conn.execute(query).df()

    def calculate_close_racing(self) -> pd.DataFrame:
        """Calculate closeness of racing within the top ten."""
        query = """
        WITH lap_gaps AS (
            SELECT
                race_id,
                lap_number,
                position,
                time,
                LAG(time) OVER (PARTITION BY race_id, lap_number ORDER BY position) as prev_time
            FROM race_results
            WHERE deleted = FALSE
                AND time IS NOT NULL
                AND position IS NOT NULL
                AND position <= 10
        ),
        gap_stats AS (
            SELECT
                race_id,
                lap_number,
                EXTRACT(EPOCH FROM (time - prev_time)) as gap_seconds
            FROM lap_gaps
            WHERE prev_time IS NOT NULL
        )
        SELECT
            r.race_id,
            r.year,
            r.race_name,
            c.name as circuit_name,
            c.country,
            AVG(gs.gap_seconds) as avg_gap_seconds,
            MIN(gs.gap_seconds) as min_gap_seconds,
            STDDEV(gs.gap_seconds) as gap_volatility
        FROM gap_stats gs
        JOIN races r ON gs.race_id = r.race_id
        JOIN circuits c ON r.circuit_id = c.circuit_id
        WHERE gs.gap_seconds > 0 AND gs.gap_seconds < 100
        GROUP BY r.race_id, r.year, r.race_name, c.name, c.country
        ORDER BY avg_gap_seconds ASC
        """
        return self.conn.execute(query).df()

    def calculate_unpredictability(self) -> pd.DataFrame:
        """Calculate race unpredictability by comparing qualifying and final positions."""
        query = """
        WITH final_positions AS (
            SELECT
                race_id,
                driver,
                position as final_position,
                ROW_NUMBER() OVER (PARTITION BY race_id, driver ORDER BY lap_number DESC) as rn
            FROM race_results
            WHERE position IS NOT NULL
                AND deleted = FALSE
        ),
        race_comparison AS (
            SELECT
                q.race_id,
                q.driver,
                q.position as quali_position,
                fp.final_position,
                ABS(q.position - fp.final_position) as position_change
            FROM qualifying q
            JOIN final_positions fp ON q.race_id = fp.race_id AND q.driver = fp.driver
            WHERE fp.rn = 1 AND q.position IS NOT NULL
        )
        SELECT
            r.race_id,
            r.year,
            r.race_name,
            c.name as circuit_name,
            c.country,
            AVG(rc.position_change) as avg_position_change,
            MAX(rc.position_change) as max_position_change,
            COUNT(*) as drivers_count
        FROM race_comparison rc
        JOIN races r ON rc.race_id = r.race_id
        JOIN circuits c ON r.circuit_id = c.circuit_id
        GROUP BY r.race_id, r.year, r.race_name, c.name, c.country
        ORDER BY avg_position_change DESC
        """
        return self.conn.execute(query).df()

    # python
    def calculate_battle_intensity(self) -> pd.DataFrame:
        """Calculate battle intensity — how many times drivers changed positions (exclude pit-stop laps)."""
        query = """
        WITH position_changes AS (
            SELECT
                race_id,
                driver,
                lap_number,
                position,
                LAG(position) OVER (PARTITION BY race_id, driver ORDER BY lap_number) as prev_position
            FROM race_results
            WHERE position IS NOT NULL
                AND deleted = FALSE
                AND lap_number IS NOT NULL
                AND pit_out_time IS NULL
                AND pit_in_time IS NULL
        ),
        battles AS (
            SELECT
                race_id,
                driver,
                lap_number,
                CASE
                    WHEN prev_position IS NOT NULL AND position != prev_position
                    THEN 1
                    ELSE 0
                END as position_changed
            FROM position_changes
        )
        SELECT
            r.race_id,
            r.year,
            r.race_name,
            c.name as circuit_name,
            c.country,
            SUM(b.position_changed) as total_position_changes,
            AVG(b.position_changed) as avg_position_changes_per_driver
        FROM battles b
        JOIN races r ON b.race_id = r.race_id
        JOIN circuits c ON r.circuit_id = c.circuit_id
        GROUP BY r.race_id, r.year, r.race_name, c.name, c.country
        ORDER BY total_position_changes DESC
        """
        return self.conn.execute(query).df()

    def create_composite_score(self) -> pd.DataFrame:
        """Create a composite attractiveness score from 5 key metrics."""
        # Load only relevant metrics
        lead_changes = self.calculate_lead_changes()
        top5_volatility = self.calculate_top5_volatility()
        close_racing = self.calculate_close_racing()
        unpredictability = self.calculate_unpredictability()
        battle_intensity = self.calculate_battle_intensity()

        # Merge dataframes
        result = lead_changes[['race_id', 'year', 'race_name', 'circuit_name', 'country', 'lead_changes']]

        result = result.merge(
            top5_volatility[['race_id', 'avg_top5_position_change']],
            on='race_id',
            how='left'
        )
        result = result.merge(
            close_racing[['race_id', 'avg_gap_seconds']],
            on='race_id',
            how='left'
        )
        result = result.merge(
            unpredictability[['race_id', 'avg_position_change']],
            on='race_id',
            how='left'
        )
        result = result.merge(
            battle_intensity[['race_id', 'total_position_changes']],
            on='race_id',
            how='left'
        )

        # Normalize metrics to 0-1
        metrics_to_normalize = [
            'lead_changes',
            'avg_top5_position_change',
            'avg_position_change',
            'total_position_changes'
        ]

        # Inverse normalization for avg_gap_seconds (smaller = better)
        if 'avg_gap_seconds' in result.columns:
            min_val = result['avg_gap_seconds'].min()
            max_val = result['avg_gap_seconds'].max()
            if max_val > min_val:
                result['avg_gap_seconds_norm'] = 1 - (result['avg_gap_seconds'] - min_val) / (max_val - min_val)
            else:
                result['avg_gap_seconds_norm'] = 0

        for col in metrics_to_normalize:
            if col in result.columns:
                min_val = result[col].min()
                max_val = result[col].max()
                if max_val > min_val:
                    result[f'{col}_norm'] = (result[col] - min_val) / (max_val - min_val)
                else:
                    result[f'{col}_norm'] = 0

        # Composite score with balanced weights (sum = 1)
        weights = {
            'lead_changes_norm': 0.25,  # Lead changes
            'avg_top5_position_change_norm': 0.20,  # Top5 volatility
            'avg_gap_seconds_norm': 0.15,  # Closeness of racing
            'avg_position_change_norm': 0.20,  # Unpredictability
            'total_position_changes_norm': 0.20  # Battle intensity
        }

        result['attractiveness_score'] = sum(
            result[col] * weight
            for col, weight in weights.items()
            if col in result.columns
        )

        return result.sort_values('attractiveness_score', ascending=False)

    def plot_lead_changes(self, top_n=15):
        """Plot lead changes."""
        df = self.calculate_lead_changes().head(top_n)
        df['race_label'] = df['race_name'] + ' (' + df['year'].astype(str) + ')'

        plt.figure(figsize=(14, 8))
        bars = plt.barh(df['race_label'], df['lead_changes'], color='#FFD700')
        plt.xlabel('Number of lead changes', fontsize=12)
        plt.title('Top races by number of lead changes', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, df['lead_changes'])):
            plt.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{int(val)}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('1_lead_changes.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n=== Chart 1: Lead Changes ===")
        print("This chart shows how many times the race leader changed (position 1).")
        print("For each timestamp I track who is in first place and count changes relative")
        print("to the previous timestamp. More lead changes indicate a more contested victory.\n")

    def plot_top5_volatility(self, top_n=15):
        """Plot top 5 volatility."""
        df = self.calculate_top5_volatility().head(top_n)
        df['race_label'] = df['race_name'] + ' (' + df['year'].astype(str) + ')'

        plt.figure(figsize=(14, 8))
        bars = plt.barh(df['race_label'], df['avg_top5_position_change'], color='#FF6B6B')
        plt.xlabel('Average position change within top 5', fontsize=12)
        plt.title('Top races by top-5 volatility', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, df['avg_top5_position_change'])):
            plt.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('2_top5_volatility.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n=== Chart 2: Top-5 Volatility ===")
        print("This chart measures dynamics within the top 5 positions of the race.")
        print("For drivers in positions 1-5 I track position changes between laps (excluding pit stops)")
        print("and compute the average magnitude of change. Higher values indicate more action at the front.\n")

    def plot_close_racing(self, top_n=15):
        """Plot closeness of racing (time gaps)."""
        df = self.calculate_close_racing().head(top_n)
        df['race_label'] = df['race_name'] + ' (' + df['year'].astype(str) + ')'

        plt.figure(figsize=(14, 8))
        bars = plt.barh(df['race_label'], df['avg_gap_seconds'], color='#4ECDC4')
        plt.xlabel('Average gap between adjacent drivers (seconds)', fontsize=12)
        plt.title('Top races by closeness of racing in Top 10 (smaller = closer)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, df['avg_gap_seconds'])):
            plt.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}s', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('3_close_racing.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n=== Chart 3: Closeness of Racing ===")
        print("This chart shows the average time differences between adjacent drivers in the top ten.")
        print("For each lap I compute the time gap between positions (e.g., 1st vs 2nd, 2nd vs 3rd, etc.)")
        print("and average across the race. Smaller values indicate tighter and more exciting battles.\n")

    def plot_unpredictability(self, top_n=15):
        """Plot unpredictability (qualifying → race)."""
        df = self.calculate_unpredictability().head(top_n)
        df['race_label'] = df['race_name'] + ' (' + df['year'].astype(str) + ')'

        plt.figure(figsize=(14, 8))
        bars = plt.barh(df['race_label'], df['avg_position_change'], color='#9400D3')
        plt.xlabel('Average position change (qualifying → race)', fontsize=12)
        plt.title('Top races by unpredictability of results', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, df['avg_position_change'])):
            plt.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('4_unpredictability.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n=== Chart 4: Unpredictability ===")
        print("This chart compares qualifying positions with final race results.")
        print("For each driver I compute the absolute difference between qualifying and final positions")
        print("and average across drivers. Higher values mean more surprising outcomes.\n")

    def plot_battle_intensity(self, top_n=15):
        """Plot battle intensity (position swaps)."""
        df = self.calculate_battle_intensity().head(top_n)
        df['race_label'] = df['race_name'] + ' (' + df['year'].astype(str) + ')'

        plt.figure(figsize=(14, 8))
        bars = plt.barh(df['race_label'], df['total_position_changes'], color='#00D2BE')
        plt.xlabel('Total number of position changes', fontsize=12)
        plt.title('Top races by battle intensity', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, df['total_position_changes'])):
            plt.text(val + 2, bar.get_y() + bar.get_height() / 2,
                     f'{int(val)}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('5_battle_intensity.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n=== Chart 5: Battle Intensity ===")
        print("This chart counts the total number of position changes in the race (excluding pit-outs).")
        print("For each driver I track how many times their position changed compared to the previous lap.")
        print("A high number indicates an action-packed race with lots of overtakes.\n")

    def plot_composite_score(self, top_n=15):
        """Plot composite attractiveness score."""
        df = self.create_composite_score().head(top_n)
        df['race_label'] = df['race_name'] + ' (' + df['year'].astype(str) + ')'

        plt.figure(figsize=(14, 8))
        bars = plt.barh(df['race_label'], df['attractiveness_score'], color='#003366')
        plt.xlabel('Overall attractiveness score (0-1)', fontsize=12)
        plt.title('Top most attractive F1 races - Composite score', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, df['attractiveness_score'])):
            plt.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('6_composite_score.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n=== Chart 6: Composite Attractiveness Score ===")
        print("This final chart combines all 5 metrics into a single overall score:")
        print("• Lead changes (25%) • Top-5 volatility (20%) • Closeness of racing (15%)")
        print("• Unpredictability (20%) • Battle intensity (20%)")
        print("Each metric is normalized to 0-1 and combined as a weighted average. Scores close to 1")
        print("indicate an extremely attractive race across all dimensions.\n")

    def create_all_visualizations(self, top_n=15):
        """Create all individual charts."""
        self.plot_lead_changes(top_n)
        self.plot_top5_volatility(top_n)
        self.plot_close_racing(top_n)
        self.plot_unpredictability(top_n)
        self.plot_battle_intensity(top_n)
        self.plot_composite_score(top_n)

    def close(self):
        self.conn.close()


# Usage
if __name__ == "__main__":
    analyzer = RaceAttractiveness()

    print("=" * 60)
    print("F1 RACES ATTRACTIVENESS ANALYSIS - 5 METRICS")
    print("=" * 60)

    # Create all charts
    analyzer.create_all_visualizations(top_n=15)

    # Print top 10 races
    composite = analyzer.create_composite_score()
    print("\n" + "=" * 60)
    print("TOP 10 MOST ATTRACTIVE RACES - SUMMARY")
    print("=" * 60)
    print(composite[['year', 'race_name', 'circuit_name', 'lead_changes',
                     'avg_top5_position_change', 'avg_gap_seconds',
                     'avg_position_change', 'total_position_changes',
                     'attractiveness_score']].head(10).to_string(index=False))

    analyzer.close()
