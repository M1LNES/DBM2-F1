# data_processor.py
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime


class F1DataProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect('f1_analysis.db')

        # Metadata pro závody
        self.circuit_metadata = {
            'Monaco': {'city_circuit': True, 'night_race': False},
            'Singapore': {'city_circuit': True, 'night_race': True},
            'Bahrain': {'city_circuit': False, 'night_race': False},
            'Abu Dhabi': {'city_circuit': False, 'night_race': True},
            'Jeddah': {'city_circuit': True, 'night_race': True},
            # Přidat další tratě
        }

    def extract_race_info(self, filename):
        """Extrahuje informace ze jména souboru"""
        parts = filename.stem.split('-')
        year = parts[0]
        race_name = parts[1].replace('%20', ' ')
        session_type = parts[2].split('.')[0]

        return {
            'year': int(year),
            'race_name': race_name,
            'session_type': session_type,
            'filename': filename.name
        }

    def get_circuit_type(self, race_name):
        """Určí typ okruhu"""
        for circuit, metadata in self.circuit_metadata.items():
            if circuit.lower() in race_name.lower():
                return metadata
        return {'city_circuit': False, 'night_race': False}

    def process_race_order(self, races_df):
        """Přidá pořadí závodů v rámci sezóny"""
        races_df = races_df.sort_values(['year', 'race_date'])
        races_df['race_order'] = races_df.groupby('year').cumcount() + 1
        return races_df

    def load_laps_data(self):
        """Načte data o kolech všech jezdců"""
        lap_files = list(self.data_dir.glob("*-Race.csv"))

        all_laps = []
        for file in lap_files:
            race_info = self.extract_race_info(file)
            circuit_info = self.get_circuit_type(race_info['race_name'])

            df = pd.read_csv(file)
            df['year'] = race_info['year']
            df['race_name'] = race_info['race_name']
            df['city_circuit'] = circuit_info['city_circuit']
            df['night_race'] = circuit_info['night_race']

            all_laps.append(df)

        combined = pd.concat(all_laps, ignore_index=True)

        # Vytvoření race_id a přidání race_order
        combined['race_date'] = pd.to_datetime(combined['LapStartDate'])
        combined = self.process_race_order(combined)

        return combined

    def load_weather_data(self):
        """Načte data o počasí"""
        weather_files = list(self.data_dir.glob("*-Weather.csv"))

        all_weather = []
        for file in weather_files:
            race_info = self.extract_race_info(file)

            df = pd.read_csv(file)
            df['year'] = race_info['year']
            df['race_name'] = race_info['race_name']
            df['weather_time'] = pd.to_timedelta(df['Time'])

            all_weather.append(df)

        return pd.concat(all_weather, ignore_index=True)

    def merge_lap_weather(self, laps_df, weather_df):
        """Spojí data kol s počasím podle času"""
        # Převod na DuckDB pro efektivní JOIN
        self.conn.execute("DROP TABLE IF EXISTS laps")
        self.conn.execute("DROP TABLE IF EXISTS weather")

        self.conn.register('laps', laps_df)
        self.conn.register('weather', weather_df)

        query = """
        SELECT 
            l.*,
            w.AirTemp,
            w.Humidity,
            w.Pressure,
            w.Rainfall,
            w.TrackTemp,
            w.WindSpeed
        FROM laps l
        LEFT JOIN LATERAL (
            SELECT *
            FROM weather w
            WHERE w.race_name = l.race_name 
              AND w.year = l.year
              AND w.weather_time <= l.Time
            ORDER BY w.weather_time DESC
            LIMIT 1
        ) w ON true
        """

        result = self.conn.execute(query).fetchdf()
        return result

    def create_aggregated_views(self):
        """Vytvoří agregované pohledy pro analýzu"""

        # Průměrné časy kol podle jezdce a závodu
        self.conn.execute("""
        CREATE OR REPLACE VIEW driver_race_performance AS
        SELECT 
            Driver,
            Team,
            race_name,
            year,
            race_order,
            city_circuit,
            night_race,
            COUNT(*) as laps_completed,
            AVG(LapTime_in_seconds) as avg_lap_time,
            MIN(LapTime_in_seconds) as best_lap_time,
            AVG(TrackTemp) as avg_track_temp,
            AVG(AirTemp) as avg_air_temp,
            MAX(CAST(Rainfall AS INTEGER)) as had_rain
        FROM merged_data
        WHERE LapTime_in_seconds IS NOT NULL
        GROUP BY Driver, Team, race_name, year, race_order, city_circuit, night_race
        """)

        # Analýza pneumatik
        self.conn.execute("""
        CREATE OR REPLACE VIEW tyre_performance AS
        SELECT 
            Compound,
            TyreLife,
            city_circuit,
            AVG(LapTime_in_seconds) as avg_lap_time,
            COUNT(*) as usage_count,
            AVG(TrackTemp) as avg_track_temp
        FROM merged_data
        WHERE LapTime_in_seconds IS NOT NULL
          AND Compound IS NOT NULL
        GROUP BY Compound, TyreLife, city_circuit
        """)

    def run_analysis(self):
        """Hlavní analýza"""
        print("Načítám data kol...")
        laps = self.load_laps_data()

        print("Načítám data o počasí...")
        weather = self.load_weather_data()

        print("Spojujem data...")
        merged = self.merge_lap_weather(laps, weather)

        self.conn.register('merged_data', merged)

        print("Vytvářím agregované pohledy...")
        self.create_aggregated_views()

        print(f"Zpracováno {len(merged)} kol z {merged['race_name'].nunique()} závodů")

        return merged

    def export_to_csv(self, output_dir="processed_data"):
        """Export zpracovaných dat"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Export hlavní tabulky
        self.conn.execute("COPY merged_data TO 'processed_data/merged_data.csv' (HEADER, DELIMITER ',')")

        # Export agregovaných pohledů
        self.conn.execute(
            "COPY driver_race_performance TO 'processed_data/driver_performance.csv' (HEADER, DELIMITER ',')")
        self.conn.execute("COPY tyre_performance TO 'processed_data/tyre_analysis.csv' (HEADER, DELIMITER ',')")

        print("Data exportována do složky processed_data/")


if __name__ == "__main__":
    processor = F1DataProcessor()
    merged_data = processor.run_analysis()
    processor.export_to_csv()

    # Ukázkový dotaz
    print("\n=== Ukázkový dotaz: Top 5 jezdců podle průměrného času ===")
    result = processor.conn.execute("""
    SELECT Driver, Team, AVG(avg_lap_time) as overall_avg
    FROM driver_race_performance
    GROUP BY Driver, Team
    ORDER BY overall_avg
    LIMIT 5
    """).fetchdf()
    print(result)
