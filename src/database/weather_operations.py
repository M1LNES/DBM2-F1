import pandas as pd
from .connection import DatabaseConnection


class WeatherOperations:
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection

    def batch_insert_weather_data(self, df, race_id):
        # Remove duplicate records based on the first column
        first_column = df.columns[0]
        df_unique = df.drop_duplicates(subset=[first_column], keep='first')

        print(f"Original records: {len(df)}, after duplicates removed: {len(df_unique)}")

        # Prepare data for insertion
        data_rows = []
        for _, row in df_unique.iterrows():
            data_rows.append((
                race_id,
                row['Time'] if pd.notna(row['Time']) else None,
                row['AirTemp'] if pd.notna(row['AirTemp']) else None,
                row['Humidity'] if pd.notna(row['Humidity']) else None,
                row['Pressure'] if pd.notna(row['Pressure']) else None,
                row['Rainfall'] if pd.notna(row['Rainfall']) else None,
                row['TrackTemp'] if pd.notna(row['TrackTemp']) else None,
                row['WindDirection'] if pd.notna(row['WindDirection']) else None,
                row['WindSpeed'] if pd.notna(row['WindSpeed']) else None,
            ))

        # Batch insert
        with self.db.get_cursor() as conn:
            conn.executemany("""
                INSERT INTO weather_data (
                    weather_id, race_id, time, air_temp, humidity, pressure,
                    rainfall, track_temp, wind_direction, wind_speed
                ) VALUES (nextval('seq_weather_id'),?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_rows)

        return len(df_unique)
