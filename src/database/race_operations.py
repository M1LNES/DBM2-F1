import pandas as pd
from .connection import DatabaseConnection


class RaceOperations:
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection

    def create_race_record(self, year, race_name, file_name, lap_start_date, night_race, city_circuit, circuit_id):
        with self.db.get_cursor() as conn:
            conn.execute("""
                INSERT INTO races (race_id, year, race_name, file_name, race_date, night_race, city_circuit, circuit_id)
                VALUES (nextval('seq_race_id'),?, ?, ?, CAST(? AS DATE), ?, ?, ?)
            """, (year, race_name, file_name, lap_start_date, night_race, city_circuit, circuit_id))

    def get_current_race_id(self):
        with self.db.get_cursor() as conn:
            return conn.execute("SELECT currval('seq_race_id')").fetchone()[0]

    def insert_race(self, year, race_name, file_name, race_date=None, night_race=False, city_circuit=False,
                    circuit_id=None):
        """Insert a new race into the database"""

        race_id = self.connection.execute(
            "SELECT nextval('seq_race_id')"
        ).fetchone()[0]

        self.connection.execute("""
            INSERT INTO races (race_id, year, race_name, file_name, race_date, night_race, city_circuit, circuit_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [race_id, year, race_name, file_name, race_date, night_race, city_circuit, circuit_id])

        return race_id

    def batch_insert_race_results(self, df, race_id):
        # Preparation of data and handling NaN values
        data_rows = []
        for _, row in df.iterrows():
            data_rows.append((
                race_id,
                row['Time'] if pd.notna(row['Time']) else None,
                row['Driver'],
                row['DriverNumber'] if pd.notna(row['DriverNumber']) else None,
                row['LapTime'] if pd.notna(row['LapTime']) else None,
                row['LapNumber'] if pd.notna(row['LapNumber']) else None,
                row['Stint'] if pd.notna(row['Stint']) else None,
                row['PitOutTime'] if pd.notna(row['PitOutTime']) else None,
                row['PitInTime'] if pd.notna(row['PitInTime']) else None,
                row['Sector1Time'] if pd.notna(row['Sector1Time']) else None,
                row['Sector2Time'] if pd.notna(row['Sector2Time']) else None,
                row['Sector3Time'] if pd.notna(row['Sector3Time']) else None,
                row['Sector1SessionTime'] if pd.notna(row['Sector1SessionTime']) else None,
                row['Sector2SessionTime'] if pd.notna(row['Sector2SessionTime']) else None,
                row['Sector3SessionTime'] if pd.notna(row['Sector3SessionTime']) else None,
                row['SpeedI1'] if pd.notna(row['SpeedI1']) else None,
                row['SpeedI2'] if pd.notna(row['SpeedI2']) else None,
                row['SpeedFL'] if pd.notna(row['SpeedFL']) else None,
                row['SpeedST'] if pd.notna(row['SpeedST']) else None,
                row['IsPersonalBest'] if pd.notna(row['IsPersonalBest']) else None,
                row['Compound'] if pd.notna(row['Compound']) else None,
                row['TyreLife'] if pd.notna(row['TyreLife']) else None,
                row['FreshTyre'] if pd.notna(row['FreshTyre']) else None,
                row['Team'] if pd.notna(row['Team']) else None,
                row['LapStartTime'] if pd.notna(row['LapStartTime']) else None,
                row['LapStartDate'] if pd.notna(row['LapStartDate']) else None,
                row['TrackStatus'] if pd.notna(row['TrackStatus']) else None,
                row['Position'] if pd.notna(row['Position']) else None,
                row['Deleted'] if pd.notna(row['Deleted']) else None,
                row['DeletedReason'] if pd.notna(row['DeletedReason']) else None,
                row['FastF1Generated'] if pd.notna(row['FastF1Generated']) else None,
                row['IsAccurate'] if pd.notna(row['IsAccurate']) else None,
                row['LapTime_in_seconds'] if pd.notna(row['LapTime_in_seconds']) else None,
                row['laptime_sum_sectortimes'] if pd.notna(row['laptime_sum_sectortimes']) else None,
            ))

        # Batch insert
        with self.db.get_cursor() as conn:
            conn.executemany("""
                INSERT INTO race_results (
                    result_id, race_id, time, driver, driver_number, lap_time,
                    lap_number, stint, pit_out_time, pit_in_time,
                    sector1_time, sector2_time, sector3_time,
                    sector1_session_time, sector2_session_time, sector3_session_time,
                    speed_i1, speed_i2, speed_fl, speed_st,
                    is_personal_best, compound, tyre_life, fresh_tyre,
                    team, lap_start_time, lap_start_date, track_status,
                    position, deleted, deleted_reason, fast_f1_generated,
                    is_accurate, lap_time_seconds, laptime_sum_sectortimes
                ) VALUES (nextval('seq_result_id'),?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_rows)
