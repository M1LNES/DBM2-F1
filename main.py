import os

import duckdb
import pandas as pd
from datetime import datetime
from src.db_init import create_race_database


def is_city_circuit(race_name):
    names = ['Monaco', 'Singapore', 'Azerbaijan', 'Saudi', 'Australian', 'Vegas', 'Miami', 'Canadian']

    for name in names:
        if name in race_name:
            return True
    return False

def is_night_race(race_name):
    names = ['Bahrain', 'Saudi Arabian', 'Abu Dhabi', 'Singapore', 'Qatar', 'Las Vegas', 'Sakhir']
    for name in names:
        if name in race_name:
            return True
    return False

def create_race_record(file_path, db_path):
    file_name = os.path.basename(file_path)
    year = file_name.split('-')[1]
    race_name = file_name.split('-')[2]
    df = pd.read_csv(file_path, nrows=1)
    lap_start_date = df['LapStartDate'].iloc[0]
    city_circuit = is_city_circuit(race_name)
    night_race = is_night_race(race_name)

    conn = duckdb.connect(db_path)
    conn.execute("""
        INSERT INTO races (race_id, year, race_name, file_name, race_date, night_race, city_circuit)
        VALUES (nextval('seq_race_id'),?, ?, ?, CAST(? AS DATE), ?, ?)
    """, (year, race_name, file_name, lap_start_date, night_race, city_circuit))
    conn.close()
    pass


def create_race_results_records(file_path, db_path):
    conn = duckdb.connect(db_path)

    try:
        # Získání aktuálního race_id ze sekvence
        current_race_id = conn.execute("SELECT currval('seq_race_id')").fetchone()[0] - 1

        # Načtení CSV souboru
        df = pd.read_csv(file_path)

        # Vložení dat do tabulky race_results
        for _, row in df.iterrows():
            conn.execute("""
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
            """, (
                current_race_id,
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

        print(f"Vloženo {len(df)} záznamů pro race_id: {current_race_id}")

    except Exception as e:
        print(f"Chyba při vkládání výsledků závodu: {e}")
    finally:
        conn.close()



def process_race_result_file(file_path, db_path):
    # Firstly, create the record in races table
    create_race_record(file_path, db_path)

    # And now, create records for race results
    create_race_results_records(file_path, db_path)
    pass


def create_weather_records(file_path, db_path):
    conn = duckdb.connect(db_path)

    try:
        # Získání aktuálního race_id ze sekvence
        current_race_id = conn.execute("SELECT currval('seq_race_id')").fetchone()[0] - 1
        print("Aktuální race id: ", current_race_id)

        # Načtení CSV souboru
        df = pd.read_csv(file_path)

        # Odebrání duplicit podle prvního sloupce (index)
        # Předpokládám, že první sloupec je unnamed index
        first_column = df.columns[0]
        df_unique = df.drop_duplicates(subset=[first_column], keep='first')

        print(f"Původní počet záznamů: {len(df)}, po odstranění duplicit: {len(df_unique)}")

        # Vložení dat do tabulky weather_data
        for _, row in df_unique.iterrows():
            conn.execute("""
                INSERT INTO weather_data (
                    weather_id, race_id, time, air_temp, humidity, pressure,
                    rainfall, track_temp, wind_direction, wind_speed
                ) VALUES (nextval('seq_weather_id'),?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                current_race_id,
                row['Time'] if pd.notna(row['Time']) else None,
                row['AirTemp'] if pd.notna(row['AirTemp']) else None,
                row['Humidity'] if pd.notna(row['Humidity']) else None,
                row['Pressure'] if pd.notna(row['Pressure']) else None,
                row['Rainfall'] if pd.notna(row['Rainfall']) else None,
                row['TrackTemp'] if pd.notna(row['TrackTemp']) else None,
                row['WindDirection'] if pd.notna(row['WindDirection']) else None,
                row['WindSpeed'] if pd.notna(row['WindSpeed']) else None,
            ))

        print(f"Vloženo {len(df_unique)} unikátních záznamů počasí pro race_id: {current_race_id}")

    except Exception as e:
        print(f"Chyba při vkládání dat o počasí: {e}")
    finally:
        conn.close()


def process_file(file_path, db_path):
    """Function for processing a single file and ingesting its data into the database."""
    if (file_path.endswith('weather.csv')):
        print('Processing weather file :', file_path)
        create_weather_records(file_path, db_path)
    else:
        print(f'Processing race-result file: {file_path}')
        process_race_result_file(file_path, db_path)
    pass


def ingest_race_data(data_folder_path='./data', db_path='race_database.db'):
    """Function for ingesting"""
    print(f"Starting data ingestion from folder : {data_folder_path}")

    if not os.path.exists(data_folder_path):
        print(f"Folder {data_folder_path} does not exist!")
        return False


    files = os.listdir(data_folder_path)
    print(f"Find {len(files)} files to ingest.")

    for file in files:
        process_file(os.path.join(data_folder_path, file), db_path)

    print("Data ingestion completed.")
    return True


def main():
    print("=== Initialization of DB ===")

    # 1. Create DB schema
    create_race_database()

    # 2. Ingest data
    print("\n=== Ingest data ===")
    ingest_race_data()


if __name__ == "__main__":
    main()
