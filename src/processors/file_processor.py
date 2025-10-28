import os
import pandas as pd
from ..database.race_operations import RaceOperations
from ..database.weather_operations import WeatherOperations
from ..utils.race_utils import is_city_circuit, is_night_race
from script.circuit_mapper import create_circuit_mapper, find_circuit_id


class FileProcessor:
    def __init__(self, race_ops: RaceOperations, weather_ops: WeatherOperations):
        self.race_ops = race_ops
        self.weather_ops = weather_ops
        self.circuits, self.keyword_to_circuit = create_circuit_mapper()


    def process_race_file(self, file_path):
        try:
            # Create race record
            file_name = os.path.basename(file_path)
            year = file_name.split('-')[1]
            race_name = file_name.split('-')[2]

            df = pd.read_csv(file_path, nrows=1)
            lap_start_date = df['LapStartDate'].iloc[0]
            city_circuit = is_city_circuit(race_name)
            night_race = is_night_race(race_name)
            circuit_id = find_circuit_id(race_name, self.keyword_to_circuit)

            self.race_ops.create_race_record(year, race_name, file_name, lap_start_date, night_race, city_circuit, circuit_id)

            # Ingest race results
            df_full = pd.read_csv(file_path)
            current_race_id = self.race_ops.get_current_race_id()
            self.race_ops.batch_insert_race_results(df_full, current_race_id)

            print(f"Ingested {len(df_full)} records for race_id: {current_race_id}")

        except Exception as e:
            print(f"Error during processing file {file_path}: {e}")

    def process_weather_file(self, file_path):
        try:
            df = pd.read_csv(file_path)
            current_race_id = self.race_ops.get_current_race_id()
            count = self.weather_ops.batch_insert_weather_data(df, current_race_id)

            print(f"Ingested {count} unique weather records for race_id: {current_race_id}")

        except Exception as e:
            print(f"Error during processing weather file {file_path}: {e}")
