import os

from script.process_qualifying import process_qualifying_data
from src.db_init import create_race_database
from src.database.connection import DatabaseConnection
from src.database.race_operations import RaceOperations
from src.database.weather_operations import WeatherOperations
from src.processors.file_processor import FileProcessor
from script.load_circuits import load_circuits


def ingest_race_data(data_folder_path='./data', db_path='race_database.db'):
    print(f"Starting data ingestion from folder: {data_folder_path}")

    if not os.path.exists(data_folder_path):
        print(f"Folder {data_folder_path} does not exist!")
        return False

    # Initialize database connection and operations
    db_connection = DatabaseConnection(db_path)
    race_ops = RaceOperations(db_connection)
    weather_ops = WeatherOperations(db_connection)
    processor = FileProcessor(race_ops, weather_ops)

    try:
        files = os.listdir(data_folder_path)
        print(f"Found {len(files)} files to ingest.")

        for file in files:
            file_path = os.path.join(data_folder_path, file)

            if file.endswith('weather.csv'):
                print(f'Processing weather file: {file_path}')
                processor.process_weather_file(file_path)
            elif file.endswith('Race.csv'):
                print(f'Processing race-result file: {file_path}')
                processor.process_race_file(file_path)
            else:
                print(f'Skipping unrecognized file: {file_path}')

        print("Data ingestion completed.")
        return True

    finally:
        db_connection.close()


def post_process_data():
    print("\n=== Post-processing: Qualifying data ===")

    try:
        process_qualifying_data()

    except Exception as e:
        print(f"Error running qualifying processor: {e}")


def main():
    print("=== Initialization of DB ===")
    create_race_database()

    print("\n=== Loading circuits ===")
    circuits_ok = load_circuits()

    if not circuits_ok:
        print("Failed to load circuits, aborting!")
        return

    print("\n=== Ingest data ===")
    success = ingest_race_data()

    if success:
        post_process_data()


if __name__ == "__main__":
    main()
