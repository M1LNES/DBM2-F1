import duckdb


def create_race_database(db_path='race_database.db'):
    """Creates DuckDB database"""
    # Create a connection to DuckDB
    conn = duckdb.connect(db_path)

    try:
        # Creating table for races
        conn.execute("""
        CREATE TABLE races (
            race_id INTEGER PRIMARY KEY,
            year INTEGER NOT NULL,
            race_name VARCHAR NOT NULL,
            file_name VARCHAR NOT NULL,
            race_date DATE,
            night_race BOOLEAN DEFAULT FALSE,
            city_circuit BOOLEAN DEFAULT FALSE
        );
        """)

        # Creating table for race results
        conn.execute("""
        CREATE TABLE race_results (
            result_id INTEGER PRIMARY KEY,
            race_id INTEGER NOT NULL,
            time INTERVAL NOT NULL,
            driver VARCHAR NOT NULL,
            driver_number INTEGER NOT NULL,
            lap_time INTERVAL,
            lap_number DOUBLE,
            stint DOUBLE,
            pit_out_time INTERVAL,
            pit_in_time INTERVAL,
            sector1_time DOUBLE,
            sector2_time DOUBLE,
            sector3_time DOUBLE,
            sector1_session_time INTERVAL,
            sector2_session_time INTERVAL,
            sector3_session_time INTERVAL,
            speed_i1 DOUBLE,
            speed_i2 DOUBLE,
            speed_fl DOUBLE,
            speed_st DOUBLE,
            is_personal_best BOOLEAN,
            compound VARCHAR,
            tyre_life DOUBLE,
            fresh_tyre BOOLEAN,
            team VARCHAR,
            lap_start_time INTERVAL,
            lap_start_date TIMESTAMP,
            track_status INTEGER,
            position DOUBLE,
            deleted BOOLEAN,
            deleted_reason VARCHAR,
            fast_f1_generated BOOLEAN,
            is_accurate BOOLEAN,
            lap_time_seconds DOUBLE,
            laptime_sum_sectortimes DOUBLE,
            FOREIGN KEY (race_id) REFERENCES races(race_id)
        );
        """)

        # Creating table for weather data
        conn.execute("""
        CREATE TABLE weather_data (
            weather_id INTEGER PRIMARY KEY,
            race_id INTEGER NOT NULL,
            time INTERVAL NOT NULL,
            air_temp DOUBLE,
            humidity DOUBLE,
            pressure DOUBLE,
            rainfall BOOLEAN,
            track_temp DOUBLE,
            wind_direction INTEGER,
            wind_speed DOUBLE,
            FOREIGN KEY (race_id) REFERENCES races(race_id)
        );
        """)

        conn.execute("CREATE SEQUENCE seq_race_id START 1;")
        conn.execute("CREATE SEQUENCE seq_weather_id START 1;")
        conn.execute("CREATE SEQUENCE seq_result_id START 1;")


        # Creating indexes for performance optimization
        conn.execute("CREATE INDEX idx_race_results_race_id ON race_results(race_id);")
        conn.execute("CREATE INDEX idx_race_results_driver ON race_results(race_id, driver);")
        conn.execute("CREATE INDEX idx_weather_race_id ON weather_data(race_id);")

        print("Database was created successfully.")

    finally:
        conn.close()


if __name__ == "__main__":
    create_race_database()
