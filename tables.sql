-- Tabulka závodů (metadata)
CREATE TABLE races (
    race_id INTEGER PRIMARY KEY,
    year INTEGER NOT NULL,
    race_name VARCHAR NOT NULL,
    file_name VARCHAR NOT NULL,
    race_date DATE,
    night_race BOOLEAN DEFAULT FALSE,
    city_circuit BOOLEAN DEFAULT FALSE
);

CREATE TABLE race_results (
    result_id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    time INTERVAL NOT NULL, -- time from the beginning of the race
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

CREATE TABLE weather_data (
    weather_id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    time INTERVAL NOT NULL, -- time from the beginning of the race
    air_temp DOUBLE,
    humidity DOUBLE,
    pressure DOUBLE,
    rainfall BOOLEAN,
    track_temp DOUBLE,
    wind_direction INTEGER,
    wind_speed DOUBLE,
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

CREATE TABLE qualifying (
    id INTEGER PRIMARY KEY,
    race_id INTEGER NOT NULL,
    driver VARCHAR NOT NULL,
    position INTEGER,
    q1 VARCHAR,
    q2 VARCHAR,
    q3 VARCHAR,
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

-- Indexes for performance optimization
CREATE INDEX idx_race_results_race_time ON race_results(race_id);
CREATE INDEX idx_race_results_driver ON race_results(race_id, driver);
CREATE INDEX idx_weather_race_time ON weather_data(race_id);
CREATE INDEX idx_qualifying_race ON qualifying(race_id);
CREATE INDEX idx_qualifying_driver ON qualifying(race_id, driver);