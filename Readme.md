# F1 Race Data Analysis Project

## Overview

This project analyzes Formula 1 race data from the 2020-2025 (Singapore) seasons, including race results, qualifying sessions, weather conditions, and circuit characteristics. The analysis includes data processing, database creation, and various statistical visualizations to understand race dynamics, driver performance, and the impact of different factors on race outcomes.

## Requirements

### Python Version
- Python 3.8+

### Dependencies
```bash
pip install duckdb pandas matplotlib seaborn numpy scikit-learn
```

### Optional Dependencies (for ML analysis)
```bash
pip install xgboost lightgbm
```

## Data Setup

### ⚠️ Important: Data Download Required

The `/data` folder is **NOT included** in this repository due to its size (~50 MB). 

**Please download the data from Google Drive:**

🔗 **[Download Data Folder from Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)**

After downloading:
1. Extract the ZIP file
2. Place the `data` folder in the project root directory
3. Verify the structure:
   ```
   DBM2-sem/
   ├── data/
   │   ├── 001-2020-Austrian Grand Prix-Race.csv
   │   ├── 002-2020-Austrian Grand Prix-Race-weather.csv
   │   └── ... (more CSV files)
   ├── main.py
   └── ...
   ```

## Getting Started

### 1. Initialize Database and Import Data

Run the main initialization script to create the database and import all race data:

```bash
python main.py
```

This will:
- Create the database schema (`race_database.db`)
- Load circuit information
- Import all race and weather CSV files from `/data`
- Post-process qualifying data

⏱️ **Time estimate:** 5-15 minutes depending on your system.

### 2. Run Visualizations

After database initialization, run visualization scripts from the `/visualisation` folder:

```bash
cd visualisation

# Race attractiveness analysis (6 charts)
python 1_atractivity.py

# Tire stint analysis (2 charts)
python 2_tire_stint.py

# Tire lap time analysis (2 charts)
python 3_tire_lap_time.py

# Qualifying correlation analysis (5 charts)
python 4_qualy_correlation.py

# General correlation analysis (3 charts)
python 5_correlation.py

# Driver performance by race conditions (3 charts)
python 5_drivers_night.py

# Driver performance by weather (3 charts)
python 5_drivers_weather.py

# Machine learning predictions (7+ charts)
python 6_machine_learning.py
```

### Run All Visualizations at Once

```bash
cd visualisation
python 1_atractivity.py && python 2_tire_stint.py && python 3_tire_lap_time.py && python 4_qualy_correlation.py && python 5_correlation.py && python 5_drivers_night.py && python 5_drivers_weather.py && python 6_machine_learning.py
```

All charts are saved as PNG files in the current directory or `/plots` subdirectory.

## Project Structure

```
DBM2-sem/
├── main.py                    # Main initialization script
├── race_database.db           # SQLite database (created after initialization)
├── data/                      # Raw CSV data files (download separately)
├── script/
│   ├── load_circuits.py       # Circuit data loader
│   └── process_qualifying.py  # Qualifying data processor
├── src/
│   ├── db_init.py            # Database schema creation
│   ├── database/             # Database operations
│   └── processors/           # File processors
└── visualisation/            # Analysis and visualization scripts
    ├── 1_atractivity.py      # Race attractiveness analysis
    ├── 2_tire_stint.py       # Tire stint analysis
    ├── 3_tire_lap_time.py    # Tire lap time analysis
    ├── 4_qualy_correlation.py # Qualifying correlation analysis
    ├── 5_correlation.py      # General correlation analysis
    ├── 5_drivers_night.py    # Driver performance by race conditions
    ├── 5_drivers_weather.py  # Driver performance by weather
    └── 6_machine_learning.py # ML prediction models
```

## Database Schema

The database includes the following main tables:

- **circuits** - F1 circuit information (name, location, type)
- **races** - Race metadata (year, race name, circuit)
- **race_results** - Lap-by-lap race data (position, lap times, tire info)
- **qualifying** - Qualifying session results
- **weather_data** - Weather conditions during races
