import csv
import duckdb


def create_mapping_dictionaries():
    """Vytvoří mapovací slovníky"""

    # 1. Jezdci: external_driver_id -> driver_code
    driver_csv_path = "./data/metadata/drivers.csv"
    target_codes = [
        'AIT', 'ALB', 'ALO', 'ANT', 'BEA', 'BOR', 'BOT', 'COL', 'DEV', 'DOO',
        'FIT', 'GAS', 'GIO', 'GRO', 'HAD', 'HAM', 'HUL', 'KUB', 'KVY', 'LAT',
        'LAW', 'LEC', 'MAG', 'MAZ', 'MSC', 'NOR', 'OCO', 'PER', 'PIA', 'RAI',
        'RIC', 'RUS', 'SAI', 'SAR', 'STR', 'TSU', 'VER', 'VET', 'ZHO'
    ]

    driver_map = {}
    with open(driver_csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            code = row['code'].strip('"') if row['code'] != '\\N' else None
            if code in target_codes:
                driver_map[row['driverId']] = code

    # 2. Závody: their_raceId -> my_race_id
    # Krok 1: jejich races.csv → raceId -> (year, round)
    their_race_to_year_round = {}
    with open('./data/metadata/races.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            race_id = row['raceId']
            year = int(row['year'])
            round_num = int(row['round'])
            their_race_to_year_round[race_id] = (year, round_num)

    # Krok 2: moje DB → (year, round) -> my_race_id
    conn = duckdb.connect("./race_database.db")
    query = "SELECT race_id, year FROM races ORDER BY race_date"
    my_races = conn.execute(query).fetchall()

    year_round_to_my_id = {}
    year_counters = {}

    for my_race_id, year in my_races:
        if year not in year_counters:
            year_counters[year] = 0
        year_counters[year] += 1
        round_num = year_counters[year]
        year_round_to_my_id[(year, round_num)] = my_race_id

    conn.close()

    # Krok 3: spojit → their_raceId -> my_race_id
    race_map = {}
    for their_race_id, (year, round_num) in their_race_to_year_round.items():
        if (year, round_num) in year_round_to_my_id:
            race_map[their_race_id] = year_round_to_my_id[(year, round_num)]

    return driver_map, race_map


def process_qualifying_data(db_path='./race_database.db', csv_path='./data/metadata/qualifying.csv'):
    driver_map, race_map = create_mapping_dictionaries()

    conn = duckdb.connect(db_path)
    inserted = 0
    skipped = 0

    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            their_race_id = row['raceId']
            their_driver_id = row['driverId']

            if their_race_id in race_map and their_driver_id in driver_map:
                my_race_id = race_map[their_race_id]
                driver_code = driver_map[their_driver_id]

                position = int(row['position']) if row['position'] != '\\N' else None
                q1 = row['q1'] if row['q1'] != '\\N' else None
                q2 = row['q2'] if row['q2'] != '\\N' else None
                q3 = row['q3'] if row['q3'] != '\\N' else None

                conn.execute("""
                    INSERT INTO qualifying (id, race_id, driver, position, q1, q2, q3)
                    VALUES (nextval('seq_qual_id'), ?, ?, ?, ?, ?, ?)
                """, (my_race_id, driver_code, position, q1, q2, q3))
                inserted += 1
            else:
                skipped += 1

    conn.close()
    print(f"Inserted: {inserted}, Skipped: {skipped}")


if __name__ == "__main__":
    process_qualifying_data()
