import duckdb
import csv


def get_race_external_ids(db_path='../race_database.db', csv_path='../data/metadata/races.csv'):
    """Vytvoří mapu race_id -> external_id (raceId z CSV)"""
    race_map = {}

    # Načtení dat z CSV souboru
    external_races = {}
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            year = int(row['year'])
            round_num = int(row['round'])
            external_id = row['raceId']

            # Klíč pro porovnání: (rok, kolo)
            key = (year, round_num)
            external_races[key] = external_id

    # Načtení dat z databáze (seřazeno chronologicky)
    conn = duckdb.connect(db_path)

    query = "SELECT race_id, year, race_name, race_date FROM races ORDER BY race_date"
    db_races = conn.execute(query).fetchall()

    # Mapování podle pozice v roce
    year_counters = {}

    for race_id, year, race_name, race_date in db_races:
        if year not in year_counters:
            year_counters[year] = 0

        year_counters[year] += 1
        round_num = year_counters[year]

        key = (year, round_num)

        if key in external_races:
            race_map[race_id] = external_races[key]
            # print(f"Mapováno: {race_id} -> {external_races[key]} ({year} Round {round_num}: {race_name})")
        else:
            print(f"Nenalezen external ID pro: {race_id} - {year} Round {round_num} {race_name}")

    conn.close()
    return race_map


if __name__ == "__main__":
    try:
        results = get_race_external_ids()

        print("\nMapa race_id -> external_id:")
        print("-" * 40)

        for race_id in sorted(results.keys()):
            external_id = results[race_id]
            print(f'"{race_id}":"{external_id}"')

        print(f"\nCelkem namapováno: {len(results)} závodů")

    except FileNotFoundError as e:
        print(f"Soubor nebyl nalezen: {e}")
    except Exception as e:
        print(f"Chyba při zpracování: {e}")
