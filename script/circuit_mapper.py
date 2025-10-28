import csv
import duckdb


def create_circuit_mapper(circuits_csv='./data/metadata/circuits.csv'):
    """Vytvoří mapovací slovník pro circuit_id podle města/země v názvu závodu"""

    circuits = {}
    with open(circuits_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            circuit_id = int(row['circuitId'])
            name = row['name'].strip('"')
            location = row['location'].strip('"')
            country = row['country'].strip('"')
            circuits[circuit_id] = {
                'name': name,
                'location': location,
                'country': country,
                'ref': row['circuitRef'].strip('"')
            }

    keyword_to_circuit = {
        # Města
        'Singapore': 15,
        'São Paulo': 18,
        'Săo Paulo': 18,
        'Miami': 79,
        'Abu Dhabi': 24,
        'Las Vegas': 80,

        # Země
        'Canada': 7,
        'Canadian': 7,
        'Spain': 4,
        'Spanish': 4,
        'Japan': 22,
        'Japanese': 22,
        'Australia': 1,
        'Australian': 1,
        'France': 34,
        'French': 34,
        'Portugal': 27,
        'Portuguese': 27,
        'Monaco': 6,
        'China': 17,
        'Chinese': 17,
        'Saudi': 77,
        'Austria': 70,
        'Austrian': 70,
        'Azerbaijan': 73,
        'Turkey': 5,
        'Turkish': 5,
        'Netherlands': 39,
        'Dutch': 39,
        'Belgium': 13,
        'Belgian': 13,
        'Hungary': 11,
        'Hungarian': 11,
        'Mexico': 32,
        'Mexican': 32,
        'Qatar': 78,
        'Bahrain': 3,

        # Specifické závody
        'United States Grand Prix': 69,
        'Styrian': 70,
        'British': 9,
        '70th Anniversary': 9,
        'Italian': 14,
        'Tuscan': 76,
        'Russian': 71,
        'Eifel': 20,
        'Emilia Romagna': 21,
        'Sakhir': 3,
    }

    return circuits, keyword_to_circuit


def find_circuit_id(race_name, keyword_to_circuit):
    """Najde circuit_id podle klíčového slova v názvu závodu"""
    race_name_lower = race_name.lower()

    # Nejprve úplná shoda
    for keyword, circuit_id in keyword_to_circuit.items():
        if keyword.lower() == race_name_lower:
            return circuit_id

    # Pak částečná shoda
    for keyword, circuit_id in keyword_to_circuit.items():
        if keyword.lower() in race_name_lower:
            return circuit_id

    return None


def update_race_circuits(db_path='../race_database.db'):
    """Aktualizuje circuit_id v tabulce races podle mapování"""

    circuits, keyword_to_circuit = create_circuit_mapper()

    conn = duckdb.connect(db_path)

    try:
        # Dočasně vypni kontrolu cizích klíčů
        conn.execute("SET enable_object_cache=false;")

        races = conn.execute("SELECT race_id, race_name FROM races").fetchall()

        updated = 0
        unmapped = []

        for race_id, race_name in races:
            circuit_id = find_circuit_id(race_name, keyword_to_circuit)

            if circuit_id:
                conn.execute(
                    "UPDATE races SET circuit_id = ? WHERE race_id = ?",
                    [circuit_id, race_id]
                )
                updated += 1
            else:
                unmapped.append((race_id, race_name))

        conn.commit()

        print(f"\n{'=' * 80}")
        print(f"Updated: {updated}/{len(races)} races")

        if unmapped:
            print(f"\n⚠ Unmapped races ({len(unmapped)}):")
            for race_id, race_name in unmapped:
                print(f"  - {race_id}: {race_name}")

        return updated, len(unmapped)

    finally:
        conn.close()

if __name__ == "__main__":
    print("=== Updating circuit_id in races table ===\n")
    updated, unmapped = update_race_circuits()

    if unmapped == 0:
        print("\n✓ All races successfully mapped!")
    else:
        print(f"\n⚠ {unmapped} races need manual mapping")
