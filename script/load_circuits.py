import csv
import duckdb


def load_circuits(db_path='race_database.db', csv_path='./data/metadata/circuits.csv'):
    """Načte okruhy do tabulky circuits PŘED nahráním závodů"""
    print("Loading circuits...")

    conn = duckdb.connect(db_path)

    try:
        inserted = 0
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                circuit_id = int(row['circuitId'])
                circuit_ref = row['circuitRef'].strip('"')
                name = row['name'].strip('"')
                location = row['location'].strip('"')
                country = row['country'].strip('"')
                lat = float(row['lat']) if row['lat'] else None
                lng = float(row['lng']) if row['lng'] else None
                alt = int(row['alt']) if row['alt'] and row['alt'] != '\\N' else None

                conn.execute("""
                    INSERT INTO circuits (circuit_id, circuit_ref, name, location, country, lat, lng, alt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (circuit_id, circuit_ref, name, location, country, lat, lng, alt))
                inserted += 1

        print(f"Inserted {inserted} circuits")
        return True

    except Exception as e:
        print(f"Error loading circuits: {e}")
        return False

    finally:
        conn.close()


if __name__ == "__main__":
    load_circuits()
