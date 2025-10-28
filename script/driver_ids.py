import csv


def find_driver_ids(csv_file_path, driver_codes):
    """Najde driverId pro zadané kódy jezdců"""
    driver_map = {}

    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            code = row['code'].strip('"') if row['code'] != '\\N' else None
            if code in driver_codes:
                driver_map[code] = row['driverId']

    return driver_map


# Seznam kódů jezdců
target_codes = [
    'AIT', 'ALB', 'ALO', 'ANT', 'BEA', 'BOR', 'BOT', 'COL', 'DEV', 'DOO',
    'FIT', 'GAS', 'GIO', 'GRO', 'HAD', 'HAM', 'HUL', 'KUB', 'KVY', 'LAT',
    'LAW', 'LEC', 'MAG', 'MAZ', 'MSC', 'NOR', 'OCO', 'PER', 'PIA', 'RAI',
    'RIC', 'RUS', 'SAI', 'SAR', 'STR', 'TSU', 'VER', 'VET', 'ZHO'
]

# Spuštění skriptu
if __name__ == "__main__":
    csv_path = "./data/metadata/drivers.csv"

    try:
        results = find_driver_ids(csv_path, target_codes)

        # Výpis pouze kombinace code:driverId
        for code in target_codes:
            if code in results:
                print(f'"{code}":"{results[code]}"')

    except FileNotFoundError:
        print(f"Soubor {csv_path} nebyl nalezen!")
    except Exception as e:
        print(f"Chyba při zpracování: {e}")
