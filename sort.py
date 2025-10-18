import os
import glob
from datetime import datetime


def sort_csv_files():
    # Získej všechny CSV soubory ze složky data
    csv_files = glob.glob("data/*.csv")

    if not csv_files:
        print("Ve složce data nebyly nalezeny žádné CSV soubory.")
        return

    # Seskup soubory podle základního názvu (bez -weather/-Race)
    file_groups = {}

    for file_path in csv_files:
        filename = os.path.basename(file_path)

        # Odstraň příponu a -weather/-Race část
        if filename.endswith("-weather.csv"):
            base_name = filename[:-12]  # odstraň "-weather.csv"
        elif filename.endswith("-Race.csv"):
            base_name = filename[:-9]  # odstraň "-Race.csv"
        else:
            base_name = filename[:-4]  # odstraň jen ".csv"

        # Extrahuj rok ze začátku názvu
        year = base_name[:4] if base_name[:4].isdigit() else "0000"

        # Získej čas poslední úpravy
        mod_time = os.path.getmtime(file_path)

        if base_name not in file_groups:
            file_groups[base_name] = {
                'year': year,
                'mod_time': mod_time,
                'files': []
            }

        file_groups[base_name]['files'].append(filename)
        # Aktualizuj čas na nejnovější z páru souborů
        file_groups[base_name]['mod_time'] = max(file_groups[base_name]['mod_time'], mod_time)

    # Seřaď skupiny podle roku a pak podle času úpravy
    sorted_groups = sorted(file_groups.items(),
                           key=lambda x: (x[1]['year'], x[1]['mod_time']))

    # Vypiš seřazené skupiny a přejmenuj soubory
    print("Seřazené CSV soubory:")
    print("=" * 50)

    for i, (base_name, info) in enumerate(sorted_groups, 1):
        mod_datetime = datetime.fromtimestamp(info['mod_time'])
        print(f"{i}. {base_name} (rok: {info['year']}, změněno: {mod_datetime.strftime('%d.%m.%Y %H:%M:%S')})")

        # Seřaď soubory - nejdříve -Race.csv, pak -weather.csv
        sorted_files = sorted(info['files'], key=lambda x: (0 if x.endswith('-Race.csv') else 1, x))

        for file in sorted_files:
            old_path = os.path.join("data", file)
            new_filename = f"{i:03d}-{file}"
            new_path = os.path.join("data", new_filename)

            try:
                os.rename(old_path, new_path)
                print(f"   - {file} -> {new_filename}")
            except OSError as e:
                print(f"   - Chyba při přejmenování {file}: {e}")
        print()


if __name__ == "__main__":
    sort_csv_files()
