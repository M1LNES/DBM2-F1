# simple_viz.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Nastavení stylu
plt.style.use('default')
sns.set_palette("husl")


def load_and_visualize_single_file(filename):
    """Načte a vizualizuje jeden CSV soubor"""

    # Načtení dat
    df = pd.read_csv(filename)
    print(f"Načteno {len(df)} řádků ze souboru {filename}")
    print(f"Sloupce: {list(df.columns)}")
    print(f"Základní info:")
    print(df.info())

    # Určení typu dat podle názvů sloupců
    if 'Driver' in df.columns:
        # Data o kolech jezdců
        visualize_lap_data(df)
    elif 'AirTemp' in df.columns:
        # Data o počasí
        visualize_weather_data(df)
    else:
        print("Neznámý typ dat - zobrazuji základní přehled")
        print(df.head())


def visualize_lap_data(df):
    """Vizualizace dat o kolech"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Graf 1: Časy kol jednotlivých jezdců (prvních 10)
    if 'LapTime_in_seconds' in df.columns and 'Driver' in df.columns:
        top_drivers = df['Driver'].value_counts().head(10).index
        lap_data = df[df['Driver'].isin(top_drivers)]

        axes[0, 0].boxplot([lap_data[lap_data['Driver'] == driver]['LapTime_in_seconds'].dropna()
                            for driver in top_drivers],
                           labels=top_drivers)
        axes[0, 0].set_title('Rozložení časů kol podle jezdců')
        axes[0, 0].set_ylabel('Čas kola (s)')
        axes[0, 0].tick_params(axis='x', rotation=45)

    # Graf 2: Histogram časů kol
    if 'LapTime_in_seconds' in df.columns:
        valid_times = df['LapTime_in_seconds'].dropna()
        axes[0, 1].hist(valid_times, bins=30, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Distribuce časů kol')
        axes[0, 1].set_xlabel('Čas kola (s)')
        axes[0, 1].set_ylabel('Počet kol')

    # Graf 3: Vývoj času během závodu
    if 'LapNumber' in df.columns and 'LapTime_in_seconds' in df.columns:
        # Vezmu jen prvních 5 jezdců pro čitelnost
        sample_drivers = df['Driver'].value_counts().head(5).index
        for driver in sample_drivers:
            driver_data = df[df['Driver'] == driver].sort_values('LapNumber')
            axes[1, 0].plot(driver_data['LapNumber'],
                            driver_data['LapTime_in_seconds'],
                            marker='o', label=driver, alpha=0.7)

        axes[1, 0].set_title('Vývoj časů kol během závodu')
        axes[1, 0].set_xlabel('Číslo kola')
        axes[1, 0].set_ylabel('Čas kola (s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Graf 4: Analýza pneumatik (pokud jsou data)
    if 'Compound' in df.columns and 'LapTime_in_seconds' in df.columns:
        compound_data = df.groupby('Compound')['LapTime_in_seconds'].mean().dropna()
        axes[1, 1].bar(compound_data.index, compound_data.values, color='coral')
        axes[1, 1].set_title('Průměrné časy podle směsi pneumatik')
        axes[1, 1].set_ylabel('Průměrný čas kola (s)')
        axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Základní statistiky
    print("\n=== ZÁKLADNÍ STATISTIKY ===")
    if 'LapTime_in_seconds' in df.columns:
        print(f"Nejrychlejší kolo: {df['LapTime_in_seconds'].min():.3f}s")
        print(f"Nejpomalejší kolo: {df['LapTime_in_seconds'].max():.3f}s")
        print(f"Průměrný čas: {df['LapTime_in_seconds'].mean():.3f}s")

    print(f"Počet jezdců: {df['Driver'].nunique()}")
    print(f"Celkem kol: {len(df)}")


def visualize_weather_data(df):
    """Vizualizace dat o počasí"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Graf 1: Teplota vzduchu v čase
    if 'Time' in df.columns and 'AirTemp' in df.columns:
        df['Time_minutes'] = pd.to_timedelta(df['Time']).dt.total_seconds() / 60
        axes[0, 0].plot(df['Time_minutes'], df['AirTemp'], color='red', alpha=0.7)
        axes[0, 0].set_title('Teplota vzduchu během závodu')
        axes[0, 0].set_xlabel('Čas (minuty)')
        axes[0, 0].set_ylabel('Teplota vzduchu (°C)')
        axes[0, 0].grid(True, alpha=0.3)

    # Graf 2: Teplota trati vs. vzduchu
    if 'TrackTemp' in df.columns and 'AirTemp' in df.columns:
        axes[0, 1].scatter(df['AirTemp'], df['TrackTemp'], alpha=0.6, color='orange')
        axes[0, 1].set_title('Teplota trati vs. teplota vzduchu')
        axes[0, 1].set_xlabel('Teplota vzduchu (°C)')
        axes[0, 1].set_ylabel('Teplota trati (°C)')
        axes[0, 1].grid(True, alpha=0.3)

    # Graf 3: Vlhkost a tlak
    if 'Humidity' in df.columns:
        axes[1, 0].plot(df.index, df['Humidity'], color='blue', alpha=0.7)
        axes[1, 0].set_title('Vlhkost během závodu')
        axes[1, 0].set_xlabel('Měření')
        axes[1, 0].set_ylabel('Vlhkost (%)')
        axes[1, 0].grid(True, alpha=0.3)

    # Graf 4: Rychlost větru
    if 'WindSpeed' in df.columns:
        axes[1, 1].plot(df.index, df['WindSpeed'], color='green', alpha=0.7)
        axes[1, 1].set_title('Rychlost větru během závodu')
        axes[1, 1].set_xlabel('Měření')
        axes[1, 1].set_ylabel('Rychlost větru (m/s)')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Základní statistiky
    print("\n=== STATISTIKY POČASÍ ===")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(df[numeric_cols].describe())


if __name__ == "__main__":
    # Zadej cestu k souboru
    filename = input("Zadej cestu k CSV souboru: ")

    try:
        load_and_visualize_single_file(filename)
    except FileNotFoundError:
        print(f"Soubor {filename} nebyl nalezen!")
    except Exception as e:
        print(f"Chyba při zpracování: {e}")
