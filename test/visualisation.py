# data_visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class F1Visualizer:
    def __init__(self, data_path="processed_data"):
        self.data_path = Path(data_path)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

    def load_data(self):
        """Načte zpracovaná data"""
        self.merged = pd.read_csv(self.data_path / "merged_data.csv")
        self.driver_perf = pd.read_csv(self.data_path / "driver_performance.csv")
        self.tyre_perf = pd.read_csv(self.data_path / "tyre_analysis.csv")

    def plot_driver_comparison(self, drivers=None, save=True):
        """Porovnání výkonnosti jezdců"""
        if drivers is None:
            # Top 10 jezdců podle průměrného času
            top_drivers = (self.driver_perf.groupby('Driver')['avg_lap_time']
                           .mean().nsmallest(10).index.tolist())
        else:
            top_drivers = drivers

        data = self.driver_perf[self.driver_perf['Driver'].isin(top_drivers)]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Graf 1: Průměrné časy podle jezdce
        avg_by_driver = data.groupby('Driver')['avg_lap_time'].mean().sort_values()
        axes[0].barh(avg_by_driver.index, avg_by_driver.values, color='steelblue')
        axes[0].set_xlabel('Průměrný čas kola (s)')
        axes[0].set_title('Průměrné časy kol - Top 10 jezdců')
        axes[0].grid(axis='x', alpha=0.3)

        # Graf 2: Vývoj během sezóny
        for driver in top_drivers[:5]:  # Top 5 pro čitelnost
            driver_data = data[data['Driver'] == driver]
            axes[1].plot(driver_data['race_order'],
                         driver_data['avg_lap_time'],
                         marker='o', label=driver)

        axes[1].set_xlabel('Pořadí závodu v sezóně')
        axes[1].set_ylabel('Průměrný čas kola (s)')
        axes[1].set_title('Vývoj výkonnosti během sezóny')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig('viz_driver_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_weather_impact(self, save=True):
        """Vliv počasí na časy kol"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Teplota trati vs. čas kola
        axes[0, 0].scatter(self.merged['TrackTemp'],
                           self.merged['LapTime_in_seconds'],
                           alpha=0.3, s=1)
        axes[0, 0].set_xlabel('Teplota trati (°C)')
        axes[0, 0].set_ylabel('Čas kola (s)')
        axes[0, 0].set_title('Vliv teploty trati na čas kola')

        # Teplota vzduchu vs. čas kola
        axes[0, 1].scatter(self.merged['AirTemp'],
                           self.merged['LapTime_in_seconds'],
                           alpha=0.3, s=1, color='orange')
        axes[0, 1].set_xlabel('Teplota vzduchu (°C)')
        axes[0, 1].set_ylabel('Čas kola (s)')
        axes[0, 1].set_title('Vliv teploty vzduchu na čas kola')

        # Vlhkost vs. čas kola
        axes[1, 0].scatter(self.merged['Humidity'],
                           self.merged['LapTime_in_seconds'],
                           alpha=0.3, s=1, color='green')
        axes[1, 0].set_xlabel('Vlhkost (%)')
        axes[1, 0].set_ylabel('Čas kola (s)')
        axes[1, 0].set_title('Vliv vlhkosti na čas kola')

        # Rychlost větru vs. čas kola
        axes[1, 1].scatter(self.merged['WindSpeed'],
                           self.merged['LapTime_in_seconds'],
                           alpha=0.3, s=1, color='red')
        axes[1, 1].set_xlabel('Rychlost větru (m/s)')
        axes[1, 1].set_ylabel('Čas kola (s)')
        axes[1, 1].set_title('Vliv rychlosti větru na čas kola')

        plt.tight_layout()
        if save:
            plt.savefig('viz_weather_impact.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_tyre_degradation(self, save=True):
        """Degradace pneumatik"""
        fig, ax = plt.subplots(figsize=(14, 7))

        for compound in self.tyre_perf['Compound'].unique():
            if pd.notna(compound):
                data = self.tyre_perf[self.tyre_perf['Compound'] == compound]
                data = data.sort_values('TyreLife')
                ax.plot(data['TyreLife'], data['avg_lap_time'],
                        marker='o', label=compound, linewidth=2)

        ax.set_xlabel('Stáří pneumatiky (kola)')
        ax.set_ylabel('Průměrný čas kola (s)')
        ax.set_title('Degradace pneumatik podle směsi')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig('viz_tyre_degradation.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_circuit_types(self, save=True):
        """Porovnání městských a klasických okruhů"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Městské vs. klasické okruhy
        circuit_comparison = self.driver_perf.groupby('city_circuit')['avg_lap_time'].mean()
        labels = ['Klasický okruh', 'Městský okruh']
        axes[0].bar(labels, [circuit_comparison[False], circuit_comparison[True]],
                    color=['steelblue', 'coral'])
        axes[0].set_ylabel('Průměrný čas kola (s)')
        axes[0].set_title('Porovnání typů okruhů')
        axes[0].grid(axis='y', alpha=0.3)

        # Noční vs. denní závody
        night_comparison = self.driver_perf.groupby('night_race')['avg_lap_time'].mean()
        labels = ['Denní závod', 'Noční závod']
        axes[1].bar(labels, [night_comparison[False], night_comparison[True]],
                    color=['gold', 'navy'])
        axes[1].set_ylabel('Průměrný čas kola (s)')
        axes[1].set_title('Porovnání denních a nočních závodů')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig('viz_circuit_types.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self):
        """Vygeneruje kompletní vizuální report"""
        print("Generuji vizualizace...")
        self.plot_driver_comparison()
        self.plot_weather_impact()
        self.plot_tyre_degradation()
        self.plot_circuit_types()
        print("Vizualizace uloženy!")


if __name__ == "__main__":
    viz = F1Visualizer()
    viz.load_data()
    viz.generate_report()
