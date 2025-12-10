import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge, BayesianRidge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# Check availability of advanced libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def prepare_ml_data():
    """Prepare data for machine learning."""
    conn = duckdb.connect('../race_database.db')

    query = """
    WITH race_weather AS (
        SELECT DISTINCT
            wd.race_id,
            CASE
                WHEN SUM(CASE WHEN wd.rainfall = TRUE THEN 1 ELSE 0 END)::FLOAT / COUNT(*) > 0.3
                THEN TRUE
                ELSE FALSE
            END as is_rain
        FROM weather_data wd
        GROUP BY wd.race_id
    ),
    driver_stats AS (
        SELECT
            driver,
            AVG(position) as avg_position,
            STDDEV(position) as stddev_position,
            COUNT(*) as total_races
        FROM race_results
        WHERE position IS NOT NULL
            AND deleted = FALSE
            AND lap_number = (
                SELECT MAX(lap_number)
                FROM race_results rr2
                WHERE rr2.race_id = race_results.race_id
                    AND rr2.driver = race_results.driver
            )
        GROUP BY driver
    ),
    race_final_positions AS (
        SELECT
            race_id,
            driver,
            position as final_position
        FROM race_results
        WHERE lap_number = (
            SELECT MAX(lap_number)
            FROM race_results rr2
            WHERE rr2.race_id = race_results.race_id
            AND rr2.driver = race_results.driver
        )
        AND position IS NOT NULL
        AND deleted = FALSE
    )
    SELECT
        q.position as quali_position,
        rfp.final_position,
        r.city_circuit::INTEGER as is_city_circuit,
        r.night_race::INTEGER as is_night_race,
        COALESCE(rw.is_rain::INTEGER, 0) as is_rain,
        ds.avg_position as driver_avg_position,
        COALESCE(ds.stddev_position, 0) as driver_stddev_position,
        ds.total_races as driver_total_races,
        r.race_name
    FROM qualifying q
    JOIN race_final_positions rfp
        ON q.race_id = rfp.race_id
        AND q.driver = rfp.driver
    JOIN races r ON q.race_id = r.race_id
    LEFT JOIN race_weather rw ON q.race_id = rw.race_id
    LEFT JOIN driver_stats ds ON q.driver = ds.driver
    WHERE q.position IS NOT NULL
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate different ML algorithms."""
    models = {
        'Bayesian Ridge': BayesianRidge(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'SVM': SVR(kernel='rbf', C=10, epsilon=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring='neg_mean_absolute_error', n_jobs=-1)

        results.append({
            'Model': name,
            'MAE Train': mean_absolute_error(y_train, y_pred_train),
            'MAE Test': mean_absolute_error(y_test, y_pred_test),
            'RMSE Test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'R² Test': r2_score(y_test, y_pred_test),
            'CV MAE': -cv_scores.mean()
        })

    return pd.DataFrame(results)


def plot_model_comparison(results_df):
    """Simplified model comparison - only MAE and R²."""
    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. MAE comparison (Train vs Test)
    x = np.arange(len(results_df))
    width = 0.35
    ax1.bar(x - width/2, results_df['MAE Train'], width, label='Train', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, results_df['MAE Test'], width, label='Test', alpha=0.8, color='coral')
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Mean Absolute Error (positions)', fontsize=11)
    ax1.set_title('Mean Absolute Error Comparison\n(lower is better)', fontsize=13, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. R² Score comparison
    colors = plt.cm.RdYlGn((results_df['R² Test'] - results_df['R² Test'].min()) /
                           (results_df['R² Test'].max() - results_df['R² Test'].min()))
    ax2.barh(results_df['Model'], results_df['R² Test'], color=colors, alpha=0.8)
    ax2.set_xlabel('R² Score', fontsize=11)
    ax2.set_title('R² Score Comparison\n(higher is better, 1.0 = perfect)', fontsize=13, weight='bold')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax2.axvline(x=0.5, color='orange', linestyle='--', alpha=0.3, linewidth=1.5)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(plots_dir / 'ml_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_table(results_df):
    """Main summary table with all metrics."""
    plots_dir = Path('plots')
    sorted_df = results_df.sort_values('MAE Test').reset_index(drop=True)

    stats_data = []
    for idx, row in sorted_df.iterrows():
        rating = "★★★" if idx < 3 else "★★" if idx < 5 else "★"
        gap = abs(row['MAE Test'] - row['MAE Train'])

        stats_data.append([
            f"#{idx + 1}",
            row['Model'],
            f"{row['MAE Test']:.3f}",
            f"{row['RMSE Test']:.3f}",
            f"{row['R² Test']:.3f}",
            f"{row['CV MAE']:.3f}",
            f"{gap:.3f}",
            rating
        ])

    fig, ax = plt.subplots(figsize=(14, len(sorted_df) * 0.7 + 2))
    ax.axis('tight')
    ax.axis('off')

    headers = ['Rank', 'Model', 'MAE\nTest', 'RMSE\nTest', 'R²\nTest',
               'CV MAE', 'Overfit\nGap', 'Rating']

    table = ax.table(cellText=stats_data, colLabels=headers,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.8)

    # Header styling
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#2E75B6')
        cell.set_text_props(weight='bold', color='white', ha='center')

    # Row coloring by performance
    colors = ['#70AD47', '#92D050', '#C5E0B4']  # green for top 3
    for i in range(len(stats_data)):
        color = colors[i] if i < 3 else '#FFF2CC' if i < 5 else '#F2F2F2'
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(color)

        # Highlight large overfitting in red
        if float(stats_data[i][6]) > 0.5:
            table[(i + 1, 6)].set_facecolor('#FFC7CE')

    plt.title('Model Performance Summary Table\n' +
              '(sorted by MAE Test, lower MAE/RMSE = better, higher R² = better)',
              fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(plots_dir / 'ml_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_feature_importance(X_train, y_train, feature_names):
    """Feature importance analysis - separate file per model."""
    plots_dir = Path('plots')

    tree_models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10)
    }

    if XGBOOST_AVAILABLE:
        from xgboost import XGBRegressor
        tree_models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    if LIGHTGBM_AVAILABLE:
        from lightgbm import LGBMRegressor
        tree_models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)

    importances_dict = {}

    for name, model in tree_models.items():
        model.fit(X_train, y_train)

        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        importances_dict[name] = importances

        # Individual plot for each model
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(importances['Importance'] / importances['Importance'].max())
        bars = ax.barh(importances['Feature'], importances['Importance'], color=colors)

        # Add values at the end of bars
        for i, (bar, importance) in enumerate(zip(bars, importances['Importance'])):
            ax.text(importance + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', va='center', fontsize=9)

        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f'Feature Importance - {name}', fontsize=13, weight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        filename = f"ml_feature_importance_{name.replace(' ', '_').lower()}.png"
        fig.savefig(plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    return importances_dict


def main():
    # Load data
    df = prepare_ml_data()

    # Prepare
    feature_cols = ['quali_position', 'is_city_circuit', 'is_night_race', 'is_rain',
                    'driver_avg_position', 'driver_stddev_position', 'driver_total_races']
    X = df[feature_cols]
    y = df['final_position']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training
    results_df = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # Visualizations - only key charts
    create_summary_table(results_df)                # 1. Summary table
    plot_model_comparison(results_df)               # 2. MAE and R² comparison
    analyze_feature_importance(X_train, y_train, feature_cols)  # 3. Feature importance

    # Best model
    best = results_df.loc[results_df['MAE Test'].idxmin()]

    # Top 3 models
    top3 = results_df.nsmallest(3, 'MAE Test')

    return {
        "results": results_df,
        "best": best,
        "top3": top3
    }


if __name__ == "__main__":
    main()
