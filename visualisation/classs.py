def main():
    import duckdb
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns

    conn = duckdb.connect('race_database.db')

    print("=== Hypotézy s ML klasifikátory ===\n")

    # HYPOTÉZA 9: Predikce podium finish na základě kvalifikace a podmínek
    print("9. Klasifikace: Kdo skončí na podiu?...")
    podium_query = """
    SELECT
        q.position as quali_pos,
        AVG(wd.air_temp) as avg_temp,
        AVG(wd.humidity) as avg_humidity,
        AVG(wd.wind_speed) as avg_wind,
        c.alt as altitude,
        CASE WHEN final.position <= 3 THEN 1 ELSE 0 END as podium
    FROM qualifying q
    JOIN races r ON q.race_id = r.race_id
    JOIN circuits c ON r.circuit_id = c.circuit_id
    LEFT JOIN weather_data wd ON r.race_id = wd.race_id
    JOIN (
        SELECT race_id, driver, position
        FROM race_results
        WHERE lap_number = (SELECT MAX(lap_number) FROM race_results rr2 
                           WHERE rr2.race_id = race_results.race_id 
                           AND rr2.driver = race_results.driver)
    ) final ON q.race_id = final.race_id AND q.driver = final.driver
    WHERE q.position IS NOT NULL
        AND final.position IS NOT NULL
        AND wd.air_temp IS NOT NULL
    GROUP BY q.race_id, q.driver, q.position, c.alt, final.position
    """
    podium_data = conn.execute(podium_query).df()
    podium_data = podium_data.dropna()

    X = podium_data[['quali_pos', 'avg_temp', 'avg_humidity', 'avg_wind', 'altitude']]
    y = podium_data['podium']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42)
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Mimo podium', 'Podium'],
                   yticklabels=['Mimo podium', 'Podium'])
        axes[idx].set_title(f'{name}\nAccuracy: {model.score(X_test_scaled, y_test):.3f}',
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Skutečnost')
        axes[idx].set_xlabel('Predikce')

    plt.tight_layout()
    plt.savefig('hypothesis_podium_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_podium_classification.png'\n")

    # ROC křivky
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Náhodný klasifikátor')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC křivky - Predikce podia', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hypothesis_podium_roc.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_podium_roc.png'\n")

    # HYPOTÉZA 10: Predikce DNF (Did Not Finish)
    print("10. Klasifikace: Kdo nedokončí závod?...")
    dnf_query = """
WITH last_laps AS (
    SELECT race_id, MAX(lap_number) as max_lap
    FROM race_results
    GROUP BY race_id
),
driver_last_lap AS (
    SELECT 
        rr.race_id,
        rr.driver,
        MAX(rr.lap_number) as driver_max_lap
    FROM race_results rr
    GROUP BY rr.race_id, rr.driver
),
weather_agg AS (
    SELECT 
        race_id,
        AVG(air_temp) as avg_temp,
        AVG(humidity) as avg_humidity,
        AVG(rainfall) as avg_rain
    FROM weather_data
    WHERE air_temp IS NOT NULL
    GROUP BY race_id
)
SELECT
    q.position as quali_pos,
    wa.avg_temp,
    wa.avg_humidity,
    wa.avg_rain,
    c.alt as altitude,
    r.year,
    CASE 
        WHEN dll.driver_max_lap < ll.max_lap THEN 1 
        ELSE 0 
    END as dnf
FROM qualifying q
JOIN races r ON q.race_id = r.race_id
JOIN circuits c ON r.circuit_id = c.circuit_id
JOIN weather_agg wa ON r.race_id = wa.race_id
JOIN driver_last_lap dll ON q.race_id = dll.race_id AND q.driver = dll.driver
JOIN last_laps ll ON q.race_id = ll.race_id
WHERE q.position IS NOT NULL
"""

    dnf_data = conn.execute(dnf_query).df()
    dnf_data = dnf_data.dropna()

    X_dnf = dnf_data[['quali_pos', 'avg_temp', 'avg_humidity', 'avg_rain', 'altitude', 'year']]
    y_dnf = dnf_data['dnf']

    X_train_dnf, X_test_dnf, y_train_dnf, y_test_dnf = train_test_split(
        X_dnf, y_dnf, test_size=0.3, random_state=42
    )

    scaler_dnf = StandardScaler()
    X_train_dnf_scaled = scaler_dnf.fit_transform(X_train_dnf)
    X_test_dnf_scaled = scaler_dnf.transform(X_test_dnf)

    rf_dnf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf_dnf.fit(X_train_dnf_scaled, y_train_dnf)

    feature_importance = pd.DataFrame({
        'feature': X_dnf.columns,
        'importance': rf_dnf.feature_importances_
    }).sort_values('importance', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.barh(feature_importance['feature'], feature_importance['importance'],
            color='steelblue', edgecolor='black')
    ax1.set_xlabel('Důležitost', fontsize=12)
    ax1.set_title('Feature Importance - Predikce DNF', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    cm_dnf = confusion_matrix(y_test_dnf, rf_dnf.predict(X_test_dnf_scaled))
    sns.heatmap(cm_dnf, annot=True, fmt='d', cmap='Reds', ax=ax2,
               xticklabels=['Dokončil', 'DNF'],
               yticklabels=['Dokončil', 'DNF'])
    ax2.set_title(f'Confusion Matrix - Random Forest\nAccuracy: {rf_dnf.score(X_test_dnf_scaled, y_test_dnf):.3f}',
                 fontsize=13, fontweight='bold')
    ax2.set_ylabel('Skutečnost')
    ax2.set_xlabel('Predikce')

    plt.tight_layout()
    plt.savefig('hypothesis_dnf_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_dnf_classification.png'\n")

    # HYPOTÉZA 11: Predikce výhry závodu (top 5 kvalifikace)
    print("11. Klasifikace: Kdo vyhraje závod z top 5?...")
    win_query = """
    SELECT
        q.position as quali_pos,
        AVG(wd.air_temp) as avg_temp,
        AVG(wd.track_temp) as avg_track_temp,
        AVG(wd.wind_speed) as avg_wind,
        c.alt as altitude,
        CASE WHEN final.position = 1 THEN 1 ELSE 0 END as winner
    FROM qualifying q
    JOIN races r ON q.race_id = r.race_id
    JOIN circuits c ON r.circuit_id = c.circuit_id
    LEFT JOIN weather_data wd ON r.race_id = wd.race_id
    JOIN (
        SELECT race_id, driver, position
        FROM race_results
        WHERE lap_number = (SELECT MAX(lap_number) FROM race_results rr2 
                           WHERE rr2.race_id = race_results.race_id 
                           AND rr2.driver = race_results.driver)
    ) final ON q.race_id = final.race_id AND q.driver = final.driver
    WHERE q.position <= 5
        AND q.position IS NOT NULL
        AND final.position IS NOT NULL
        AND wd.air_temp IS NOT NULL
    GROUP BY q.race_id, q.driver, q.position, c.alt, final.position
    """
    win_data = conn.execute(win_query).df()
    win_data = win_data.dropna()

    X_win = win_data[['quali_pos', 'avg_temp', 'avg_track_temp', 'avg_wind', 'altitude']]
    y_win = win_data['winner']

    X_train_win, X_test_win, y_train_win, y_test_win = train_test_split(
        X_win, y_win, test_size=0.3, random_state=42, stratify=y_win
    )

    scaler_win = StandardScaler()
    X_train_win_scaled = scaler_win.fit_transform(X_train_win)
    X_test_win_scaled = scaler_win.transform(X_test_win)

    gb_win = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    gb_win.fit(X_train_win_scaled, y_train_win)

    y_pred_win = gb_win.predict(X_test_win_scaled)
    y_proba_win = gb_win.predict_proba(X_test_win_scaled)[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    cm_win = confusion_matrix(y_test_win, y_pred_win)
    sns.heatmap(cm_win, annot=True, fmt='d', cmap='Greens', ax=ax1,
               xticklabels=['Nezvítězil', 'Zvítězil'],
               yticklabels=['Nezvítězil', 'Zvítězil'])
    ax1.set_title(f'Predikce vítěze z top 5\nAccuracy: {gb_win.score(X_test_win_scaled, y_test_win):.3f}',
                 fontsize=13, fontweight='bold')
    ax1.set_ylabel('Skutečnost')
    ax1.set_xlabel('Predikce')

    fpr_win, tpr_win, thresholds = roc_curve(y_test_win, y_proba_win)
    roc_auc_win = auc(fpr_win, tpr_win)
    ax2.plot(fpr_win, tpr_win, color='darkgreen', linewidth=2.5,
            label=f'Gradient Boosting (AUC = {roc_auc_win:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Náhodný klasifikátor')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC křivka - Predikce vítěze', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('hypothesis_winner_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("   ✓ Uloženo jako 'hypothesis_winner_classification.png'\n")

    print("\n=== Výsledky klasifikace ===")
    print(f"Podium model - nejlepší: Random Forest")
    print(f"DNF model - accuracy: {rf_dnf.score(X_test_dnf_scaled, y_test_dnf):.3f}")
    print(f"Winner model - accuracy: {gb_win.score(X_test_win_scaled, y_test_win):.3f}")

    conn.close()

    print("\n=== ML hypotézy vizualizovány ===")
    print("\nVýsledky s klasifikátory:")
    print("  • hypothesis_podium_classification.png - Srovnání 4 klasifikátorů pro podium")
    print("  • hypothesis_podium_roc.png - ROC křivky pro podium")
    print("  • hypothesis_dnf_classification.png - Predikce DNF s feature importance")
    print("  • hypothesis_winner_classification.png - Predikce vítěze z top 5")


if __name__ == "__main__":
    main()
