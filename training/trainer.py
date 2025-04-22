def run_training_pipeline():
    from demo_data.synthetic_loader import load_synthetic_dataset
    from app.services.feature_generator import generate_complex_features
    from training.train_vae import train_vae
    from training.train_risk import train_risk_classifier
    from training.train_limit import train_limit_regressor

    data = load_synthetic_dataset()
    X, y_risk, y_limit = [], [], []
    for row in data:
        flat = generate_complex_features(row)
        X.append(list(flat.values())[1:])
        y_risk.append(1 if flat['normalized_score'] < 60 else 0)
        y_limit.append(flat['salary_mean_6mo'] * 6)

    X = np.array(X, dtype=np.float32)
    y_risk = np.array(y_risk)
    y_limit = np.array(y_limit)

    train_vae(X)
    train_risk_classifier(X, y_risk)
    train_limit_regressor(X, y_limit)
    print("✅ All models trained successfully.")
