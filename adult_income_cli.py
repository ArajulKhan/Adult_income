import argparse
import os
import pickle
import joblib
import numpy as np
import pandas as pd

TARGET = 'income'
NUMERIC_COLUMNS = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL_COLUMNS = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

MODEL_CANDIDATES = ["model.pkl"]
SCALER_CANDIDATES = ["adult_scaler.pkl"]


def load_artifact(candidates):
   
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            return joblib.load(path), path
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f), path
    return None, None


def clean_input(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    df = df.replace("?", np.nan)
    return df


def validate_input(df):
    required = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing) +
            "\nExpected feature columns: " + ", ".join(required)
        )


def prepare_features(df):
    validate_input(df)
    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()

    
    for col in NUMERIC_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def predict_with_saved_artifacts(X):
    model, model_path = load_artifact(MODEL_CANDIDATES)
    scaler, scaler_path = load_artifact(SCALER_CANDIDATES)

    if model is None:
        raise FileNotFoundError(
            "No saved model was found. Run adult_income.ipynb first"
        )

   
    if hasattr(model, "predict") and hasattr(model, "named_steps"):
        predictions = model.predict(X)
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X)[:, 1]
            except Exception:
                probabilities = None
        return predictions, probabilities, model_path

    
    X_work = X.copy()

    if scaler is not None:
        try:
            X_work[NUMERIC_COLUMNS] = scaler.transform(X_work[NUMERIC_COLUMNS])
        except Exception as exc:
            raise ValueError(
                "The saved scaler could not transform the input. "
                f"Scaler: {scaler_path}. Original error: {exc}"
            ) from exc

    try:
        predictions = model.predict(X_work)
    except Exception as exc:
        raise ValueError(
            "The saved model expects a preprocessed feature matrix." \
            " Re-run the notebook's model-saving cell after "
        
        ) from exc

    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_work)[:, 1]
        except Exception:
            pass

    return predictions, probabilities, model_path


def normalize_prediction(value):
    text = str(value).strip()
    if text in ('>50K', '>50K.'):
        return ">50K"
    if text in ('<=50K', '<=50K.'):
        return "<=50K"
    return text


def run_csv(path, output):
    df = clean_input(pd.read_csv(path))
    X = prepare_features(df)

    predictions, probabilities, model_path = predict_with_saved_artifacts(X)

    result = df.copy()
    result["predicted_income"] = [normalize_prediction(x) for x in predictions]

    if probabilities is not None:
        result["probability_gt_50K"] = probabilities

    result.to_csv(output, index=False)

    print("\n=== Adult Income Working Prototype ===")
    print(f"Model loaded : {model_path}")
    print(f"Input rows   : {len(result)}")
    print(f"Output file  : {output}")
    print("\nPredictions:")
    print(result["predicted_income"].value_counts())


def run_interactive():
    print("\n=== Adult Income Working Prototype ===")
    print("Enter one person's values. Press Ctrl+C to cancel.\n")

    data = {}
    for col in NUMERIC_COLUMNS:
        data[col] = input(f"{col}: ").strip()

    for col in CATEGORICAL_COLUMNS:
        data[col] = input(f"{col}: ").strip()

    df = pd.DataFrame([data])
    df = clean_input(df)
    X = prepare_features(df)

    predictions, probabilities, model_path = predict_with_saved_artifacts(X)
    prediction = normalize_prediction(predictions[0])

    print(f"\nModel loaded: {model_path}")
    print(f"Predicted income: {prediction}")

    if probabilities is not None:
        print(f"Probability of >50K: {probabilities[0] * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Working CLI prototype for the Adult Income ML model."
    )
    parser.add_argument("--csv", help="CSV file containing new records.")
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Output CSV file (default: predictions.csv).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter one record manually.",
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.csv:
        run_csv(args.csv, args.output)
    else:
        parser.error("Use --csv <file> or --interactive.")


if __name__ == "__main__":
    main()