# One-Page Summary — Adult Income ML Project

## Project Overview

The Adult Income project uses the **original `adult_income.ipynb` notebook as the Machine Learning and model-training component**. The notebook performs the data preparation, feature processing, model training, evaluation, and model saving required for the classification task.

A separate Python file, `adult_income_cli.py`, acts as the **working prototype**. It does not replace the notebook or retrain the model. Instead, it loads the trained model artifacts produced by the notebook and uses them to generate predictions for new input data.

## Machine Learning Component

The original notebook is retained as:

```text
adult_income.ipynb
```

The notebook's workflow is the source of truth for the ML implementation. The main classification task is to predict the income category using the features used in the notebook.

The model identified in the notebook is:

```text
LogisticRegression
```

The target is:

```text
income
```

The feature groups identified from the notebook are:

**Numerical:** age, fnlwgt, educational-num, capital-gain, capital-loss, hours-per-week

**Categorical:** workclass, education, marital-status, occupation, relationship, race, gender, native-country

## Working Prototype

The prototype is:

```text
adult_income_cli.py
```

Its purpose is to demonstrate how the trained ML model can be used outside the notebook.

It supports two input methods:

1. **CSV input** for multiple records.
2. **Interactive input** for one person's information.

Example:

```bash
python adult_income_cli.py --csv new_people.csv
```

or:

```bash
python adult_income_cli.py --interactive
```

The prototype produces the predicted income class and, when supported by the saved model, the probability associated with the `>50K` class.

## System Flow

```text
Adult Income Dataset
        ↓
adult_income.ipynb
        ↓
Data preprocessing
        ↓
Model training
        ↓
Model evaluation
        ↓
Saved model/scaler artifacts
        ↓
adult_income_cli.py
        ↓
New CSV / Manual Input
        ↓
Income Prediction
```

## Separation of Responsibilities

The two files have different purposes:

| File | Purpose |
|---|---|
| `adult_income.ipynb` | Machine Learning, training, evaluation |
| `adult_income_cli.py` | Working prediction prototype |
| `README.md` | Setup and usage documentation |
| `one_page_summary.md` | Project overview |

The original notebook is therefore kept as the **ML/model training file**, while the `.py` file provides the practical prototype for using the trained model.
