# Dataset

**Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — collected and analysed by the Machine Learning Group at Université Libre de Bruxelles (ULB). Transactions made by European cardholders over two days in September 2013.

The raw CSV is **not committed to this repository** (284,807 rows, ~150MB, and redistribution terms favor pointing to the canonical source rather than mirroring it). It is fetched at setup time instead.

## Fetching the data

```bash
python scripts/fetch_dataset.py
```

This downloads the dataset via [OpenML](https://www.openml.org/) (`data_id=1597`), which mirrors the same ULB data and requires no account or API key. It's written to `data/raw/creditcard.csv` (gitignored).

If OpenML is unreachable, download `creditcard.csv` manually from the Kaggle link above and place it at `data/raw/creditcard.csv`.

**Note:** the OpenML mirror omits the original `Time` column (seconds since the first transaction in the dataset). This project does not use `Time` as a model feature, so it has no effect on preprocessing or training — the feature set is `V1`–`V28` (PCA-transformed, anonymized) plus `Amount`.

## Inspecting the data

```bash
python scripts/inspect_dataset.py
```

### Measured characteristics (run 2026-08-20)

| Metric | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 |
| Fraud percentage | 0.1727% |
| Class imbalance | 1 : 577.9 (fraud:legit) |
| Features | 29 (V1–V28, Amount) |
| Missing values | 0 |
| Duplicate rows | 9,144 |

The duplicate rows are a documented characteristic of this dataset (widely noted in prior work using it) and are dropped during preprocessing — see `src/data/preprocessing.py`.

## Preprocessing

```bash
python -m src.data.preprocessing
```

Produces deterministic (seed=42) stratified train/val/test splits under `data/processed/` (also gitignored — regenerate locally):

- `X_{train,val,test}.npy`, `y_{train,val,test}.npy`
- `amount_scaler.json` — StandardScaler parameters fit on the train split only
- `metadata.json` — split sizes and fraud rates

See `tests/test_preprocessing.py` for correctness tests (schema validation, no train/test leakage, deterministic splitting).
