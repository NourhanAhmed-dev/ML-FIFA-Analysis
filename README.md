# FIFA Players — Performance Tier Classification & Market Value Regression

Two linked ML assignments on a single FIFA player dataset: EDA → classical baselines → tuned/ensembled models → a final production-style pipeline.

## Dataset

`Fifa.csv` — ~19,700 players with attributes including `Age`, `Overall_Rating`, `Future Potential`, `Total_Stats Score`, `Position`, `Country`, `Team`, and `Value Per M$`.

Two targets are predicted:
- **Regression** — `Value Per M$` (log-transformed during modeling to handle skew)
- **Classification** — `performance Tier` (`Low` / `Mid` / `High` / `Elite`), derived from quartiles of `Overall_Rating`

## Structure

1. **EDA** — missing values, duplicates, outliers, distributions, correlations
2. **Feature engineering** — target encoding for `Country`/`Team`, one-hot encoding for `Position`, scaling
3. **Assignment 2 — Regression** — Linear/Polynomial Regression, Ridge & Lasso
4. **Assignment 2 — Classification** — Logistic Regression, Naive Bayes (Gaussian/Bernoulli/Complement)
5. **Cross-validation** — K-Fold / Stratified K-Fold stability comparison
6. **Assignment 3 — Tuned models** — KNN, SVM, Random Forest with `GridSearchCV` + learning curves
7. **Ensembles** — Voting and Stacking (KNN + SVM + RF)
8. **Final pipeline** — a single `sklearn.Pipeline` per task, taking raw player data in and returning a value + tier prediction

## Results

| Task | Assignment 2 Baseline | Assignment 3 Final Pipeline |
|---|---|---|
| Classification Accuracy | 0.79 | 0.8640 |
| Regression R² Score | 0.7870 | 0.9547 |

## Avoiding data leakage

`Overall_Rating` is the value `performance Tier` is derived from, so:
- It is **excluded** from every classification model's features (kept for regression, where it's a legitimate predictor of value).
- The `Country`/`Team` encodings are fit on `Overall_Rating`, **never** on the actual prediction target (`y_clf`/`y_reg`) — this avoids leaking the answer into the encoded features.
- The encoder uses smoothing (`min_samples_leaf=20`, `smoothing=10`) so teams/countries with very few players don't get an encoding that's effectively just that one player's own rating.

This is a soft mitigation, not a perfect one — teams with very few players still carry a small amount of residual signal. Increasing `min_samples_leaf` or lowering `smoothing` further would shrink this risk more aggressively, at the cost of less informative encodings for small-but-real groups.

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn category_encoders
```

Place `Fifa.csv` in the same directory as the notebook, then run top to bottom.

## Notes for reproducing results

- All train/test splits and cross-validation folds use `random_state=42`.
- `GridSearchCV` searches and the final `StackingClassifier`/`StackingRegressor` use 5-fold CV internally; expect a noticeable runtime for those cells on the full dataset.
- Before publishing further changes, do a full **Restart & Run All** and re-check that any hardcoded result numbers in markdown cells still match the printed output.
