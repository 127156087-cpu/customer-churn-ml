# Customer Churn Prediction

Predict whether a telecom customer will leave (`Churn` = Yes/No) using the IBM Telco dataset (7,043 customers).

## How to run
```
pip install -r requirements.txt
python churn_project.py        # or open churn_project.ipynb
```
Outputs: `reports/model_comparison.csv` and charts in `reports/figures/`.

## Results (random seed 42)

| Model | Train F1 | Validation F1 | Gap | Val recall | Val ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.628 | 0.624 | 0.004 | 0.814 | 0.845 |
| Decision Tree (no limits) | 0.996 | 0.462 | 0.534 | 0.475 | 0.633 |
| Random Forest (default) | 0.996 | 0.554 | 0.442 | 0.511 | 0.808 |
| **Random Forest (tuned)** | n/a | **0.641** | n/a | 0.811 | 0.844 |

Final test result (tuned Random Forest, threshold 0.5): precision 0.54, recall 0.75, F1 0.63, ROC-AUC 0.84.

## Structure of the code (what to say for each part)

| Step | What it does | Concept shown |
|---|---|---|
| 1 | Load data, fix `TotalCharges` (text -> number), drop `customerID`, define X and y | Features/target, supervised learning |
| 2 | Stratified 70/15/15 split **before** anything else | Train/validation/test, leakage |
| 3 | Pipeline: impute -> scale/one-hot -> model. Baseline = Logistic Regression | Preprocessing, baseline, leakage prevention |
| 4 | Logistic Regression vs Decision Tree vs Random Forest, train vs validation | Model comparison, overfitting |
| 5a | Sweep `C` (LR) and `max_depth` (tree) | Regularization, overfitting curves |
| 5b | GridSearchCV on Random Forest with 5-fold stratified CV (training data only) | Cross-validation, tuning |
| 6 | Pick model and threshold using validation data only | Model/metric justification |
| 7 | One-time test evaluation: confusion matrix, report, ROC | Final evaluation |
| 8 | LR coefficients and RF feature importances | Interpretability |
| Extra A | Linear regression predicting `MonthlyCharges` (R^2 = 0.955) | Linear regression |
| Extra B | KMeans on tenure + monthly charges, no labels | Unsupervised learning |

## Key explanations

**Why F1 and not accuracy?** Only 26.5% of customers churn. Predicting "no churn" for everyone gives about 73% accuracy and catches zero churners. F1 balances precision (don't waste retention offers) and recall (don't miss leavers). ROC-AUC (0.84) shows the model ranks risky customers well regardless of threshold.

**Why `class_weight="balanced"`?** It makes mistakes on the rare churn class cost more, so the models do not just favour "no churn".

**Overfitting evidence:** the unlimited tree scores 0.996 on training but 0.462 on validation (it memorised the data). Limiting depth to about 3 fixes this. Random Forest with `min_samples_leaf=10` rose from 0.544 to 0.633 CV F1, consistently across all 5 folds.

**Why Logistic Regression is competitive:** the churn signal is mostly simple (contract type, tenure, charges), which a linear model captures well. The tuned Random Forest was only slightly better on validation (0.641 vs 0.624), so be ready to say the difference is small and the simpler model is a valid choice for interpretability.

**Leakage checklist**
1. Split before preprocessing.
2. Imputer, scaler and encoder live inside the Pipeline, so they are fitted on training data only (and per fold in CV).
3. Tuning used training data with CV; model and threshold were chosen on validation.
4. Test set used once at the end.
5. `customerID` dropped; no duplicate IDs; no columns that are recorded after churn happens (checked by looking at column meanings).

**Honest limitations to mention:** F1 around 0.63 is moderate; the threshold was chosen to maximize F1 rather than from real business costs; one random split means results vary slightly with the seed; the models were not retrained on train+validation before testing (kept simple).

