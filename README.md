# Customer-Segmentation-and-churn-prediction
A churn prediction project on telecom customer data using Random Forest, then re-using the model's churn probabilities to segment customers with K-Means. Started as a straightforward classification task, but the feature importance results turned out to be just as useful for grouping customers, so both pieces are part of this repo.

## About the Data

Dataset: Telco Customer Churn (`Telco_customer_churn.xlsx`) — 7043 rows. Includes tenure, contract type, monthly/total charges, internet & phone add-ons, and a churn label.

Class balance: **5174 stayed** vs **1869 churned** (~73.5% / 26.5%), so imbalance was a factor throughout.

## Exploratory Analysis

- Churned customers have much shorter average tenure: **~18 months** vs **~37.6 months** for non-churners.
- Monthly charges range from **$18.25 to $118.75**, averaging **$64.76**.
- Contract type is one of the strongest churn signals:

  | Contract | Churn Rate |
  |---|---|
  | Month-to-month | 42.7% |
  | One year | 11.3% |
  | Two year | 2.8% |

- Tenure has a moderate negative correlation with churn (**-0.35**) — the longer someone stays, the less likely they are to leave.

## Cleaning the Data

- `Total Charges` was stored as text; converting it with `pd.to_numeric` produced 11 NaNs, all belonging to customers with 0 tenure (new sign-ups with no bill yet). Filled with 0.
- Dropped columns that don't help modeling or leak the target: CustomerID, Count, location fields (Country/State/City/Zip/Lat-Long), Churn Label, Churn Score, CLTV, Churn Reason.
- One-hot encoded the remaining categorical columns (`pd.get_dummies`), ending up with **30 features**.
- Standard 80/20 train-test split on `Churn Value` (5634 train / 1409 test rows).

## Modeling Results

| Stage | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) |
|---|---|---|---|---|
| Baseline Random Forest | 78.6% | 0.66 | **0.51** | 0.58 |
| Balanced (`class_weight='balanced'`) | 79.2% | 0.67 | 0.52 | 0.59 |
| Tuned (300 trees, depth 10) | 78.3% | 0.59 | 0.75 | 0.66 |
| **After feature selection (final model)** | 78.0% | 0.56 | **0.74** | 0.63 |

The baseline and balanced models barely moved recall — both were stuck missing roughly half of all customers who actually churned. Hyperparameter tuning alone didn't fix it either.

**Feature selection was the real turning point.** Top features by importance: Tenure, Total Charges, Contract (two-year), Monthly Charges, Dependents. Two redundant columns — `Phone Service_Yes` and `Multiple Lines_No phone service` — were dropped, and retraining pushed recall from **0.52 → 0.74**, by far the biggest jump in the whole project, for a small (~1pt) accuracy trade-off.

**Final model:** `RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced')`

**5-fold cross-validation on the final model:**
| Metric | Mean | Fold scores |
|---|---|---|
| Accuracy | 77.9% | 0.767, 0.798, 0.762, 0.786, 0.784 |
| Recall | 73.4% | 0.709, 0.765, 0.741, 0.743, 0.710 |

So the recall improvement held up across folds, not just one lucky split.

**ROC-AUC: 0.857** — the model separates churners from non-churners reasonably well.

## Customer Segmentation (K-Means)

Took the churn probability output from the final Random Forest, combined it with Tenure, Monthly Charges, and Total Charges, and scaled everything with `StandardScaler` before clustering (K-Means is distance-based and sensitive to scale).

Used the elbow method to pick **k = 3** — WCSS dropped sharply up to that point and flattened out after.

| Cluster | Avg Tenure | Avg Monthly Charges | Avg Total Charges | Avg Churn Prob | Segment |
|---|---|---|---|---|---|
| 0 | ~32 mo | ~$32.8 | ~$1,048 | 0.12 | Budget Loyal Customers |
| 1 | ~11 mo | ~$72.0 | ~$884 | **0.69** | High Risk New Customers |
| 2 | ~58 mo | ~$90.4 | ~$5,278 | 0.23 | Loyal Premium Customers |

- **Cluster 1 (High Risk New Customers)** stands out — short tenure, mid-high charges, and by far the highest churn probability. This is the group a retention team should prioritize.
- **Cluster 0 (Budget Loyal)** — low spend, low risk.
- **Cluster 2 (Loyal Premium)** — long tenure, highest spend, but still relatively safe — likely satisfied or locked into longer contracts.

## Takeaways

New customers on pricier plans (Cluster 1) are clearly the most likely to churn, and contract length plays a major role across the board. Practically, this points toward investing in onboarding for new customers and incentivizing month-to-month customers to move to longer contracts, since that's where most of the churn risk concentrates.

## Built With

Python · pandas · numpy · matplotlib · seaborn · scikit-learn (`RandomForestClassifier`, `KMeans`, `StandardScaler`, `train_test_split`, `cross_val_score`)

## What I'd Try Next

Feature selection mattered far more here than hyperparameter tuning — a good lesson to check feature importance early rather than tuning blindly first. Next steps could include SMOTE for the class imbalance, or trying XGBoost to see if recall improves further without sacrificing precision.
