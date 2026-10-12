# Car Price Prediction

Predicts the price of a car from its specifications using regression models.

## Problem Statement
Estimate the market price of a car based on features like engine size, horsepower, body type, drive wheel and fuel system.

## Dataset
- 205 rows, 26 columns (`CarPrice_Assignment.csv`)
- Target: `price` (ranges from about 5,118 to 45,400)
- No missing values
- File not included in this repo; add your own copy

## Approach
1. **EDA**: describe, correlation heatmap, boxplots for numeric features, countplots for categorical features
2. **Feature cleaning**:
   - Dropped `citympg` (about 97% correlated with `highwaympg`)
   - Dropped `car_ID` and `CarName` (identifiers, not useful for prediction)
3. **Encoding**:
   - Ordinal text columns (`doornumber`, `cylindernumber`) mapped to numbers
   - Remaining categoricals encoded with `LabelEncoder`
4. **Split**: 70/30 train-test split (`random_state=1`)
5. **Models**: Linear Regression, KNN Regressor (k=6), SVR (kernel tuned with `GridSearchCV`)

## Results (test set)
| Model | R² Score |
|---|---|
| Linear Regression | 81.04% (RMSE ≈ 3,383) |
| KNN Regressor | 77.23% |
| **SVR (tuned kernel)** | **81.40%** |

SVR performed best, though Linear Regression was very close.

## Tech Stack
Python, pandas, seaborn, matplotlib, scikit-learn

## How to Run
1. Place `CarPrice_Assignment.csv` in your working directory and update the path in the notebook
2. Open `CarPrice_Project.ipynb` and run all cells

## Possible Improvements
- Scale features before SVR and KNN
- Try tree-based models (Random Forest, Gradient Boosting)
- Use brand extracted from `CarName` as a feature
