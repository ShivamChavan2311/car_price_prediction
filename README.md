<h1 align="center">🚗 Ford Used-Car Price Prediction</h1>

<p align="center">
  <i>Predicting the resale price of used Ford cars from age, mileage, fuel type, transmission and more, using Linear Regression, KNN and SVR.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-4C8CBF?style=flat-square" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" />
</p>

---

## 📋 Problem Statement

Pricing a used car is hard. Sellers often guess, buyers can't tell whether a listing is fair, and prices depend on many factors at once (age, mileage, engine, fuel, transmission).

**Goal:** build a regression model that predicts the **asking price (£)** of a used Ford from its listed attributes, and compare a few algorithms to see which one fits best.

## 📊 Dataset

| Property | Value |
|---|---|
| Records | 17,966 used Ford listings |
| Columns | 9 (8 features + the `price` target) |
| Missing values | None |
| Target | `price` (£, from £495 to £54,995) |

| Feature | Type | Description |
|---|---|---|
| `model` | categorical | Ford model (24 values, e.g. Fiesta, Focus, Kuga) |
| `year` | numeric | Year of registration |
| `transmission` | categorical | Manual, Automatic, Semi-Auto |
| `mileage` | numeric | Miles driven |
| `fuelType` | categorical | Petrol, Diesel, Hybrid, Electric, Other |
| `tax` | numeric | Road tax (£) |
| `mpg` | numeric | Fuel efficiency (miles per gallon) |
| `engineSize` | numeric | Engine size (litres) |
| `price` (target) | numeric | Listing price (£) |

**Data notes**
- The mix is mostly **petrol (12,179)** and **diesel (5,762)**, with only a handful of hybrid, electric and other cars.
- **Manual** cars dominate (15,518), against 1,361 automatic and 1,087 semi-auto.
- The data contains outliers and errors (for example a registration year of 2060), which the pipeline handles with IQR clipping.

## 🔑 Pipeline

1. **Load and inspect:** `head`, `describe`, null checks and value counts
2. **Outlier detection:** boxplots of `year`, `price` and `engineSize`
3. **Outlier handling:** IQR clipping (1.5 × IQR) on `year`, `price`, `mileage`, `tax`, `mpg` and `engineSize`
4. **Feature selection:** drop `model` and the target `price` from the features
5. **Train-test split:** 80% train, 20% test (`random_state=0`)
6. **Preprocessing** with a `ColumnTransformer`:
   - `transmission`: ordinal encoding (Manual → Automatic → Semi-Auto)
   - `fuelType`: one-hot encoding (first category dropped)
   - `year`, `mileage`, `tax`, `mpg`: standard scaling
   - `engineSize`: passed through unchanged
7. **Model training:** Linear Regression, KNN Regressor, Support Vector Regressor
8. **Evaluation:** R² score on the held-out test set

## 🤖 Models and Results

| Model | R² Score (test set) |
|---|---|
| Linear Regression | 0.766 |
| **K-Nearest Neighbors Regressor** | **0.894** |
| Support Vector Regressor (default settings) | 0.202 |

🏆 **Best model: KNN Regressor with R² ≈ 0.89**, meaning it explains about 89% of the variance in used-car prices on unseen data.

**Takeaways**
- KNN beats Linear Regression, which suggests the relationship between features and price is **non-linear** (for example, depreciation is not a straight line with age or mileage).
- SVR with default hyperparameters performs poorly. The target values are in the thousands, so default `C` and `epsilon` leave the model underfit. Tuning `C` (and scaling the target) is the obvious next step.
- Only R² is reported so far. Adding MAE and RMSE would show the typical error in pounds.

## 🛠️ Installation

**Prerequisites:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/ShivamChavan2311/ford-car-price-prediction.git
cd ford-car-price-prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## 🚀 Usage

Run the full notebook:

```bash
jupyter notebook ford_pred.ipynb
```

Or use the core code directly:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# Load data
ford = pd.read_csv("ford.csv")

# Clip outliers using the IQR rule
for col in ["year", "price", "mileage", "tax", "mpg", "engineSize"]:
    Q1, Q3 = ford[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    ford[col] = ford[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

# Features and target
X = ford.drop(columns=["model", "price"])
y = ford["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Preprocessing
transformer = ColumnTransformer(
    transformers=[
        ("transmission", OrdinalEncoder(categories=[["Manual", "Automatic", "Semi-Auto"]]), ["transmission"]),
        ("fuel", OneHotEncoder(sparse_output=False, drop="first"), ["fuelType"]),
        ("scale", StandardScaler(), ["year", "mileage", "tax", "mpg"]),
    ],
    remainder="passthrough",
)

X_train_t = transformer.fit_transform(X_train)
X_test_t = transformer.transform(X_test)

# Train and evaluate the best model (KNN)
knn = KNeighborsRegressor()
knn.fit(X_train_t, y_train)
print("KNN R2 Score:", r2_score(y_test, knn.predict(X_test_t)))
```

## 📁 Project Structure

```
ford-car-price-prediction/
│
├── ford.csv          # Dataset (17,966 used Ford listings)
├── ford_pred.ipynb   # Notebook: EDA, preprocessing, modelling, evaluation
└── README.md         # This file
```

## 📚 What I Learned

- Spotting and handling outliers with boxplots and IQR clipping
- Building a `ColumnTransformer` that applies ordinal, one-hot and scaling steps to the right columns
- Why encoding choices matter (ordinal for ordered categories, one-hot for unordered ones)
- Comparing regression models and reading R² to judge fit
- Why default hyperparameters can make a good algorithm look bad (SVR here)

## ⚠️ Limitations

- Outlier clipping is applied to the **whole dataset before the split**, so the clipping bounds use test-set information. Fitting the bounds on the training set only is cleaner.
- The `model` column (Fiesta, Focus, etc.) is **not used**, even though it is likely one of the strongest price drivers.
- `price` was also clipped, so very expensive cars (for example some Mustangs) are capped, and predictions for them will be understated.
- Results come from a single train-test split with no cross-validation.

## 🔭 Future Improvements

- Add the `model` feature with one-hot or target encoding
- Try tree-based models: Random Forest, Gradient Boosting, XGBoost
- Tune hyperparameters (KNN's `n_neighbors`, SVR's `C` and `epsilon`) with `GridSearchCV`
- Use k-fold cross-validation and report MAE and RMSE as well as R²
- Wrap preprocessing and model in one `Pipeline` and deploy it as a small Streamlit price-estimator app

## 👤 Author

**Shivam Chavan**

[LinkedIn](https://www.linkedin.com/in/shivamchavansppu/) · [Portfolio](https://shivamchavan2311.github.io/portfolio/) · [GitHub](https://github.com/ShivamChavan2311) · [chavanshivam160@gmail.com](mailto:chavanshivam160@gmail.com)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center"><b>⭐ If you found this useful, consider starring the repo.</b></p>
