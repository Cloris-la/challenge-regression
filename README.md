# Machine Learning Models for Real Estate Price Prediction

## Description

This project evaluates various machine learning models to predict real estate prices. The goal is to assess the accuracy of different models and select the best one. The following models were used for evaluation:

- **Linear Regression**
- **Random Forest Regression**
- **Gradient Boost Regression**
- **CatBoost Regression**
- **XGBoost Regression**

---

## Dataset

The dataset used in this project was scraped from **ImmoEliza** via **Immoweb** using web scraping techniques. The dataset can be found at:

data/dataset_wout_price_outliers.csv


---

## Installation

To run this Python code, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/Cloris-la/challenge-regression.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

The `requirements.txt` file includes the following libraries:

- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `jupyter`
- `catboost`
- `xgboost`

---

## Usage

This project uses the **scikit-learn** Python library to accommodate multivariable regression models. The target variable, **price**, contains both continuous and categorical determinant variables. Data preprocessing was carried out using **StandardScaler** to improve model prediction accuracy.

### Data Preprocessing Example:

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

df = pd.read_csv('data/dataset_wout_price_outliers.csv')
df_prepro = df.copy()

columns = ['habitableSurface', 'price']
for col in columns:
    df_prepro[col] = StandardScaler().fit_transform(np.array(df_prepro[col]).reshape(-1, 1))

df_LR = df_prepro

## Model Training and Evaluation:
```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Setting feature and target variables
feature_names = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_start)

# Model training
model_LR = LinearRegression(fit_intercept=True)
model_LR.fit(X_train, y_train)

# Model evaluation
y_pred_LR = model_LR.predict(X_test)
MSE = mean_squared_error(y_test, y_pred_LR)
r2 = r2_score(y_test, y_pred_LR)

# Feature importances (for Random Forest model)
coefficients_gl = best_model.coef_
importance_gl = pd.DataFrame({
    'Feature': feature_names_lg,
    'Importance': coefficients_gl
}).sort_values(by='Importance', key=abs, ascending=False)
'''

## Visualization:
To visually compare the actual vs predicted prices, we use Matplotlib for plotting the data.

```
import matplotlib.pyplot as plt

# Plotting the actual vs predicted prices
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Fit Line')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True)
plt.show()
```
## Timeline
This project was completed as part of a five-day challenge:

Day 1: Data Preprocessing

Days 2-5: Code Development, Model Training, Validation, Documentation, and Presentation

## Contributors
The following individuals contributed to this project:

- Hanieh

- Estifania

- Fang

- Mengstu

Each contributor worked on different parts of the code, from data preprocessing to implementing the various machine learning models. Afterward, the files were merged into a single script, main.py.

## Personal Reflection
This project was an excellent opportunity for us to deepen our understanding of machine learning models and real estate price prediction. As AI Data Science trainees, we gained valuable experience working with various regression models like Linear Regression, XGBoost, CatBoost, and Random Forest.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

### Key Changes:
- I structured the README for clarity and organization, making it easier to follow.
- I've added code blocks where needed, and formatted Python code, visualizations, and installation steps properly.
- The reflection section gives a brief summary of the team's learning and progress.
  
You can now use this as your `README.md` for the project repository! Let me know if you'd like any further changes.






