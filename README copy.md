==== Machine Learning Models====

Description 

The project work has approached various machine models to evaluate the price of real estates.
Five models have been utilized to examine their accuracy and to select the best Model
These include:
  - Linear regression 
  - Random Forest Regression 
  - Gradient Boost Regression 
  - Cat Boost Regression 
  - xg Boost Regression 
  - 
-------------------------------------------------------------------------------
Dataset

A Dataset was taken from ImmoEliza through Immoweb scapying.
The Dataset is found at data/dataset_wout_price_outlierscsv


Installation

To run this python code, you have to follow these few procedures: 

   -clone this repository from https://github.com/Cloris-la/challenge-regression.git
   - install the requirement.txt file through
   
        pip install -r requirements.txt 

            This file includes:
                scikit-learn
                pandas
                numpy
                matplotlib
                jupyter
                catboost
                xgboost


Usage

These machine learning modes use the scikit learn python library to accommodate multivariable functions. 
The price (the target variable) contains determinant variables of which some are continuous and others are 
categorical. Therefore, preprocessing data was carried out using standard Scaler.

        from sklearn.preprocessing import StandardScaler
        df= pd.read_csv('data/dataset_wout_price_outliers.csv')
        df_prepro = df.copy()
        columns = ['habitableSurface',  'price']
        for col in columns:
            df_prepro[col] = StandardScaler().fit_transform(np.array(df_prepro[col]).reshape(-1, 1))
        df_LR= df_prepro
 
 This way, the model prediction accuracy was improved. Here is an inference to illustrate how to use it.

    # calling important libraries and methods

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time

    # setting feature 
    feature_names = X.columns

    # model training and fitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = random_start)
    model_LR = LinearRegression(fit_intercept=True, copy_X=True, tol=1e-06, n_jobs=None, positive=False)   
    model_LR.fit(X_train, y_train)

    # measuring the accuracy
    MSE = mean_squared_error(y_test, Y_pred_LR)
    r2 = r2_score(y_test, Y_pred_LR)

    # Feature importances from best RandomForest
    coefficients_gl = best_model.coef_
    importance_gl = pd.DataFrame({
        'Feature': feature_names_lg,
        'Importance': coefficients_gl
    }).sort_values(by='Importance', key=abs, ascending=False)
 

Visuals

The model predictions were also plotted to have visual control over the patterns.
The Matplotlib is used to visualize the patterns on the test and the predicted values. Here 
is how the plot code is produced. 
---------------------------------------------
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predicted vs Actual')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Fit Line')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted House Prices')
        plt.legend()
        plt.grid(True)
        plt.show()
------------------------------------------------------

Contributors

Four persons have participated in the project work: Hanieh, Estifania, Fang, and Mengstu. After data preprocessing, every person developed his code to run these five models and finally combined the files into one py code file named main.py.


Timeline

This project was a five-day challenge work. The first day was dedicated to data preprocessing and the remaining for the code development and validation, documentation, and presentation.

Personal situation

We have participated in this project to deepen our understanding in machine learning. As AI data science trainees, we have learned a lot in using the various machine learning models. 

