# Healthcare Weight Prediction using Linear Regression

This project demonstrates a supervised learning approach to predict an individual's weight based on height, age, and exercise level using a Linear Regression model.

## Problem Statement

The goal is to build a regression model that can accurately predict a person's `Weight_kg` given their `Height_cm`, `Age_years`, and `Exercise_Level`. This is a classic regression problem, suitable for supervised learning algorithms.

## Dataset

For this demonstration, a synthetic dataset has been generated to mimic realistic healthcare data related to human attributes and weight.
*(If you used a real dataset from Kaggle or UCI, replace this section with details about that dataset, including a link to its source. For example: "The dataset used is 'Body Fat Prediction' from Kaggle, available at: [https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction](https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction)")*

## Key Steps and Results

The `healthcare_weight_prediction.ipynb` notebook performs the following:

1.  **Data Loading and Preprocessing:**
    * A synthetic dataset (`pandas.DataFrame`) is created with `Height_cm`, `Age_years`, `Exercise_Level` as features and `Weight_kg` as the target variable.
    * Basic data inspection (`.head()`, `.info()`, `.describe()`, `.isnull().sum()`) is performed to understand the data structure and check for missing values.
    * The data is split into training and testing sets (80% training, 20% testing) to ensure robust model evaluation on unseen data.

    **--- Dataset Head ---**
    ```
       Height_cm  Age_years  Exercise_Level   Weight_kg
    0  177.450712         21               1  112.773243
    1  167.926035         45               3  128.127780
    2  179.715328         36               1  137.103811
    3  192.845448         59               1  158.307046
    4  166.487699         52               1  131.249809
    ```

    **--- Dataset Info ---**
    ```
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200 entries, 0 to 199
    Data columns (total 4 columns):
     #   Column          Non-Null Count  Dtype
    ---  ------          --------------  -----
     0   Height_cm       200 non-null    float64
     1   Age_years       200 non-null    int32
     2   Exercise_Level  200 non-null    int32
     3   Weight_kg       200 non-null    float64
    dtypes: float64(2), int32(2)
    memory usage: 4.8 KB
    ```

    **--- Descriptive Statistics ---**
    ```
           Height_cm   Age_years  Exercise_Level   Weight_kg
    count  200.000000  200.000000       200.00000  200.000000
    mean   169.388436   45.290000         3.09500  131.817692
    std     13.965059   14.222775         1.51905   12.943309
    min    130.703823   20.000000         1.00000   95.733915
    25%    159.423085   33.000000         2.00000  123.487852
    50%    169.937122   45.000000         3.00000  132.924295
    75%    177.512787   56.000000         5.00000  140.272285
    max    210.802537   69.000000         5.00000  166.962184
    ```

    **--- Missing Values Check ---**
    ```
    Height_cm         0
    Age_years         0
    Exercise_Level    0
    Weight_kg         0
    dtype: int64
    ```

    **Dataset Split:**
    ```
    Training set shape: (160, 3)
    Testing set shape: (40, 3)
    ```

2.  **Model Training:**
    * A `LinearRegression` model from `sklearn.linear_model` is initialized.
    * The model is trained (`.fit()`) on the training features (`X_train`) and their corresponding target values (`y_train`).

    **--- Model Training Complete ---**
    ```
    Model Coefficients: [ 0.68112694  0.15536528 -2.06694484]
    Model Intercept: 15.305702619713443
    ```
    * **Interpretation of Coefficients:**
        * For every 1 cm increase in `Height_cm`, `Weight_kg` is predicted to increase by approximately 0.68 kg (holding `Age_years` and `Exercise_Level` constant).
        * For every 1 year increase in `Age_years`, `Weight_kg` is predicted to increase by approximately 0.16 kg (holding `Height_cm` and `Exercise_Level` constant).
        * For every 1 unit increase in `Exercise_Level`, `Weight_kg` is predicted to decrease by approximately 2.07 kg (holding `Height_cm` and `Age_years` constant).

3.  **Evaluation:**
    * Predictions are made on the test set (`X_test`).
    * The model's performance is evaluated using **Mean Squared Error (MSE)**. MSE is a common metric for regression problems, representing the average of the squared differences between predicted and actual values. A lower MSE indicates a better fit of the model to the data.

    ```
    Mean Squared Error (MSE) on the test set: 72.52
    ```
    * **Interpretation of MSE:** An MSE of 72.52 suggests that, on average, the squared difference between the model's predictions and the actual weights is 72.52. The Root Mean Squared Error (RMSE), which is $\sqrt{72.52} \approx 8.52$ kg, provides an error measure in the original units of weight, indicating that typical prediction errors are around 8.52 kg.

## Reflection on the Problem and Solution

**Problem Analysis:**
Predicting weight is a continuous output problem, making it a regression task. Features like height, age, and exercise level are intuitively related to weight, suggesting that a regression model could find these relationships.

**Chosen Solution - Linear Regression:**
Linear Regression was chosen as a foundational model due to its simplicity, interpretability, and as a good baseline for regression problems. It assumes a linear relationship between the independent variables and the dependent variable.

**Strengths:**
* **Simplicity and Speed:** Easy to implement and computationally efficient.
* **Interpretability:** The coefficients of the linear regression model directly show the impact of each feature on the predicted weight, allowing for clear insights (e.g., how many kg weight changes for each cm increase in height).

**Limitations:**
* **Linearity Assumption:** Linear Regression performs well when the relationship between features and the target is approximately linear. In reality, biological systems might have more complex, non-linear interactions (e.g., the relationship between age and weight might not be strictly linear across all age ranges).
* **Sensitivity to Outliers:** Outliers can disproportionately influence the regression line, affecting model accuracy.

**Potential Improvements and Next Steps:**
* **Feature Engineering:** Explore creating new features (e.g., polynomial features if non-linear relationships are suspected, or interaction terms between existing features).
* **Other Regression Algorithms:** Experiment with more advanced algorithms like Decision Tree Regressors, Random Forest Regressors, Gradient Boosting Regressors (e.g., XGBoost, LightGBM), or Support Vector Regressors, which can capture more complex patterns.
* **Regularization:** For datasets with many features or high dimensionality, apply regularization techniques (Ridge or Lasso Regression) to prevent overfitting.
* **More Data:** Access to a larger, more diverse, and real-world dataset (e.g., from a medical study) would significantly improve the model's generalizability and robustness.
* **Hyperparameter Tuning:** For more complex models, tuning hyperparameters would be crucial to optimize performance.
* **Error Analysis:** Delve deeper into the residuals to understand where the model makes errors and identify potential areas for improvement (e.g., if errors are systematically higher for certain age groups).

This project serves as a solid foundation for understanding and implementing a basic regression task using supervised learning.
