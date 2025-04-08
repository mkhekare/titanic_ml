# Titanic Survival Prediction

## Overview

This project aims to predict the survival of passengers aboard the Titanic using various features from the dataset. The data includes information about passengers such as age, gender, ticket class, and more.

## Data Description

- **Survived**: 0 = Did not survive, 1 = Survived
- **Pclass**: Ticket class (1 = First class, 2 = Second class, 3 = Third class). This can also be seen as a proxy for socio-economic status.
- **Sex**: Male or female
- **Age**: Age in years, fractional if less than 1
- **SibSp**: Number of siblings or spouses aboard the Titanic
- **Parch**: Number of parents or children aboard the Titanic
- **Ticket**: Passenger ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Point of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Libraries Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning models and metrics.
- **Seaborn & Matplotlib**: For data visualization.

## Steps in the Project

1. **Import Libraries**: Load necessary libraries for data manipulation and modeling.
   
   ```python
   import pandas as pd
   import numpy as np
   from sklearn.preprocessing import LabelEncoder
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import classification_report, confusion_matrix
   import seaborn as sns
   import matplotlib.pyplot as plt
   ```

2. **Load Data**: Read the training and test datasets.

   ```python
   train = pd.read_csv("titanic_train.csv")
   test = pd.read_csv("titanic_test.csv")
   ```

3. **Explore Data**: Check the shape and structure of the datasets.

4. **Handle Missing Values**: Identify and fill missing values, particularly in the `Age` and `Embarked` columns.

5. **Feature Engineering**: Drop unnecessary features (e.g., `Ticket`, `Cabin`, `Name`, `PassengerId`) and encode categorical variables (`Sex`, `Embarked`).

6. **Data Splitting**: Separate the combined dataset back into training and test sets.

7. **Model Training**: Train a Logistic Regression model on the training dataset.

   ```python
   lg = LogisticRegression()
   lg.fit(X, y1)
   ```

8. **Model Evaluation**: Evaluate the model using confusion matrix and classification report.

9. **Visualize Predictions**: Create visualizations to understand the predictions made by the model.

   ```python
   sns.countplot(x=y_pred, edgecolor='black')
   plt.title("Countplot of Predictions")
   plt.xlabel("Predicted Values")
   plt.ylabel("Count")
   plt.show()
   ```

## Logical Error Detection and Rectification

### Task

1. **Identify** the logical error in the code.
2. **Explain** why it is an error and the impact it has on the results.
3. **Correct** the code to eliminate the logical error.
4. **Compare Results**:
   - Show the difference in the output before and after fixing the error.
   - Provide a brief analysis of how the correction improves the results.

## Conclusion

This project provides insights into the factors affecting survival on the Titanic and demonstrates the application of machine learning techniques to real-world problems. The logical error detection and correction process is crucial for improving model accuracy and ensuring reliable predictions.

## License

This project is licensed under the MIT License.
