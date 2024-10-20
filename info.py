import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\gadda\OneDrive\Desktop\diabetes project\diabetes_prediction_dataset.csv")

# Check for missing values and handle them (optional: fill missing values with the median)
print(df.isnull().sum())


# Drop duplicate rows
df = df.drop_duplicates()

# Descriptive statistics and data preview
print(df.describe())
print(df.head())

# Convert categorical variables to dummy variables

# df = df.fillna(df.mean())
# df.drop('gender', axis=1,errors='ignore')
# df.drop('smoking_history',axis=1,errors='ignore')
# df = df.dropna(axis=0)

# Split the dataset into features (X) and target variable (y)
x = df.drop(['diabetes', 'gender','smoking_history'],axis=1)
y = df['diabetes']

# Scale the features using StandardScaler


# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = 0, strategy='mean')

# Initialize and train the logistic regression model
import xgboost

classifier = xgboost.XGBClassifier()


# Initialize and train the logistic regression model
 

classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(x_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(classifier, r"C:\Users\gadda\OneDrive\Desktop\diabetes project\mod.pkl")



