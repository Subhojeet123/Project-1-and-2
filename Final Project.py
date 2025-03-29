#Project 1
#Drugs, Side Effects and Medical Condition arrow_drop_up


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r'C:\Users\HP\Downloads\drugs_side_effects_drugs_com.csv')

# Display basic information
print(df.head())
print(df.info())

# Data Preprocessing
df.dropna(inplace=True)  # Remove missing values

# Exploratory Data Analysis (EDA)
sns.countplot(y=df['medical_condition'], order=df['medical_condition'].value_counts().index[:10])
plt.title('Top 10 Medical Conditions')
plt.show()

# Encoding categorical data
df['drug'] = df['drug'].astype('category').cat.codes

# Encode target variable
le = LabelEncoder()
df['medical_condition'] = le.fit_transform(df['medical_condition'])

# Splitting dataset
X = df[['drug']]  # You can add more features here
y = df['medical_condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Decoding predictions back to original labels (optional)
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# Print some example predictions
print(pd.DataFrame({'Actual': y_test_labels[:10], 'Predicted': y_pred_labels[:10]}))


#Project 2
#Personalized Healthcare Recommendations


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(r'C:\Users\HP\Downloads\blood.csv')

# Display dataset info
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
df.dropna(inplace=True)

# Rename columns for clarity
df.columns = ["Recency", "Frequency", "Monetary", "Time", "Class"]

# Exploratory Data Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['Recency'], bins=20, kde=True)
plt.title("Recency Distribution")
plt.show()

# Define features and target
X = df.drop("Class", axis=1)  # 'Class' is the target variable
y = df["Class"]

# Identify numerical features (all columns are numerical)
numerical_features = ["Recency", "Frequency", "Monetary", "Time"]

# Create preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Function to generate recommendations
def generate_recommendations(patient_data):
    prediction = model_pipeline.predict(patient_data)
    recommendation_mapping = {
        0: "May not donate at this time",
        1: "Eligible for donation"
    }
    return recommendation_mapping.get(prediction[0], "Unknown recommendation")

# Example patient data
example_patient_data = pd.DataFrame({
    "Recency": [2],      # Months since last donation
    "Frequency": [10],   # Number of donations
    "Monetary": [2500],  # Total blood donated
    "Time": [60]         # Months since first donation
})

print("Personalized Recommendation:", generate_recommendations(example_patient_data))


