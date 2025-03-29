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
