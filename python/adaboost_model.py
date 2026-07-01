import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the shuffled dataset
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cp2_shuffled.csv"
df = pd.read_csv(DATA_PATH)

# Print the column names to identify the target variable
print(df.columns)

# Separate features and target variable
X = df.drop(columns=['Heart_ stroke', 'Unnamed: 16'])  # Drop irrelevant column if necessary
y = df['Heart_ stroke']

# Print dataset shape before handling NaN
print("Shape of features (X) before NaN handling:", X.shape)
print("Shape of target (y) before NaN handling:", y.shape)

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Handle missing values for numeric columns using mean strategy
imputer_numeric = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(imputer_numeric.fit_transform(X[numeric_cols]), columns=numeric_cols)

# Handle missing values for categorical columns using the most frequent strategy
imputer_categorical = SimpleImputer(strategy='most_frequent')
X_categorical = pd.DataFrame(imputer_categorical.fit_transform(X[categorical_cols]), columns=categorical_cols)

# Concatenate numeric and categorical columns back together
X_clean = pd.concat([X_numeric, X_categorical], axis=1)

# Fill any missing values in the target variable (if necessary)
y.fillna(y.mode()[0], inplace=True)  # Filling NaN in target column with mode

# Print dataset shape after handling NaN
print("Shape of features (X) after NaN handling:", X_clean.shape)
print("Shape of target (y) after NaN handling:", y.shape)

# Convert categorical variables to numeric (one-hot encoding)
X_encoded = pd.get_dummies(X_clean, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Check the shape of the training and testing data
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# AdaBoost ensemble modeling implementation
# Initialize AdaBoost classifier
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the model on the training data
ada_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = ada_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("AdaBoost Model Accuracy: {:.2f}%".format(accuracy * 100))

# Display classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



#rotation forest





