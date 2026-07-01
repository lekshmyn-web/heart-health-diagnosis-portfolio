import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import resample

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cp2_shuffled.csv"
df = pd.read_csv(DATA_PATH)


# Separate features and target variable
X = df.drop(columns=['Heart_ stroke', 'Unnamed: 16'])  # Drop irrelevant column if necessary
y = df['Heart_ stroke']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Handle missing values
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Handle missing values for numeric columns
imputer_numeric = SimpleImputer(strategy='mean')
X_numeric = pd.DataFrame(imputer_numeric.fit_transform(X[numeric_cols]), columns=numeric_cols)

# Handle missing values for categorical columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
X_categorical = pd.DataFrame(imputer_categorical.fit_transform(X[categorical_cols]), columns=categorical_cols)

# Concatenate cleaned columns
X_clean = pd.concat([X_numeric, X_categorical], axis=1)

# Convert categorical variables to numeric (one-hot encoding)
X_encoded = pd.get_dummies(X_clean, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### AdaBoost Model
# Initialize AdaBoost classifier
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train the model on the training data
ada_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred_ada = ada_model.predict(X_test_scaled)

# Evaluate the AdaBoost model
ada_accuracy = accuracy_score(y_test, y_pred_ada)
print("AdaBoost Model Accuracy: {:.2f}%".format(ada_accuracy * 100))

# Display classification report and confusion matrix for AdaBoost
print("\nAdaBoost Classification Report:")
print(classification_report(y_test, y_pred_ada))

print("\nAdaBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ada))


### Rotation Forest Model
# Custom Rotation Forest Classifier
class RotationForest:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape

        for _ in range(self.n_estimators):
            # Bootstrap sample
            X_sample, y_sample = resample(X, y)

            # PCA for rotation
            pca = PCA(n_components=n_features)
            X_rotated = pca.fit_transform(X_sample)

            # Train a decision tree on the rotated data
            tree = DecisionTreeClassifier()
            tree.fit(X_rotated, y_sample)
            self.trees.append((tree, pca))

    def predict(self, X):
        # Collect predictions from each tree
        predictions = np.zeros((X.shape[0], self.n_estimators))

        for i, (tree, pca) in enumerate(self.trees):
            X_rotated = pca.transform(X)
            predictions[:, i] = tree.predict(X_rotated)

        # Return the majority vote
        return np.array([np.bincount(pred.astype(int)).argmax() for pred in predictions])


# Initialize and train the Rotation Forest model
rot_forest_model = RotationForest(n_estimators=10)
rot_forest_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_rot_pred = rot_forest_model.predict(X_test_scaled)

# Evaluate the Rotation Forest model
rot_accuracy = accuracy_score(y_test, y_rot_pred)
print("\nRotation Forest Model Accuracy: {:.2f}%".format(rot_accuracy * 100))

# Display classification report and confusion matrix for Rotation Forest
print("\nRotation Forest Classification Report:")
print(classification_report(y_test, y_rot_pred))

print("\nRotation Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_rot_pred))

