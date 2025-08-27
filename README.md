# iris-flower-classification
A machine learning project for classifying iris flower species using scikit-learn.
# iris_classifier.py
# A simple machine learning project for classifying Iris flower species using scikit-learn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Iris classification accuracy: {accuracy:.2f}")

# Predict species for new sample (optional)
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example features
predicted_class = iris.target_names[clf.predict(sample)[0]]
print(f"Predicted class for sample {sample}: {predicted_class}")
