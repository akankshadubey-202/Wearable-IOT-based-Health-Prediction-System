import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load the synthetic temperature dataset
data = pd.read_csv('synthetic_temperature_data.csv')

# Define features and labels
X = data[['Temperature_Celsius', 'Temperature_Fahrenheit']]
y = data['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with the desired number of neighbors (e.g., 3)
knn_model = KNeighborsClassifier(n_neighbors=3)

# Train the KNN model
knn_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)
# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 7))  # Increase the figure size for a 4x4 matrix
plt.imshow(confusion_mat, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Set the tick positions and labels for x and y axes for 4 classes
tick_positions = np.arange(4)
class_names = ['Normal', 'Mild', 'Moderate', 'Sever']  # Replace with your class names

plt.xticks(tick_positions, class_names)
plt.yticks(tick_positions, class_names)

# Add text annotations within the cells
for i in range(4):
    for j in range(4):
        plt.text(j, i, str(confusion_mat[i, j]), ha='center', va='center',
                 color='white' if confusion_mat[i, j] > confusion_mat.max() / 2 else 'black')

plt.show()

# Test with a single input case
input_case = pd.DataFrame({
    'Temperature_Celsius': [36],
    'Temperature_Fahrenheit':[96.8]
})

# Predict the label for the input case
predicted_label = knn_model.predict(input_case)
print("Predicted Label for Input Case:", predicted_label[0])

import joblib
joblib.dump(knn_model, 'temp_model.pkl')