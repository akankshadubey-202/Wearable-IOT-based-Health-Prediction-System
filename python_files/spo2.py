import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load synthetic SpO2 data and labels
data = pd.read_csv('synthetic_spo2_data.csv')

# Define features (SpO2 level) and labels
X = data[['SpO2 Level']]
y = data['Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier (e.g., with k=5)
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Print confusion matrix and accuracy
print("Confusion Matrix:")
print(confusion)
print(f"Accuracy: {accuracy:.2f}")

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
    'SpO2 Level': [91]
})

# Predict the label for the input case
predicted_label = knn.predict(input_case)
print("Predicted Label for Input Case:", predicted_label[0])
import joblib
joblib.dump(knn, 'spo2_model.pkl')