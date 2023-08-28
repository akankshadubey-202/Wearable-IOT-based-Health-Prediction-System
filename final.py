import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('patient_1.csv')  # Replace with the actual file name

# Define features (X) and target (y)
X = data[['Respiratory_Rate', 'Spo2', 'Heart_Rate', 'Temperature', 'High_BP']]
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the XGBoost classifier
xgb_classifier = xgb.XGBClassifier(random_state=42)

# Train the XGBoost model
xgb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = xgb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Normal','low' 'Moderate', 'Severe'])

# Print confusion matrix, accuracy, precision, and recall
print("Confusion Matrix:")
print(confusion)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(class_report)
# Extract precision, recall, and accuracy from the classification report
report_dict = classification_report(y_test, y_pred, target_names=['Normal', 'Moderate', 'Severe'], output_dict=True)

precision = [report_dict[label]['precision'] for label in ['Normal', 'Moderate', 'Severe']]
recall = [report_dict[label]['recall'] for label in ['Normal', 'Moderate', 'Severe']]
accuracy = accuracy_score(y_test, y_pred)
# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 4))  # Increase the figure size for a 4x4 matrix
plt.imshow(confusion_mat, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Set the tick positions and labels for x and y axes for 4 classes
tick_positions = np.arange(3)
class_names = ['Normal', 'Moderate', 'Severe']  # Replace with your class names

plt.xticks(tick_positions, class_names)
plt.yticks(tick_positions, class_names)

# Add text annotations within the cells
for i in range(3):
    for j in range(3):
        plt.text(j, i, str(confusion_mat[i, j]), ha='center', va='center',
                 color='white' if confusion_mat[i, j] > confusion_mat.max() / 2 else 'black')

plt.show()
# Display the classification report as text on the plot
# Create a bar chart for precision, recall, and accuracy
labels = ['Normal', 'Moderate', 'Severe']
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, [accuracy] * 3, width, label='Accuracy')

ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and Accuracy by Class')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
# Input case for prediction
input_data = pd.DataFrame({
    'Respiratory_Rate': [20],  # Replace with the desired value
    'Spo2': [95],  # Replace with the desired value
    'Heart_Rate': [75],  # Replace with the desired value
    'Temperature': [36.5],  # Replace with the desired value
    'High_BP': [120]  # Replace with the desired value
})

# Use the trained XGBoost model to predict the label for the input case
predicted_label = xgb_classifier.predict(input_data)

# Print the predicted label
print("Predicted Label for Input Case:", predicted_label[0])
