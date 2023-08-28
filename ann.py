import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Load your data
data = pd.read_csv('patient_1.csv')  # Replace with the actual file name

# Define features (X) and target (y)
X = data[['Respiratory_Rate', 'Spo2', 'Heart_Rate', 'Temperature', 'High_BP']]
y = data['label']

# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Sequential model
model = Sequential()

# Add an input layer
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))

# Add one or more hidden layers (customize as needed)
model.add(Dense(units=32, activation='relu'))

# Add the output layer with appropriate units for your classes (3 in this case)
model.add(Dense(units=3, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=-1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_classes)
confusion = confusion_matrix(y_test, y_pred_classes)
class_report = classification_report(y_test, y_pred_classes, target_names=['Normal', 'Moderate', 'Severe'])

# Print confusion matrix, accuracy, precision, and recall
print("Confusion Matrix:")
print(confusion)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(class_report)
# Plot the confusion matrix
plt.figure(figsize=(4, 4))
plt.imshow(confusion, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
tick_positions = np.arange(3)
class_names = ['Normal', 'Moderate', 'Severe']  # Replace with your class names
plt.xticks(tick_positions, class_names)
plt.yticks(tick_positions, class_names)
for i in range(3):
    for j in range(3):
        plt.text(j, i, str(confusion[i, j]), ha='center', va='center',
                 color='white' if confusion[i, j] > confusion.max() / 2 else 'black')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot training history (optional)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()
