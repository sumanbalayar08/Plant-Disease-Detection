import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model_save_location = r"D:\MajorProject\Plant-Disease-Detection\EfficientNet"
model = tf.keras.models.load_model(model_save_location)

# Path to your test data
test_path = r"D:\MajorProject/Plant-Disease-Detection/test/"

# Create a data generator for testing
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

# Get true labels
test_labels = test_batches.classes
print("Test Labels",test_labels)
print(test_batches.class_indices)

# Predict using the model
predictions = model.predict(test_batches, steps=len(test_batches), verbose=0)

# Calculate accuracy
acc = 0
for i in range(len(test_labels)):
    actual_class = test_labels[i]
    if predictions[i][actual_class] > 0.5:
        acc += 1

accuracy = (acc / len(test_labels)) * 100
print("Accuracy:", accuracy, "%")
