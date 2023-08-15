import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Import the Desired Version of EfficientNet
from tensorflow.keras.applications import EfficientNetB0

NUM_CLASSES = 10

train_path = r"D:\MajorProject/Plant-Disease-Detection/train/"
valid_path = r"D:\MajorProject/Plant-Disease-Detection/validation/"
test_path = r"D:\MajorProject/Plant-Disease-Detection/test/"

epochs = 2
steps_per_epoch = 100  # Set the desired number of steps per epoch

model_save_location = r"D:\MajorProject\Plant-Disease-Detection\EfficientNet"

#Building the Model

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

#Plotting the Accuracy and Loss

def plot_accuracy_and_loss(history):
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


def build_model(NUM_CLASSES):
    inputs = layers.Input(shape=(224, 224, 3))
    x = img_augmentation(inputs)

    #Using the imported version of EfficientNet
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


#Train Model

model = build_model(NUM_CLASSES)

unfreeze_model(model)

# Load previously trained model weights if available
try:
    model = tf.keras.models.load_model(model_save_location)
    print("Loaded previously trained model weights.")
except:
    print("No previous model weights found. Training from scratch.")


#Data Generators
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)


# Callback to save model weights during training
checkpoint = ModelCheckpoint(
    model_save_location,
    monitor='val_accuracy',  # Save based on validation accuracy
    save_best_only=True,     # Only save the best model
    mode='max',              # Maximize validation accuracy
    verbose=1
)

hist = model.fit(train_batches, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=valid_batches, verbose=1)

#Saving Model after training
model.save(model_save_location )
plot_accuracy_and_loss(hist)

print("Model Trained and Saved Successfully")