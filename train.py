import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("Loading MNIST dataset...")

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Building CNN model...")

# Build CNN model
model = models.Sequential([
   layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
   layers.BatchNormalization(),
   layers.Conv2D(32, (3,3), activation='relu'),
   layers.MaxPooling2D((2,2)),
   layers.Dropout(0.25),

   layers.Conv2D(64, (3,3), activation='relu'),
   layers.BatchNormalization(),
   layers.Conv2D(64, (3,3), activation='relu'),
   layers.MaxPooling2D((2,2)),
   layers.Dropout(0.25),

   layers.Flatten(),
   layers.Dense(128, activation='relu'),
   layers.Dropout(0.5),
   layers.Dense(10, activation='softmax') 
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training model...")

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(x_train)
# Train model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=12,
    validation_data=(x_test, y_test)
)

# Save model
model.save("digit_model.h5")

print("✅ Model saved as digit_model.h5")