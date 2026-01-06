import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================

# Load dataset
(xtrain, ytrain), (xtest, ytest) = fashion_mnist.load_data()

# Normalization: Scale pixel values to be between 0 and 1
# This helps the neural network converge faster and achieve better accuracy.
xtrain = xtrain.astype("float32") / 255.0
xtest = xtest.astype("float32") / 255.0

# Reshape data to (28, 28, 1)
# CNNs expect a 3D input (Height, Width, Channels). Since these are grayscale, channels=1.
xtrain = np.expand_dims(xtrain, -1)
xtest = np.expand_dims(xtest, -1)

print(f"Training data shape: {xtrain.shape}")
print(f"Test data shape: {xtest.shape}")

# Visualizing first 25 images to verify data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(xtrain[i].reshape(28,28), cmap="gray")
    plt.axis('off')
    plt.title(ytrain[i])
plt.show()

# ==========================================
# 2. Model Architecture
# ==========================================

def build_model():
    input_shape = (28, 28, 1)
    inputs = layers.Input(shape=input_shape)

    # First Convolutional Block
    # Padding='same' preserves dimensions; Relu introduces non-linearity
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    # Second Convolutional Block
    x = layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    # Regularization (Prevents Overfitting)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Dropout(rate=0.1)(x)

    # Classification Head (Fully Connected Layers)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Output Layer: 10 units for 10 classes, Softmax for probability distribution
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Fashion_MNIST_CNN")
    return model

model = build_model()
model.summary()

# ==========================================
# 3. Training
# ==========================================

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Training the model
history = model.fit(
    xtrain, ytrain,
    batch_size=64, # Standard batch size
    epochs=10,
    validation_split=0.1, # Monitor performance on unseen data during training
    verbose=1
)

# ==========================================
# 4. Evaluation & Saving
# ==========================================

print("\nEvaluating on Test Set...")
test_loss, test_acc = model.evaluate(xtest, ytest)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model so it can be used later without retraining
model.save("fashion_mnist_model.keras")
print("Model saved successfully as 'fashion_mnist_model.keras'")