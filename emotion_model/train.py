import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Base directory (emotion_model folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct paths based on your structure
train_dir = os.path.join(BASE_DIR, '..', 'data', 'train')
test_dir = os.path.join(BASE_DIR, '..', 'data', 'test')
model_path = os.path.join(BASE_DIR, '..', 'models', 'emotion_model.keras')

# Create models folder if not exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# If model exists → load
if os.path.exists(model_path):
    print("✅ Loading existing model...")
    model = load_model(model_path)

else:
    print("🚀 Training new model...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical'
    )


    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48,48),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical'
    )

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_data,
        epochs=10,
        validation_data=test_data
    )

    model.save(model_path)
    print("💾 Model saved at:", model_path)