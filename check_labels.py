import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get current file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct path
train_dir = os.path.join(BASE_DIR, 'data', 'train')

datagen = ImageDataGenerator(rescale=1./255)

data = datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

print(data.class_indices)

# {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}