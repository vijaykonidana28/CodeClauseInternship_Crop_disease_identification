import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
DATASET_DIR = 'PlantVillage'
MODEL_PATH = 'model/crop_disease_model.h5'

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=(150, 150), batch_size=16, class_mode='categorical', subset='training')

val_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=(150, 150), batch_size=16, class_mode='categorical', subset='validation')

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save
os.makedirs('model', exist_ok=True)
model.save(MODEL_PATH)
