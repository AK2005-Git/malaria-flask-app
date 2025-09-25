import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
train_dir = "C:/Users/ACT-AIDS-T3-23/Desktop/DL1/Dataset/Train"
test_dir = "C:/Users/ACT-AIDS-T3-23/Desktop/DL1/Dataset/Test"

# Image augmentation and rescale for training + validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2  # 20% validation
)

# Only rescale for test
test_datagen = ImageDataGenerator(rescale=1./255)

# Train generator (80% data)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Print the class indices mapping
print(train_generator.class_indices)

# Validation generator (20% data)
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Checkpoint to save best weights
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train with validation and checkpoint
model.fit(
    train_generator,
    epochs=35,
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

# Save final model (best weights would be in best_model.h5)
model.save('malaria_model.h5')
