import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Keras/TensorFlow imports
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers.legacy import Adam # Explicitly use legacy Adam if preferred


# --- Configuration ---
# THIS IS THE ABSOLUTELY CRITICAL SETTING FOR YOUR MODEL'S INPUT SHAPE.
# It MUST be (224, 224, 3) for your VGG16 model to expect 224x224 images.
IMAGE_SIZE = (224, 224, 3) # Image size for better accuracy with reduced data

# IMPORTANT: SET YOUR LOCAL DATASET ROOT PATH HERE
# This is where your 'train', 'val', 'test' subdirectories are located.
# The .h5 model file will also be saved directly into this DATASET_ROOT_PATH.
DATASET_ROOT_PATH = r"C:\Users\vuyyu\Downloads\archive (16)\data\data" # <--- **CHANGE THIS TO YOUR ACTUAL PATH**
# Double-check that this path is correct and contains 'train', 'val', 'test' subdirectories.

# Define the paths to your train, validation, and test sets within the root path
train_data_dir = os.path.join(DATASET_ROOT_PATH, "train")
val_data_dir = os.path.join(DATASET_ROOT_PATH, "val")
test_data_dir = os.path.join(DATASET_ROOT_PATH, "test")

# --- Performance Optimizations ---
# Reduce BATCH_SIZE to prevent Out-Of-Memory (OOM) errors.
# Start with a very small batch size (e.g., 32 or 16), and if it runs,
# gradually increase it to find the maximum your GPU can handle.
BATCH_SIZE = 32 # Reduced for better memory stability, you can increase if stable.

# Configure GPU memory growth: This prevents TensorFlow from allocating all GPU memory at once,
# allowing it to grow as needed. This can help prevent OOM errors.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured with memory growth.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Error configuring GPU memory growth: {e}")
else:
    print("No GPU devices found. Training will proceed on CPU.")


# Enable Mixed Precision Training (if GPU supports it) - essential for speed on modern GPUs
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled. Training will use mixed_float16 where possible.")
except Exception as e:
    print(f"Could not enable mixed precision: {e}. Ensure you have a compatible GPU and TensorFlow version.")
    print("Proceeding with default float32 precision.")

# --- Data Loading and Preprocessing ---

# Function to create DataFrame from a directory (infers labels from subfolder names)
def create_dataframe_from_directory(data_dir):
    filepaths = []
    labels = []
    if not os.path.exists(data_dir):
        print(f"Warning: Directory not found: {data_dir}. Please check DATASET_ROOT_PATH and subdirectories.")
        return pd.DataFrame({'path': [], 'label': []}) # Return empty DataFrame to avoid errors

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    filepaths.append(img_path)
                    labels.append(class_name)
    df = pd.DataFrame({'path': filepaths, 'label': labels})
    return df

print("Creating DataFrames from local directories (this might take a while for large datasets)...")
train_df_full = create_dataframe_from_directory(train_data_dir)
val_df_full = create_dataframe_from_directory(val_data_dir)
test_df_full = create_dataframe_from_directory(test_data_dir)

# *** IMPORTANT: Data Sub-sampling for faster testing ***
# These percentages are small to make training faster for testing purposes.
# Adjust these higher for full training accuracy.
TRAIN_DATA_PERCENT = 0.10 # 10% of full train data
VAL_DATA_PERCENT = 0.025 # 2.5% of full val data
TEST_DATA_PERCENT = 0.025 # 2.5% of full test data

# Ensure we don't try to sample more data than available
train_df = train_df_full.sample(frac=min(TRAIN_DATA_PERCENT, 1.0), random_state=42).reset_index(drop=True)
val_df = val_df_full.sample(frac=min(VAL_DATA_PERCENT, 1.0), random_state=42).reset_index(drop=True)
test_df = test_df_full.sample(frac=min(TEST_DATA_PERCENT, 1.0), random_state=42).reset_index(drop=True)

print(f"Using {len(train_df)} images for training (approx {TRAIN_DATA_PERCENT*100}% of full train data).")
print(f"Using {len(val_df)} images for validation (approx {VAL_DATA_PERCENT*100}% of full val data).")
print(f"Using {len(test_df)} images for testing (approx {TEST_DATA_PERCENT*100}% of full test data).")

# Image Data Generators for augmentation and batching
# The target_size here is directly IMAGE_SIZE[0], IMAGE_SIZE[1] which is (224, 224).
gen = ImageDataGenerator(rescale=1./255)

train_gen = gen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), # This ensures images are resized to 224x224
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    batch_size=BATCH_SIZE,
    seed=123
)

test_gen = gen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label',
    target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), # This ensures images are resized to 224x224
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    batch_size=BATCH_SIZE,
    seed=123
)

val_gen = gen.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='label',
    target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]), # This ensures images are resized to 224x224
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False,
    batch_size=BATCH_SIZE,
    seed=123
)

# Get the number of classes and their names from the training generator
num_classes = len(train_gen.class_indices)
labels = list(train_gen.class_indices.keys())
print(f"\nNumber of classes detected by ImageDataGenerator: {num_classes}")
print(f"Detected class labels (in alphabetical order based on subfolder names): {labels}")
# --- IMPORTANT: Print class_indices for app.py configuration ---
# You NEED to use this exact order for CLASS_LABELS in app.py
print("\n--- COPY THIS FOR CLASS_LABELS IN app.py ---")
print(train_gen.class_indices) # This dictionary shows the exact mapping of labels to integer indices
print("----------------------------------------------\n")


# Callbacks for training (EarlyStopping is crucial for faster, optimized training)
# EarlyStopping will halt training if validation loss doesn't improve for 'patience' epochs.
# This prevents overfitting and saves time, even if 'epochs' is set high.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=5e-5, verbose=1)


# --- VGG16 Model Definition and Training ---
print("\n--- Initializing and Training VGG16 Model ---")
# The input_shape here directly uses IMAGE_SIZE (224, 224, 3), ensuring the model expects this size.
vgg = VGG16(input_shape=IMAGE_SIZE, weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False # Freeze base VGG16 layers to use pre-trained features

x = vgg.output
x = GlobalAveragePooling2D()(x) # Reduces spatial dimensions to 1x1, keeps channels
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax', dtype='float32')(x) # Output layer for classification

model_vgg16 = Model(inputs=vgg.input, outputs=predictions, name="VGG16_Model")

model_vgg16.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\n--- VGG16 Model Summary (CONFIRM INPUT SHAPE HERE: (None, 224, 224, 3)) ---")
model_vgg16.summary() # Crucial: Check the first layer's input shape here!

# --- Training Block ---
try:
    print(f"\nStarting VGG16 model training for up to {15} epochs.")
    print("Training will stop earlier if validation loss does not improve for 5 consecutive epochs (EarlyStopping).")
    history_vgg16 = model_vgg16.fit(
        train_gen,
        validation_data=val_gen,
        epochs=1, # Set a reasonable number of epochs. EarlyStopping handles stopping.
        callbacks=[early_stopping, reduce_lr]
    )
    print("\nTraining completed successfully or stopped by EarlyStopping.")

    # Save the VGG16 model
    # Model will be saved directly into your DATASET_ROOT_PATH.
    # Make sure this path is accessible and writeable.
    model_vgg16_path = os.path.join(DATASET_ROOT_PATH, "vgg16_model_balanced_run2.h5")
    model_vgg16.save(model_vgg16_path)
    print(f"\n--- VGG16 Model saved SUCCESSFULLY to: {model_vgg16_path} ---")

    # Plot training history (accuracy and loss over epochs) - only if history is available
    if history_vgg16.history:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history_vgg16.history['accuracy'], label='Training Accuracy')
        plt.plot(history_vgg16.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history_vgg16.history['loss'], label='Training Loss')
        plt.plot(history_vgg16.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"\nERROR during model training: {e}")
    print("Please check your DATASET_ROOT_PATH, ensure enough memory (RAM/GPU VRAM), and verify dataset integrity.")
    print("If you're running out of memory, try reducing BATCH_SIZE or TRAIN_DATA_PERCENT further.")


# --- Predictor Function (for evaluating on test set within main2.py) ---
def predictor(model_to_evaluate, test_generator_for_prediction):
    classes = list(test_generator_for_prediction.class_indices.keys())
    class_count_len = len(classes)

    print(f"\n--- Making predictions with {model_to_evaluate.name} on test data ---")
    test_generator_for_prediction.reset() # Reset generator to ensure predictions start from the beginning
    preds = model_to_evaluate.predict(test_generator_for_prediction, verbose=1)

    loss, accuracy = model_to_evaluate.evaluate(test_generator_for_prediction, verbose=1)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    pred_indices = np.argmax(preds, axis=1) # Get index of max probability
    true_labels = test_generator_for_prediction.labels # Get true labels from generator

    ytrue = np.array(true_labels, dtype='int')
    ypred = np.array(pred_indices, dtype='int')

    # Plot the confusion matrix
    cm = confusion_matrix(ytrue, ypred)
    plt.figure(figsize=(class_count_len * 2, class_count_len * 2))
    sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_to_evaluate.name}")
    plt.show()

    # Classification Report
    clr = classification_report(ytrue, ypred, target_names=classes, digits=4)
    print("\nClassification Report:\n----------------------\n", clr)


# Evaluate the VGG16 model (only if training was successful)
print("\n--- Evaluating VGG16 Model on Test Data ---")
if 'model_vgg16' in locals() and model_vgg16:
    predictor(model_vgg16, test_gen)
else:
    print("Model not available for evaluation due to prior error during training.")


# --- Example of single image prediction (using a sample from your test dataset) ---
if not test_df.empty:
    sample_image_path = test_df['path'].iloc[0] # Get the path of the first image in the test set
    print(f"\n--- Predicting for a sample image from the test set: {sample_image_path} ---")

    def get_model_prediction_single_image(model_to_use, image_path, class_labels):
        try:
            # Load and resize the image to (224, 224) - same as model input
            img = load_img(image_path, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]))
            x = img_to_array(img) # Convert to numpy array
            x = np.expand_dims(x, axis=0) # Add batch dimension (1, H, W, C)
            x = x / 255.0 # Normalize pixel values

            predictions = model_to_use.predict(x, verbose=0)
            predicted_class_index = np.argmax(predictions) # Get the index of the highest probability
            predicted_label = class_labels[predicted_class_index] # Map index to label
            confidence = predictions[0][predicted_class_index] * 100 # Get confidence percentage

            return predicted_label, f"{confidence:.2f}%"
        except Exception as e:
            print(f"Error predicting for image {image_path}: {e}")
            return "Prediction Failed", "N/A"

    # Only predict if model is available
    if 'model_vgg16' in locals() and model_vgg16:
        predicted_label, conf = get_model_prediction_single_image(model_vgg16, sample_image_path, labels)
        print(f"Prediction by VGG16 for '{sample_image_path}': {predicted_label} (Confidence: {conf})")
    else:
        print("Model not available for sample prediction.")
else:
    print("\nNo images available in the reduced test dataset for sample prediction.")
