import os
from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from PIL import Image # Import Pillow for original image dimension checking

# Determine the base directory of the application
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Initialize Flask app
# --- MODIFIED: Use absolute path for template_folder for robustness ---
app = Flask(__name__, template_folder=BASE_DIR)
# For local development only: A simple, non-secure secret key.
# In a production environment, this MUST be a strong, randomly generated key:
# app.config['SECRET_KEY'] = os.urandom(24) or a complex string.
app.config['SECRET_KEY'] = 'dev_key_for_local_use_only'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads') # Ensure uploads folder is also relative to BASE_DIR
app.config['MODEL_PATH'] = os.path.join(BASE_DIR, 'vgg16_model_balanced_run2.h5') # Ensure model path is also relative to BASE_DIR
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define IMAGE_SIZE for preprocessing. This MUST match the input size your model expects.
# Your main2.py currently trains a model with (224, 224, 3) input.
IMAGE_SIZE = (224, 224, 3)

# Define class labels in the exact order your model was trained on.
# !!! IMPORTANT: YOU MUST COPY THE OUTPUT OF 'CLASS INDICES FOR APP.PY CONFIGURATION'
# from main2.py's console after it finishes running and paste it here
# to ensure correct mapping of predictions to labels. Example:
# If main2.py output was {'Coccidiosis': 0, 'Healthy': 1, 'Newcastle Disease': 2, 'Salmonella': 3},
# then your list should be: ['Coccidiosis', 'Healthy', 'Newcastle Disease', 'Salmonella']
CLASS_LABELS = ['Coccidiosis', 'Healthy', 'Newcastle Disease', 'Salmonella'] # <--- **UPDATE THIS ACCORDING TO main2.py OUTPUT**


# Load the trained model once when the app starts
model = None
try:
    # Ensure the model file exists before attempting to load
    if not os.path.exists(app.config['MODEL_PATH']):
        raise FileNotFoundError(f"Model file not found at: {app.config['MODEL_PATH']}")

    loaded_model = load_model(app.config['MODEL_PATH'])

    # --- Input Shape Verification: THIS IS CRUCIAL FOR AVOIDING YOUR ERROR ---
    # Get the expected input shape of the first input layer of the loaded model
    model_input_shape = loaded_model.input_shape
    # We expect a shape like (None, 224, 224, 3)
    # The 'None' indicates the batch size, which can vary.
    expected_input_hw = (IMAGE_SIZE[0], IMAGE_SIZE[1]) # (224, 224)
    found_input_hw = (model_input_shape[1], model_input_shape[2]) # e.g., (128, 128) if old model

    if found_input_hw == expected_input_hw:
        model = loaded_model
        print(f"\n--- Model '{app.config['MODEL_PATH']}' loaded SUCCESSFULLY. ---")
        print(f"Model's EXPECTED Input Shape: {model_input_shape}")
        model.summary() # Print model summary to console for verification
    else:
        print(f"\n--- ERROR: Mismatched Model Input Shape! ---")
        print(f"Model '{app.config['MODEL_PATH']}' loaded, but its input shape {model_input_shape} "
              f"does not match the expected IMAGE_SIZE ({IMAGE_SIZE}).")
        print(f"Expected image dimensions for app.py processing: {expected_input_hw}")
        print(f"But the loaded model ACTUALLY expects: {found_input_hw}")
        print("Please ensure your 'vgg16_model_balanced_run2.h5' file was created by a main2.py run")
        print("where IMAGE_SIZE was (224, 224, 3) and that it's the correct, updated file.")
        model = None # Set model to None if loading fails, to prevent further errors gracefully

except Exception as e:
    print(f"\n--- CRITICAL ERROR: Failed to load model from '{app.config['MODEL_PATH']}' ---")
    print(f"Error details: {e}")
    print("Please ensure the model file exists, is in the correct path relative to app.py, and is a valid Keras model (.h5).")
    model = None # Set model to None if loading fails, to prevent further errors


# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to get original image dimensions (before Keras preprocessing)
def get_original_image_dimensions(image_path):
    """
    Loads an image using Pillow and returns its original width, height, and number of color channels.
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        img_array = np.array(img)
        # Check if it's a color image (H, W, C) or grayscale (H, W)
        channels = img_array.shape[2] if len(img_array.shape) == 3 else 1
        return width, height, channels
    except Exception as e:
        print(f"Error getting original dimensions for {image_path}: {e}")
        return None, None, None

# Function to preprocess the image and make a prediction
def predict_image(image_path):
    if model is None:
        return "Error: Prediction model not loaded correctly. Check server logs.", None

    # Before prediction, ensure CLASS_LABELS are configured
    if not CLASS_LABELS:
        return "Error: CLASS_LABELS not configured in app.py. Please check server logs.", None

    try:
        # Load and resize the image to the expected model input size (224x224)
        # This function handles the resizing, regardless of the original image's size.
        img = load_img(image_path, target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]))
        # Convert image to numpy array
        x = img_to_array(img)
        # Add batch dimension (1, H, W, C) required by model.predict
        x = np.expand_dims(x, axis=0)
        # Normalize pixel values (0-255 -> 0-1), as done during training
        x = x / 255.0

        # Make prediction
        predictions = model.predict(x, verbose=0)
        # Get the predicted class index (the index with the highest probability)
        predicted_class_index = np.argmax(predictions)
        # Get the class label using the index from your configured CLASS_LABELS
        predicted_label = CLASS_LABELS[predicted_class_index]
        # Get the confidence (probability) for the predicted class, formatted as percentage
        confidence = predictions[0][predicted_class_index] * 100

        return predicted_label, f"{confidence:.2f}%"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Prediction Failed: {e}", None

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/blog')
def blog_page():
    """Renders the blog page."""
    return render_template('blog.html')

@app.route('/blog-single')
def blog_single_page():
    """Renders a single blog post page."""
    return render_template('blog-single.html')

@app.route('/portfolio')
def portfolio_page():
    """Renders the portfolio page."""
    return render_template('portfolio-details.html')

@app.route('/predict')
def predict_page_route():
    """Renders the prediction page with the upload form."""
    return render_template('ipython.html',
                           uploaded_image=None,
                           detected_label=None,
                           confidence=None,
                           message=None,
                           original_width=None,
                           original_height=None,
                           original_channels=None)

@app.route('/predict', methods=['POST'])
def predict_post_route():
    """Processes the uploaded image and displays prediction."""
    detected_label = None
    uploaded_image_filename = None
    confidence = None
    message = None
    original_width = None
    original_height = None
    original_channels = None

    # Check if a file was uploaded
    if 'file' not in request.files:
        message = 'No file part in the request.'
    else:
        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            message = 'No selected file.'
        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(uploaded_file_path)
            uploaded_image_filename = filename # Store filename for template

            # --- Get original image dimensions of the uploaded file ---
            original_width, original_height, original_channels = get_original_image_dimensions(uploaded_file_path)

            # Perform prediction
            detected_label, confidence = predict_image(uploaded_file_path)

            # Handle errors from predict_image
            if detected_label and "Error:" in detected_label:
                message = detected_label
                detected_label = None # Clear these so they don't display a partial result
                confidence = None
            # Optional: Clean up the uploaded file after prediction (uncomment if you want to delete them)
            # os.remove(uploaded_file_path)

        else:
            message = 'Allowed image types are: png, jpg, jpeg, gif.'

    return render_template('ipython.html',
                           uploaded_image=uploaded_image_filename,
                           detected_label=detected_label,
                           confidence=confidence,
                           message=message,
                           original_width=original_width,
                           original_height=original_height,
                           original_channels=original_channels)

if __name__ == '__main__':
    # When running locally, you might want to enable debug mode for development
    # In a production environment, set debug=False and use a production-ready WSGI server
    app.run(debug=True)
