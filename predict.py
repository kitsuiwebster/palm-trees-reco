import numpy as np
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Set up logging
logging.basicConfig(level=logging.INFO)

# Emojis for outputs
emojis = {
    'start': '🚀',
    'load_success': '✅',
    'error': '❌',
    'process_image': '📸',
    'prediction': '🔍'
}

# Default paths
DEFAULT_MODEL_PATH = './models/palm_trees_reco.h5'
DEFAULT_IMAGE_PATH = './test-data/test.jpg'

# Load the trained model
def load_model_from_file(model_path=DEFAULT_MODEL_PATH):
    try:
        model = load_model(model_path)
        logging.info("Model loaded successfully.")
        print(f"{emojis['load_success']} Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        print(f"{emojis['error']} Failed to load model: {e}")
        exit()

# Load and preprocess the image
def load_and_preprocess_image(img_path=DEFAULT_IMAGE_PATH):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        logging.info("Image preprocessed successfully.")
        print(f"{emojis['process_image']} Image preprocessed successfully.")
        return img_array
    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")
        print(f"{emojis['error']} Error processing image {img_path}: {e}")
        exit()

# Make a prediction
def predict_image_class(model, img_array):
    try:
        predictions = model.predict(img_array)
        class_names = ['Phoenix canariensis', 'Phoenix dactylifera']
        predicted_class = class_names[np.argmax(predictions)]
        return predicted_class
    except Exception as e:
        logging.error(f"Failed to make prediction: {e}")
        print(f"{emojis['error']} Failed to make prediction: {e}")
        exit()
    

if __name__ == "__main__":
    print(f"{emojis['start']} Starting the palm species prediction system...")
    model = load_model_from_file()
    img_array = load_and_preprocess_image()
    prediction = predict_image_class(model, img_array)
    print(f"{emojis['prediction']} The predicted species is: {prediction} 🌴")
