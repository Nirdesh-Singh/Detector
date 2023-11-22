import tensorflow as tf
from flask import Flask, render_template, request
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('cnn_model.h5')

# Define the target image size expected by the model
img_height, img_width = 180, 180

# Define the class names
class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array /= 255.0  # Normalize pixel values to be between 0 and 1
    return img_array

# Function to make predictions
def make_prediction(img_array):
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return predicted_class, confidence

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image upload and classification
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        # Save the uploaded file
        file_path = "uploads/" + file.filename
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make predictions
        predicted_class, confidence = make_prediction(img_array)

        # Display the result
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'file_path': file_path
        }
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
