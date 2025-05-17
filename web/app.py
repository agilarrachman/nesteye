# ... existing imports ...
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask import redirect, request, render_template, url_for
from werkzeug.utils import secure_filename
# ... other imports ...

# Set up model path with proper path handling
model_path = os.path.join(os.path.dirname(__file__), 'model', 'nest_model.h5')
print(os.path.dirname(__file__))

# Try-except block for model loading to handle errors gracefully
try:
    model = load_model(model_path)
    print(f"Model successfully loaded from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Provide a fallback or exit gracefully

# Define class names based on your model's output classes
class_names = ['Elang Jawa', 'Jalak Bali', 'Kakatua Putih', 'Nuri Hitam']

# Function to preprocess image and make prediction
def predict_image(img_path):
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Apply softmax to get proper probability distribution
        from scipy.special import softmax
        probabilities = softmax(predictions[0])
        
        predicted_class_index = np.argmax(probabilities)
        predicted_class = class_names[predicted_class_index]
        confidence = float(probabilities[predicted_class_index]) * 100
        
        # Jika confidence terlalu tinggi untuk gambar yang tidak seharusnya,
        # kita bisa menambahkan threshold deteksi
        if confidence > 95:
            # Periksa distribusi probabilitas, jika terlalu yakin pada satu kelas
            # padahal gambar mungkin bukan dari kelas manapun
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            if entropy < 0.1:  # Entropy rendah = terlalu yakin
                print(f"Warning: Model terlalu yakin ({confidence}%) tetapi entropy rendah ({entropy})")
                confidence = min(confidence, 70.0)  # Batasi confidence
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Unknown", 0.0


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Make prediction using the model
        predicted_class, confidence = predict_image(file_path)
        
        # Create crop data for the template
        crops_data = [{
            'img': filename,
            'label': predicted_class,
            'confidence': round(confidence, 2),
            'is_highest': True
        }]
        
        # Render the results template with the prediction data
        return render_template('results.html', crops_data=crops_data)
    
    return redirect(url_for('index'))