from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model('malaria_model.h5')

if not os.path.exists('uploads'):
    os.makedirs('uploads')

def prepare_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Simple function to check if uploaded image is likely a blood smear
def is_blood_smear_like(img_path):
    img = Image.open(img_path)
    width, height = img.size
    # Blood smear images are mostly square or roundish
    if abs(width - height) > min(width, height) * 0.3:
        return False
    # Check average pixel value (blood smears are usually not too dark or too bright)
    avg = img.convert('L').resize((32, 32)).getdata()
    mean_val = sum(avg) / len(avg)
    # Typical blood smear is not very dark or very bright
    return 80 < mean_val < 230

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ''
    image_url = ''
    if request.method == 'POST':
        f = request.files['image']
        filepath = os.path.join('uploads', f.filename)
        f.save(filepath)

        # New: check if image is likely a blood smear before prediction
        if not is_blood_smear_like(filepath):
            result = "Invalid image: Please upload ONLY blood smear microscope images."
            image_url = "/uploads/" + f.filename
            return render_template('index.html', result=result, image_url=image_url)

        img = prepare_image(filepath)
        prediction = model.predict(img)[0][0]

        # Updated prediction logic based on label mapping {'Parasite': 0, 'Uninfected': 1}
        if prediction < 0.5:
            result = "Parasitized (Malaria Positive)"
        else:
            result = "Uninfected (Malaria Negative)"

        image_url = "/uploads/" + f.filename

    return render_template('index.html', result=result, image_url=image_url)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

